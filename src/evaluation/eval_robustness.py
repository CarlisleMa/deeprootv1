"""Eval 2: Robustness to Edge Shuffling (Perturbation Test).

For each perturbation level p%:
  1. Clone the primary Neo4j graph to Instance01 (Aura)
  2. Shuffle p% of TARGETS and RELATES_TO endpoints in Instance01 in-place
     (delete edges, recreate with swapped endpoints — Neo4j endpoints are immutable)
  3. Run the full Critic (KG + Gemini) against Instance01 for the first
     N_CORPUS_SOURCES sources from eval_corpus.txt
  4. Store verdict + graph signals per source per perturbation level
  5. Wipe Instance01 before the next trial

Output JSON: {pct: [{source, malady, shuffle, verdict, graph_score, n_closed_loops}]}
Expected result: verdict quality degrades as p increases.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from src.graph.client import GraphClient
from src.graph import queries
from src.agents.phase2.task_a_critic import CriticAgent

load_dotenv()

logger = logging.getLogger(__name__)

PERTURBATION_LEVELS = [0, 20, 40, 60, 80, 100]
BATCH_SIZE = 500
N_CORPUS_SOURCES = 25

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CORPUS_PATH = _REPO_ROOT / "data" / "eval" / "eval_corpus.txt"

# Clone Neo4j Aura instance for in-place perturbation. Credentials are read
# from the environment at runtime (see CLONE_NEO4J_* in .env.example), not at
# import time, so this module can be imported without a clone instance.

NODE_LABELS = [
    "Source",
    "Chemical_Compound",
    "Biological_Target",
    "Modern_Disease",
    "Traditional_Malady",
]

EDGE_TYPES = [
    ("TREATS_TRADITIONALLY", "Source", "Traditional_Malady"),
    ("IS_EXTRACTED_FROM", "Chemical_Compound", "Source"),
    ("TARGETS", "Chemical_Compound", "Biological_Target"),
    ("RELATES_TO", "Biological_Target", "Modern_Disease"),
    ("MAPS_TO", "Traditional_Malady", "Modern_Disease"),
    ("KNOWN_TREATS", "Chemical_Compound", "Modern_Disease"),
]

VERDICT_ORDER = ["UNSUPPORTED", "WEAK", "PLAUSIBLE", "VALIDATED"]
VERDICT_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _load_corpus_sources(n: int = N_CORPUS_SOURCES) -> list[str]:
    """Return the first n source names from eval_corpus.txt."""
    if not EVAL_CORPUS_PATH.exists():
        raise FileNotFoundError(f"Eval corpus not found at {EVAL_CORPUS_PATH}")
    names = []
    for line in EVAL_CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("### SOURCE:"):
            names.append(line.removeprefix("### SOURCE:").strip())
            if len(names) >= n:
                break
    return names


# ---------------------------------------------------------------------------
# Clone helpers
# ---------------------------------------------------------------------------

def _make_clone_client() -> GraphClient:
    try:
        uri = os.environ["CLONE_NEO4J_URI"]
        user = os.environ["CLONE_NEO4J_USER"]
        password = os.environ["CLONE_NEO4J_PASSWORD"]
    except KeyError as exc:
        raise RuntimeError(
            "The edge-perturbation eval requires a separate 'clone' Neo4j "
            "instance. Set CLONE_NEO4J_URI / CLONE_NEO4J_USER / "
            "CLONE_NEO4J_PASSWORD (see .env.example)."
        ) from exc
    database = os.environ.get("CLONE_NEO4J_DATABASE", user)
    return GraphClient(uri=uri, user=user, password=password, database=database)


def _batch_write(client: GraphClient, query: str, items: list, batch_size: int = BATCH_SIZE) -> None:
    for i in range(0, len(items), batch_size):
        client.run_write(query, {"batch": items[i : i + batch_size]})


def _ensure_indexes(clone: GraphClient) -> None:
    for label in NODE_LABELS:
        try:
            clone.run_write(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)")
        except Exception as exc:
            logger.debug("Index skipped for %s: %s", label, exc)


def _clone_primary_to_instance01(primary: GraphClient, clone: GraphClient) -> None:
    for label in NODE_LABELS:
        rows = primary.run(f"MATCH (n:{label}) RETURN properties(n) AS props")
        if not rows:
            continue
        _batch_write(
            clone,
            f"UNWIND $batch AS props CREATE (n:{label}) SET n = props",
            [r["props"] for r in rows],
        )
        logger.info("  cloned %d %s nodes", len(rows), label)

    for rel_type, from_label, to_label in EDGE_TYPES:
        rows = primary.run(
            f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) "
            f"RETURN a.name AS from_name, b.name AS to_name, properties(r) AS props"
        )
        if not rows:
            continue
        _batch_write(
            clone,
            f"UNWIND $batch AS e "
            f"MATCH (a:{from_label} {{name: e.from_name}}), (b:{to_label} {{name: e.to_name}}) "
            f"CREATE (a)-[r:{rel_type}]->(b) SET r = e.props",
            rows,
        )
        logger.info("  cloned %d %s edges", len(rows), rel_type)


def _wipe_clone(clone: GraphClient) -> None:
    clone.run_write("MATCH (n) DETACH DELETE n")
    logger.info("  wiped Instance01")


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def _shuffle_edges(edges: list[dict], to_key: str, pct: float, rng: random.Random) -> list[dict]:
    result = [dict(e) for e in edges]
    if pct == 0 or len(result) < 2:
        return result

    n_swap = int(len(result) * pct / 100.0)
    if n_swap % 2 != 0:
        n_swap -= 1

    indices = list(range(len(result)))
    rng.shuffle(indices)
    swap_idx = indices[:n_swap]

    for i in range(0, len(swap_idx) - 1, 2):
        a, b = swap_idx[i], swap_idx[i + 1]
        result[a][to_key], result[b][to_key] = result[b][to_key], result[a][to_key]

    return result


def _perturb_clone(clone: GraphClient, pct: float, rng: random.Random) -> None:
    # TARGETS
    targets = clone.run(
        "MATCH (c:Chemical_Compound)-[r:TARGETS]->(t:Biological_Target) "
        "RETURN c.name AS from_name, t.name AS to_name, properties(r) AS props"
    )
    shuffled_targets = _shuffle_edges(targets, "to_name", pct, rng)
    clone.run_write("MATCH ()-[r:TARGETS]->() DELETE r")
    if shuffled_targets:
        _batch_write(
            clone,
            "UNWIND $batch AS e "
            "MATCH (c:Chemical_Compound {name: e.from_name}), (t:Biological_Target {name: e.to_name}) "
            "CREATE (c)-[r:TARGETS]->(t) SET r = e.props",
            shuffled_targets,
        )

    # RELATES_TO
    relates = clone.run(
        "MATCH (t:Biological_Target)-[r:RELATES_TO]->(d:Modern_Disease) "
        "RETURN t.name AS from_name, d.name AS to_name, properties(r) AS props"
    )
    shuffled_relates = _shuffle_edges(relates, "to_name", pct, rng)
    clone.run_write("MATCH ()-[r:RELATES_TO]->() DELETE r")
    if shuffled_relates:
        _batch_write(
            clone,
            "UNWIND $batch AS e "
            "MATCH (t:Biological_Target {name: e.from_name}), (d:Modern_Disease {name: e.to_name}) "
            "CREATE (t)-[r:RELATES_TO]->(d) SET r = e.props",
            shuffled_relates,
        )

    logger.info("  perturbed %d TARGETS, %d RELATES_TO at %d%%", len(targets), len(relates), int(pct))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    perturbation_levels: list[int] = PERTURBATION_LEVELS,
    out_dir: str = "src/evaluation/results",
    plot_dir: str = "src/evaluation/plots",
    force: bool = False,
) -> dict:
    """Run Eval 2 and save results + plots.

    If cached results exist and force=False, skips computation and just re-plots.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    out_path = Path(out_dir) / "eval2_results.json"
    if not force and out_path.exists():
        logger.info("Loading cached eval2 results (use force=True to re-run)")
        with open(out_path) as f:
            cached = json.load(f)
        _plot_verdicts(cached["perturbation_levels"], cached["results"], plot_dir)
        _plot_graph_score(cached["perturbation_levels"], cached["results"], plot_dir)
        return cached

    # Load first N_CORPUS_SOURCES sources from eval_corpus
    corpus_sources = set(_load_corpus_sources(N_CORPUS_SOURCES))
    print(f"[Eval 2] Using {len(corpus_sources)} sources from eval_corpus")

    # Get best (source, malady) pair per corpus source from primary
    with GraphClient() as primary:
        rows = primary.run(queries.TASK_A_VALIDATE_CLAIM)

    seen: set[str] = set()
    hypotheses: list[tuple[str, str]] = []
    for r in rows:
        src = r["source"]
        if src in corpus_sources and src not in seen:
            seen.add(src)
            hypotheses.append((src, r["traditional_malady"]))

    print(f"[Eval 2] {len(hypotheses)} hypotheses matched to corpus sources")

    results: dict[str, list[dict[str, Any]]] = {str(p): [] for p in perturbation_levels}
    total_iters = len(perturbation_levels)

    with GraphClient() as primary:
        clone = _make_clone_client()
        clone.connect()
        try:
            print("[Eval 2] Creating indexes on Instance01...")
            _ensure_indexes(clone)
            _wipe_clone(clone)

            for i, pct in enumerate(perturbation_levels, 1):
                print(f"[Eval 2] [{i}/{total_iters}] pct={pct}%")

                print("[Eval 2]   Cloning primary → Instance01...")
                _clone_primary_to_instance01(primary, clone)

                print(f"[Eval 2]   Perturbing {pct}% of edges...")
                rng = random.Random(int(np.random.randint(0, 2**31)))
                _perturb_clone(clone, pct, rng)

                print(f"[Eval 2]   Running critic on {len(hypotheses)} sources...")
                agent = CriticAgent(clone)
                for j, (source, malady) in enumerate(hypotheses, 1):
                    print(f"[Eval 2]     [{j}/{len(hypotheses)}] {source}")
                    result = agent.run(
                        source_name=source,
                        malady_name=malady,
                        skip_llm=False,
                        write_to_graph=False,
                    )
                    llm = result.get("llm_evaluation") or {}
                    results[str(pct)].append({
                        "source": source,
                        "malady": malady,
                        "verdict": llm.get("verdict"),
                        "mechanistic_coherence": llm.get("mechanistic_coherence"),
                        "specificity": llm.get("specificity"),
                        "graph_score": result["signals"]["graph_score"],
                        "n_closed_loops": result["signals"]["n_closed_loops"],
                        "n_gold_tier_paths": result["signals"]["n_gold_tier_paths"],
                    })

                print("[Eval 2]   Wiping Instance01...")
                _wipe_clone(clone)

        finally:
            clone.close()

    out_data = {
        "perturbation_levels": perturbation_levels,
        "hypotheses": [{"source": s, "malady": m} for s, m in hypotheses],
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    logger.info("Saved results to %s", out_path)

    _plot_verdicts(perturbation_levels, results, plot_dir)
    _plot_graph_score(perturbation_levels, results, plot_dir)
    return out_data


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_verdicts(
    perturbation_levels: list[int],
    results: dict[str, list[dict]],
    plot_dir: str,
) -> None:
    """Grouped bar chart: verdict distribution per perturbation level."""
    # Count verdicts per pct (aggregate across all shuffles)
    counts: dict[str, dict[str, int]] = {}
    for pct in perturbation_levels:
        pct_str = str(pct)
        counts[pct_str] = {v: 0 for v in VERDICT_ORDER}
        for entry in results.get(pct_str, []):
            v = (entry.get("verdict") or "").upper()
            if v in counts[pct_str]:
                counts[pct_str][v] += 1

    n_levels = len(perturbation_levels)
    n_verdicts = len(VERDICT_ORDER)
    group_width = 0.7
    bar_width = group_width / n_verdicts
    x = np.arange(n_levels)
    offsets = np.linspace(-group_width / 2 + bar_width / 2, group_width / 2 - bar_width / 2, n_verdicts)

    _, ax = plt.subplots(figsize=(12, 5))
    for i, (verdict, color) in enumerate(zip(VERDICT_ORDER, VERDICT_COLORS)):
        vals = [counts[str(p)][verdict] for p in perturbation_levels]
        bars = ax.bar(x + offsets[i], vals, bar_width, color=color, alpha=0.85, label=verdict)
        for bar, h in zip(bars, vals):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(h),
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in perturbation_levels])
    ax.set_xlabel("% Edges Shuffled")
    ax.set_ylabel("Count")
    ax.legend(title="Verdict", loc="upper right", fontsize=9)
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()

    plot_path = Path(plot_dir) / "eval2_verdicts.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved verdict plot to %s", plot_path)


def _plot_graph_score(
    perturbation_levels: list[int],
    results: dict[str, list[dict]],
    plot_dir: str,
) -> None:
    """Line chart: mean graph_score (left axis) + UNSUPPORTED count (right axis, red)."""
    means, stds, unsupported_counts = [], [], []
    for pct in perturbation_levels:
        entries = results.get(str(pct), [])
        scores = [e["graph_score"] for e in entries if e.get("graph_score") is not None]
        means.append(float(np.mean(scores)) if scores else 0.0)
        stds.append(float(np.std(scores, ddof=0)) if len(scores) > 1 else 0.0)
        n_unsupported = sum(1 for e in entries if (e.get("verdict") or "").upper() == "UNSUPPORTED")
        unsupported_counts.append(100.0 * n_unsupported / len(entries) if entries else 0.0)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(perturbation_levels, means, marker="o", linewidth=2, color="#4C72B0", label="Mean graph score")
    ax1.fill_between(
        perturbation_levels,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.3, color="#4C72B0", label="±std dev",
    )
    ax1.set_xlabel("% Edges Shuffled")
    ax1.set_ylabel("Mean Graph Score", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xticks(perturbation_levels)

    ax2 = ax1.twinx()
    ax2.plot(perturbation_levels, unsupported_counts, marker="s", linewidth=2,
             color="#d62728", linestyle="--", label="% Unsupported")
    ax2.set_ylabel("% Unsupported Claims", color="#d62728")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.yaxis.get_major_locator().set_params(integer=True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    plot_path = Path(plot_dir) / "eval2_graph_score.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved graph score plot to %s", plot_path)


def plot_only(
    out_dir: str = "src/evaluation/results",
    plot_dir: str = "src/evaluation/plots",
) -> None:
    """Re-plot from cached eval2_results.json without re-running the eval."""
    out_path = Path(out_dir) / "eval2_results.json"
    if not out_path.exists():
        raise FileNotFoundError(f"No cached results at {out_path}. Run eval_robustness.run() first.")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path) as f:
        data = json.load(f)
    _plot_verdicts(data["perturbation_levels"], data["results"], plot_dir)
    _plot_graph_score(data["perturbation_levels"], data["results"], plot_dir)
    print(f"[Eval 2] Plots saved to {plot_dir}/")
