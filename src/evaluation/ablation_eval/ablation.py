"""Eval 2: Robustness to Edge Shuffling (Perturbation Test).

For each perturbation level p%:
  1. Clone the primary Neo4j graph to Instance01 (Aura)
  2. Shuffle p% of TARGETS and RELATES_TO endpoints in Instance01 in-place
  3. Run TaskAValidator + CriticAgent against Instance01 for the corpus pairs
  4. Capture the Critic's biological_plausibility (0..1) per pair
  5. Wipe Instance01 before the next trial

Baseline arm (graph-free, run once): hand the same (source, malady) to Gemini
without any KG context and ask for a 0..1 plausibility. Plotted as horizontal
reference — the Critic should regress toward this floor as p → 100%.

Output JSON: {pct: [{source, malady, plausibility, ...}], baseline: [...]}
Expected: mean plausibility decays from p=0 (KG-grounded) toward the baseline
mean as graph signal is shuffled away.
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

from src.config import GEMINI_MODEL
from src.graph.client import GraphClient
from src.graph import queries
from src.agents.phase2.task_a_validator import TaskAValidator
from src.agents.phase2.task_a_critic import CriticAgent

load_dotenv()

logger = logging.getLogger(__name__)

PERTURBATION_LEVELS = [0, 20, 40, 60, 80, 100]
BATCH_SIZE = 500
N_CORPUS_SOURCES = 50

EVAL_CORPUS_PATH = Path("data/historical_corpus/eval_corpus.txt")

CLONE_URI = os.environ["CLONE_NEO4J_URI"]
CLONE_USER = os.environ["CLONE_NEO4J_USER"]
CLONE_PASSWORD = os.environ["CLONE_NEO4J_PASSWORD"]
CLONE_DATABASE = os.environ.get("CLONE_NEO4J_DATABASE", CLONE_USER)

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


# ---------------------------------------------------------------------------
# Corpus + hypothesis selection
# ---------------------------------------------------------------------------

def _load_corpus_sources(n: int = N_CORPUS_SOURCES) -> list[str]:
    if not EVAL_CORPUS_PATH.exists():
        raise FileNotFoundError(f"Eval corpus not found at {EVAL_CORPUS_PATH}")
    names = []
    for line in EVAL_CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("### SOURCE:"):
            names.append(line.removeprefix("### SOURCE:").strip())
            if len(names) >= n:
                break
    return names


def _select_hypotheses(primary: GraphClient, corpus_sources: set[str]) -> list[tuple[str, str]]:
    """For each corpus source, pick the malady with the strongest
    converging mechanistic evidence (first row from TASK_A_VALIDATE_CLAIM).

    Corpus uses common names ('Dipsacus root') while the KG uses Latin
    binomials. Map each corpus name to a KG source via fuzzy name_match
    against the full active source list, then pick the top malady."""
    from scripts._eval_utils import name_match

    kg_rows = primary.run(
        "MATCH (s:Source) WHERE coalesce(s.archived,false)=false RETURN s.name AS name"
    )
    kg_sources = [r["name"] for r in kg_rows if r.get("name")]

    matched_kg: set[str] = set()
    for cs in corpus_sources:
        for kg in kg_sources:
            if name_match(cs, kg):
                matched_kg.add(kg)
                break

    rows = primary.run(queries.TASK_A_VALIDATE_CLAIM)
    seen: set[str] = set()
    hypotheses: list[tuple[str, str]] = []
    for r in rows:
        src = r["source"]
        if src in matched_kg and src not in seen:
            seen.add(src)
            hypotheses.append((src, r["traditional_malady"]))
    return hypotheses


# ---------------------------------------------------------------------------
# Clone helpers
# ---------------------------------------------------------------------------

def _make_clone_client() -> GraphClient:
    return GraphClient(uri=CLONE_URI, user=CLONE_USER, password=CLONE_PASSWORD, database=CLONE_DATABASE)


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
# Perturbation
# ---------------------------------------------------------------------------

def _shuffle_edges(edges: list[dict], to_key: str, pct: float, rng: random.Random) -> list[dict]:
    """Fisher-Yates derangement over `to_key` for a fraction pct of edges.

    Pick pct% of edges, shuffle their to_key values among themselves
    (random permutation, not pair-swap). This redistributes endpoints
    without changing the multiset of targets — strictly stronger than
    pairwise swap for breaking loop closure on dense subgraphs."""
    result = [dict(e) for e in edges]
    n = len(result)
    if pct <= 0 or n < 2:
        return result

    n_perm = int(n * pct / 100.0)
    if n_perm < 2:
        return result

    indices = list(range(n))
    rng.shuffle(indices)
    perm_idx = indices[:n_perm]

    to_values = [result[i][to_key] for i in perm_idx]
    rng.shuffle(to_values)
    for i, idx in enumerate(perm_idx):
        result[idx][to_key] = to_values[i]

    return result


_SHUFFLE_EDGE_TYPES = [
    # (rel_type, from_label, to_label) — `to_name` (= disease/target) is shuffled.
    ("TARGETS", "Chemical_Compound", "Biological_Target"),
    ("RELATES_TO", "Biological_Target", "Modern_Disease"),
    ("KNOWN_TREATS", "Chemical_Compound", "Modern_Disease"),
    ("MAPS_TO", "Traditional_Malady", "Modern_Disease"),
]


def _perturb_clone(clone: GraphClient, pct: float, rng: random.Random) -> None:
    counts = {}
    for rel_type, from_label, to_label in _SHUFFLE_EDGE_TYPES:
        edges = clone.run(
            f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) "
            f"RETURN a.name AS from_name, b.name AS to_name, properties(r) AS props"
        )
        shuffled = _shuffle_edges(edges, "to_name", pct, rng)
        clone.run_write(f"MATCH ()-[r:{rel_type}]->() DELETE r")
        if shuffled:
            _batch_write(
                clone,
                f"UNWIND $batch AS e "
                f"MATCH (a:{from_label} {{name: e.from_name}}), (b:{to_label} {{name: e.to_name}}) "
                f"CREATE (a)-[r:{rel_type}]->(b) SET r = e.props",
                shuffled,
            )
        counts[rel_type] = len(edges)

    summary = ", ".join(f"{n} {t}" for t, n in counts.items())
    logger.info("  perturbed at %d%%: %s", int(pct), summary)
    print(f"[Eval 2]   perturbed at {int(pct)}%: {summary}")


# ---------------------------------------------------------------------------
# Critic arm (KG-grounded) — run validator then critic on the clone
# ---------------------------------------------------------------------------

def _run_critic_arm(
    clone: GraphClient,
    hypotheses: list[tuple[str, str]],
    critic_model: str | None,
) -> list[dict[str, Any]]:
    validator = TaskAValidator(clone)
    critic = CriticAgent(
        clone,
        gemini_model=critic_model or GEMINI_MODEL,
        skip_non_actionable=False,
    )
    print(f"[Eval 2]   critic model: {critic._model}")

    pass1: list[dict] = []
    by_pair: dict[tuple[str, str], dict] = {}
    for j, (source, malady) in enumerate(hypotheses, 1):
        try:
            res = validator.run(
                source_name=source, malady_name=malady,
                write_graph=False, keep_top_paths=20,
            )
        except Exception as e:
            logger.warning("validator failed for (%s, %s): %s", source, malady, e)
            continue
        for v in res.get("verdicts") or []:
            pass1.append(v)
            by_pair[(v.get("source", ""), v.get("malady", ""))] = v

    print(f"[Eval 2]   validator produced {len(pass1)} verdicts")

    crit = critic.run(verdicts=pass1, write_graph=False)
    crit_results = crit.get("results") or []

    out: list[dict[str, Any]] = []
    for cv in crit_results:
        out.append({
            "source": cv.get("source"),
            "malady": cv.get("malady"),
            "plausibility": float(cv.get("biological_plausibility") or 0.0),
            "evidence_coherence": float(cv.get("evidence_coherence") or 0.0),
            "verdict": cv.get("verdict"),
            "pass1_verdict": cv.get("pass1_verdict"),
            "rationale": cv.get("rationale") or "",
            "key_evidence": cv.get("key_evidence") or [],
            "concerns": cv.get("concerns") or [],
            "requires_human_review": bool(cv.get("requires_human_review") or False),
            "skipped": bool(cv.get("skipped")),
            "error": cv.get("error"),
        })
    return out


# ---------------------------------------------------------------------------
# Baseline arm (graph-free) — prompt patterned after eval_lead_recovery.py
# ---------------------------------------------------------------------------

BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "plausibility": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["plausibility"],
}


def _baseline_prompt(source: str, malady: str) -> str:
    return (
        "You are evaluating a claim from historical Chinese herbal medicine. "
        "The claim asserts that the medicinal source below treats the "
        "traditional malady below. Without access to any structured knowledge "
        "graph, judge the biological plausibility of the claim on the basis "
        "of background knowledge alone (chemistry, pharmacology, ethnobotany).\n\n"
        f"SOURCE: {source}\n"
        f"TRADITIONAL_MALADY: {malady}\n\n"
        "Return a single JSON object with:\n"
        "  plausibility ∈ [0.0, 1.0]   0=incoherent, 1=biologically obvious\n"
        "  reasoning: one sentence explaining your score.\n"
    )


def _run_baseline_arm(hypotheses: list[tuple[str, str]]) -> list[dict[str, Any]]:
    from scripts._eval_utils import GeminiCaller  # local import; eval-only dep
    caller = GeminiCaller(model=GEMINI_MODEL, verbose=False)
    print(f"[Eval 2] baseline arm: {len(hypotheses)} pairs (model={GEMINI_MODEL})")
    out: list[dict[str, Any]] = []
    for j, (source, malady) in enumerate(hypotheses, 1):
        raw = caller.call(_baseline_prompt(source, malady), schema=BASELINE_SCHEMA) or {}
        p = raw.get("plausibility")
        try:
            p = max(0.0, min(1.0, float(p)))
        except (TypeError, ValueError):
            p = 0.0
        out.append({
            "source": source,
            "malady": malady,
            "plausibility": p,
            "reasoning": raw.get("reasoning", ""),
        })
        if j % 10 == 0 or j == len(hypotheses):
            print(f"[Eval 2]   baseline {j}/{len(hypotheses)}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    perturbation_levels: list[int] = PERTURBATION_LEVELS,
    n_sources: int = N_CORPUS_SOURCES,
    limit: int | None = None,
    out_dir: str = "data/eval2",
    plot_dir: str = "data/eval2",
    critic_model: str | None = None,
    force: bool = False,
    skip_baseline: bool = False,
) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    out_path = Path(out_dir) / "eval2_results.json"
    if not force and out_path.exists():
        logger.info("Loading cached eval2 results (use force=True to re-run)")
        with open(out_path) as f:
            cached = json.load(f)
        _plot_plausibility(cached, plot_dir)
        return cached

    corpus_sources = set(_load_corpus_sources(n_sources))
    print(f"[Eval 2] Using {len(corpus_sources)} corpus sources")

    with GraphClient() as primary:
        hypotheses = _select_hypotheses(primary, corpus_sources)
    print(f"[Eval 2] {len(hypotheses)} (source, malady) hypotheses matched")
    if limit is not None and limit > 0:
        hypotheses = hypotheses[:limit]
        print(f"[Eval 2] limited to {len(hypotheses)} hypotheses (--limit)")

    # Resume from checkpoint if present.
    ckpt_path = Path(out_dir) / "eval2_results.checkpoint.json"
    baseline: list[dict[str, Any]] = []
    results: dict[str, list[dict[str, Any]]] = {str(p): [] for p in perturbation_levels}
    if ckpt_path.exists():
        try:
            ck = json.load(open(ckpt_path))
            baseline = ck.get("baseline") or []
            for k, v in (ck.get("results") or {}).items():
                if k in results and v:
                    results[k] = v
            done = [k for k, v in results.items() if v]
            print(f"[Eval 2] resumed checkpoint: baseline={len(baseline)}, done levels={done}")
        except Exception as e:
            print(f"[Eval 2] checkpoint load failed ({e}); starting fresh")

    def _save_ckpt():
        out_data = {
            "perturbation_levels": perturbation_levels,
            "hypotheses": [{"source": s, "malady": m} for s, m in hypotheses],
            "results": results,
            "baseline": baseline,
        }
        ckpt_path.write_text(json.dumps(out_data, indent=2))

    if not skip_baseline and not baseline:
        baseline = _run_baseline_arm(hypotheses)
        _save_ckpt()

    with GraphClient() as primary:
        clone = _make_clone_client()
        clone.connect()
        try:
            print("[Eval 2] Creating indexes on Instance01...")
            _ensure_indexes(clone)
            _wipe_clone(clone)

            for i, pct in enumerate(perturbation_levels, 1):
                if results[str(pct)]:
                    print(f"[Eval 2] [{i}/{len(perturbation_levels)}] pct={pct}% — already in checkpoint, skipping")
                    continue
                print(f"[Eval 2] [{i}/{len(perturbation_levels)}] pct={pct}%")
                print("[Eval 2]   Cloning primary -> Instance01...")
                _clone_primary_to_instance01(primary, clone)

                print(f"[Eval 2]   Perturbing {pct}% of TARGETS+RELATES_TO...")
                rng = random.Random(int(np.random.randint(0, 2**31)))
                _perturb_clone(clone, pct, rng)

                print(f"[Eval 2]   KG critic arm on {len(hypotheses)} pairs...")
                results[str(pct)] = _run_critic_arm(clone, hypotheses, critic_model)

                _save_ckpt()
                print(f"[Eval 2]   Checkpoint saved -> {ckpt_path}")

                print("[Eval 2]   Wiping Instance01...")
                _wipe_clone(clone)
        finally:
            clone.close()

    out_data = {
        "perturbation_levels": perturbation_levels,
        "hypotheses": [{"source": s, "malady": m} for s, m in hypotheses],
        "results": results,
        "baseline": baseline,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    logger.info("Saved results to %s", out_path)

    _plot_plausibility(out_data, plot_dir)
    return out_data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _scored_entries(entries: list[dict]) -> list[float]:
    return [
        float(e["plausibility"])
        for e in entries
        if not e.get("skipped") and not e.get("error")
        and e.get("plausibility") is not None
    ]


def _plot_plausibility(data: dict, plot_dir: str) -> None:
    perturbation_levels = data["perturbation_levels"]
    results = data["results"]
    baseline = data.get("baseline") or []

    means, stds, ns = [], [], []
    for pct in perturbation_levels:
        vals = _scored_entries(results.get(str(pct), []))
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0)
        ns.append(len(vals))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(perturbation_levels, means, marker="o", linewidth=2, color="#495F24",
            label="DeepRoot Critic Agent")
    ax.fill_between(
        perturbation_levels,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25, color="#495F24",
    )

    if baseline:
        bvals = [float(b["plausibility"]) for b in baseline if b.get("plausibility") is not None]
        if bvals:
            bmean = float(np.mean(bvals))
            bstd = float(np.std(bvals, ddof=0))
            ax.axhline(bmean, linestyle="--", color="#c45a5a",
                       label=f"LLM Baseline")

    ax.set_xlabel("% Edges Shuffled")
    ax.set_ylabel("Plausibility Score")
    ax.set_xticks(perturbation_levels)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 100)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = Path(plot_dir) / "eval2_plausibility.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved plausibility plot to %s", plot_path)


def plot_only(
    out_dir: str = "data/eval2",
    plot_dir: str = "data/eval2",
) -> None:
    out_path = Path(out_dir) / "eval2_results.json"
    if not out_path.exists():
        raise FileNotFoundError(f"No cached results at {out_path}. Run eval2_robustness.run() first.")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path) as f:
        data = json.load(f)
    _plot_plausibility(data, plot_dir)
    print(f"[Eval 2] Plot saved to {plot_dir}/eval2_plausibility.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=str, default=None,
                   help="Comma-separated perturbation pcts, e.g. 0,50,100")
    p.add_argument("--n", type=int, default=N_CORPUS_SOURCES,
                   help="Corpus source headers to read (upstream cap).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on (source, malady) hypotheses tested per pct level.")
    p.add_argument("--critic-model", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    if args.plot_only:
        plot_only()
    else:
        levels = [int(x) for x in args.levels.split(",")] if args.levels else PERTURBATION_LEVELS
        run(perturbation_levels=levels, n_sources=args.n, limit=args.limit,
            critic_model=args.critic_model, force=args.force,
            skip_baseline=args.skip_baseline)
