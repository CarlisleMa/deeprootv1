"""Shared utilities for the recovery-style evaluations.

Provides:
- JSONL load/dump helpers
- A rate-limited Gemini call wrapper with structured-output support
- Standard recovery metrics (precision/recall/F1, recall@k, MRR, top-k accuracy)
- Light name-normalization for fuzzy source/compound matching
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from pathlib import Path
from typing import Any, Iterable

from google.genai import types

from src.config import GEMINI_MODEL, make_gemini_client

logger = logging.getLogger(__name__)

# Single eval model, sourced from .env (GEMINI_MODEL).
DEFAULT_FLASH_MODEL = GEMINI_MODEL


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


class TeeStdout:
    """Mirror writes to stdout AND a log file."""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("a", encoding="utf-8", buffering=1)
        import sys as _sys
        self._orig = _sys.stdout
        _sys.stdout = self
        self._sys = _sys
        self.path = log_path

    def write(self, s):
        self._orig.write(s)
        self._file.write(s)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def close(self):
        self.flush()
        self._sys.stdout = self._orig
        self._file.close()


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Gemini caller (shared, rate-limited)
# ---------------------------------------------------------------------------

class GeminiCaller:
    """Minimal rate-limited Gemini client for evals."""

    def __init__(self, model: str | None = None, min_interval_s: float = 1.0,
                 timeout_s: float = 180.0, verbose: bool = True):
        self.client = make_gemini_client()
        self.model = model or DEFAULT_FLASH_MODEL
        self._min_interval = min_interval_s
        self._timeout_ms = int(timeout_s * 1000)
        self._verbose = verbose
        self._last_call = 0.0

    def call(
        self,
        prompt: str,
        schema: dict | None = None,
        system: str | None = None,
        max_retries: int = 3,
    ) -> dict | None:
        if not self.client:
            logger.warning("Gemini client unavailable; returning None.")
            return None
        for attempt in range(max_retries):
            try:
                elapsed = _time.time() - self._last_call
                wait = max(0, self._min_interval - elapsed)
                if wait > 0:
                    _time.sleep(wait)
                self._last_call = _time.time()

                cfg_kwargs: dict[str, Any] = {
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "http_options": types.HttpOptions(timeout=self._timeout_ms),
                }
                if schema:
                    cfg_kwargs["response_schema"] = schema
                if system:
                    cfg_kwargs["system_instruction"] = system

                if self._verbose:
                    print(f"    [gemini call attempt={attempt+1} model={self.model}]", flush=True)
                t0 = _time.time()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                if self._verbose:
                    print(f"    [gemini ok in {_time.time()-t0:.1f}s]", flush=True)
                return json.loads(resp.text)
            except Exception as e:
                err = str(e)
                transient = any(c in err for c in ("429", "500", "502", "503", "504", "DEADLINE_EXCEEDED", "UNAVAILABLE"))
                if transient and attempt < max_retries - 1:
                    wait_s = 5 * (attempt + 1)
                    if self._verbose:
                        print(f"    [gemini transient: {err[:120]}; sleeping {wait_s}s, retry {attempt+2}/{max_retries}]", flush=True)
                    _time.sleep(wait_s)
                    continue
                logger.warning("Gemini call failed: %s", e)
                if self._verbose:
                    print(f"    [gemini FAIL: {err[:200]}]", flush=True)
                return None
        return None


# ---------------------------------------------------------------------------
# Name normalization for fuzzy matching
# ---------------------------------------------------------------------------

_PHARM_PARTS = {
    "pericarpium", "fructus", "radix", "rhizoma", "folium", "flos",
    "semen", "herba", "cortex", "exocarpium", "endocarpium", "mesocarpium",
    "bulbus", "tuber", "lignum", "ramulus", "spica", "stigma", "gummi",
    "resina", "oleum", "succus", "aqua", "pollen",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, drop pharmaceutical part words."""
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", name).lower()
    toks = [t for t in s.split() if t and t not in _PHARM_PARTS]
    return " ".join(toks)


def _levenshtein(a: str, b: str, cap: int = 2) -> int:
    """Tiny Levenshtein with early-out at `cap` for 'is distance <= cap?' use."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
            row_min = min(row_min, cur[j])
        if row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def name_match(a: str, b: str) -> bool:
    """Loose match: equality on normalized names, token overlap >=2,
    or fuzzy fallback for 1-edit typos in one of the remaining tokens
    (covers KG typos like 'Acoti' vs 'Acori').
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if len(ta & tb) >= 2:
        return True
    # Fuzzy: |ta| == |tb| and exactly one mismatched token on each side
    # whose Levenshtein distance is <=1. Catches 'acori graminei' vs
    # 'acoti graminei' (one-letter typo). Restrict to ≥2 chars per token
    # to avoid matching short particles.
    if len(ta) >= 2 and len(tb) >= 2 and len(ta & tb) >= 1:
        diff_a = ta - tb
        diff_b = tb - ta
        if len(diff_a) == 1 and len(diff_b) == 1:
            xa = next(iter(diff_a))
            xb = next(iter(diff_b))
            if min(len(xa), len(xb)) >= 4 and _levenshtein(xa, xb, cap=1) <= 1:
                return True
    return False


def match_in_set(query: str, gold: Iterable[str]) -> str | None:
    """Return the gold entry that matches query, else None."""
    for g in gold:
        if name_match(query, g):
            return g
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf(predicted: Iterable[str], gold: Iterable[str]) -> dict:
    """Precision/recall/F1 over name-matched sets."""
    pred = list(predicted)
    gold_l = list(gold)
    matched_gold: set[str] = set()
    tp = 0
    for p in pred:
        m = match_in_set(p, [g for g in gold_l if g not in matched_gold])
        if m:
            tp += 1
            matched_gold.add(m)
    fp = len(pred) - tp
    fn = len(gold_l) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def recall_at_k(ranked: list[str], gold: Iterable[str], ks: list[int]) -> dict:
    gold_l = list(gold)
    if not gold_l:
        return {f"recall@{k}": 0.0 for k in ks}
    out = {}
    for k in ks:
        topk = ranked[:k]
        hits = sum(1 for g in gold_l if match_in_set(g, topk))
        out[f"recall@{k}"] = hits / len(gold_l)
    return out


def mrr(ranked: list[str], gold: Iterable[str]) -> float:
    gold_l = list(gold)
    for i, r in enumerate(ranked, 1):
        if match_in_set(r, gold_l):
            return 1.0 / i
    return 0.0


def top1_acc(ranked: list[str], gold: Iterable[str]) -> float:
    if not ranked:
        return 0.0
    return 1.0 if match_in_set(ranked[0], gold) else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns 0.0 on degenerate input."""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    n = len(xs)

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        for i, idx in enumerate(order, 1):
            r[idx] = i
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def plot_bar_compare(metrics_pairs: list[tuple[str, float, float]],
                     title: str, out_path: Path,
                     labels: tuple[str, str] = ("KG arm", "Baseline")) -> None:
    """Side-by-side bar chart for KG vs baseline metrics.

    metrics_pairs: list of (metric_label, kg_value, baseline_value).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [plot skipped: matplotlib not installed]", flush=True)
        return
    if not metrics_pairs:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [m[0] for m in metrics_pairs]
    kg_vals = [m[1] for m in metrics_pairs]
    base_vals = [m[2] for m in metrics_pairs]

    x = list(range(len(names)))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    ax.bar([i - w / 2 for i in x], kg_vals, w, label=labels[0], color="#3b6ea5")
    ax.bar([i + w / 2 for i in x], base_vals, w, label=labels[1], color="#c45a5a")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, max(1.0, max(kg_vals + base_vals) * 1.15) if (kg_vals + base_vals) else 1.0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot saved: {out_path}]", flush=True)


def plot_box_compare(metric_pairs: list[tuple[str, list[float], list[float]]],
                     title: str, out_path: Path,
                     labels: tuple[str, str] = ("DeepRoot Discovery", "LLM Baseline")) -> None:
    """Side-by-side boxplots per metric. Outliers shown as dots. Mean shown
    as a white-filled circle inside each box.

    metric_pairs: list of (label, kg_values, base_values).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [plot skipped: matplotlib not installed]", flush=True)
        return
    if not metric_pairs:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(metric_pairs)
    fig, ax = plt.subplots(figsize=(3, 4))
    positions, data, colors = [], [], []
    tick_pos, tick_labels = [], []
    group_stride = 2   # smaller = tighter spacing between metric groups
    for i, (lab, kg, base) in enumerate(metric_pairs):
        p_kg, p_base = i * group_stride + 1, i * group_stride + 1.8
        positions.extend([p_kg, p_base])
        data.extend([kg or [0.0], base or [0.0]])
        colors.extend(["#495F24", "#c45a5a"])
        tick_pos.append((p_kg + p_base) / 2)
        tick_labels.append(lab)
    
    # tick_labels already built from metric_pairs labels above

    bp = ax.boxplot(
        data, positions=positions, widths=0.65, patch_artist=True,
        flierprops=dict(marker="o", markersize=4, markerfacecolor="black",
                        markeredgecolor="black", alpha=0.7),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # White-circle mean marker.
    for pos, vals in zip(positions, data):
        if vals:
            mu = sum(vals) / len(vals)
            ax.scatter(pos, mu, s=40, marker="o",
                       facecolor="white", edgecolor="black",
                       linewidth=1, zorder=4)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=4, rotation=30, ha="right")
    ax.set_ylim(min(0.0, min((min(d) for d in data), default=0.0)) - 0.02,
                max(1.0, max((max(d) for d in data), default=1.0)) + 0.05)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(facecolor="#495F24", alpha=0.6, label=labels[0]),
        Patch(facecolor="#c45a5a", alpha=0.6, label=labels[1]),
    ], loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [plot saved: {out_path}]", flush=True)


def plot_scatter(xs: list[float], ys: list[float], title: str, out_path: Path,
                 xlabel: str = "KG Arm", ylabel: str = "LLM Baseline") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [plot skipped: matplotlib not installed]", flush=True)
        return
    if not xs:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys, alpha=0.6, color="#3b6ea5")
    lim = max(max(xs, default=1.0), max(ys, default=1.0), 1.0)
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot saved: {out_path}]", flush=True)


def aggregate_dicts(rows: list[dict]) -> dict:
    """Mean of numeric keys across a list of metric dicts."""
    if not rows:
        return {}
    keys: set[str] = set()
    for r in rows:
        keys.update(k for k, v in r.items() if isinstance(v, (int, float)))
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    out["n"] = len(rows)
    return out
