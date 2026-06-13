"""Build a multi-source recovery eval corpus.

Output rows are *regions* (>=5 paragraphs, >=2 KG sources mentioned). Each
row carries gold sources, paragraph→source map, and KG-derived gold compounds
and maladies for each source. Used by the two recovery evals.

Usage:
    python scripts/build_recovery_eval_corpus.py [--target N] [--out PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._eval_utils import (
    DEFAULT_FLASH_MODEL,
    GeminiCaller,
    write_jsonl,
)
from scripts.build_eval_corpus import (
    EXTRA_KEYWORDS,
    _genus_from_source_name,
    _normalize_full_text,
    _search_keywords_for_genus,
)
from src.graph.client import GraphClient


def _search_keywords_for_source(source_name: str) -> list[str]:
    """Pharma-Latin tolerant keywords from every meaningful word in the name.

    Drops part terms (Radix, Fructus, ...), normalizes Latin genitive
    suffixes, and folds in EXTRA_KEYWORDS aliases. Used by the region
    tagger so 'Pericarpium Citri Reticulatae' still matches passages that
    say 'Citrus reticulata'.
    """
    pharm_parts = {
        "pericarpium", "fructus", "radix", "rhizoma", "folium", "flos",
        "semen", "herba", "cortex", "exocarpium", "endocarpium",
        "mesocarpium", "bulbus", "tuber", "lignum", "ramulus", "spica",
        "stigma", "gummi", "resina", "oleum", "succus", "aqua", "pollen",
    }
    genitive_fixes = {
        "zanthoxyli": "zanthoxylum", "cnidii": "cnidium", "citri": "citrus",
        "ephedrae": "ephedra", "glycyrrhizae": "glycyrrhiza",
        "astragali": "astragalus", "rhei": "rheum", "puerariae": "pueraria",
        "coptidis": "coptis", "scutellariae": "scutellaria",
        "phellodendri": "phellodendron", "salviae": "salvia",
        "curcumae": "curcuma", "zingiberis": "zingiber",
        "cinnamomi": "cinnamomum", "angelicae": "angelica",
        "ligustici": "ligusticum", "rehmanniae": "rehmannia",
        "atractylodis": "atractylodes", "alismatis": "alisma",
        "sophorae": "sophora", "fritillariae": "fritillaria",
        "anemarrhenae": "anemarrhena", "artemisiae": "artemisia",
    }
    words = [w.lower() for w in re.split(r"\s+", source_name.strip()) if w]
    kws: set[str] = set()
    for w in words:
        if w in pharm_parts:
            continue
        kws.add(w)
        nom = genitive_fixes.get(w)
        if nom:
            kws.add(nom)
            kws.update(EXTRA_KEYWORDS.get(nom, []))
        for suf in ("i", "ae", "is"):
            if w.endswith(suf) and len(w) > len(suf) + 3:
                kws.add(w[: -len(suf)])
        kws.update(EXTRA_KEYWORDS.get(w, []))
    if not kws:
        kws.update(_search_keywords_for_genus(_genus_from_source_name(source_name)))
    return [k for k in kws if len(k) >= 4]


CORPUS_FULL = Path("data/historical_corpus/shen_nong_ben_cao_jing_full.txt")
OUT_DEFAULT = Path("data/eval/recovery_corpus.jsonl")

WINDOW_SIZE = 1
WINDOW_STRIDE = 1
MIN_REGION_CHARS = 200
MIN_PARAS_PER_SOURCE = 1
MIN_SOURCES_PER_REGION = 2

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "kept_sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Subset of the candidate sources actually discussed (not merely cross-referenced).",
        }
    },
    "required": ["kept_sources"],
}


# ---------------------------------------------------------------------------
# KG queries
# ---------------------------------------------------------------------------

SOURCES_WITH_COMPOUNDS_QUERY = """
MATCH (c:Chemical_Compound)-[r:IS_EXTRACTED_FROM]->(s:Source)
WHERE coalesce(s.archived,false)=false
  AND coalesce(c.archived,false)=false
  AND coalesce(r.archived,false)=false
RETURN s.name AS source,
       collect(DISTINCT c.name)[..50] AS compounds
ORDER BY source
"""


# All Source nodes — including those without IS_EXTRACTED_FROM compounds
# (mineral / animal materials etc). Used to broaden the keyword tagger so
# the distractor pool can include any source mentioned in the corpus,
# closed-loop or not.
ALL_SOURCES_QUERY = """
MATCH (s:Source)
WHERE coalesce(s.archived,false)=false
RETURN s.name AS source
ORDER BY source
"""


# Sources with at least one closed-loop mechanism chain
# (Source <- Compound -> Target -> Disease, with Source-TREATS-Malady-MAPS_TO-Disease).
# Used to filter the eval corpus: regions about sources lacking ANY full chain
# (e.g. Cinnabar) make the KG arm look bad even when the absence is real.
SOURCES_WITH_CLOSED_LOOP_QUERY = """
MATCH (s:Source)-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady)-[:MAPS_TO]->(d:Modern_Disease)
WHERE coalesce(s.archived,false)=false
  AND coalesce(m.archived,false)=false
  AND coalesce(d.archived,false)=false
MATCH (c:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s)
MATCH (c)-[:TARGETS]->(t:Biological_Target)-[:RELATES_TO]->(d)
WHERE coalesce(c.archived,false)=false
  AND coalesce(t.archived,false)=false
RETURN DISTINCT s.name AS source
ORDER BY source
"""

MALADIES_FOR_SOURCE_QUERY = """
MATCH (s:Source {name: $source})-[r:TREATS_TRADITIONALLY]->(m:Traditional_Malady)
WHERE coalesce(s.archived,false)=false
  AND coalesce(m.archived,false)=false
  AND coalesce(r.archived,false)=false
RETURN m.name AS malady
"""


# Gold compounds = compounds that close a mechanism loop for THIS source:
# Compound -> TARGETS -> Target -> RELATES_TO -> Disease where the source
# also TREATS_TRADITIONALLY -> Malady -> MAPS_TO -> SAME Disease.
# These are the compounds the KG actually validates as therapeutic for the
# source. Much smaller than IS_EXTRACTED_FROM (which can be 50+ unrelated
# constituents) -> recall@k metric becomes meaningful.
CLOSED_LOOP_COMPOUNDS_QUERY = """
MATCH (s:Source {name: $source})-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady)-[:MAPS_TO]->(d:Modern_Disease)
WHERE coalesce(s.archived,false)=false
  AND coalesce(m.archived,false)=false
  AND coalesce(d.archived,false)=false
MATCH (c:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s)
MATCH (c)-[:TARGETS]->(t:Biological_Target)-[:RELATES_TO]->(d)
WHERE coalesce(c.archived,false)=false
  AND coalesce(t.archived,false)=false
RETURN DISTINCT c.name AS compound
ORDER BY compound
"""


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    text = _normalize_full_text(text)
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip() and len(p.strip()) > 50]


def _canonicalize_sources(src_compounds: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Merge case-insensitive duplicate source names.

    Returns (canonical_src_compounds, alias_to_canonical).
    Canonical = the variant with the most compounds (ties: lexically first).
    """
    groups: dict[str, list[str]] = {}
    for s in src_compounds:
        groups.setdefault(s.lower().strip(), []).append(s)
    canonical: dict[str, list[str]] = {}
    alias_map: dict[str, str] = {}
    for _, variants in groups.items():
        chosen = max(variants, key=lambda v: (len(src_compounds.get(v, [])), -ord(v[0]) if v else 0))
        merged: set[str] = set()
        for v in variants:
            merged.update(src_compounds.get(v, []))
            alias_map[v] = chosen
        canonical[chosen] = sorted(merged)
    return canonical, alias_map


def _build_keyword_index(sources: list[str]) -> dict[str, list[str]]:
    """source name → list of search keywords."""
    return {s: _search_keywords_for_source(s) for s in sources}


_KW_PATTERN_CACHE: dict[str, "re.Pattern"] = {}


def _kw_regex(kw: str) -> "re.Pattern":
    if kw not in _KW_PATTERN_CACHE:
        _KW_PATTERN_CACHE[kw] = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
    return _KW_PATTERN_CACHE[kw]


def _tag_paragraph(para: str, kw_index: dict[str, list[str]]) -> list[str]:
    """Tag paragraph with sources whose keywords appear as whole words."""
    hits: list[str] = []
    for src, kws in kw_index.items():
        if any(_kw_regex(k).search(para) for k in kws):
            hits.append(src)
    return hits


_HERB_ENTRY_SIGNALS = re.compile(
    r"\b("
    r"radix|rhizoma|fructus|pericarpium|herba|folium|cortex|semen|flos|"
    r"bulbus|exocarpium|ramulus|spica|stigma|"
    r"is acrid|is bitter|is sweet|is sour|is salty|is warm|is cold|is neutral|"
    r"mainly treats|protracted taking|grows in|grows on"
    r")\b",
    re.IGNORECASE,
)


# Markers of therapeutic content — what makes a passage useful for our eval.
# A passage with any of these is making a (source, malady) treatment claim,
# not a (source, source) compatibility claim.
_THERAPEUTIC_SIGNALS = re.compile(
    r"\b("
    r"mainly treats|treats hundreds|treats|cures|indicated for|"
    r"is acrid|is bitter|is sweet|is sour|is salty|"
    r"is warm|is cold|is cool|is neutral|is balanced|is hot|nontoxic|"
    r"disinhibits|disperses|tonifies|nourishes|relieves|brightens|"
    r"warms the|clears (heat|the)|harmonizes|invigorates|"
    r"protracted taking|long-term taking|lightens the body|"
    r"calms the spirit|quiets the spirit|stops bleeding|stops pain"
    r")\b",
    re.IGNORECASE,
)


# Markers of pure compatibility prose (Seven Affinities — 七情). These passages
# describe source-source interactions, NOT therapeutic claims. Our eval extracts
# (source, malady) pairs; compat-only passages contain no maladies → KG arm
# extracts nothing → metrics destroyed.
_COMPAT_SIGNALS = re.compile(
    r"\b("
    r"envoy of|envoys of|envoys are|"
    r"averse to|clashes with|fears|kills the toxin|kills the toxins|"
    r"will become better if it acquires|"
    r"is the envoy of|is the assistant of"
    r")\b",
    re.IGNORECASE,
)


def _is_herb_region(text: str) -> bool:
    """Reject front matter / preface / index by requiring herb-entry markers."""
    return len(_HERB_ENTRY_SIGNALS.findall(text)) >= 2


def _is_compat_only(text: str) -> bool:
    """True if passage is dominated by compatibility prose with no treatment claims.

    Drop these — they make (source, source) interaction claims (Seven Affinities)
    rather than (source, malady) treatment claims. The eval extractor finds no
    maladies, KG arm gets 0 pairs, metrics collapse.
    """
    n_compat = len(_COMPAT_SIGNALS.findall(text))
    n_therap = len(_THERAPEUTIC_SIGNALS.findall(text))
    if n_compat == 0:
        return False
    # If there's substantially more compat prose than therapeutic prose, drop.
    return n_compat >= 1 and n_therap < max(1, n_compat // 2)


def _has_therapeutic_content(text: str) -> bool:
    """True if passage contains at least one therapeutic-claim marker."""
    return _THERAPEUTIC_SIGNALS.search(text) is not None


def _candidate_regions(paragraphs: list[str], kw_index: dict[str, list[str]]) -> list[dict]:
    """Sliding window over paragraphs; keep windows with >=2 sources."""
    regions = []
    para_tags = [_tag_paragraph(p, kw_index) for p in paragraphs]
    rid = 0
    for start in range(0, len(paragraphs) - WINDOW_SIZE + 1, WINDOW_STRIDE):
        window_paras = paragraphs[start : start + WINDOW_SIZE]
        window_tags = para_tags[start : start + WINDOW_SIZE]
        text = "\n\n".join(window_paras)
        if len(text) < MIN_REGION_CHARS:
            continue
        if not _is_herb_region(text):
            continue
        if _is_compat_only(text):
            continue

        # Count distinct sources, requiring MIN_PARAS_PER_SOURCE evidence
        src_counts: dict[str, int] = {}
        for tags in window_tags:
            for s in tags:
                src_counts[s] = src_counts.get(s, 0) + 1
        kept = [s for s, c in src_counts.items() if c >= MIN_PARAS_PER_SOURCE]
        if len(kept) < MIN_SOURCES_PER_REGION:
            continue

        para_map = {str(i): [s for s in tags if s in kept] for i, tags in enumerate(window_tags)}
        regions.append({
            "region_id": f"rgn_{rid:04d}",
            "text": text,
            "paragraphs": window_paras,
            "candidate_sources": sorted(kept),
            "paragraph_source_map": para_map,
            "_window_start": start,
        })
        rid += 1
    return regions


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------

def _build_verify_prompt(text: str, candidates: list[str]) -> str:
    return (
        "Read this passage from a historical Chinese herbal text. "
        "From the candidate list, return ONLY sources that the passage "
        "DESCRIBES THERAPEUTICALLY — i.e. the passage states what diseases / "
        "symptoms / conditions THIS source treats, or describes its flavor / "
        "nature / medicinal action.\n\n"
        "EXCLUDE every source that:\n"
        "  - appears only as a cross-reference inside another source's entry,\n"
        "  - is named only in a Seven Affinities / compatibility list "
        "(envoy of, averse to, fears, clashes with, kills the toxins of, "
        "becomes better if it acquires),\n"
        "  - appears as a one-word parenthetical translation,\n"
        "  - is mentioned but has no therapeutic indication of its own in "
        "this passage.\n\n"
        "If NONE of the candidates is described therapeutically, return an "
        "empty list.\n\n"
        f"Candidates: {', '.join(candidates)}\n\n"
        f"PASSAGE:\n{text}\n"
    )


def _verify_region(caller: GeminiCaller, region: dict) -> list[str]:
    prompt = _build_verify_prompt(region["text"], region["candidate_sources"])
    raw = caller.call(prompt, schema=VERIFY_SCHEMA)
    if not raw:
        return []
    kept = raw.get("kept_sources", []) if isinstance(raw, dict) else []
    return [s for s in kept if s in region["candidate_sources"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _synthesize_multi_source_regions(
    verified_singles_loop: list[dict],
    verified_singles_distractor: list[dict],
    target_n: int,
    combine_k: int = 3,
    distractor_n: int = 7,
    seen_signatures: set[frozenset[str]] | None = None,
    rid_start: int = 0,
    seed: int = 42,
) -> list[dict]:
    """Build synthetic multi-source eval cases with **diversity-maximizing**
    selection. Each case: K gold + N distractor paragraphs.

    Selection algorithm (greedy, diversity-first):
      1. Maintain a `usage` counter across all already-emitted regions for
         every source (gold or distractor). Sources used more often are
         penalized; least-used sources picked first.
      2. For each new region:
           a. Pick K gold sources by least-usage (closed-loop pool, breaking
              ties randomly with a fixed seed).
           b. Pick N distractor sources from the remaining pool by the same
              least-usage criterion. Distractor source set must not overlap
              the gold set.
           c. Skip the region if the gold-set signature already exists.
           d. Update usage counters.
      3. Stop after target_n regions or when pools are exhausted.

    All paragraphs (gold + distractor) are interleaved by deterministic
    shuffle so gold paragraphs don't always appear at the start.
    """
    import random
    if seen_signatures is None:
        seen_signatures = set()

    by_source_loop: dict[str, list[str]] = {}
    for s in verified_singles_loop:
        by_source_loop.setdefault(s["source"], []).append(s["paragraph"])
    distinct_loop = sorted(by_source_loop.keys())

    by_source_dist: dict[str, list[str]] = {}
    for s in verified_singles_distractor:
        by_source_dist.setdefault(s["source"], []).append(s["paragraph"])
    distinct_dist = sorted(by_source_dist.keys())

    if len(distinct_loop) < combine_k:
        return []
    # Need at least N distinct distractor sources outside any gold combo.
    if len(distinct_dist) < distractor_n + combine_k:
        # Allow distractor reuse only as a last resort; warn caller.
        print(f"  WARN: distractor pool ({len(distinct_dist)}) < "
              f"distractor_n+gold_k ({distractor_n + combine_k}); "
              f"distractor reuse will occur.")

    rng = random.Random(seed)
    usage_loop: dict[str, int] = {s: 0 for s in distinct_loop}
    usage_dist: dict[str, int] = {s: 0 for s in distinct_dist}

    def _pick_least_used(sources: list[str], usage: dict[str, int],
                         k: int, exclude: set[str]) -> list[str]:
        # Sort by (usage_count, random_tiebreak); stable selection.
        cands = [s for s in sources if s not in exclude]
        cands.sort(key=lambda s: (usage[s], rng.random()))
        return cands[:k]

    out: list[dict] = []
    rid = rid_start
    while len(out) < target_n:
        gold_combo = _pick_least_used(distinct_loop, usage_loop,
                                       combine_k, exclude=set())
        if len(gold_combo) < combine_k:
            break
        sig = frozenset(gold_combo)
        if sig in seen_signatures:
            # Force diversity: bump usage on this combo so we don't pick it
            # again next iteration.
            for s in gold_combo:
                usage_loop[s] += 1
            # If we keep cycling, eventually distinct_loop is exhausted vs
            # combinations(K), break to avoid infinite loop.
            if all(usage_loop[s] > len(out) + 2 for s in distinct_loop):
                break
            continue

        gold_set = set(gold_combo)
        distractors = _pick_least_used(distinct_dist, usage_dist,
                                        distractor_n, exclude=gold_set)
        if len(distractors) < distractor_n:
            # Pool exhausted; pad with random draws (with replacement).
            extra_pool = [s for s in distinct_dist if s not in gold_set]
            while len(distractors) < distractor_n and extra_pool:
                distractors.append(rng.choice(extra_pool))

        gold_paras = [by_source_loop[s][0] for s in gold_combo]
        distractor_paras = [by_source_dist[s][0] for s in distractors]

        items = (
            [(p, [s], False) for p, s in zip(gold_paras, gold_combo)]
            + [(p, [s], True)  for p, s in zip(distractor_paras, distractors)]
        )
        rng.shuffle(items)

        paragraphs = [t[0] for t in items]
        para_map = {
            str(i): ([] if is_distract else srcs)
            for i, (_, srcs, is_distract) in enumerate(items)
        }
        text = "\n\n".join(paragraphs)

        out.append({
            "region_id": f"syn_{rid:04d}",
            "text": text,
            "paragraphs": paragraphs,
            "gold_sources": sorted(gold_combo),
            "paragraph_source_map": para_map,
            "distractor_sources": sorted(set(distractors)),
            "synthetic": True,
        })
        seen_signatures.add(sig)
        for s in gold_combo:
            usage_loop[s] += 1
        for s in distractors:
            usage_dist[s] += 1
        rid += 1
    return out


def build(target: int = 25, buffer: int = 10, out_path: Path = OUT_DEFAULT,
          require_closed_loop: bool = True,
          synthesize: bool = True, synth_k: int = 3,
          synth_distractors: int = 7) -> None:
    print("[1] Loading sources from KG ...")
    with GraphClient() as g:
        rows = g.run(SOURCES_WITH_COMPOUNDS_QUERY)
        all_rows = g.run(ALL_SOURCES_QUERY)
        loop_rows = (
            g.run(SOURCES_WITH_CLOSED_LOOP_QUERY) if require_closed_loop else []
        )
    src_compounds_raw = {r["source"]: r["compounds"] for r in rows}
    src_compounds, alias_map = _canonicalize_sources(src_compounds_raw)

    # Tag passages using EVERY source in the KG (compound-having or not),
    # so the distractor pool can include mineral / animal / closed-loop-less
    # sources. Closed-loop set used only as the gold filter.
    all_source_names = sorted({r["source"] for r in all_rows if r.get("source")})
    # Fold no-compound sources into the alias map (identity) so canonical
    # lookups don't miss them.
    for s in all_source_names:
        alias_map.setdefault(s, s)
    sources_for_tagging = sorted(set(all_source_names) | set(src_compounds.keys()))

    loop_canon: set[str] = set()
    if require_closed_loop:
        loop_set_raw = {r["source"] for r in loop_rows}
        loop_canon = {alias_map.get(s, s) for s in loop_set_raw}
        print(f"    closed-loop sources: {len(loop_canon)} canonical "
              f"({len(loop_set_raw)} raw)")

    print(f"    sources w/ compounds: {len(src_compounds)} | "
          f"all sources (tagger pool): {len(sources_for_tagging)}")
    sources = sources_for_tagging

    if not CORPUS_FULL.exists():
        print(f"ERROR: corpus missing at {CORPUS_FULL}")
        return

    text = CORPUS_FULL.read_text(encoding="utf-8")
    paragraphs = _split_paragraphs(text)
    print(f"[2] Parsed {len(paragraphs)} paragraphs from corpus")

    kw_index = _build_keyword_index(sources)
    candidates = _candidate_regions(paragraphs, kw_index)
    print(f"[3] {len(candidates)} candidate windows w/ >=2 sources")

    if not candidates:
        print("No candidate regions. Aborting.")
        return

    cap = target + buffer
    print(f"[4] Verifying with {DEFAULT_FLASH_MODEL} (cap={cap}) ...")
    caller = GeminiCaller(model=DEFAULT_FLASH_MODEL)

    confirmed: list[dict] = []
    seen_signatures: set[frozenset[str]] = set()
    covered_sources: set[str] = set()
    # Two banks: closed-loop (gold candidates) + everything verified
    # (distractors). A loop single goes into BOTH so distractor sampling
    # can also pick already-seen sources for unrelated regions.
    verified_loop_singles: list[dict] = []
    verified_distractor_singles: list[dict] = []
    for region in candidates:
        if len(confirmed) >= cap:
            break
        kept = _verify_region(caller, region)
        # Bank verified single-source paragraphs for synthesis.
        if len(kept) == 1 and len(region["paragraphs"]) == 1:
            src = kept[0]
            if _has_therapeutic_content(region["text"]):
                if (not require_closed_loop) or src in loop_canon:
                    verified_loop_singles.append({
                        "paragraph": region["text"], "source": src,
                    })
                # Every therapeutic verified single is a valid distractor too.
                verified_distractor_singles.append({
                    "paragraph": region["text"], "source": src,
                })
        if len(kept) < MIN_SOURCES_PER_REGION:
            print(f"  {region['region_id']}: dropped (LLM kept {len(kept)} sources)",
                  flush=True)
            continue
        # Closed-loop filter applied here: gold sources must have a full
        # mechanism chain. Region keeps full-text + tags but gold restricts
        # to closed-loop sources only.
        if require_closed_loop:
            kept_loop = [s for s in kept if s in loop_canon]
            if len(kept_loop) < MIN_SOURCES_PER_REGION:
                print(f"  {region['region_id']}: dropped (closed-loop kept "
                      f"{len(kept_loop)}/{len(kept)}: {kept})", flush=True)
                continue
            kept = kept_loop
        sig = frozenset(kept)
        # Dedup: skip identical source-set signatures and skip regions whose
        # gold sources are a strict subset of the already-covered union (no
        # new sources introduced — likely sliding-window neighbor).
        new_sources = sig - covered_sources
        if sig in seen_signatures or not new_sources:
            print(f"  {region['region_id']}: dropped (overlap; no new sources, kept={list(kept)})",
                  flush=True)
            continue
        seen_signatures.add(sig)
        covered_sources |= sig
        para_map = {k: [s for s in v if s in kept]
                    for k, v in region["paragraph_source_map"].items()}
        confirmed.append({
            "region_id": region["region_id"],
            "text": region["text"],
            "paragraphs": region["paragraphs"],
            "gold_sources": sorted(kept),
            "paragraph_source_map": para_map,
        })
        print(f"  {region['region_id']}: kept {len(kept)} sources -> {kept} "
              f"(new={sorted(new_sources)})", flush=True)

    n_natural = len(confirmed)
    print(f"    natural multi-source regions: {n_natural} "
          f"(loop singles banked: {len(verified_loop_singles)}, "
          f"distractor pool: {len(verified_distractor_singles)})")

    if synthesize and len(confirmed) < target:
        need = target - len(confirmed)
        print(f"[4b] Synthesizing multi-source regions "
              f"(gold_k={synth_k}, distractors={synth_distractors}, "
              f"need {need} to reach target={target}) ...")
        synth = _synthesize_multi_source_regions(
            verified_singles_loop=verified_loop_singles,
            verified_singles_distractor=verified_distractor_singles,
            target_n=need,
            combine_k=synth_k,
            distractor_n=synth_distractors,
            seen_signatures=seen_signatures,
            rid_start=0,
        )
        for r in synth:
            print(f"  {r['region_id']}: gold={r['gold_sources']} "
                  f"distractors={r.get('distractor_sources', [])}", flush=True)
        confirmed.extend(synth)
        print(f"    synthetic added: {len(synth)} -> total: {len(confirmed)}")

    # Trim to exact target if natural overshoot.
    if len(confirmed) > target:
        confirmed = confirmed[:target]
        print(f"    trimmed natural overshoot -> {len(confirmed)}")

    if len(confirmed) < target:
        print(f"WARNING: only {len(confirmed)} confirmed regions (target {target}).")

    print(f"[5] Fetching gold (closed-loop) compounds + maladies per source ...")
    # Reverse alias map: canonical -> all variants (so we query every Neo4j label).
    canon_to_variants: dict[str, list[str]] = {}
    for variant, canon in alias_map.items():
        canon_to_variants.setdefault(canon, []).append(variant)

    with GraphClient() as g:
        for region in confirmed:
            gc, gm, gc_all = {}, {}, {}
            for s in region["gold_sources"]:
                loop_compounds: set[str] = set()
                for variant in canon_to_variants.get(s, [s]):
                    rows = g.run(CLOSED_LOOP_COMPOUNDS_QUERY, {"source": variant})
                    loop_compounds.update(r["compound"] for r in rows if r.get("compound"))
                gc[s] = sorted(loop_compounds)
                gc_all[s] = src_compounds.get(s, [])
                maladies: set[str] = set()
                for variant in canon_to_variants.get(s, [s]):
                    rows = g.run(MALADIES_FOR_SOURCE_QUERY, {"source": variant})
                    maladies.update(r["malady"] for r in rows if r.get("malady"))
                gm[s] = sorted(maladies)
            region["gold_compounds_per_source"] = gc            # closed-loop only
            region["gold_compounds_extracted"] = gc_all          # superset (audit)
            region["gold_maladies_per_source"] = gm

            # Distractor compounds: IS_EXTRACTED_FROM for each distractor source.
            dc: dict[str, list[str]] = {}
            for s in region.get("distractor_sources", []):
                dc[s] = src_compounds.get(s, [])
            region["distractor_compounds_per_source"] = dc

    n = write_jsonl(out_path, confirmed)
    print(f"[6] Wrote {n} regions to {out_path}")
    # Diagnostic: average gold_compounds size per region.
    if confirmed:
        sizes = [sum(len(v) for v in r["gold_compounds_per_source"].values())
                 for r in confirmed]
        print(f"    gold compounds per region: min={min(sizes)} max={max(sizes)} "
              f"mean={sum(sizes)/len(sizes):.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=25)
    p.add_argument("--buffer", type=int, default=10)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--no-closed-loop", action="store_true",
                   help="Don't restrict to sources with full mechanism chains.")
    p.add_argument("--no-synthesize", action="store_true",
                   help="Disable synthetic multi-source region generation.")
    p.add_argument("--synth-k", type=int, default=3,
                   help="Distinct GOLD sources per synthetic region (default 3).")
    p.add_argument("--synth-distractors", type=int, default=7,
                   help="Distractor paragraphs per synthetic region (default 7).")
    args = p.parse_args()
    build(target=args.target, buffer=args.buffer, out_path=args.out,
          require_closed_loop=not args.no_closed_loop,
          synthesize=not args.no_synthesize, synth_k=args.synth_k,
          synth_distractors=args.synth_distractors)
