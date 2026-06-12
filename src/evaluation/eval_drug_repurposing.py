"""Eval A — Drug Repurposing Recovery (held-out KNOWN_TREATS).

The marquee quantitative evaluation for the Phase-2 Task B pipeline.

Protocol:
  1. Test set: every (compound, disease) pair where
       (Chemical_Compound)-[:KNOWN_TREATS]->(Modern_Disease)
     AND the compound is reachable through at least one historical
     `IS_EXTRACTED_FROM` Source. The KNOWN_TREATS layer is FDA / clinical-
     trial ground truth populated by Phase 1's Compound→Disease Linker
     from ChEMBL drug_indication, so it functions as held-out clinical
     truth for the historical-text→mechanistic-loop discovery pipeline.

  2. For each test pair (c*, d*):
       a. Identify ALL InChIKeys sharing c*'s planar prefix (first 14
          chars). COCONUT stores each stereo-form of a natural product
          as a distinct InChIKey, so masking just the exact full key
          would leak the compound's other stereos to the nominator.
       b. Mask every (ik_i, d*) pair where ik_i shares the planar.
          The nominator's in-memory `masked_known_treats` argument
          treats them as novel for THIS trial only — no graph writes.
       c. Run TaskBNominator(disease_name=d*, top_k=K * 3).
       d. Dedup the nominations by planar key (keep highest-ranked
          stereo per planar). Take the first `top_k` distinct planars.
       e. The trial succeeds at rank R iff the test compound's planar
          appears at position R in the deduped list. R = None if absent.

  3. Metrics aggregated over all trials:
       - recall@1, recall@5, recall@10, recall@K
       - MRR (mean reciprocal rank; missing = 0)
       - rank distribution (histogram of ranks)
       - per-disease and per-compound-source breakdowns

Read-only — no graph writes. The nominator instance is reused across
trials for cache locality.

This eval doesn't depend on Pass 2 (CriticAgent). The proposed-system
"with multi-agent + with graph" baseline IS the Pass 3 nominator's
output. Eval B layers in baselines for the configuration ablation.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agents.phase2.baseline_llm_passages import BaselineLLMPassages
from src.agents.phase2.task_b_critic import TaskBCritic
from src.agents.phase2.task_b_nominator import TaskBNominator
from src.graph.client import GraphClient

MODES = ("pass3", "pass3_critic", "llm_passages")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher (read-only)
# ---------------------------------------------------------------------------

# Three reachability modes for the test set. Each measures something
# distinct, so the harness exposes all three and reports both the
# coverage diagnostic ("how many broad pairs survive the constraint?")
# and the recall metrics on the chosen mode.
#
#   broad       — any compound that has any Source path. Big test set
#                 (~301 pairs). Most pairs are unreachable in principle
#                 (Source doesn't treat the disease), so 0% recall on
#                 those is a Phase-1 coverage failure, not a Phase-2
#                 ranking failure. Only useful as a denominator for
#                 the coverage statistic.
#
#   historical  — DEFAULT. The compound must be reachable from the
#                 disease via the historical chain
#                 (Source → Malady → MAPS_TO → Disease). Tests Phase-2
#                 ranking quality without conflating it with Phase-1
#                 corpus coverage. ~21 pairs in the live graph.
#
#   strict      — historical AND forward loop closure
#                 (Compound → Target → RELATES_TO → Disease) exists.
#                 Strictest test of "the system has unambiguous
#                 evidence to work with." ~13 pairs.

_TEST_SET_BROAD_QUERY = """
MATCH (c:Chemical_Compound)-[k:KNOWN_TREATS]->(d:Modern_Disease)
WHERE (c.archived IS NULL OR c.archived = false)
  AND (d.archived IS NULL OR d.archived = false)
  AND EXISTS { (c)-[:IS_EXTRACTED_FROM]->(:Source) }
RETURN
  c.inchikey AS test_inchikey,
  c.name AS test_compound,
  c.linker_chembl_id AS test_chembl_id,
  d.name AS test_disease,
  k.clinical_phase AS clinical_phase,
  k.evidence_type AS kt_evidence_type
ORDER BY d.name, c.name
"""


_TEST_SET_HISTORICAL_QUERY = """
MATCH (c:Chemical_Compound)-[k:KNOWN_TREATS]->(d:Modern_Disease)
WHERE (c.archived IS NULL OR c.archived = false)
  AND (d.archived IS NULL OR d.archived = false)
  AND EXISTS {
    MATCH (c)-[:IS_EXTRACTED_FROM]->(s:Source)-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady)-[r_map:MAPS_TO]->(d)
    WHERE coalesce(r_map.is_primary, true) = true
      AND (s.archived IS NULL OR s.archived = false)
      AND (m.archived IS NULL OR m.archived = false)
  }
RETURN
  c.inchikey AS test_inchikey,
  c.name AS test_compound,
  c.linker_chembl_id AS test_chembl_id,
  d.name AS test_disease,
  k.clinical_phase AS clinical_phase,
  k.evidence_type AS kt_evidence_type
ORDER BY d.name, c.name
"""


_TEST_SET_STRICT_QUERY = """
MATCH (c:Chemical_Compound)-[k:KNOWN_TREATS]->(d:Modern_Disease)
WHERE (c.archived IS NULL OR c.archived = false)
  AND (d.archived IS NULL OR d.archived = false)
  AND EXISTS {
    MATCH (c)-[:IS_EXTRACTED_FROM]->(s:Source)-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady)-[r_map:MAPS_TO]->(d)
    WHERE coalesce(r_map.is_primary, true) = true
      AND (s.archived IS NULL OR s.archived = false)
      AND (m.archived IS NULL OR m.archived = false)
  }
  AND EXISTS {
    MATCH (c)-[:TARGETS]->(t:Biological_Target)-[:RELATES_TO]->(d)
    WHERE (t.archived IS NULL OR t.archived = false)
  }
RETURN
  c.inchikey AS test_inchikey,
  c.name AS test_compound,
  c.linker_chembl_id AS test_chembl_id,
  d.name AS test_disease,
  k.clinical_phase AS clinical_phase,
  k.evidence_type AS kt_evidence_type
ORDER BY d.name, c.name
"""


_TEST_SET_QUERIES: dict[str, str] = {
    "broad":      _TEST_SET_BROAD_QUERY,
    "historical": _TEST_SET_HISTORICAL_QUERY,
    "strict":     _TEST_SET_STRICT_QUERY,
}


# Find all stereo-isomer InChIKeys sharing a planar prefix. Used to
# mask every stereo-form of the test compound during a trial — without
# this the nominator would still find a sibling stereo and the eval
# would over-report recall.
_PLANAR_SIBLINGS_QUERY = """
MATCH (c:Chemical_Compound)
WHERE c.inchikey STARTS WITH $planar
  AND (c.archived IS NULL OR c.archived = false)
RETURN c.inchikey AS inchikey, c.name AS name
"""


# Sanity check: how many distinct active diseases does the test set cover?
_DISEASE_COUNT_QUERY = """
MATCH (d:Modern_Disease)
WHERE (d.archived IS NULL OR d.archived = false)
  AND EXISTS { (:Chemical_Compound)-[:KNOWN_TREATS]->(d) }
  AND EXISTS {
    (:Chemical_Compound)-[:KNOWN_TREATS]->(d)
    WHERE EXISTS { (:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(:Source) }
  }
RETURN count(DISTINCT d) AS disease_count
"""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """One held-out (compound, disease) trial."""
    test_inchikey: str
    test_compound: str
    test_disease: str
    test_chembl_id: str | None
    clinical_phase: str | None
    planar_key: str
    masked_inchikey_count: int           # how many stereo-form IDs we masked
    candidates_total: int                # nominator's pre-dedup candidate count
    nominations_returned: int            # post-dedup unique-planar nominations
    rank: int | None                     # first rank where any same-planar appeared
    found: bool
    matched_inchikey: str | None         # which exact InChIKey at the matched rank
    matched_compound: str | None
    has_loop_closure_at_match: bool | None
    forward_bucket_at_match: str | None


@dataclass
class EvalASummary:
    test_set_size: int
    top_k: int
    reachability_mode: str
    config_mode: str                      # pass3 | pass3_critic | llm_passages
    use_critic: bool
    critic_input_top_n: int
    coverage_diagnostic: dict[str, int]   # mode → test set size
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_top_k: float
    mrr: float
    found_count: int
    not_found_count: int
    rank_histogram: dict[int, int]       # rank → count
    per_disease_recall_at_top_k: dict[str, float]
    duration_s: float
    trials: list[TrialResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _planar_of(inchikey: str) -> str:
    """First 14 chars of an InChIKey — the constitutional / 2-D structural
    hash. Different stereoisomers share this prefix; truly different
    structures don't."""
    return (inchikey or "")[:14]


def _dedup_by_planar(nominations: list[dict]) -> list[dict]:
    """Keep the highest-ranked nomination per planar key. Preserves the
    nominator's existing ordering — first occurrence wins."""
    seen: set[str] = set()
    out: list[dict] = []
    for n in nominations:
        ik = n.get("compound_inchikey") or ""
        pk = _planar_of(ik)
        if not pk or pk in seen:
            continue
        seen.add(pk)
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------


class DrugRepurposingEval:
    """Eval A driver. Read-only, no graph writes."""

    def __init__(
        self,
        client: GraphClient,
        *,
        mode: str = "pass3",
        use_critic: bool = False,
        critic_input_top_n: int = 50,
        critic_gemini_model: str | None = None,
        baseline_corpus_path: str | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        self.client = client
        self.mode = mode
        # Back-compat: if someone still passes use_critic=True we map it
        # to mode="pass3_critic" (callers should prefer mode= now).
        if use_critic and mode == "pass3":
            self.mode = "pass3_critic"
        self.use_critic = self.mode == "pass3_critic"
        self.critic_input_top_n = critic_input_top_n
        self.nominator = TaskBNominator(client) if self.mode != "llm_passages" else None
        self.critic = (
            TaskBCritic(client, gemini_model=critic_gemini_model)
            if self.mode == "pass3_critic" else None
        )
        self.llm_baseline = (
            BaselineLLMPassages(
                client,
                gemini_model=critic_gemini_model,
                corpus_path=baseline_corpus_path,
            )
            if self.mode == "llm_passages" else None
        )
        self._planar_cache: dict[str, set[str]] = {}

    # ------------------------------------------------------------------

    def pull_test_set(self, mode: str = "historical") -> list[dict]:
        """Pull the test set under the specified reachability mode.

        Modes (see module docstring):
          broad       — any KNOWN_TREATS pair where compound has a Source
          historical  — DEFAULT. compound reaches disease via the
                        historical chain
          strict      — historical AND forward loop closure
        """
        if mode not in _TEST_SET_QUERIES:
            raise ValueError(
                f"Unknown reachability mode: {mode!r}. "
                f"Expected one of: {sorted(_TEST_SET_QUERIES.keys())}"
            )
        return self.client.run(_TEST_SET_QUERIES[mode])

    def pull_coverage_diagnostic(self) -> dict[str, int]:
        """Quick stat of test-set sizes under each reachability mode.
        Useful for the paper's coverage section."""
        out: dict[str, int] = {}
        for mode in _TEST_SET_QUERIES:
            out[mode] = len(self.pull_test_set(mode))
        return out

    def _compound_names_for(self, inchikeys: set[str]) -> list[str]:
        """Look up display names for the masked stereo siblings, so the
        LLM-passages baseline knows which compound names to exclude."""
        if not inchikeys:
            return []
        rows = self.client.run("""
            UNWIND $iks AS ik
            MATCH (c:Chemical_Compound {inchikey: ik})
            WHERE c.archived IS NULL OR c.archived = false
            RETURN DISTINCT c.name AS name
        """, {"iks": list(inchikeys)})
        return sorted({r.get("name") for r in rows if r.get("name")})

    def pull_planar_siblings(self, planar: str) -> set[str]:
        """All InChIKeys (active) sharing the given planar prefix."""
        if planar in self._planar_cache:
            return self._planar_cache[planar]
        rows = self.client.run(_PLANAR_SIBLINGS_QUERY, {"planar": planar})
        siblings = {r["inchikey"] for r in rows if r.get("inchikey")}
        self._planar_cache[planar] = siblings
        return siblings

    # ------------------------------------------------------------------

    def run(
        self,
        *,
        top_k: int = 20,
        reachability_mode: str = "historical",
        require_loop_closure: bool = False,
        limit: int | None = None,
        progress_every: int = 25,
    ) -> EvalASummary:
        """Run the held-out KNOWN_TREATS recovery test.

        Args:
          top_k: distinct compounds in the deduped nomination list to
              evaluate. Recall@K is computed at K=1,5,10,top_k.
          reachability_mode: which test set to use.
              "broad"      — every clinical-truth pair where the compound
                             has a Source path (~301).
              "historical" — DEFAULT. compound is reachable from the
                             disease via the full backward chain (~21).
              "strict"     — historical AND forward loop closure (~13).
          require_loop_closure: pass through to the nominator. If True,
              only loop-closed candidates are considered nominations
              (stricter test).
          limit: cap the test set (useful for spot-checks).
          progress_every: log a heartbeat every N trials.
        """
        t_start = time.time()

        coverage_diag = self.pull_coverage_diagnostic()
        test_pairs = self.pull_test_set(reachability_mode)
        if limit is not None:
            test_pairs = test_pairs[:limit]

        logger.info(
            "Eval A — mode=%s, %d trials (broad=%d, historical=%d, strict=%d), "
            "top_k=%d, require_loop_closure=%s",
            reachability_mode, len(test_pairs),
            coverage_diag.get("broad", 0),
            coverage_diag.get("historical", 0),
            coverage_diag.get("strict", 0),
            top_k, require_loop_closure,
        )

        # Internal nominator top-k buffer: pull more than we need so the
        # planar-dedup still has top_k unique compounds left after collapse.
        # Empirical buffer: 5x is comfortable; 11 stereos for aconitine
        # was the worst observed case in the live graph.
        internal_top_k = max(top_k * 5, 50)

        trials: list[TrialResult] = []
        rank_histogram: dict[int, int] = {}
        per_disease_recall: dict[str, list[bool]] = {}

        for i, pair in enumerate(test_pairs, 1):
            test_ik = pair["test_inchikey"]
            test_compound = pair["test_compound"]
            test_disease = pair["test_disease"]
            planar = _planar_of(test_ik)

            siblings = self.pull_planar_siblings(planar)
            masked = {(ik, test_disease) for ik in siblings}

            if self.mode == "llm_passages":
                # No graph walk; the LLM gets the corpus and the disease
                # name. We do NOT pass the masked stereo siblings as an
                # "exclude" instruction — that would literally tell the
                # LLM not to nominate the test compound, defeating the
                # eval. Pass-3's masking semantics (override the novelty
                # filter) don't translate to an LLM that has no novelty
                # filter; the LLM is free to nominate any compound and
                # we honestly disclose its training-data access as a
                # baseline limitation.
                try:
                    bres = self.llm_baseline.run(
                        disease_name=test_disease,
                        excluded_compound_names=[],
                        top_k=top_k,
                        resolve_in_graph=True,
                    )
                except Exception as e:
                    logger.warning(
                        "LLM baseline failed for (%s, %s): %s",
                        test_ik[:14], test_disease, e,
                    )
                    bres = None

                if bres is None or bres.error:
                    trials.append(TrialResult(
                        test_inchikey=test_ik,
                        test_compound=test_compound,
                        test_disease=test_disease,
                        test_chembl_id=pair.get("test_chembl_id"),
                        clinical_phase=pair.get("clinical_phase"),
                        planar_key=planar,
                        masked_inchikey_count=len(masked),
                        candidates_total=0,
                        nominations_returned=0,
                        rank=None, found=False,
                        matched_inchikey=None, matched_compound=None,
                        has_loop_closure_at_match=None,
                        forward_bucket_at_match=None,
                    ))
                    continue

                # Build a Pass-3-shaped nominations list. Resolved planar
                # key (when present) drives recall computation. LLM-only
                # has no loop closure / forward bucket info.
                nominations = []
                for n in bres.nominations:
                    if n.matched_inchikey is None:
                        continue   # LLM proposed a name not in our graph
                    nominations.append({
                        "compound": n.matched_compound_name or n.compound_name,
                        "compound_inchikey": n.matched_inchikey,
                        "has_loop_closure": None,
                        "forward_bucket": None,
                    })
                result = {"candidates_total": len(bres.nominations)}
            else:
                try:
                    result = self.nominator.run(
                        disease_name=test_disease,
                        top_k=internal_top_k,
                        apply_novelty_filter=True,
                        require_loop_closure=require_loop_closure,
                        masked_known_treats=masked,
                        keep_top_paths_per_kind=2,
                        write_graph=False,
                    )
                except Exception as e:
                    logger.warning(
                        "Nominator failed for (%s, %s): %s",
                        test_ik[:14], test_disease, e,
                    )
                    trials.append(TrialResult(
                        test_inchikey=test_ik,
                        test_compound=test_compound,
                        test_disease=test_disease,
                        test_chembl_id=pair.get("test_chembl_id"),
                        clinical_phase=pair.get("clinical_phase"),
                        planar_key=planar,
                        masked_inchikey_count=len(masked),
                        candidates_total=0,
                        nominations_returned=0,
                        rank=None,
                        found=False,
                        matched_inchikey=None,
                        matched_compound=None,
                        has_loop_closure_at_match=None,
                        forward_bucket_at_match=None,
                    ))
                    continue

                nominations = result.get("nominations") or []

            # Optional Pass 4: LLM critic re-ranks the top-N nominations.
            # We dedup by planar BEFORE feeding the LLM so the critic
            # doesn't waste tokens scoring redundant stereo-siblings.
            if self.use_critic and self.critic is not None and nominations:
                pre_critic_deduped = _dedup_by_planar(nominations)[: self.critic_input_top_n]
                try:
                    critic_result = self.critic.run(
                        disease_name=test_disease,
                        nominations=pre_critic_deduped,
                        input_top_n=self.critic_input_top_n,
                        output_top_k=top_k,
                    )
                except Exception as e:
                    logger.warning(
                        "Critic failed for (%s, %s): %s — falling back to Pass-3 ranking",
                        test_ik[:14], test_disease, e,
                    )
                    critic_result = None

                if critic_result is not None and critic_result.reranked:
                    # The critic returns deduped output; re-attach a small
                    # set of original Pass-3 fields for the rank lookup.
                    deduped = [
                        {
                            "compound": r.compound,
                            "compound_inchikey": r.compound_inchikey,
                            "has_loop_closure": r.pass3_has_loop_closure,
                            "forward_bucket": r.pass3_forward_bucket,
                        }
                        for r in critic_result.reranked
                    ][:top_k]
                else:
                    deduped = _dedup_by_planar(nominations)[:top_k]
            else:
                deduped = _dedup_by_planar(nominations)[:top_k]

            rank = None
            matched_ik = None
            matched_compound = None
            matched_loop = None
            matched_bucket = None
            for r_idx, nom in enumerate(deduped, 1):
                if _planar_of(nom.get("compound_inchikey") or "") == planar:
                    rank = r_idx
                    matched_ik = nom.get("compound_inchikey")
                    matched_compound = nom.get("compound")
                    matched_loop = nom.get("has_loop_closure")
                    matched_bucket = nom.get("forward_bucket")
                    break

            found = rank is not None
            if rank is not None:
                rank_histogram[rank] = rank_histogram.get(rank, 0) + 1

            per_disease_recall.setdefault(test_disease, []).append(found)

            trials.append(TrialResult(
                test_inchikey=test_ik,
                test_compound=test_compound,
                test_disease=test_disease,
                test_chembl_id=pair.get("test_chembl_id"),
                clinical_phase=pair.get("clinical_phase"),
                planar_key=planar,
                masked_inchikey_count=len(masked),
                candidates_total=result.get("candidates_total", 0),
                nominations_returned=len(deduped),
                rank=rank,
                found=found,
                matched_inchikey=matched_ik,
                matched_compound=matched_compound,
                has_loop_closure_at_match=matched_loop,
                forward_bucket_at_match=matched_bucket,
            ))

            if i % progress_every == 0 or i == len(test_pairs):
                running_recall = sum(1 for t in trials if t.found) / len(trials)
                logger.info(
                    "  %d/%d  running recall@%d = %.3f",
                    i, len(test_pairs), top_k, running_recall,
                )

        return self._summarize(
            trials=trials,
            top_k=top_k,
            reachability_mode=reachability_mode,
            config_mode=self.mode,
            use_critic=self.use_critic,
            critic_input_top_n=self.critic_input_top_n,
            coverage_diagnostic=coverage_diag,
            rank_histogram=rank_histogram,
            per_disease_recall=per_disease_recall,
            duration_s=round(time.time() - t_start, 2),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _summarize(
        *,
        trials: list[TrialResult],
        top_k: int,
        reachability_mode: str,
        config_mode: str,
        use_critic: bool,
        critic_input_top_n: int,
        coverage_diagnostic: dict[str, int],
        rank_histogram: dict[int, int],
        per_disease_recall: dict[str, list[bool]],
        duration_s: float,
    ) -> EvalASummary:
        n = len(trials)
        if n == 0:
            return EvalASummary(
                test_set_size=0, top_k=top_k,
                reachability_mode=reachability_mode,
                config_mode=config_mode,
                use_critic=use_critic,
                critic_input_top_n=critic_input_top_n,
                coverage_diagnostic=coverage_diagnostic,
                recall_at_1=0.0, recall_at_5=0.0, recall_at_10=0.0,
                recall_at_top_k=0.0, mrr=0.0, found_count=0, not_found_count=0,
                rank_histogram={}, per_disease_recall_at_top_k={},
                duration_s=duration_s, trials=[],
            )

        def _recall_at(k: int) -> float:
            return sum(1 for t in trials if t.rank is not None and t.rank <= k) / n

        mrr = sum(1.0 / t.rank for t in trials if t.rank is not None) / n
        per_disease = {
            d: round(sum(rs) / len(rs), 4)
            for d, rs in per_disease_recall.items()
        }

        return EvalASummary(
            test_set_size=n,
            top_k=top_k,
            reachability_mode=reachability_mode,
            config_mode=config_mode,
            use_critic=use_critic,
            critic_input_top_n=critic_input_top_n,
            coverage_diagnostic=coverage_diagnostic,
            recall_at_1=round(_recall_at(1), 4),
            recall_at_5=round(_recall_at(5), 4),
            recall_at_10=round(_recall_at(10), 4),
            recall_at_top_k=round(_recall_at(top_k), 4),
            mrr=round(mrr, 4),
            found_count=sum(1 for t in trials if t.found),
            not_found_count=sum(1 for t in trials if not t.found),
            rank_histogram=dict(sorted(rank_histogram.items())),
            per_disease_recall_at_top_k=dict(
                sorted(per_disease.items(), key=lambda kv: -kv[1])
            ),
            duration_s=duration_s,
            trials=trials,
        )


def summary_to_dict(s: EvalASummary) -> dict:
    out = asdict(s)
    out["trials"] = [asdict(t) for t in s.trials]
    return out
