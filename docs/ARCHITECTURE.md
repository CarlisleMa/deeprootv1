# DeepRoot Architecture

This document describes the system internals: the knowledge-graph schema, the
two-stage agent pipeline, the identity/grounding model, and the Neo4j storage
conventions.

---

## Overview

DeepRoot has two stages that share a typed Neo4j knowledge graph as working
memory:

1. **Assembly** populates the graph from raw historical text, grounding every
   entity against curated biomedical databases.
2. **Discovery** reasons over the assembled graph to validate therapeutic
   claims and rank candidate compounds.

A therapeutic claim is **verifiable** when its mechanistic loop closes:
a `Source` treats a `Traditional_Malady` that `MAPS_TO` a `Modern_Disease`;
the source contains a `Chemical_Compound` that `TARGETS` a `Biological_Target`
which `RELATES_TO` that same `Modern_Disease`.

## Knowledge-graph schema

**Six node types:** `Source`, `Traditional_Malady`, `Preparation`,
`Chemical_Compound`, `Biological_Target`, `Modern_Disease`.

**Seven edge types:** `IS_EXTRACTED_FROM`, `TARGETS`, `RELATES_TO`,
`TREATS_TRADITIONALLY`, `MAPS_TO`, `KNOWN_TREATS`, `PREPARED_AS`.

**Identity (so equivalent entities from different routes collapse to one node):**
- `Chemical_Compound` identity = RDKit-computed **InChIKey**.
- `Biological_Target` identity = curated **ChEMBL ID**.

On the *Shen Nong Ben Cao Jing* (71 chunks), the assembled graph contains:

| Nodes (21,111) | | Edges (52,467) | |
|---|---|---|---|
| Sources | 415 | IS_EXTRACTED_FROM | 32,909 |
| Maladies | 294 | TARGETS | 16,696 |
| Modern diseases | 129 | RELATES_TO | 1,841 |
| Compounds | 18,012 | TREATS_TRADITIONALLY | 431 |
| Targets | 2,211 | MAPS_TO | 257 |
| Preparations | 50 | KNOWN_TREATS | 301 |
| | | PREPARED_AS | 32 |

The authoritative schema (constraints, indexes, property-level definitions)
lives in [`src/graph/schema.py`](../src/graph/schema.py) and the Pydantic node/
edge models in [`src/graph/models.py`](../src/graph/models.py).

## Stage 1 — Assembly (`src/agents/phase1/`)

Seven specialized agents populate the graph in dependency order. Build the
shared graph layer (`src/graph/`) first; everything depends on it.

```
src/graph/ (schema, client, models, queries)        ← shared foundation
    │
    ├── extraction.py            emits Source / Malady / Preparation nodes from raw text
    ├── extraction_auditor.py    canonicalizes sources; archives evidence spans that
    │                            fail substring verification against their source chunk
    ├── source_compound.py       Source → Compound (organisms → COCONUT, chemicals → PubChem)
    ├── compound_target.py       Compound → Target via ChEMBL
    ├── compound_disease(_batched).py   Compound → Disease KNOWN_TREATS (ChEMBL indications)
    ├── target_disease.py        Target → Disease RELATES_TO via Open Targets / EFO
    ├── malady_disease.py        Malady → Modern_Disease (generate-then-verify; codes
    │                            recovered only by tolerant exact match)
    └── reviewer.py              deterministic post-pass: orphan dedup, off-domain archival
```

Note: `compound_disease_batched.py` is the production batched linker and imports
helper definitions (confidence priors, link status, disease index) from
`compound_disease.py`; both modules are required.

## Stage 2 — Discovery (`src/agents/phase2/`)

- `task_a_validator.py` — Pass 1, deterministic claim validation; emits the
  per-claim metapath signals consumed downstream (`results/task_a_pass1.json`).
- `task_a_critic.py` — Pass 2, the KG-grounded semantic **Critic** that scores
  therapeutic plausibility; this is the proposed system, graded at three LLM
  tiers in the reasoning-quality evaluation.
- `task_b_nominator.py` / `task_b_critic.py` — Pass 3/4 candidate **nomination**
  and re-ranking for drug-repurposing (the held-out recovery experiment).

Baselines used for the ablations live alongside the agents:
`task_a_baseline_graph_only.py` (no LLM), `task_a_baseline_text_only.py`
(LLM + corpus passages, no graph), `task_a_baseline_tool_call.py`
(agentic LLM with direct ChEMBL/Open Targets/PubMed/MeSH access), and
`baseline_llm_passages.py` (raw-corpus single-LLM repurposing baseline).

## Grounding sources

| Edge / mapping | Source database |
|----------------|-----------------|
| Source → Compound | COCONUT (natural products), PubChem (chemicals) |
| Compound → Target, Compound → Disease | ChEMBL |
| Target → Disease | Open Targets (human-disease GraphQL); NCBI Taxonomy + OLS4 for pathogenic-organism targets |
| Malady → Modern_Disease codes | ICD-10, MeSH, SNOMED, MONDO, DOID via NLM/EBI lookup |

## Neo4j storage conventions

- **MERGE, not CREATE.** All writes use `MERGE` on the identity key so repeated
  runs and concurrent writers converge to a single node rather than duplicating.
- **Uniqueness constraints** are declared in `src/graph/schema.py` and enforced
  at the database level (InChIKey for compounds, ChEMBL ID for targets, etc.).
- **Concurrent-write safety** relies on constraint-backed `MERGE`, so multiple
  assembly agents can populate the graph in parallel.
- Local development uses the bundled `docker-compose.yml` (Neo4j 5-community
  with APOC); production runs used a hosted AuraDB instance.

## Evaluation (`src/evaluation/`)

- `eval_drug_repurposing.py` — held-out `KNOWN_TREATS` recovery (Table 1).
- `eval_task_a_reasoning_judge.py` + `judge_cases.py` + `judge_prompts.py` —
  the LLM-as-judge harness (Table 2); stratified sampling, payload
  reconstruction, deterministic checks, seed 42.
- `eval_robustness.py` — the edge-perturbation ablation (Fig. 2A); requires a
  second "clone" Neo4j instance configured via `CLONE_NEO4J_*` in `.env`.
