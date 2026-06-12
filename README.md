# DeepRoot

**A knowledge-graph-coordinated multi-agent system for therapeutic reasoning over historical medical texts.**

DeepRoot converts pre-ontological medical prose (e.g. traditional materia medica) into an auditable biomedical knowledge graph, then reasons over that graph to surface and justify therapeutic candidates. The central finding: **grounding and reasoning are separable axes**, and building a *verified* knowledge graph suppresses hallucination in a way that giving an LLM live tool/API access at inference time does not.

All outputs are **for research hypothesis generation only — not medical advice, diagnosis, or treatment protocols.** See [Disclaimer](#disclaimer).

---

## What it does

Given a historical medical corpus, DeepRoot runs in two stages:

- **DeepRoot Assembly** — seven specialized LLM agents extract entities and populate a typed Neo4j graph, grounding every entity against curated biomedical databases (COCONUT, PubChem, ChEMBL, Open Targets, NCBI Taxonomy) and disease ontologies (ICD-10, MeSH, SNOMED, MONDO, DOID). LLMs *propose* canonical names; identifiers are only accepted by tolerant exact match, eliminating hallucinated codes.
- **DeepRoot Discovery** — a validator, critic, and nominator reason over the assembled graph, tracing `Source → Compound → Target → Disease` mechanistic loops to evaluate therapeutic claims and rank candidate compounds, all grounded in graph evidence.

Applied to the *Shen Nong Ben Cao Jing* (71 chunks), Assembly yields **21,111 active nodes** and **52,467 active edges**. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full schema and agent protocols.

## Repository layout

```
src/
  config.py                 # central config; all credentials read from env
  agents/
    base.py                 # BaseAgent
    phase1/                 # Assembly: extraction, auditor, 4 linkers, reviewer
    phase2/                 # Discovery: validator, critic, nominator + baselines
  data/                     # ChEMBL, COCONUT, PubChem, Open Targets, ontology clients
  graph/                    # Neo4j client, Pydantic models, Cypher queries, schema
  evaluation/               # eval harness, LLM-judge, robustness ablation
scripts/                    # entry points: assembly run_* stages + evals + setup
data/                       # small eval extracts (full corpus NOT shipped; see data/README.md)
docs/ARCHITECTURE.md        # system internals
```

This is a lean, code-first release: the system and the scripts needed to run it.
Result artifacts and the source datasets are not bundled — running the pipeline
regenerates them.

## Setup

```bash
# 1. Install dependencies (Python 3.11+ recommended)
pip install -r requirements.txt

# 2. Start a local Neo4j (or point at AuraDB)
docker compose up -d            # Neo4j 5-community on bolt://localhost:7687

# 3. Configure credentials
cp .env.example .env            # then fill in Neo4j + LLM API keys

# 4. Obtain the COCONUT compound snapshot (not shipped — ~40 MB derived dataset)
python scripts/prepare_coconut.py --source <COCONUT_bulk_export>
#   see data/README.md for where to download COCONUT 2.0 and the required schema

# 5. Initialize the graph schema
python scripts/setup_neo4j.py
```

LLM tiers and all credentials are configured via `.env` (see `.env.example`); the
default models are Gemini 3.1 Flash Lite (extraction/batch), Gemini 3.1 Pro
(critic/mechanistic reasoning), and Claude Sonnet 4.6 (LLM-as-judge).

## Data and third-party resources

DeepRoot derives from several external databases, each under its own license:
COCONUT, PubChem, ChEMBL, Open Targets, NCBI Taxonomy, and the ontology
services (ICD-10, MeSH, SNOMED, MONDO, DOID). The historical corpus is
copyrighted and **not redistributed** — see [`data/README.md`](data/README.md)
for how to supply it and for the COCONUT snapshot schema.

## License

Source code is released under the [MIT License](LICENSE). The license does
**not** cover the third-party databases or the historical corpus; see the note
in `LICENSE` and `data/README.md`.

## Disclaimer

DeepRoot is a research tool for **hypothesis generation only**. Its outputs are
not medical advice, diagnosis, or treatment recommendations. Graph-supported
plausibility does not establish safety or efficacy. Historical therapeutic
claims may be ineffective, toxic, or culturally specific. Any downstream use
requires expert review, provenance tracking, toxicity assessment, and
experimental validation.
