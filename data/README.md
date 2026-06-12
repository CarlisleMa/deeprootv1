# Data

This directory holds the **small, derived evaluation extracts** that ship with
the repository. The large and/or copyrighted source datasets are **not
redistributed** — this file documents how to obtain them.

## What ships here

| Path | Description |
|------|-------------|
| `data/eval/eval_corpus.txt` | Derived evaluation corpus used by the held-out recovery eval. |
| `data/eval/shen_nong_ben_cao_jing_sample.txt` | Small illustrative sample of the source corpus. |

> The derived extracts above contain short excerpts of the source translation.
> Confirm they fall within your permitted use before publishing (see corpus note
> below).

## What does NOT ship (and how to get it)

### 1. Historical corpus — *Shen Nong Ben Cao Jing*

The full corpus used in the paper is a **copyrighted translation** and is **not
included** in this repository. To reproduce the full Assembly pipeline, supply
your own licensed copy of the source text and place it at:

```
data/historical_corpus/shen_nong_ben_cao_jing_full.txt
```

(plain UTF-8 text). This path is `.gitignore`d so it is never accidentally
committed. Cite the specific translation you use.

### 2. COCONUT compound snapshot

`src/data/coconut.py` loads a trimmed COCONUT 2.0 parquet at import time from:

```
src/data/coconut_02-2026-trimmed.parquet      (~40 MB, .gitignore'd)
```

This file is derived from the open **COCONUT 2.0** natural-products database
(https://coconut.naturalproducts.net, CC BY 4.0 — attribute accordingly).
Build it once from a COCONUT bulk export:

```bash
python scripts/prepare_coconut.py --source <path-or-url-to-COCONUT-export>
```

The script trims the full export down to only the columns the loader needs and
writes the parquet to the path above. Required output schema:

| Column | Type | Used for |
|--------|------|----------|
| `organisms` | str | `\|`-separated organism list; drives the species → compound index |
| `canonical_smiles` | str | compound structure |
| `name` (or `iupac_name`) | str | display name |
| `np_likeness` | float | raw COCONUT score (0–5), passed through |
| `annotation_level` | int | raw COCONUT level (0–5), passed through |

If the parquet is missing, importing `src.data.coconut` raises a
`FileNotFoundError` pointing back here.

## Third-party databases queried at runtime

Assembly also queries these public APIs live (no bulk download needed), each
under its own terms: **PubChem**, **ChEMBL**, **Open Targets**, **NCBI
Taxonomy / OLS4**, and the ontology services (**ICD-10, MeSH, SNOMED, MONDO,
DOID**). Set optional API keys (`NCBI_API_KEY`, `S2_API_KEY`) in `.env` to raise
rate limits.
