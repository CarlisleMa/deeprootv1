#!/usr/bin/env python3
"""Prepare the trimmed COCONUT snapshot consumed by ``src/data/coconut.py``.

The full COCONUT 2.0 export is large; the DeepRoot loader only needs a handful
of columns. This script reads a COCONUT bulk export (CSV or Parquet), keeps the
required columns, and writes the trimmed Parquet that the loader expects at
``src/data/coconut_02-2026-trimmed.parquet``.

COCONUT 2.0 is open data (CC BY 4.0) — https://coconut.naturalproducts.net.
Attribute it in any downstream use.

Usage
-----
    python scripts/prepare_coconut.py --source path/to/coconut_export.csv
    python scripts/prepare_coconut.py --source coconut.parquet --out src/data/coconut_02-2026-trimmed.parquet

If the source uses different column names, pass overrides, e.g.::

    python scripts/prepare_coconut.py --source export.csv \\
        --organisms-col organisms --smiles-col canonical_smiles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "src" / "data" / "coconut_02-2026-trimmed.parquet"

# Output columns the loader (src/data/coconut.py) relies on.
REQUIRED_OUT_COLUMNS = [
    "organisms",          # "|"-separated organism list
    "canonical_smiles",   # compound structure
    "name",               # display name (falls back to iupac_name)
    "np_likeness",        # raw COCONUT score (0-5), passed through
    "annotation_level",   # raw COCONUT level (0-5), passed through
]


def _load(source: str) -> pd.DataFrame:
    path = Path(source)
    if not path.exists():
        sys.exit(f"error: source not found: {source}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    sys.exit(f"error: unsupported source extension: {path.suffix} (use .csv/.tsv/.parquet)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="COCONUT bulk export (.csv/.tsv/.parquet)")
    ap.add_argument("--out", default=str(_DEFAULT_OUT), help=f"output parquet (default: {_DEFAULT_OUT})")
    ap.add_argument("--organisms-col", default="organisms")
    ap.add_argument("--smiles-col", default="canonical_smiles")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--iupac-col", default="iupac_name", help="fallback when name is empty")
    ap.add_argument("--np-likeness-col", default="np_likeness")
    ap.add_argument("--annotation-level-col", default="annotation_level")
    args = ap.parse_args()

    df = _load(args.source)
    print(f"Loaded {len(df):,} rows / {len(df.columns)} columns from {args.source}")

    colmap = {
        args.organisms_col: "organisms",
        args.smiles_col: "canonical_smiles",
        args.name_col: "name",
        args.np_likeness_col: "np_likeness",
        args.annotation_level_col: "annotation_level",
    }
    missing = [src for src in colmap if src not in df.columns]
    if missing:
        sys.exit(
            "error: source is missing expected column(s): "
            + ", ".join(missing)
            + f"\navailable columns: {list(df.columns)}"
            + "\nPass --<field>-col overrides to map your export's column names."
        )

    out = df[list(colmap)].rename(columns=colmap).copy()

    # Fill empty display names from the IUPAC column when available.
    if args.iupac_col in df.columns:
        empty = out["name"].isna() | (out["name"].astype(str).str.strip() == "")
        out.loc[empty, "name"] = df.loc[empty, args.iupac_col]

    out = out[REQUIRED_OUT_COLUMNS]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(out):,} rows -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
