"""One-time conversion: raw MIMIC-IV zips -> the parquet files the pipeline reads.

You have the credentialed MIMIC-IV distribution as zip files:
    mimic-iv-3.1.zip        (core: hosp/patients, hosp/admissions, hosp/diagnoses_icd)
    mimic-iv-notes-2.2.zip  (notes: note/discharge)

This script reads only the four tables the pipeline needs, straight out of the
zips (no full extraction to disk), keeps only the columns we use, and writes:
    <out-dir>/patients.parquet
    <out-dir>/admissions.parquet
    <out-dir>/diagnoses_icd.parquet
    <out-dir>/discharge_notes.parquet

The big discharge table (~1.7 GB compressed CSV) is converted in chunks with a
streaming parquet writer, so peak RAM stays low -- safe on a laptop.

Run once, then point config.yaml -> paths.drive_root at <out-dir> and run the
normal pipeline.

Usage:
    python scripts/prepare_data.py \
        --core-zip  /path/to/mimic-iv-3.1.zip \
        --notes-zip /path/to/mimic-iv-notes-2.2.zip \
        --out-dir   ./data/mimic
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# table -> (basename inside the zip, columns to keep)
SMALL_TABLES = {
    "patients.parquet": (
        "patients.csv.gz",
        ["subject_id", "anchor_age", "gender", "anchor_year_group"],
    ),
    "admissions.parquet": (
        "admissions.csv.gz",
        ["subject_id", "hadm_id", "admittime", "dischtime"],
    ),
    "diagnoses_icd.parquet": (
        "diagnoses_icd.csv.gz",
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    ),
}
NOTES_BASENAME = "discharge.csv.gz"
NOTES_COLS = ["subject_id", "hadm_id", "note_type", "text"]
NOTES_OUT = "discharge_notes.parquet"


def _find_member(zf: zipfile.ZipFile, basename: str) -> str:
    """Locate a file by basename anywhere inside the zip (folder-name agnostic)."""
    for name in zf.namelist():
        if name.endswith("/" + basename) or name == basename:
            return name
    raise FileNotFoundError(
        f"'{basename}' not found in {zf.filename}. "
        f"Members include: {zf.namelist()[:8]} ..."
    )


def convert_small(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for out_name, (basename, cols) in SMALL_TABLES.items():
            member = _find_member(zf, basename)
            print(f"  {basename} -> {out_name}")
            with zf.open(member) as f:
                # read only the columns we need; low_memory=False for clean dtypes
                df = pd.read_csv(f, compression="gzip", usecols=lambda c: c in cols)
            df.columns = [c.lower() for c in df.columns]
            df.to_parquet(out_dir / out_name, index=False)
            print(f"    wrote {len(df):,} rows")


def convert_notes(zip_path: Path, out_dir: Path, chunksize: int = 20000) -> None:
    out_path = out_dir / NOTES_OUT
    with zipfile.ZipFile(zip_path) as zf:
        member = _find_member(zf, NOTES_BASENAME)
        print(f"  {NOTES_BASENAME} -> {NOTES_OUT} (chunked, low-memory)")
        writer = None
        total = 0
        with zf.open(member) as f:
            reader = pd.read_csv(
                f,
                compression="gzip",
                usecols=lambda c: c in NOTES_COLS,
                chunksize=chunksize,
            )
            for chunk in reader:
                chunk.columns = [c.lower() for c in chunk.columns]
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                total += len(chunk)
                print(f"    ... {total:,} notes written", end="\r")
        if writer is not None:
            writer.close()
    print(f"\n    wrote {total:,} notes total")


def main() -> None:
    ap = argparse.ArgumentParser(description="MIMIC-IV zips -> parquet for the pipeline")
    ap.add_argument("--core-zip", required=True, help="mimic-iv-3.1.zip")
    ap.add_argument("--notes-zip", required=True, help="mimic-iv-notes-2.2.zip")
    ap.add_argument("--out-dir", default="./data/mimic")
    ap.add_argument("--chunksize", type=int, default=20000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Core tables from {args.core_zip}")
    convert_small(Path(args.core_zip), out_dir)

    print(f"[2/2] Discharge notes from {args.notes_zip}")
    convert_notes(Path(args.notes_zip), out_dir, args.chunksize)

    print(f"\nDone. Parquet files are in {out_dir.resolve()}")
    print("Next: set paths.drive_root in config/config.yaml to this folder, then")
    print("      python -m src.run_pipeline --config config/config.yaml --no-mount")


if __name__ == "__main__":
    main()
