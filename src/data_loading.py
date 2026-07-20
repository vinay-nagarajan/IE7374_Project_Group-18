"""Load MIMIC-IV parquet tables from Google Drive without blowing up RAM.

The discharge_notes.parquet file is ~1.74 GB. Reading its full `text` column
into a pandas DataFrame can use 5-8 GB of RAM and will crash a free Colab
runtime (~13 GB). The core trick in this module: we NEVER read the whole notes
table. We first build the cohort from the small tables (patients, admissions,
diagnoses -- all < 25 MB), then read only the note rows for the cohort's
hadm_ids using pyarrow predicate pushdown. Memory stays in the low hundreds
of MB the whole time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def mount_drive() -> None:
    """Mount Google Drive when running in Colab. No-op elsewhere."""
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        print("Google Drive mounted at /content/drive")
    except ImportError:
        print("Not running in Colab -- assuming files are already accessible.")


def _resolve(root: str, name: str) -> Path:
    p = Path(root) / name
    if not p.exists():
        raise FileNotFoundError(
            f"Expected file not found: {p}\n"
            "Check paths.drive_root in config.yaml and that Drive is mounted."
        )
    return p


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MIMIC parquet columns are lower-case, but be defensive."""
    df.columns = [str(c).lower() for c in df.columns]
    return df


def load_small_table(
    root: str, name: str, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read a small parquet table (patients / admissions / diagnoses).

    `columns` prunes at read time so we only pull what we need.
    """
    path = _resolve(root, name)
    df = pd.read_parquet(path, columns=list(columns) if columns else None)
    df = _normalise_columns(df)
    print(f"Loaded {name}: {len(df):,} rows, {df.shape[1]} cols "
          f"({path.stat().st_size / 1e6:.1f} MB on disk)")
    return df


def load_notes_for_hadms(
    root: str,
    name: str,
    hadm_ids: Iterable[int],
    columns: Sequence[str] = ("subject_id", "hadm_id", "note_type", "text"),
) -> pd.DataFrame:
    """Read ONLY the discharge-note rows whose hadm_id is in `hadm_ids`.

    Uses pyarrow's dataset filter so matching happens during the scan; the full
    1.74 GB text column is never materialised in memory. This is the single most
    important step for keeping the T4 runtime alive.
    """
    import pyarrow.parquet as pq  # lazy: only needed for the filtered scan

    path = _resolve(root, name)
    hadm_list = [int(h) for h in hadm_ids]
    print(f"Reading notes for {len(hadm_list):,} target admissions "
          f"(filtered scan, not a full load)...")

    # `filters` triggers predicate pushdown; only matching rows are built.
    table = pq.read_table(
        path,
        columns=list(columns),
        filters=[("hadm_id", "in", hadm_list)],
    )
    df = table.to_pandas()
    df = _normalise_columns(df)
    print(f"  -> materialised {len(df):,} note rows "
          f"(~{df['text'].str.len().sum() / 1e6:.0f} MB of text)")
    return df
