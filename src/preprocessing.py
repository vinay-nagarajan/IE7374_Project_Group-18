"""Clean MIMIC-IV discharge summaries before tokenisation.

MIMIC-IV-Note de-identifies PHI by replacing it with underscore runs ("___")
and, in some exports, bracketed tags like "[**Hospital1 18**]". Both are noise
for the model, so we strip them. We deliberately do NOT truncate here -- the
tokenizer handles the 512-token cap at embedding time, which is the correct
place to enforce BERT's limit (truncation on characters would be wrong).
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd

# MIMIC-III-style bracket tags: [** ... **]
_BRACKET_DEID = re.compile(r"\[\*\*.*?\*\*\]")
# MIMIC-IV-style underscore runs (2+ underscores)
_UNDERSCORE_DEID = re.compile(r"_{2,}")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str, cfg: SimpleNamespace) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    if cfg.preprocessing.remove_deid:
        t = _BRACKET_DEID.sub(" ", t)
        t = _UNDERSCORE_DEID.sub(" ", t)
    if cfg.preprocessing.collapse_whitespace:
        t = _MULTISPACE.sub(" ", t)
        t = _MULTINEWLINE.sub("\n\n", t)
        t = t.strip()
    if cfg.preprocessing.lowercase:
        t = t.lower()
    return t


def preprocess_notes(
    cohort: pd.DataFrame, notes: pd.DataFrame, cfg: SimpleNamespace
) -> pd.DataFrame:
    """Join cohort labels to note text, clean, and drop empties/dupes.

    Returns columns: subject_id, hadm_id, label, text
    """
    notes = notes.copy()

    # Keep only discharge summaries if a note_type column is present.
    if "note_type" in notes.columns:
        mask = notes["note_type"].astype(str).str.upper().str.startswith("DS")
        if mask.any():
            notes = notes[mask]

    # One note per admission (deterministic: first occurrence).
    if cfg.cohort.one_note_per_admission:
        notes = notes.drop_duplicates("hadm_id", keep="first")

    merged = cohort.merge(
        notes[["hadm_id", "text"]], on="hadm_id", how="inner"
    )

    merged["text"] = merged["text"].apply(lambda x: clean_text(x, cfg))
    before = len(merged)
    merged = merged[merged["text"].str.len() >= cfg.cohort.min_note_chars]
    print(f"Preprocessing: {before:,} -> {len(merged):,} notes after "
          f"cleaning + min-length filter ({cfg.cohort.min_note_chars} chars)")

    out = merged[["subject_id", "hadm_id", "label", "text"]].reset_index(drop=True)
    n_pos = int((out["label"] == 1).sum())
    print(f"Final dataset: {len(out):,} notes "
          f"({n_pos:,} positive / {len(out) - n_pos:,} negative)")
    return out
