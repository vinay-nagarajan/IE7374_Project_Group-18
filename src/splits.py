"""Create stratified train/val/test splits grouped by patient.

Two constraints, both important for a credible clinical result:
  1. Stratified: preserve the positive/negative ratio in every split
     (the cohort is imbalanced).
  2. Grouped by subject_id: a single patient may have several admissions;
     if the same patient appeared in both train and test, the model could
     memorise patient-specific phrasing -> leakage -> inflated scores.

We use StratifiedGroupKFold twice: once to carve out the test set, then again
on the remainder to carve out validation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _one_holdout(
    df: pd.DataFrame, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rest_idx, holdout_idx) with `frac` of rows held out."""
    n_splits = max(2, round(1.0 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, hold_idx = next(
        sgkf.split(df, y=df["label"].values, groups=df["subject_id"].values)
    )
    return rest_idx, hold_idx


def make_splits(df: pd.DataFrame, cfg: SimpleNamespace) -> pd.DataFrame:
    """Add a 'split' column with values train / val / test."""
    seed = cfg.seed
    df = df.reset_index(drop=True).copy()

    # Step 1: hold out the test set.
    rest_pos, test_pos = _one_holdout(df, cfg.split.test, seed)
    test_idx = df.index[test_pos]
    rest = df.iloc[rest_pos].reset_index(drop=True)

    # Step 2: hold out validation from what remains.
    val_frac_of_rest = cfg.split.val / (cfg.split.train + cfg.split.val)
    _, val_local = _one_holdout(rest, val_frac_of_rest, seed + 1)
    val_subjects = set(rest.iloc[val_local]["subject_id"])

    split = np.array(["train"] * len(df), dtype=object)
    split[test_idx] = "test"
    split[df["subject_id"].isin(val_subjects).values & (split != "test")] = "val"
    df["split"] = split

    summary = (
        df.groupby("split")["label"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "pos_rate"})
    )
    print("Split summary (n rows, positive rate):")
    print(summary.to_string())

    # Sanity check: no subject spans two splits.
    overlap = df.groupby("subject_id")["split"].nunique().max()
    assert overlap == 1, "Patient leakage detected across splits!"
    print("Leakage check passed: every patient is in exactly one split.")
    return df
