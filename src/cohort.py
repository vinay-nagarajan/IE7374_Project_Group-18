"""Build the labelled cohort from ICD codes with age-matched controls.

Positive class  : admissions carrying an Alzheimer's / dementia ICD code.
Negative class  : patients who NEVER carry any dementia code in ANY admission,
                  age-matched to the positives (avoids label leakage).

The cohort is computed entirely from the small tables, producing the exact set
of hadm_ids we then hand to data_loading.load_notes_for_hadms().
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


def normalise_icd(code: str) -> str:
    """Upper-case, strip whitespace and dots -> matches MIMIC storage format."""
    return str(code).upper().replace(".", "").strip()


def _matches(code: str, prefixes: list[str]) -> bool:
    c = normalise_icd(code)
    return any(c.startswith(p.upper()) for p in prefixes)


def identify_dementia(diagnoses: pd.DataFrame, cfg: SimpleNamespace):
    """Return (dementia_hadm_ids, dementia_subject_ids) sets.

    diagnoses columns used: subject_id, hadm_id, icd_code, icd_version
    """
    df = diagnoses.copy()
    df["icd_code"] = df["icd_code"].astype(str)

    is_v9 = df["icd_version"] == 9
    is_v10 = df["icd_version"] == 10

    hit9 = is_v9 & df["icd_code"].apply(
        lambda c: _matches(c, list(cfg.cohort.icd9_prefixes))
    )
    hit10 = is_v10 & df["icd_code"].apply(
        lambda c: _matches(c, list(cfg.cohort.icd10_prefixes))
    )
    dementia_rows = df[hit9 | hit10]

    dementia_hadm = set(dementia_rows["hadm_id"].dropna().astype(int))
    dementia_subj = set(dementia_rows["subject_id"].dropna().astype(int))
    print(f"Dementia diagnoses found: {len(dementia_hadm):,} admissions, "
          f"{len(dementia_subj):,} unique patients")
    return dementia_hadm, dementia_subj


def _age_match_controls(
    pos_patients: pd.DataFrame,
    control_pool: pd.DataFrame,
    ratio: int,
    tolerance: int,
    seed: int,
) -> pd.DataFrame:
    """For each positive patient sample `ratio` controls within +/- tolerance yrs.

    Sampling is without replacement so no control is used twice.
    Both frames need columns: subject_id, anchor_age.
    """
    rng = np.random.default_rng(seed)
    pool = control_pool[["subject_id", "anchor_age"]].dropna().copy()
    pool["anchor_age"] = pool["anchor_age"].astype(int)
    used: set[int] = set()
    chosen: list[int] = []

    # Shuffle positives so age competition isn't order-biased.
    pos = pos_patients.sample(frac=1.0, random_state=seed)
    for age in pos["anchor_age"].astype(int):
        cand = pool[
            (pool["anchor_age"].between(age - tolerance, age + tolerance))
            & (~pool["subject_id"].isin(used))
        ]
        if len(cand) == 0:
            continue
        take = cand.sample(n=min(ratio, len(cand)), random_state=rng.integers(1e9))
        ids = take["subject_id"].tolist()
        used.update(ids)
        chosen.extend(ids)

    matched = control_pool[control_pool["subject_id"].isin(chosen)].copy()
    print(f"Age-matched controls selected: {len(matched):,} patients")
    return matched


def _pick_admissions(
    subjects: pd.DataFrame, admissions: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """One admission per control patient (earliest, for determinism)."""
    adm = admissions[admissions["subject_id"].isin(subjects["subject_id"])].copy()
    if "admittime" in adm.columns:
        adm = adm.sort_values("admittime")
    adm = adm.drop_duplicates("subject_id", keep="first")
    return adm[["subject_id", "hadm_id"]]


def build_cohort(
    diagnoses: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Return a frame: subject_id, hadm_id, anchor_age, gender, label.

    This defines exactly which notes we need to pull.
    """
    seed = cfg.seed
    dementia_hadm, dementia_subj = identify_dementia(diagnoses, cfg)

    # ---- Positives (admission level) ----
    hadm_to_subj = (
        diagnoses[["hadm_id", "subject_id"]].dropna().drop_duplicates()
    )
    pos = pd.DataFrame({"hadm_id": sorted(dementia_hadm)})
    pos = pos.merge(hadm_to_subj, on="hadm_id", how="left")
    pos = pos.merge(
        patients[["subject_id", "anchor_age", "gender"]], on="subject_id", how="left"
    )
    pos["label"] = 1

    # ---- Controls (patient level, then one admission each) ----
    control_pool = patients[~patients["subject_id"].isin(dementia_subj)]
    pos_patients = pos.drop_duplicates("subject_id")[["subject_id", "anchor_age"]]
    control_patients = _age_match_controls(
        pos_patients,
        control_pool,
        ratio=cfg.cohort.control_ratio,
        tolerance=cfg.cohort.age_tolerance,
        seed=seed,
    )
    ctrl_adm = _pick_admissions(control_patients, admissions, seed)
    neg = ctrl_adm.merge(
        patients[["subject_id", "anchor_age", "gender"]], on="subject_id", how="left"
    )
    neg["label"] = 0

    cols = ["subject_id", "hadm_id", "anchor_age", "gender", "label"]
    cohort = pd.concat([pos[cols], neg[cols]], ignore_index=True)
    cohort = cohort.dropna(subset=["hadm_id"]).drop_duplicates("hadm_id")
    cohort["hadm_id"] = cohort["hadm_id"].astype(int)

    n_pos = int((cohort["label"] == 1).sum())
    n_neg = int((cohort["label"] == 0).sum())
    print(f"Cohort assembled: {len(cohort):,} admissions "
          f"({n_pos:,} positive / {n_neg:,} negative)")
    return cohort.reset_index(drop=True)
