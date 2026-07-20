# Data

This project uses **MIMIC-IV v3.1** (Medical Information Mart for Intensive Care),
a de-identified EHR dataset from Beth Israel Deaconess Medical Center, maintained
by the MIT Laboratory for Computational Physiology.

## Access & licensing

MIMIC-IV is **credentialed** data. To obtain it you must:

1. Become a credentialed PhysioNet user.
2. Complete the CITI "Data or Specimens Only Research" training.
3. Sign the data use agreement at
   <https://physionet.org/content/mimiciv/>.

**No MIMIC data is stored in this repository**. The pipeline reads parquet exports which can be obtained by running prepare_data.py upon obtaining MIMIC-IV access and pointing the .parquet file(s) locations set via `paths.drive_root` in
`config/config.yaml`.

## Tables used

Only four tables are needed. File names match the parquet files:

| File | Rows (approx) | Columns used | Purpose |
|---|---|---|---|
| `patients.parquet` | ~300k | `subject_id, anchor_age, gender` | demographics, age matching |
| `admissions.parquet` | ~430k | `subject_id, hadm_id, admittime` | pick one control admission |
| `diagnoses_icd.parquet` | ~6M | `subject_id, hadm_id, icd_code, icd_version` | ICD-based labelling |
| `discharge_notes.parquet` | ~330k | `subject_id, hadm_id, note_type, text` | discharge summary text |

## ICD codes for the positive class

- ICD-9: `331.0` (Alzheimer's), `290.x`, `294.1x`, `294.2x` (dementias) →
  stored as `3310`, `290...`, `2941...`, `2942...`
- ICD-10: `G30.x` (Alzheimer's), `F00`–`F03` (dementias) → stored as `G30...`,
  `F00`–`F03`

Codes are configurable in `config/config.yaml → cohort`.

## Ethics

All notes are de-identified (PHI replaced with placeholders). Use is restricted
to the terms of the PhysioNet data use agreement. Do not attempt re-identification
and do not redistribute the data.
