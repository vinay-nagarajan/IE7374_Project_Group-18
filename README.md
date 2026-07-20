# Detecting Alzheimer's Disease from Clinical Discharge Notes

**Using Pre-Trained Transformer Models and Generative Explanations**

Milestone 3 — Data Pipeline · Group 18

**Team:** Vinay Manikandan Nagarajan

---

## Overview

This project detects Alzheimer's disease (AD) from free-text hospital discharge
summaries in **MIMIC-IV**. The approach:

1. Use **Bio_ClinicalBERT** (frozen) as a feature extractor — each note is
   turned into its 768-dim `[CLS]` embedding, with no fine-tuning.
2. Train a **logistic-regression** classifier on those embeddings.
3. Compare against a classic **TF-IDF + logistic-regression baseline** on the
   exact same splits (RQ1).
4. (Later milestone) feed the most-important note segments into a **generative
   LLM** to produce plain-language explanations of each prediction (RQ3).

This milestone delivers the full **data pipeline and model implementation**:
cohort construction from ICD codes, note extraction, preprocessing, splitting,
embedding extraction, baseline, training, and evaluation.

## Research questions

- **RQ1** — Do frozen Bio_ClinicalBERT embeddings beat a TF-IDF baseline at
  distinguishing AD patients from controls?
- **RQ2** — Which linguistic/clinical features are most predictive of AD?
- **RQ3** — Can a generative model produce accurate, readable explanations of a
  prediction?

## Repository structure

```
alzheimers-clinical-nlp/
├── README.md                  # this file
├── requirements.txt           # pip dependencies
├── environment.yml            # conda alternative
├── config/
│   └── config.yaml            # all paths + hyperparameters (edit here, not code)
├── data/
│   └── prepare_data.py              # to prepare data by creating .parquet from dataset
├── docs/
│   ├── methods.md             # method selection, literature, benchmarking
│   └── pipeline.md            # stage-by-stage pipeline walkthrough
├── experiments/
│   └── milestone3_pipeline.ipynb   # Colab notebook (mounts Drive, runs all)
├── results/                   # metrics.json + confusion/ROC plots (generated)
└── src/
    ├── config.py              # config loader + seeding
    ├── data_loading.py        # Drive mount + memory-safe parquet reads
    ├── cohort.py              # ICD labelling + age-matched controls
    ├── preprocessing.py       # de-id removal + text cleaning
    ├── splits.py              # stratified, patient-grouped train/val/test
    ├── embeddings.py          # T4-safe Bio_ClinicalBERT [CLS] extraction
    ├── baseline.py            # TF-IDF features
    ├── train.py               # logistic-regression training
    ├── evaluate.py            # metrics + plots
    └── run_pipeline.py        # end-to-end orchestrator
```

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

## Quickstart

1. Open `experiments/milestone3_pipeline.ipynb` in a GPU environment.
3. Run the cells top to bottom. The notebook mounts your Drive, installs
   `transformers`, and runs the whole pipeline via `src.run_pipeline.run()`.


## Quickstart (local / CLI)

```bash
git clone <your-repo-url> alzheimers-clinical-nlp
cd alzheimers-clinical-nlp
pip install -r requirements.txt

python -m src.run_pipeline --config config/config.yaml --no-mount
```

## Reproducibility

- One config file drives everything; a fixed `seed` (42) seeds Python, NumPy and
  Torch.
- `python -m src.run_pipeline` reproduces the full run end to end.
- Intermediate artifacts are written to disk so any stage can be inspected.

## Rubric mapping

| Rubric criterion | Where |
|---|---|
| 1. Research & selection of methods | `docs/methods.md` |
| 2. Model implementation | `src/` (modular), `config/config.yaml`, `results/` |
| 3. Documentation & reproducibility | this README, `docs/` |

## References

See `docs/methods.md` for the full literature review and citations.
