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
│   └── README.md              # dataset documentation + access instructions
├── docs/
│   ├── methods.md             # method selection, literature, benchmarking
│   └── pipeline.md            # stage-by-stage pipeline walkthrough
├── experiments/
│   └── milestone3_pipeline.ipynb   # Colab notebook (mounts Drive, runs all)
├── models/                    # trained classifiers (generated)
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

## Data access

MIMIC-IV is **credentialed** data (PhysioNet + CITI training). It is **not**
included in this repo and must never be committed. This project reads the
parquet exports stored in the author's Google Drive at
`My Drive/ADRD_IDR/mimic/`. See [`data/README.md`](data/README.md) for the full
list of tables and how to obtain access.

## Quickstart (Google Colab, T4 GPU) — recommended

1. Open `experiments/milestone3_pipeline.ipynb` in Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run the cells top to bottom. The notebook mounts your Drive, installs
   `transformers`, and runs the whole pipeline via `src.run_pipeline.run()`.

The pipeline is engineered so the **1.74 GB notes table is never fully loaded**
and embedding extraction stays well within the T4's 16 GB — see
[Why it fits on a T4](#why-it-fits-on-a-t4).

## Quickstart (local / CLI)

```bash
git clone <your-repo-url> alzheimers-clinical-nlp
cd alzheimers-clinical-nlp
pip install -r requirements.txt

# edit config/config.yaml -> paths.drive_root to point at your MIMIC folder
python -m src.run_pipeline --config config/config.yaml --no-mount
```

Outputs land in `artifacts/` (cohort, dataset, cached embeddings),
`models/` (trained `.joblib` files), and `results/` (`metrics.json`, plots).

## Why it fits on a T4

| Risk | Mitigation |
|---|---|
| `discharge_notes.parquet` is 1.74 GB | Cohort is built from the small tables first; notes are read with a **pyarrow `hadm_id` filter** so only the few-thousand cohort rows are materialised. |
| BERT activations exhaust 16 GB VRAM | `torch.no_grad()`, `model.eval()`, **fp16** weights, **batch size 16**, embeddings moved to CPU each batch. |
| GPU memory fragments over a long run | `torch.cuda.empty_cache()` + `gc.collect()` periodically; model freed after extraction. |
| Kernel restart loses hours of work | Embeddings **cached to `.npz`**; re-running skips recomputation. |
| Class imbalance | Stratified splits + `class_weight="balanced"` + F1/AUC-ROC reporting. |
| Patient leakage across splits | Splits are **grouped by `subject_id`** (StratifiedGroupKFold) with an explicit leakage assertion. |

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
| 3. GitHub repository | this structure, meaningful commits, `.gitignore` |
| 4. Documentation & reproducibility | this README, `docs/`, `data/README.md` |

## References

See `docs/methods.md` for the full literature review and citations.
