# Pipeline Walkthrough

Seven stages, orchestrated by `src/run_pipeline.py`. Each stage prints progress
and writes an artifact so runs are transparent and resumable.

```
Drive (parquet)
   │
   ▼
[1] load small tables ──► patients / admissions / diagnoses  (< 25 MB each)
   │
   ▼
[2] build cohort ───────► ICD labelling + age-matched controls
   │                       → artifacts/cohort.parquet
   ▼
[3] load notes ─────────► filtered scan on cohort hadm_ids ONLY
   │                       (1.74 GB file, but only cohort rows materialised)
   ▼
[4] preprocess ─────────► de-id removal, cleaning, min-length filter
   │                       + stratified patient-grouped split
   │                       → artifacts/dataset.parquet
   ▼
[5] embeddings ─────────► frozen Bio_ClinicalBERT [CLS] (fp16, batched)
   │                       → artifacts/clinicalbert_cls_embeddings.npz (cached)
   ▼
[6] baseline ───────────► TF-IDF (fit on train only)
   │
   ▼
[7] train + evaluate ───► logistic regression on both feature sets
                           → models/*.joblib, results/metrics.json + plots
```

## Stage details

**[1] Load small tables** — `data_loading.load_small_table` prunes to only the
columns each stage needs (e.g. diagnoses → `subject_id, hadm_id, icd_code,
icd_version`).

**[2] Cohort** — `cohort.build_cohort` normalises ICD codes, flags dementia
admissions/patients, and samples age-matched controls without replacement. The
output is the exact set of `hadm_id`s whose notes we need.

**[3] Notes** — `data_loading.load_notes_for_hadms` uses a pyarrow
`filters=[("hadm_id", "in", ...)]` predicate so the scan only builds cohort
rows. This is the key memory-safety step for the 1.74 GB table.

**[4] Preprocess + split** — `preprocessing.preprocess_notes` strips MIMIC
de-identification markers (`[** **]`, `___`), collapses whitespace, lowercases,
and drops near-empty notes. `splits.make_splits` produces train/val/test that
are both stratified by label and grouped by patient, with an explicit leakage
assertion.

**[5] Embeddings** — `embeddings.extract_cls_embeddings` runs the frozen encoder
in fp16 with `no_grad`, small batches, and per-batch CPU offload, caching the
result to `.npz`.

**[6] Baseline** — `baseline.build_tfidf` fits TF-IDF (uni+bigrams, 10k
features) on the training texts only.

**[7] Train + evaluate** — `train.train_logreg` fits logistic regression on each
feature set; `evaluate.evaluate_model` reports F1/precision/recall/accuracy/
AUC-ROC and saves confusion-matrix and ROC plots per model.

## Outputs

| Path | Contents |
|---|---|
| `artifacts/cohort.parquet` | labelled admissions (subject_id, hadm_id, age, gender, label) |
| `artifacts/dataset.parquet` | cleaned notes + labels + split assignment |
| `artifacts/clinicalbert_cls_embeddings.npz` | cached 768-dim embeddings |
| `models/logreg_clinicalbert.joblib` | trained classifier (embedding features) |
| `models/logreg_tfidf.joblib`, `tfidf_vectorizer.joblib` | baseline artifacts |
| `results/metrics.json` | all metrics, both models, val + test |
| `results/cm_*.png`, `results/roc_*.png` | plots per model |
