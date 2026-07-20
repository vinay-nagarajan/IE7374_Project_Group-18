# Methods: Research & Selection

This document covers rubric criterion 1 (Research and Selection of Methods):
objectives, literature review, benchmarking, and preliminary experiments.

## 1. Objectives

The task is **binary text classification** over clinical discharge summaries:
label a note as coming from a patient with Alzheimer's disease / related
dementia (positive) or not (negative). A secondary objective is
**interpretability** — surfacing which parts of a note drove a prediction and
translating them into plain language with a generative model.

## 2. Literature review

- **Bio_ClinicalBERT** (Alsentzer et al., 2019) further pre-trains BERT on ~2M
  MIMIC-III clinical notes, giving representations tuned to clinical
  abbreviations, note structure, and medical vocabulary. It is the natural
  encoder for MIMIC discharge text.
- **ClinicalBERT** (Huang et al., 2019) established that clinical-domain
  pretraining improves downstream tasks (e.g. readmission) over general BERT.
- **Linguistic markers of AD** (Fraser et al., 2016) show that cognitive decline
  produces measurable linguistic signals, motivating a text-only detector.
- **MIMIC-IV** (Johnson et al., 2023) is the source EHR dataset; the Note module
  supplies the discharge summaries.

**Why frozen, not fine-tuned.** Fine-tuning BERT on a few thousand long notes is
GPU-heavy and prone to overfitting a small, imbalanced positive class. Using the
encoder frozen and training only a linear head is (a) feasible on a free T4,
(b) a clean test of how much AD signal the pretrained representation already
contains (RQ1), and (c) far more reproducible.

## 3. Benchmarking / method comparison

| Method | Accuracy potential | Compute | Pretrained available | Interpretability |
|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | Moderate | Very low | n/a | High (feature weights) |
| **Bio_ClinicalBERT (frozen) + LogReg** | High | Low (inference only) | Yes | Medium (attention + head) |
| Fine-tuned Bio_ClinicalBERT | Highest | High (backprop, OOM risk on T4) | Yes | Low |
| Decoder LLM zero-shot classify | Variable | High | Yes | Low, costly |

The frozen-encoder + linear-head design is the best accuracy/compute trade-off
for a T4 and directly enables the RQ1 comparison against TF-IDF.

**Frameworks chosen:** Hugging Face `transformers` for the encoder,
`scikit-learn` for TF-IDF and the logistic-regression head, `pyarrow` for
memory-safe parquet access. Justification: mature, well-documented, and all run
comfortably on Colab's T4 without custom CUDA code.

## 4. Cohort definition (ICD codes)

- **Positive:** admissions with an AD/dementia ICD code.
  - ICD-9: `331.0` (Alzheimer's); dementias `290.x`, `294.1x`, `294.2x`.
  - ICD-10: `G30.x` (Alzheimer's); `F00`–`F03` (dementias).
  - MIMIC stores codes with no decimal point and upper-cased, so matching is by
    normalised prefix (`331.0` → `3310`, `G30.9` → `G309`).
- **Negative:** patients with **no** dementia code in **any** admission,
  age-matched to positives within ±5 years, sampled at a 3:1 ratio. Defining
  controls at the patient level (not admission level) prevents label leakage.

Exact code lists live in `config/config.yaml` and can be tuned without touching
source.

## 5. Preliminary experiments / feasibility notes

- **Memory feasibility.** A full load of `discharge_notes.parquet` (1.74 GB) was
  rejected as infeasible on a free Colab host (~13 GB RAM). The filtered-read
  approach (only cohort `hadm_id`s) keeps note memory in the low hundreds of MB.
- **VRAM feasibility.** Bio_ClinicalBERT in fp16 with batch size 16 and 512
  tokens keeps peak VRAM well under the T4's 16 GB; batch size can be raised to
  32 if the runtime is stable.
- **Class imbalance.** Positives are a small fraction of admissions, so
  evaluation centres on **F1 and AUC-ROC**, and the classifier uses
  `class_weight="balanced"`.
- **Data augmentation.** Standard text augmentation (back-translation, synonym
  swap) is **not** applied: it risks corrupting clinically precise language and
  adds little with frozen embeddings. Imbalance is handled via sampling ratio,
  class weighting, and threshold-independent metrics instead. (Rubric item is
  "if applicable"; here it is not.)

## References

- Alsentzer, E. et al. (2019). *Publicly Available Clinical BERT Embeddings.*
  Clinical NLP Workshop.
- Huang, K. et al. (2019). *ClinicalBERT: Modeling Clinical Notes and Predicting
  Hospital Readmission.* arXiv:1904.05342.
- Johnson, A. et al. (2023). *MIMIC-IV, a freely accessible electronic health
  record dataset.* Scientific Data 10(1).
- Fraser, K. et al. (2016). *Linguistic Features Identify Alzheimer's Disease in
  Narrative Speech.* J. Alzheimer's Disease 49(2), 407–422.
