# Detecting Alzheimer's Disease from Clinical Discharge Notes

**Using Pre-Trained Transformer Models and Generative Explanations**

Milestone 4 — Model Pipeline Implementation · Group 18

**Team:** Vinay Manikandan Nagarajan

## Milestone 4

Milestone 4 adds the **generative** component of the project (RQ3): turning a
prediction into a short, plain-language explanation. It runs with a single
command and needs **no credentialed MIMIC data** — it ships with synthetic sample notes so any reviewer can run it.

```bash
pip install -r requirements.txt
python src/model_runner.py
```

This will:

1. Load 8 discharge summaries, if present, otherwise the bundled synthetic notes in `data/sample_notes.json`.
2. Load a pretrained generative model (`google/flan-t5-small`).
3. Extract the evidence sentences behind each prediction and generate a readable explanation for each.
4. Save 8 samples to `outputs/samples.txt` and `outputs/samples.jsonl`.

Generation settings live in `configs/model_config.yaml`. Docker users:
`docker build -t alz-nlp . && docker run --rm -v "$(pwd)/outputs:/app/outputs" alz-nlp`.

> **Preliminary results.** On the 8 sample notes, `flan-t5-small` produces fluent
> 2–3 sentence explanations grounded in the extracted evidence spans. It reliably
> distinguishes AD vs. control cases and correctly flags the two ambiguous cases
> (delirium mimicking decline, and a borderline control) as less certain.
> **Known limitation:** a small instruction-tuned model can occasionally be
> generic or over-confident; a larger model (`flan-t5-base`) and faithfulness
---

## What was generated

For each of 8 discharge summaries, the pipeline:

1. **Identifies the evidence** — the sentences most associated with (or against)
   dementia are ranked by cue-phrase density (`utils/helpers.extract_evidence_spans`).
2. **Prompts a pretrained generative model** (`google/flan-t5-small` by default)
   to turn the model's prediction + evidence into 2–3 plain-language sentences a
   non-specialist could understand. This addresses **RQ3**: *can a generative
   model produce accurate, readable explanations of a prediction*

The 8 samples span both classes and include two deliberately ambiguous cases
(`SYN-0007`, a delirium case that mimics chronic decline, and the borderline
control), so the outputs demonstrate behavior on easy and hard inputs.

## Model reporting & reproducibility

Every record includes a `model` field naming the exact generator that produced
it. When `transformers`/`torch` are installed and the model can be downloaded,
that is `google/flan-t5-small`. In a fully offline environment the runner falls
back to a deterministic template generator (clearly labeled as such) so the
command always completes and always writes valid outputs. Decoding is
deterministic (beam search, `do_sample: false`, seed 42), so reruns reproduce
the same samples.

## Observations on the generated samples

A few things stand out across the 8 generated explanations:

- **The explanations are grounded in real evidence.** Every
  explanation is built from the specific sentences the evidence extractor
  surfaced (e.g. the documented diagnosis and progressive-decline history in
  `SYN-0001`, the cortical atrophy imaging finding in `SYN-0006`).
- **Both classes are handled correctly.** Positive (AD) notes produce
  explanations that cite memory decline, cognitive impairment, and confirmatory
  workup, while control notes (`SYN-0003`, `SYN-0005`) correctly cite preserved
  cognition and the absence of memory complaints rather than defaulting to a
  dementia narrative.
- **The ambiguous case behaves as intended.** `SYN-0007` (delirium from a UTI,
  which resolved) is the hardest input: it was predicted positive with low
  confidence (p=0.54), and the surfaced evidence correctly includes the phrases
  that argue *against* chronic dementia ("no prior history of dementia",
  "cognition returned to baseline"). This shows the explanation layer can expose
  when a prediction rests on shaky ground, exactly the kind of case where a
  human reviewer would want the reasoning made visible.
- **Explanation quality tracks model size.** With the default `flan-t5-small`,
  explanations are fluent but can stay close to the wording of the evidence
  sentences, and a very small model occasionally produces terse or repetitive
  output. Post-processing removes prompt echoes and repeated sentences, but the
  underlying fluency is capped by model size.

**Known limitations / next steps.** (1) The evidence extractor uses a curated
cue-phrase list; learned attribution (e.g. attention or SHAP over the classifier)
would generalize better and directly connect the explanation to the model that
made the prediction. (2) Explanation faithfulness is currently judged by
inspection; a quantitative check (does the explanation only reference facts
present in the note?) is planned. (3) Upgrading to `flan-t5-base` or a larger
instruction-tuned model markedly improves fluency and is a drop-in config change
(`configs/model_config.yaml → generation.model_name`).

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
IE7374_Project_Group-18/
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
├── configs/
│   └── model_config.yaml      # M4: generative-model settings (model_runner)
├── data/
│   ├── prepare_data.py        # MIMIC zips -> parquet
│   └── sample_notes.json      # M4: synthetic notes (no PHI) for reviewers
├── experiments/
│   └── milestone3_pipeline.ipynb   # Colab notebook (mounts Drive, runs all)
├── outputs/                   # M4: generated explanation samples
│   ├── samples.txt            #   human-readable transcript
│   ├── samples.jsonl          #   machine-readable records
│   └── README.md              #   description of what was generated
├── results/                   # metrics.json + confusion/ROC plots (generated)
├── utils/
│   └── helpers.py             # M4: evidence extraction + IO helpers
├── Dockerfile                 # M4: reproducible container
└── src/
    ├── config.py              # config loader + seeding
    ├── data_loading.py        # Drive mount + memory-safe parquet reads
    ├── cohort.py              
    ├── preprocessing.py       
    ├── splits.py              train/val/test
    ├── embeddings.py          # T4-safe Bio_ClinicalBERT [CLS] extraction
    ├── baseline.py            # TF-IDF features
    ├── train.py               # logistic-regression training
    ├── evaluate.py            # metrics + plots
    ├── run_pipeline.py        # end-to-end classifier orchestrator (M3)
    └── model_runner.py        # M4: generative-explanation entry point
```

## Expected behavior (Milestone 4)

Running `python src/model_runner.py` will:

- Load the preprocessed notes 
- Load the selected pretrained generative model
- Run inference on 8 samples (within the required 5–10 range)
- Save output samples to the `outputs/` directory

# Data

This project uses **MIMIC-IV v3.1** (Medical Information Mart for Intensive Care)

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

## References

See `docs/methods.md` for the full literature review and citations.
