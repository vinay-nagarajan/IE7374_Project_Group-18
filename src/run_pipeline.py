"""End-to-end data pipeline: Drive -> cohort -> notes -> features -> models.

Run from the repo root:
    python -m src.run_pipeline --config config/config.yaml

Or import and call run() from the Colab notebook. Every stage prints progress
and writes artifacts so the run is transparent and resumable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baseline import build_tfidf
from src.cohort import build_cohort
from src.config import ensure_dirs, load_config, set_seed
from src.data_loading import load_notes_for_hadms, load_small_table, mount_drive
from src.embeddings import extract_cls_embeddings
from src.evaluate import evaluate_model, save_results
from src.preprocessing import preprocess_notes
from src.splits import make_splits
from src.train import train_logreg


def run(config_path: str = "config/config.yaml", do_mount: bool = True) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    root = cfg.paths.drive_root

    # -------------------------------------------------------------- stage 1
    print("\n[1/7] Loading MIMIC-IV tables from Drive")
    if do_mount:
        mount_drive()
    patients = load_small_table(
        root, cfg.paths.patients, ["subject_id", "anchor_age", "gender"]
    )
    admissions = load_small_table(
        root, cfg.paths.admissions, ["subject_id", "hadm_id", "admittime"]
    )
    diagnoses = load_small_table(
        root, cfg.paths.diagnoses,
        ["subject_id", "hadm_id", "icd_code", "icd_version"],
    )

    # -------------------------------------------------------------- stage 2
    print("\n[2/7] Building age-matched cohort from ICD codes")
    cohort = build_cohort(diagnoses, patients, admissions, cfg)
    cohort_path = Path(cfg.paths.artifacts_dir) / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)

    # -------------------------------------------------------------- stage 3
    print("\n[3/7] Reading ONLY the cohort's discharge notes (filtered scan)")
    notes = load_notes_for_hadms(
        root, cfg.paths.notes, cohort["hadm_id"].tolist()
    )

    # -------------------------------------------------------------- stage 4
    print("\n[4/7] Preprocessing note text")
    dataset = preprocess_notes(cohort, notes, cfg)
    del notes  # free the biggest object as soon as possible
    dataset = make_splits(dataset, cfg)
    dataset_path = Path(cfg.paths.artifacts_dir) / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    print(f"Saved labelled dataset -> {dataset_path}")

    train = dataset[dataset.split == "train"]
    val = dataset[dataset.split == "val"]
    test = dataset[dataset.split == "test"]

    # -------------------------------------------------------------- stage 5
    print("\n[5/7] Extracting frozen Bio_ClinicalBERT embeddings (T4-safe)")
    cache = str(Path(cfg.paths.artifacts_dir) / cfg.embeddings.cache_file)
    emb_all = extract_cls_embeddings(
        dataset["text"].tolist(), cfg, cache_path=cache
    )
    idx = {s: dataset.index[dataset.split == s] for s in ("train", "val", "test")}
    Xb_train, Xb_val, Xb_test = (emb_all[idx["train"]],
                                 emb_all[idx["val"]], emb_all[idx["test"]])

    # -------------------------------------------------------------- stage 6
    print("\n[6/7] Building TF-IDF baseline")
    vec, Xt_train = build_tfidf(train["text"].tolist(), cfg)
    Xt_val = vec.transform(val["text"].tolist())
    Xt_test = vec.transform(test["text"].tolist())

    # -------------------------------------------------------------- stage 7
    print("\n[7/7] Training + evaluating both models")
    y_train, y_val, y_test = train["label"].values, val["label"].values, test["label"].values

    clf_bert = train_logreg(Xb_train, y_train, cfg)
    clf_tfidf = train_logreg(Xt_train, y_train, cfg)

    results = {
        "clinicalbert_val": evaluate_model(
            clf_bert, Xb_val, y_val, "ClinicalBERT_val", cfg.paths.results_dir),
        "clinicalbert_test": evaluate_model(
            clf_bert, Xb_test, y_test, "ClinicalBERT_test", cfg.paths.results_dir),
        "tfidf_val": evaluate_model(
            clf_tfidf, Xt_val, y_val, "TFIDF_val", cfg.paths.results_dir),
        "tfidf_test": evaluate_model(
            clf_tfidf, Xt_test, y_test, "TFIDF_test", cfg.paths.results_dir),
    }
    save_results(results, cfg.paths.results_dir)

    # persist trained artifacts
    joblib.dump(clf_bert, Path(cfg.paths.models_dir) / "logreg_clinicalbert.joblib")
    joblib.dump(clf_tfidf, Path(cfg.paths.models_dir) / "logreg_tfidf.joblib")
    joblib.dump(vec, Path(cfg.paths.models_dir) / "tfidf_vectorizer.joblib")
    print("\nSaved trained models -> models/")
    print("\nPipeline complete.")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Alzheimer's clinical-NLP data pipeline")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--no-mount", action="store_true",
                    help="skip Google Drive mount (local runs)")
    args = ap.parse_args()
    run(args.config, do_mount=not args.no_mount)


if __name__ == "__main__":
    main()
