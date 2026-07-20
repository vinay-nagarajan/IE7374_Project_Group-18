"""Evaluate a trained classifier and persist metrics + plots.

Reports the metrics named in the proposal: F1 (primary), precision, recall,
accuracy, plus AUC-ROC for the imbalanced setting. Saves a JSON of numbers, a
confusion-matrix PNG and an ROC-curve PNG per model so results land straight in
results/ for the repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (works in Colab + CI)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
    }


def _plot_confusion(y_true, y_pred, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No AD", "AD"]); ax.set_yticklabels(["No AD", "AD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _plot_roc(y_true, y_proba, title: str, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def evaluate_model(clf, X, y_true, name: str, results_dir: str) -> dict:
    """Evaluate, print, plot, and return the metrics dict for one model."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_pred = clf.predict(X)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X)[:, 1]
    else:  # linear SVM etc.
        y_proba = clf.decision_function(X)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")

    _plot_confusion(y_true, y_pred, f"{name} - Confusion", out / f"cm_{name}.png")
    _plot_roc(y_true, y_proba, f"{name} - ROC", out / f"roc_{name}.png")
    return metrics


def save_results(all_metrics: dict, results_dir: str) -> None:
    path = Path(results_dir) / "metrics.json"
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics -> {path}")
