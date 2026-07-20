"""Train the downstream logistic-regression classifier.

This is the only component trained from scratch (per the proposal): a simple,
interpretable linear model on top of frozen features -- either Bio_ClinicalBERT
[CLS] embeddings or TF-IDF vectors.
"""

from __future__ import annotations

from types import SimpleNamespace

from sklearn.linear_model import LogisticRegression


def train_logreg(X_train, y_train, cfg: SimpleNamespace) -> LogisticRegression:
    clf = LogisticRegression(
        C=cfg.classifier.C,
        max_iter=cfg.classifier.max_iter,
        class_weight=cfg.classifier.class_weight,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf
