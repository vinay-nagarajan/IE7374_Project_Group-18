"""TF-IDF baseline features for the apples-to-apples comparison (RQ1).

Same classifier and same splits as the Bio_ClinicalBERT path -- the only thing
that changes is the feature representation. This is what lets us answer whether
pretrained clinical embeddings actually beat classic bag-of-ngrams.
"""

from __future__ import annotations

from types import SimpleNamespace

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    train_texts: list[str], cfg: SimpleNamespace
) -> tuple[TfidfVectorizer, spmatrix]:
    """Fit TF-IDF on the training texts only (no test leakage)."""
    vec = TfidfVectorizer(
        max_features=cfg.baseline.max_features,
        ngram_range=(cfg.baseline.ngram_min, cfg.baseline.ngram_max),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vec.fit_transform(train_texts)
    print(f"TF-IDF fitted: vocab={len(vec.vocabulary_):,}, "
          f"train matrix={X_train.shape}")
    return vec, X_train
