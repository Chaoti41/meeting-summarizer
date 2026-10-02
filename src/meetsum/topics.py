"""Topic discovery: k-means over sentence embeddings, labelled with TF-IDF terms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class TopicModel:
    labels: np.ndarray        # (n,) topic id per sentence
    centroids: np.ndarray     # (k, d) L2-normalised
    names: List[str]          # human-readable label per topic

    @property
    def k(self) -> int:
        return len(self.names)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def _name_topics(texts: Sequence[str], labels: np.ndarray, k: int, top_terms: int = 3) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [" ".join(t for t, lab in zip(texts, labels) if lab == c) for c in range(k)]
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=3000)
        X = vec.fit_transform(docs)
        vocab = np.array(vec.get_feature_names_out())
        names = []
        for c in range(k):
            row = X[c].toarray().ravel()
            idx = np.argsort(-row)[:top_terms]
            terms = [vocab[i] for i in idx if row[i] > 0]
            names.append(", ".join(terms) if terms else f"topic {c}")
        return names
    except ValueError:   # empty vocabulary
        return [f"topic {c}" for c in range(k)]


def build_topics(texts: Sequence[str], emb: np.ndarray, n_topics: Optional[int] = None,
                 seed: int = 13) -> TopicModel:
    n = len(texts)
    if n == 0:
        return TopicModel(np.zeros(0, dtype=int), np.zeros((0, emb.shape[1] if emb.ndim == 2 else 0)), [])
    distinct = np.unique(emb.round(5), axis=0).shape[0]
    k = n_topics if n_topics else (1 if n < 4 else max(2, min(10, int(round(math.sqrt(n / 2))))))
    k = max(1, min(k, n, distinct))
    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)
    k = int(labels.max()) + 1
    centroids = _unit(np.stack([emb[labels == c].mean(axis=0) for c in range(k)]))
    return TopicModel(labels, centroids, _name_topics(texts, labels, k))
