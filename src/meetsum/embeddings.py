"""Sentence embeddings: Sentence-BERT (default) and an offline hashing fallback."""

from __future__ import annotations

import re
import zlib
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    """Thin wrapper around sentence-transformers returning L2-normalised float32 vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for Sentence-BERT embeddings: "
                "pip install 'meetsum[sbert]'  (or use embedding_model='hash' for the offline fallback)"
            ) from e
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        emb = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        return emb.astype(np.float32)


_STOP = frozenset("a an the and or but if of to in on at for with is are was were be been it this that "
                  "i you we they he she so just um uh yeah okay ok".split())


class HashingEmbedder:
    """Deterministic bag-of-(uni+bi)grams hashing embedder. Offline; used for tests and quick demos."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOP]
            feats: List[str] = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = zlib.crc32(f.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def get_embedder(name: str = DEFAULT_MODEL):
    if name.lower() in {"hash", "hashing"}:
        return HashingEmbedder()
    return SentenceEmbedder(name)
