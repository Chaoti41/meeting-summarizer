"""Personalized PageRank (seeded by action verbs / dates) and Maximal Marginal Relevance."""

from __future__ import annotations

from typing import List, Optional, Sequence

import networkx as nx
import numpy as np

from .config import RankConfig
from .graph import sent_node
from .preprocess import SentenceRecord


def seed_weights(sentences: Sequence[SentenceRecord], cfg: RankConfig) -> np.ndarray:
    """Raw seed weight per sentence: action verbs (boosted by commitment cues) and dates."""
    w = np.zeros(len(sentences))
    for i, s in enumerate(sentences):
        if s.has_verb:
            w[i] += cfg.seed_verb * (1.0 + cfg.seed_cue_boost * float(s.has_cue))
        if s.has_date:
            w[i] += cfg.seed_date
    return w


def personalized_pagerank(G: nx.Graph, sentences: Sequence[SentenceRecord], cfg: RankConfig) -> np.ndarray:
    """PPR scores for sentence nodes. Restart distribution = (1-mix)*seeds + mix*uniform."""
    n = len(sentences)
    if n == 0:
        return np.zeros(0)
    seeds = seed_weights(sentences, cfg)
    uniform = np.full(n, 1.0 / n)
    restart = uniform if seeds.sum() <= 0 else (1 - cfg.uniform_mix) * seeds / seeds.sum() + cfg.uniform_mix * uniform
    personalization = {node: 0.0 for node in G.nodes}
    for i in range(n):
        personalization[sent_node(i)] = float(restart[i])
    pr = nx.pagerank(G, alpha=cfg.damping, personalization=personalization, weight="weight",
                     max_iter=500, tol=1e-10)
    return np.array([pr[sent_node(i)] for i in range(n)])


def _minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def mmr_select(scores: np.ndarray, emb: np.ndarray, k: int, lam: float = 0.7,
               candidates: Optional[Sequence[int]] = None) -> List[int]:
    """Greedy MMR: argmax  lam * relevance(i) - (1 - lam) * max_{j in selected} cos(i, j)."""
    pool = list(range(len(scores))) if candidates is None else list(candidates)
    if not pool or k <= 0:
        return []
    rel = _minmax(np.asarray(scores, dtype=float))
    sim = emb @ emb.T
    selected: List[int] = []
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda i: rel[i])
        else:
            best = max(pool, key=lambda i: lam * rel[i] - (1 - lam) * max(sim[i, j] for j in selected))
        selected.append(best)
        pool.remove(best)
    return selected
