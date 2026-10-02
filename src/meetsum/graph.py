"""Heterogeneous graph with sentence, speaker, topic and entity nodes."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import networkx as nx
import numpy as np

from .config import GraphConfig
from .preprocess import SentenceRecord
from .topics import TopicModel


def sent_node(i: int) -> str:
    return f"s:{i}"


def _accumulate(G: nx.Graph, u: str, v: str, w: float, etype: str) -> None:
    if w <= 0 or u == v:
        return
    if G.has_edge(u, v):
        G[u][v]["weight"] += w
    else:
        G.add_edge(u, v, weight=w, etype=etype)


def _max_edge(G: nx.Graph, u: str, v: str, w: float, etype: str) -> None:
    if w <= 0 or u == v:
        return
    if G.has_edge(u, v):
        G[u][v]["weight"] = max(G[u][v]["weight"], w)
    else:
        G.add_edge(u, v, weight=w, etype=etype)


def build_graph(sentences: List[SentenceRecord], emb: np.ndarray, topics: TopicModel,
                cfg: GraphConfig) -> nx.Graph:
    """Undirected weighted graph. Node types: sentence / speaker / topic / entity.

    Edges:
      sentence-sentence : cosine kNN (semantic) + local adjacency (discourse flow)
      sentence-speaker  : who said it
      sentence-topic    : weighted by cosine to the topic centroid
      sentence-entity   : named entities shared by >= min_entity_mentions sentences
    """
    G = nx.Graph()
    n = len(sentences)
    for s in sentences:
        G.add_node(sent_node(s.idx), type="sentence", idx=s.idx)

    # sentence - sentence (semantic kNN)
    if n > 1:
        sim = emb @ emb.T
        np.fill_diagonal(sim, -1.0)
        for i in range(n):
            for j in np.argsort(-sim[i])[: cfg.knn]:
                if sim[i, j] >= cfg.sim_threshold:
                    _max_edge(G, sent_node(i), sent_node(int(j)), cfg.w_sent_sent * float(sim[i, j]), "semantic")
    # sentence - sentence (adjacency)
    for d in range(1, cfg.adjacency_window + 1):
        for i in range(n - d):
            _accumulate(G, sent_node(i), sent_node(i + d), cfg.adjacency_weight / d, "adjacent")

    # sentence - speaker
    for s in sentences:
        p = f"p:{s.speaker.lower()}"
        if p not in G:
            G.add_node(p, type="speaker", label=s.speaker)
        _accumulate(G, sent_node(s.idx), p, cfg.w_sent_speaker, "speaker")

    # sentence - topic
    for k, name in enumerate(topics.names):
        G.add_node(f"t:{k}", type="topic", label=name)
    for s in sentences:
        k = int(topics.labels[s.idx])
        w = max(float(emb[s.idx] @ topics.centroids[k]), 0.05)
        _accumulate(G, sent_node(s.idx), f"t:{k}", cfg.w_sent_topic * w, "topic")

    # sentence - entity
    counts = Counter(k for s in sentences for k in s.entity_keys)
    for s in sentences:
        for key in s.entity_keys:
            if counts[key] >= cfg.min_entity_mentions:
                e = f"e:{key}"
                if e not in G:
                    G.add_node(e, type="entity", label=key)
                _accumulate(G, sent_node(s.idx), e, cfg.w_sent_entity, "entity")
    return G


def graph_stats(G: nx.Graph) -> Dict[str, int]:
    c = Counter(d["type"] for _, d in G.nodes(data=True))
    return {**{f"{k}_nodes": v for k, v in c.items()}, "edges": G.number_of_edges()}
