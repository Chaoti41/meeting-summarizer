"""Configuration dataclasses (optionally loaded from YAML/JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class GraphConfig:
    sim_threshold: float = 0.35      # min cosine similarity for a sentence-sentence edge
    knn: int = 5                     # each sentence links to at most this many neighbours
    adjacency_window: int = 2        # link sentences within this distance (discourse flow)
    adjacency_weight: float = 0.3
    min_entity_mentions: int = 2     # entity nodes need >= this many sentences
    w_sent_sent: float = 1.0         # edge-type multipliers
    w_sent_speaker: float = 0.5
    w_sent_topic: float = 1.0
    w_sent_entity: float = 0.7


@dataclass
class RankConfig:
    damping: float = 0.85            # PageRank alpha (probability of following an edge)
    uniform_mix: float = 0.5         # fraction of restart mass spread uniformly over sentences
    seed_verb: float = 0.5           # seed weight for sentences containing an action verb
    seed_cue_boost: float = 0.6      # extra multiplier when a commitment cue ("will", "need to") is present
    seed_date: float = 0.3           # seed weight for sentences containing a date/deadline
    mmr_lambda: float = 0.7          # MMR trade-off: 1.0 = relevance only, 0.0 = diversity only


@dataclass
class Config:
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_sentence_tokens: int = 4
    summary_min_tokens: int = 6
    n_topics: Optional[int] = None
    summary_sentences: int = 5
    max_action_items: int = 10
    dedup_overlap: float = 0.8
    dedup_cosine: float = 0.85
    seed: int = 13
    graph: GraphConfig = field(default_factory=GraphConfig)
    rank: RankConfig = field(default_factory=RankConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = dict(d)
        graph = GraphConfig(**d.pop("graph", {}) or {})
        rank = RankConfig(**d.pop("rank", {}) or {})
        return cls(graph=graph, rank=rank, **d)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)
