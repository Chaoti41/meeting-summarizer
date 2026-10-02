"""End-to-end pipeline: transcript -> summary + action items."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .actions import ActionItem, extract_action_items
from .config import Config
from .data import Transcript
from .embeddings import get_embedder
from .graph import build_graph, graph_stats
from .preprocess import Preprocessor
from .ranking import mmr_select, personalized_pagerank
from .topics import build_topics

__all__ = ["ActionItem", "MeetingSummarizer", "MeetingSummary"]


@dataclass
class MeetingSummary:
    summary_text: str
    summary_sentences: List[Dict[str, Any]]
    action_items: List[ActionItem]
    topics: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary_text,
            "summary_sentences": self.summary_sentences,
            "action_items": [a.to_dict() for a in self.action_items],
            "topics": self.topics,
            "stats": self.stats,
        }

    def to_markdown(self) -> str:
        lines = ["# Meeting Summary", "", self.summary_text or "_(empty transcript)_", ""]
        lines += ["## Action Items", ""]
        if not self.action_items:
            lines.append("_No action items detected._")
        else:
            lines += ["| # | Task | Owner | Deadline |", "|---|------|-------|----------|"]
            for i, a in enumerate(self.action_items, 1):
                dl = a.deadline or "—"
                if a.deadline_date:
                    dl += f" ({a.deadline_date})"
                lines.append(f"| {i} | {a.task.replace('|', '/')} | {a.owner} | {dl} |")
        if self.topics:
            lines += ["", "## Topics", ""] + [f"- {t}" for t in self.topics]
        return "\n".join(lines) + "\n"


class MeetingSummarizer:
    def __init__(self, config: Optional[Config] = None, embedder=None, nlp=None):
        self.config = config or Config()
        self.embedder = embedder if embedder is not None else get_embedder(self.config.embedding_model)
        self.pre = Preprocessor(nlp=nlp, model=self.config.spacy_model, min_tokens=self.config.min_sentence_tokens)

    def summarize(self, transcript: Union[Transcript, str],
                  meeting_date: Union[date, str, None] = None) -> MeetingSummary:
        cfg = self.config
        if isinstance(transcript, str):
            transcript = Transcript.from_text(transcript)
        meeting_date = meeting_date or transcript.meeting_date

        sentences = self.pre.process(transcript)
        if not sentences:
            return MeetingSummary("", [], [])

        emb = np.asarray(self.embedder.encode([s.text for s in sentences]), dtype=np.float32)
        topics = build_topics([s.text for s in sentences], emb, cfg.n_topics, cfg.seed)
        G = build_graph(sentences, emb, topics, cfg.graph)
        scores = personalized_pagerank(G, sentences, cfg.rank)

        # Summary: MMR over sufficiently long sentences, then restore chronological order.
        long_enough = [s for s in sentences if s.n_tokens >= cfg.summary_min_tokens] or list(sentences)
        statements = [s for s in long_enough if not s.text.rstrip().endswith("?")]
        pool = [s.idx for s in (statements if len(statements) >= cfg.summary_sentences else long_enough)]
        chosen = sorted(mmr_select(scores, emb, cfg.summary_sentences, cfg.rank.mmr_lambda, pool))
        summary_sents = [
            {"idx": i, "speaker": sentences[i].speaker, "text": sentences[i].text, "score": float(scores[i])}
            for i in chosen
        ]

        actions = extract_action_items(
            sentences, scores, emb, transcript.speakers,
            max_items=cfg.max_action_items, dedup_overlap=cfg.dedup_overlap,
            dedup_cosine=cfg.dedup_cosine, meeting_date=meeting_date,
        )
        return MeetingSummary(
            summary_text=" ".join(s["text"] for s in summary_sents),
            summary_sentences=summary_sents,
            action_items=actions,
            topics=topics.names,
            stats={"sentences": len(sentences), **graph_stats(G)},
        )
