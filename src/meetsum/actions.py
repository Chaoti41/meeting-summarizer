"""Action-item extraction: task text, owner and deadline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .dates import resolve_deadline
from .preprocess import SentenceRecord

UNASSIGNED = "Unassigned"
TEAM = "Team"

_FIRST_PERSON = re.compile(
    r"\b(?:i'll|i will|i'm going to|i am going to|i can|i could|i should|i need to|i have to|"
    r"i must|let me|i'd)\b", re.I)
_YOU = re.compile(r"\b(?:can you|could you|would you|will you|you(?:'ll| will| should| need to))\b", re.I)
_WE = re.compile(r"\b(?:we|let's|us)\b", re.I)
_LEAD = re.compile(
    r"^(?:(?:ok(?:ay)?|so|yeah|yes|right|great|good|sure|alright|well|and|then|thanks|thank you)\b[,.]?\s*)+",
    re.I)
_STOP = frozenset("a an the and or to of in on at for with by is are was be it this that i you we will "
                  "can should need please let's i'll then also so up".split())


@dataclass
class ActionItem:
    task: str
    owner: str
    deadline: Optional[str]            # as spoken, e.g. "Friday"
    deadline_date: Optional[str]       # ISO date when resolvable
    speaker: str
    sentence_idx: int
    score: float
    source: str

    def to_dict(self) -> Dict:
        return asdict(self)


def build_name_index(participants: Sequence[str]) -> Dict[str, str]:
    """Map lowercase full names and unambiguous first names to the canonical speaker name."""
    idx: Dict[str, str] = {}
    firsts: Dict[str, set] = {}
    for p in participants:
        low = p.lower().strip()
        if not low:
            continue
        idx[low] = p
        firsts.setdefault(low.split()[0], set()).add(p)
    for first, ps in firsts.items():
        if len(ps) == 1:
            idx.setdefault(first, next(iter(ps)))
    return idx


def _names_alt(index: Dict[str, str]) -> str:
    return "|".join(re.escape(n) for n in sorted(index, key=len, reverse=True))


def resolve_owner(sent: SentenceRecord, sentences: Sequence[SentenceRecord], index: Dict[str, str]) -> str:
    """Earliest of: 'NAME will/should/can you ...', 'NAME, can you/please ...', 'I'll ...'.
    Falls back to the next speaker for bare 'can you ...', 'Team' for 'we/let's', else Unassigned."""
    text = sent.text
    cands = []
    if index:
        names = _names_alt(index)
        a = re.compile(rf"\b(?P<n>{names})\b\s+(?:will|'ll|shall|should|needs? to|is going to|is to|has to|must|"
                       r"is responsible|can take|can handle|can you|could you|would you|will you)\b", re.I)
        b = re.compile(rf"\b(?P<n>{names})\s*,\s*(?:can|could|would|will|please|you|are you|do you mind)\b", re.I)
        for rx in (a, b):
            m = rx.search(text)
            if m:
                cands.append((m.start(), index[m.group("n").lower()]))
    m = _FIRST_PERSON.search(text)
    if m and sent.speaker != "Unknown":
        cands.append((m.start(), sent.speaker))
    if cands:
        return min(cands)[1]
    if _YOU.search(text):
        for nxt in sentences[sent.idx + 1: sent.idx + 4]:
            if nxt.speaker != sent.speaker and nxt.speaker != "Unknown":
                return nxt.speaker
    if _WE.search(text):
        return TEAM
    return UNASSIGNED


def resolve_deadline_for(sent: SentenceRecord, sentences: Sequence[SentenceRecord]) -> Optional[str]:
    if sent.dates:
        return sent.dates[0]
    for j in (sent.idx + 1, sent.idx - 1):     # date stated in an adjacent sentence of the same utterance
        if 0 <= j < len(sentences) and sentences[j].utt_idx == sent.utt_idx and sentences[j].dates \
                and not sentences[j].is_action:
            return sentences[j].dates[0]
    return None


def clean_task(text: str, index: Dict[str, str]) -> str:
    t = _LEAD.sub("", text.strip())
    if index:
        t = re.sub(rf"^(?:{_names_alt(index)})\s*,\s*", "", t, flags=re.I)
    t = re.sub(r"^(?:please\s+)?(?:(?:can|could|would|will) you\s+(?:please\s+)?)?", "", t, flags=re.I)
    t = t.strip().rstrip("?").strip() or text.strip()
    return t[:1].upper() + t[1:]


def _content_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def _overlap(a: set, b: set) -> float:
    return len(a & b) / min(len(a), len(b)) if min(len(a), len(b)) >= 3 else 0.0


def extract_action_items(
    sentences: Sequence[SentenceRecord],
    scores: np.ndarray,
    emb: np.ndarray,
    participants: Sequence[str],
    max_items: int = 10,
    dedup_overlap: float = 0.8,
    dedup_cosine: float = 0.85,
    meeting_date: Union[date, str, None] = None,
) -> List[ActionItem]:
    index = build_name_index(participants)
    order = sorted((s for s in sentences if s.is_action), key=lambda s: -scores[s.idx])
    kept: List[ActionItem] = []
    kept_tokens: List[set] = []
    for s in order:
        item = ActionItem(
            task=clean_task(s.text, index),
            owner=resolve_owner(s, sentences, index),
            deadline=resolve_deadline_for(s, sentences),
            deadline_date=None,
            speaker=s.speaker,
            sentence_idx=s.idx,
            score=float(scores[s.idx]),
            source=s.text,
        )
        item.deadline_date = resolve_deadline(item.deadline, meeting_date)
        toks = _content_tokens(item.task)
        dup = None
        for k, (other, otoks) in enumerate(zip(kept, kept_tokens)):
            cos = float(emb[s.idx] @ emb[other.sentence_idx])
            if _overlap(toks, otoks) >= dedup_overlap or cos >= dedup_cosine:
                dup = k
                break
        if dup is not None:                         # merge missing fields into the higher-scoring item
            other = kept[dup]
            if other.owner in {UNASSIGNED, TEAM} and item.owner not in {UNASSIGNED, TEAM}:
                other.owner = item.owner
            if not other.deadline and item.deadline:
                other.deadline, other.deadline_date = item.deadline, item.deadline_date
            continue
        if len(kept) < max_items:
            kept.append(item)
            kept_tokens.append(toks)
    return sorted(kept, key=lambda a: a.sentence_idx)
