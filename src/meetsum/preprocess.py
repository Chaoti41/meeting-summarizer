"""spaCy preprocessing: sentence splitting, NER, action-verb / cue / date detection."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

from .data import Transcript
from .lexicons import (
    ACTION_VERBS,
    CUE_RE,
    DATE_RE,
    IGNORED_ENTITY_LABELS,
    NON_ENTITY_CAPS,
    PAST_RE,
    PHRASAL_RE,
)


@dataclass
class SentenceRecord:
    idx: int
    text: str
    speaker: str
    utt_idx: int
    n_tokens: int
    entities: List[Tuple[str, str]] = field(default_factory=list)   # (surface, label)
    dates: List[str] = field(default_factory=list)
    has_verb: bool = False     # contains an action verb
    has_cue: bool = False      # contains a commitment cue ("will", "need to", ...)

    @property
    def has_date(self) -> bool:
        return bool(self.dates)

    @property
    def is_action(self) -> bool:
        """Action-item candidate: an action verb plus a commitment cue (modal/obligation or imperative).
        Dates alone do not qualify ("The goal today is to review ...") but do seed PageRank and set deadlines."""
        return self.has_verb and self.has_cue

    @property
    def entity_keys(self) -> List[str]:
        keys = []
        for surface, _ in self.entities:
            k = surface.lower().strip()
            if k.startswith("the "):
                k = k[4:]
            if k and k not in keys:
                keys.append(k)
        return keys


def normalize_text(text: str) -> str:
    return (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))


def load_nlp(model: str = "en_core_web_sm"):
    """Load a spaCy pipeline; fall back to a blank English pipeline (rule-based sentences, no NER/POS)."""
    import spacy

    try:
        nlp = spacy.load(model)
        nlp.meta["meetsum_fallback"] = False
    except OSError:
        warnings.warn(
            f"spaCy model '{model}' not found; falling back to a blank pipeline without NER/POS. "
            f"Install it with: python -m spacy download {model}",
            stacklevel=2,
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.meta["meetsum_fallback"] = True
    return nlp


class Preprocessor:
    def __init__(self, nlp=None, model: str = "en_core_web_sm", min_tokens: int = 4):
        self.nlp = nlp if nlp is not None else load_nlp(model)
        self.min_tokens = min_tokens
        self.has_ner = self.nlp.has_pipe("ner")

    def process(self, transcript: Transcript) -> List[SentenceRecord]:
        texts = [normalize_text(u.text) for u in transcript.utterances]
        records: List[SentenceRecord] = []
        for utt, doc in zip(transcript.utterances, self.nlp.pipe(texts)):
            tagged = doc.has_annotation("TAG")
            for sent in doc.sents:
                tokens = [t for t in sent if not t.is_space]
                words = [t for t in tokens if not t.is_punct]
                if len(words) < self.min_tokens:
                    continue
                text = sent.text.strip()
                records.append(
                    SentenceRecord(
                        idx=len(records),
                        text=text,
                        speaker=utt.speaker,
                        utt_idx=utt.idx,
                        n_tokens=len(words),
                        entities=self._entities(sent, tokens),
                        dates=self._dates(sent, text),
                        has_verb=self._has_action_verb(sent, tokens, tagged, text),
                        has_cue=bool(CUE_RE.search(text)) or self._is_imperative(sent, tokens, tagged),
                    )
                )
        return records

    # ------------------------------------------------------------------ helpers
    def _has_action_verb(self, sent, tokens, tagged: bool, text: str) -> bool:
        if PHRASAL_RE.search(text):
            return True
        for t in tokens:
            if tagged:
                is_base_verb = t.tag_ in {"VB", "VBP"} and t.lemma_.lower() in ACTION_VERBS
                imperative = t.i == sent.start and t.lower_ in ACTION_VERBS   # "Send the report ..."
                if is_base_verb or imperative:
                    return True
            elif t.lower_ in ACTION_VERBS:
                return True
        return False

    @staticmethod
    def _is_imperative(sent, tokens, tagged: bool) -> bool:
        """Sentence starts with a base-form action verb ("Send the report ...")."""
        first = next((t for t in tokens if not t.is_punct), None)
        if first is None or first.lower_ not in ACTION_VERBS:
            return False
        return first.tag_ in {"VB", "VBP"} if tagged else True

    def _entities(self, sent, tokens) -> List[Tuple[str, str]]:
        if self.has_ner:
            return [(e.text.strip(), e.label_) for e in sent.ents
                    if e.label_ not in IGNORED_ENTITY_LABELS and e.text.strip()]
        # Fallback: mid-sentence capitalised alphabetic tokens act as proper-noun entities.
        out = []
        for t in tokens:
            if (t.i > sent.start and t.is_alpha and t.text[:1].isupper() and not t.is_stop
                    and t.lower_ not in NON_ENTITY_CAPS):
                out.append((t.text, "PROPN"))
        return out

    def _dates(self, sent, text: str) -> List[str]:
        spans: List[Tuple[int, int, str]] = [(m.start(), m.end(), m.group(0)) for m in DATE_RE.finditer(text)]
        if self.has_ner:
            base = sent.start_char
            for e in sent.ents:
                if e.label_ in {"DATE", "TIME"} and not PAST_RE.search(e.text):
                    s, en = e.start_char - base, e.end_char - base
                    if not any(s < b and a < en for a, b, _ in spans):   # skip overlaps with regex hits
                        spans.append((s, en, e.text))
        spans.sort()
        seen, out = set(), []
        for _, _, surface in spans:
            k = surface.lower().strip()
            if k not in seen:
                seen.add(k)
                out.append(surface.strip())
        return out
