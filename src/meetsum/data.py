"""Transcript data structures and parsers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_LINE_RE = re.compile(
    r"^\s*(?:[\[(]?\d{1,2}:\d{2}(?::\d{2})?[\])]?\s*[-–]?\s*)?"   # optional [00:12] timestamp
    r"(?P<speaker>[A-Za-z][\w.'\- ]{0,40}?)\s*:\s+(?P<text>\S.*)$"
)
UNKNOWN_SPEAKER = "Unknown"


@dataclass
class Utterance:
    idx: int
    speaker: str
    text: str


@dataclass
class Transcript:
    utterances: List[Utterance]
    title: Optional[str] = None
    meeting_date: Optional[str] = None   # ISO date, used to resolve relative deadlines
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def speakers(self) -> List[str]:
        seen: List[str] = []
        for u in self.utterances:
            if u.speaker != UNKNOWN_SPEAKER and u.speaker not in seen:
                seen.append(u.speaker)
        return seen

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple], **kw) -> "Transcript":
        utts = [Utterance(i, str(s).strip() or UNKNOWN_SPEAKER, str(t).strip())
                for i, (s, t) in enumerate(pairs) if str(t).strip()]
        return cls(utts, **kw)

    @classmethod
    def from_text(cls, text: str, **kw) -> "Transcript":
        """Parse 'Speaker: text' lines. Unlabelled lines continue the previous utterance."""
        pairs: List[List[str]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if m:
                pairs.append([m.group("speaker").strip(), m.group("text").strip()])
            elif pairs:
                pairs[-1][1] += " " + line
            else:
                pairs.append([UNKNOWN_SPEAKER, line])
        return cls.from_pairs(pairs, **kw)

    @classmethod
    def from_json_obj(cls, obj: Any, **kw) -> "Transcript":
        """Accepts a list of {speaker,text} dicts or {"utterances": [...], "title":..., "meeting_date":...}."""
        if isinstance(obj, dict):
            kw.setdefault("title", obj.get("title"))
            kw.setdefault("meeting_date", obj.get("meeting_date"))
            obj = obj.get("utterances", [])
        pairs = [(u.get("speaker", UNKNOWN_SPEAKER), u.get("text", "")) for u in obj]
        return cls.from_pairs(pairs, **kw)

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "Transcript":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return cls.from_json_obj(json.loads(text), **kw)
        return cls.from_text(text, **kw)
