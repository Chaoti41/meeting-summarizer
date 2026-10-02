"""Lexicons and regexes for action verbs, commitment cues and date expressions."""

from __future__ import annotations

import re

# Base-form verbs that typically start a task.
ACTION_VERBS = frozenset("""
approve ask assign book build call check circulate close collect compile complete confirm contact
coordinate create decide define deploy design determine document draft email escalate evaluate
finish fix gather handle implement investigate merge migrate notify organize own plan post prepare
prioritize prototype publish raise refactor release remind research resolve review schedule send
share ship submit test triage update verify write
""".split())

# Multi-word action expressions.
PHRASAL_RE = re.compile(
    r"\b(?:follow(?:ing)? up|set up|look(?:ing)? into|reach(?:ing)? out|get back to|"
    r"take care of|touch base|circle back|sync (?:up )?with|hand off)\b",
    re.I,
)

# Commitment / obligation cues.
CUE_RE = re.compile(
    r"(?:\b(?:will|shall|should|must|gonna|let's|please|action item|todo|to-do)\b"
    r"|'ll\b|\bneeds? to\b|\bhave to\b|\bhas to\b|\bgot to\b|\bgoing to\b"
    r"|\b(?:can|could|would) you\b|\bresponsible for\b|\bi can\b|\bi could\b|\bi'd\b)",
    re.I,
)

_WD = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_MONTH = ("january|february|march|april|may|june|july|august|september|october|november|december|"
          "jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec")
_NUM = r"a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+"

DATE_RE = re.compile(
    "|".join(
        [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
            rf"\b(?:{_MONTH})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b",
            rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH})\b",
            rf"\b(?:next|this|coming)\s+(?:{_WD}|week|month|quarter)\b",
            rf"\b(?:{_WD})\b",
            r"\b(?:today|tonight|tomorrow|eod|eow|eom|asap)\b",
            r"\bend of (?:the )?(?:day|week|month|quarter|year)\b",
            rf"\bin\s+(?:{_NUM})\s+(?:days?|weeks?|months?)\b",
        ]
    ),
    re.I,
)

# NER dates that refer to the past are not deadlines.
PAST_RE = re.compile(r"\b(?:last|ago|yesterday|previous|earlier|past)\b", re.I)

WEEKDAYS = _WD.split("|")
MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
NON_ENTITY_CAPS = frozenset(WEEKDAYS + MONTHS + ["i", "okay", "ok", "yes", "yeah", "thanks", "monday"])

NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

IGNORED_ENTITY_LABELS = frozenset({"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "DATE", "TIME"})
