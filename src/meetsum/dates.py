"""Resolve deadline expressions ("by Friday", "next Wednesday", "October 20th") to ISO dates."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Optional, Union

from .lexicons import MONTHS, NUMBER_WORDS, WEEKDAYS

_MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_NUM.update({m[:3]: i + 1 for i, m in enumerate(MONTHS)})
_MONTH_NUM["sept"] = 9
_MONTH_ALT = "|".join(sorted(_MONTH_NUM, key=len, reverse=True))

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_MD = re.compile(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$")
_MONTH_DAY = re.compile(rf"^({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?$")
_DAY_MONTH = re.compile(rf"^(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})$")
_IN_N = re.compile(r"^in\s+(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(day|week|month)s?$")


def _to_date(ref: Union[date, str, None]) -> Optional[date]:
    if ref is None or isinstance(ref, date):
        return ref
    return date.fromisoformat(ref)


def _safe(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _upcoming(ref: date, weekday: int) -> date:
    """First given weekday strictly after ref."""
    return ref + timedelta(days=(weekday - ref.weekday() - 1) % 7 + 1)


def resolve_deadline(expr: Optional[str], ref: Union[date, str, None] = None) -> Optional[str]:
    """Return an ISO date string or None.

    Absolute expressions need no reference date. Relative ones are resolved against `ref`
    (the meeting date). Conventions:
      * bare weekday          -> next such weekday strictly after the meeting date
      * "next <weekday>"      -> that weekday in the following calendar week
      * "next week"/"eow"     -> Friday of next week / this week
      * "end of month/quarter/year" -> last day of the period
    """
    if not expr:
        return None
    t = re.sub(r"\s+", " ", expr.strip().lower()).strip(".,;")
    t = re.sub(r"^(?:by|on|before|until|till)\s+", "", t)
    ref_d = _to_date(ref)

    if m := _ISO.match(t):
        d = _safe(*map(int, m.groups()))
        return d.isoformat() if d else None
    if m := _MD.match(t):
        mo, da, yr = int(m.group(1)), int(m.group(2)), m.group(3)
        year = (int(yr) + (2000 if len(yr) == 2 else 0)) if yr else (ref_d.year if ref_d else None)
        d = _safe(year, mo, da) if year else None
        if d and not yr and ref_d and d < ref_d:
            d = _safe(year + 1, mo, da)
        return d.isoformat() if d else None
    for rx, order in ((_MONTH_DAY, ("m", "d")), (_DAY_MONTH, ("d", "m"))):
        if m := rx.match(t):
            a, b = m.groups()
            mo, da = (_MONTH_NUM[a], int(b)) if order == ("m", "d") else (_MONTH_NUM[b], int(a))
            if not ref_d:
                return None
            d = _safe(ref_d.year, mo, da)
            if d and d < ref_d:
                d = _safe(ref_d.year + 1, mo, da)
            return d.isoformat() if d else None

    if ref_d is None:
        return None
    if t in {"today", "tonight", "eod", "end of day", "end of the day", "asap"}:
        return ref_d.isoformat()
    if t == "tomorrow":
        return (ref_d + timedelta(days=1)).isoformat()
    if t in {"eow", "end of week", "end of the week", "this week"}:
        d = ref_d + timedelta(days=4 - ref_d.weekday()) if ref_d.weekday() <= 4 else _upcoming(ref_d, 4)
        return d.isoformat()
    if t == "next week":
        return (ref_d - timedelta(days=ref_d.weekday()) + timedelta(days=7 + 4)).isoformat()
    if t in {"eom", "end of month", "end of the month", "this month"}:
        return date(ref_d.year, ref_d.month, calendar.monthrange(ref_d.year, ref_d.month)[1]).isoformat()
    if t == "next month":
        n = add_months(ref_d, 1)
        return date(n.year, n.month, calendar.monthrange(n.year, n.month)[1]).isoformat()
    if t in {"end of quarter", "end of the quarter", "this quarter"}:
        qm = ((ref_d.month - 1) // 3 + 1) * 3
        return date(ref_d.year, qm, calendar.monthrange(ref_d.year, qm)[1]).isoformat()
    if t in {"end of year", "end of the year"}:
        return date(ref_d.year, 12, 31).isoformat()
    if t in WEEKDAYS:
        return _upcoming(ref_d, WEEKDAYS.index(t)).isoformat()
    if m := re.match(rf"^(next|this|coming) ({'|'.join(WEEKDAYS)})$", t):
        wd = WEEKDAYS.index(m.group(2))
        if m.group(1) == "next":
            monday_next = ref_d - timedelta(days=ref_d.weekday()) + timedelta(days=7)
            return (monday_next + timedelta(days=wd)).isoformat()
        return _upcoming(ref_d, wd).isoformat()
    if m := _IN_N.match(t):
        word, unit = m.groups()
        n = int(word) if word.isdigit() else NUMBER_WORDS[word]
        if unit == "day":
            return (ref_d + timedelta(days=n)).isoformat()
        if unit == "week":
            return (ref_d + timedelta(weeks=n)).isoformat()
        return add_months(ref_d, n).isoformat()
    return None
