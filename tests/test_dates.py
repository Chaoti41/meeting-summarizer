from meetsum.dates import resolve_deadline

REF = "2026-10-05"   # a Monday


def test_absolute():
    assert resolve_deadline("2026-12-01") == "2026-12-01"
    assert resolve_deadline("October 20th", REF) == "2026-10-20"
    assert resolve_deadline("3 Jan", REF) == "2027-01-03"


def test_relative():
    assert resolve_deadline("tomorrow", REF) == "2026-10-06"
    assert resolve_deadline("Friday", REF) == "2026-10-09"
    assert resolve_deadline("Monday", REF) == "2026-10-12"
    assert resolve_deadline("next Wednesday", REF) == "2026-10-14"
    assert resolve_deadline("end of the month", REF) == "2026-10-31"
    assert resolve_deadline("next week", REF) == "2026-10-16"
    assert resolve_deadline("in two weeks", REF) == "2026-10-19"
    assert resolve_deadline("eow", REF) == "2026-10-09"
    assert resolve_deadline("by Friday", REF) == "2026-10-09"


def test_unresolvable():
    assert resolve_deadline("Friday") is None
    assert resolve_deadline("someday", REF) is None
    assert resolve_deadline(None, REF) is None
