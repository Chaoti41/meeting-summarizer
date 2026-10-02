import pytest

from meetsum.evaluate import action_item_scores, rouge_scores

GOLD = [
    {"task": "Fix the contrast issues", "owner": "Carol", "deadline": "friday"},
    {"task": "Write a rollback plan for payments", "owner": "Bob", "deadline": None},
]


def test_perfect_match():
    s = action_item_scores(GOLD, GOLD)
    assert s["task+owner+deadline"]["f1"] == 1.0 and s["owner_acc"] == 1.0


def test_owner_error_hurts_strict_modes_only():
    pred = [dict(GOLD[0], owner="Dave"), GOLD[1]]
    s = action_item_scores(pred, GOLD)
    assert s["task"]["f1"] == 1.0
    assert s["task+owner"]["precision"] == 0.5 and s["owner_acc"] == 0.5


def test_deadline_iso_matches_and_extra_prediction():
    pred = [{"task": "Fix contrast issues", "owner": "Carol", "deadline": "Friday", "deadline_date": "2026-10-09"},
            {"task": "Completely unrelated chore", "owner": "Bob", "deadline": None}]
    s = action_item_scores(pred, GOLD[:1])
    assert s["task+owner+deadline"]["tp"] == 1 and s["task"]["precision"] == 0.5 and s["task"]["recall"] == 1.0


def test_empty_cases():
    assert action_item_scores([], GOLD)["task"]["f1"] == 0.0
    assert action_item_scores([], [])["task"]["f1"] == 0.0


def test_rouge():
    pytest.importorskip("rouge_score")
    r = rouge_scores(["the team ships on friday"], ["the team ships on friday"])
    assert r["rouge1"] == pytest.approx(1.0) and r["rougeL"] == pytest.approx(1.0)
