from meetsum.data import Transcript
from meetsum.preprocess import Preprocessor


def _run(nlp, text):
    return Preprocessor(nlp=nlp).process(Transcript.from_text(text))


def test_action_features(nlp):
    s = _run(nlp, "Carol: I'll send the updated mockups to Dave by Friday.\n"
                  "Bob: The weather was quite nice during the offsite last week.")
    assert s[0].has_verb and s[0].has_cue and s[0].has_date and s[0].is_action
    assert any(d.lower() == "friday" for d in s[0].dates)
    assert not s[1].is_action


def test_short_sentences_dropped(nlp):
    assert _run(nlp, "Alice: Yes.\nBob: Okay great.") == []


def test_past_dates_not_deadlines(nlp):
    s = _run(nlp, "Alice: We shipped the release two weeks ago without incident.")
    assert not any("ago" in d for d in s[0].dates)


def test_imperative_counts_as_commitment_and_date_alone_does_not(nlp):
    s = _run(nlp, "Alice: Send the revised budget to finance before the audit.\n"
                  "Bob: The goal today is to review the status of the launch.")
    assert s[0].is_action
    assert not s[1].is_action and s[1].has_date    # date seeds PageRank but does not make an action item
