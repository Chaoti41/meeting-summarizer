from meetsum.data import Transcript


def test_parse_speakers_timestamps_and_continuations():
    t = Transcript.from_text("[00:01] Alice: Hello there.\nBob: Hi.\nstill Bob talking\n\n10:30 Carol Smith: Ok.")
    assert [u.speaker for u in t.utterances] == ["Alice", "Bob", "Carol Smith"]
    assert t.utterances[1].text == "Hi. still Bob talking"
    assert t.speakers == ["Alice", "Bob", "Carol Smith"]


def test_url_is_not_a_speaker():
    t = Transcript.from_text("Alice: see http://example.com now")
    assert len(t.utterances) == 1 and t.utterances[0].speaker == "Alice"


def test_json_input():
    t = Transcript.from_json_obj({"utterances": [{"speaker": "A", "text": "x y z w"}], "meeting_date": "2026-01-02"})
    assert t.meeting_date == "2026-01-02" and t.utterances[0].speaker == "A"
