from meetsum.actions import TEAM, UNASSIGNED, build_name_index, clean_task, resolve_owner
from meetsum.data import Transcript
from meetsum.preprocess import Preprocessor

PEOPLE = ["Alice", "Bob", "Carol"]


def _sents(nlp, text):
    return Preprocessor(nlp=nlp).process(Transcript.from_text(text))


def test_owner_first_person(nlp):
    s = _sents(nlp, "Carol: I'll send the updated mockups by Friday.")
    assert resolve_owner(s[0], s, build_name_index(PEOPLE)) == "Carol"


def test_owner_vocative_and_named(nlp):
    s = _sents(nlp, "Alice: Bob, can you review the contract this week?\n"
                    "Alice: Carol will update the roadmap before the launch.")
    idx = build_name_index(PEOPLE)
    assert resolve_owner(s[0], s, idx) == "Bob"
    assert resolve_owner(s[1], s, idx) == "Carol"


def test_owner_addressee_team_and_unassigned(nlp):
    s = _sents(nlp, "Alice: Can you send the summary to the client today?\nBob: Sure, no problem at all.\n"
                    "Alice: We should schedule a follow up with legal.\n"
                    "Alice: Somebody needs a rollback plan for payments.")
    idx = build_name_index(PEOPLE)
    assert resolve_owner(s[0], s, idx) == "Bob"
    assert resolve_owner(s[2], s, idx) == TEAM
    assert resolve_owner(s[3], s, idx) == UNASSIGNED


def test_clean_task():
    idx = build_name_index(PEOPLE)
    assert clean_task("Okay. Carol, can you fix the contrast issues by Friday?", idx) == \
        "Fix the contrast issues by Friday"
