import json
from pathlib import Path

from meetsum.cli import main
from meetsum.data import Transcript

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end(summarizer):
    tr = Transcript.from_file(ROOT / "examples" / "sample_transcript.txt")
    out = summarizer.summarize(tr, meeting_date="2026-10-05")
    assert 0 < len(out.summary_sentences) <= 5
    idxs = [s["idx"] for s in out.summary_sentences]
    assert idxs == sorted(idxs)
    assert out.action_items, "expected action items"
    carol = [a for a in out.action_items if "contrast" in a.task.lower()]
    assert carol and carol[0].owner == "Carol" and carol[0].deadline_date == "2026-10-09"
    bob = [a for a in out.action_items if "rollback" in a.task.lower()]
    assert bob and bob[0].owner == "Bob" and bob[0].deadline_date == "2026-10-14"
    assert "| Task | Owner | Deadline |" in out.to_markdown()
    json.dumps(out.to_dict())


def test_empty_transcript(summarizer):
    out = summarizer.summarize("")
    assert out.summary_text == "" and out.action_items == []


def test_cli_summarize_and_evaluate(tmp_path, capsys):
    out_file = tmp_path / "out.json"
    assert main(["summarize", str(ROOT / "examples/sample_transcript.txt"), "--embedding", "hash",
                 "--meeting-date", "2026-10-05", "--format", "json", "-o", str(out_file)]) == 0
    assert json.loads(out_file.read_text())["action_items"]
    assert main(["evaluate", str(ROOT / "examples/mini_dataset.jsonl"), "--embedding", "hash"]) == 0
    assert "action_item_micro" in capsys.readouterr().out
