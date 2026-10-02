"""Command line interface: `meetsum summarize` and `meetsum evaluate`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .data import Transcript
from .summarize import MeetingSummarizer


def _build_summarizer(args) -> MeetingSummarizer:
    cfg = Config.from_file(args.config) if args.config else Config()
    if args.embedding:
        cfg.embedding_model = args.embedding
    if getattr(args, "sentences", None):
        cfg.summary_sentences = args.sentences
    return MeetingSummarizer(cfg)


def _cmd_summarize(args) -> int:
    tr = Transcript.from_file(args.transcript)
    out = _build_summarizer(args).summarize(tr, meeting_date=args.meeting_date)
    text = json.dumps(out.to_dict(), indent=2, ensure_ascii=False) if args.format == "json" else out.to_markdown()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_evaluate(args) -> int:
    from .evaluate import evaluate_dataset, load_dataset

    result = evaluate_dataset(load_dataset(args.dataset), _build_summarizer(args),
                              use_bertscore=args.bertscore, match_threshold=args.match_threshold,
                              semantic_match=args.semantic_match)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="meetsum", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", help="YAML/JSON config file (see configs/default.yaml)")
        p.add_argument("--embedding", help="embedding model name, or 'hash' for the offline embedder")

    s = sub.add_parser("summarize", help="summarize one transcript (.txt 'Speaker: text' lines, or .json)")
    s.add_argument("transcript")
    s.add_argument("--meeting-date", help="ISO date used to resolve 'Friday', 'tomorrow', ...")
    s.add_argument("--format", choices=["md", "json"], default="md")
    s.add_argument("--sentences", type=int, help="number of summary sentences")
    s.add_argument("-o", "--output")
    common(s)
    s.set_defaults(fn=_cmd_summarize)

    e = sub.add_parser("evaluate", help="evaluate on a JSONL dataset")
    e.add_argument("dataset")
    e.add_argument("--bertscore", action="store_true", help="also compute BERTScore (downloads a model)")
    e.add_argument("--match-threshold", type=float, default=0.5, help="task-similarity threshold for action matching")
    e.add_argument("--semantic-match", action="store_true", help="match action tasks by embedding cosine instead of token-F1")
    e.add_argument("-o", "--output")
    common(e)
    e.set_defaults(fn=_cmd_evaluate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
