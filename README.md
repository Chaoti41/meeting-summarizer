# meetsum — graph-based meeting summaries and action items

Turns a meeting transcript into a short extractive **summary** plus a table of **action items with owners and deadlines**.

```
transcript ──► spaCy (sentences, NER, action verbs, dates)
           ──► Sentence-BERT embeddings
           ──► heterogeneous graph: sentence · speaker · topic · entity nodes
           ──► Personalized PageRank (restart mass seeded by action verbs / dates)
           ──► MMR  → diverse summary sentences
           └─► action-item extraction (task · owner · deadline), dedup
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"                  # core + sentence-transformers + rouge/bert-score + dev tools
python -m spacy download en_core_web_sm
```

Lightweight/offline install (no PyTorch): `pip install -e ".[dev,eval]"` and pass `--embedding hash`
to use the built-in hashing embedder (handy for tests and demos; real Sentence-BERT gives better graphs).
If the spaCy model is missing, meetsum warns and falls back to a blank pipeline (rule-based sentence
splitting, no NER/POS) so everything still runs.

## Usage

```bash
# Markdown (default) or JSON
meetsum summarize examples/sample_transcript.txt --meeting-date 2026-10-05
meetsum summarize examples/sample_transcript.txt --format json -o out.json --sentences 4

# Offline demo without PyTorch
meetsum summarize examples/sample_transcript.txt --embedding hash --meeting-date 2026-10-05

# Evaluate on a JSONL dataset
meetsum evaluate examples/mini_dataset.jsonl --bertscore -o results.json
```

Python API:

```python
from meetsum import MeetingSummarizer, Transcript

tr = Transcript.from_file("examples/sample_transcript.txt")
out = MeetingSummarizer().summarize(tr, meeting_date="2026-10-05")
print(out.summary_text)
for a in out.action_items:
    print(a.owner, "|", a.task, "|", a.deadline, a.deadline_date)
```

**Input formats:** `.txt` with `Speaker: text` lines (optional `[00:12]` timestamps; unlabelled lines continue the
previous utterance), or `.json` (`[{"speaker","text"}, ...]` / `{"utterances": [...], "meeting_date": ...}`).
`meeting_date` lets relative deadlines ("Friday", "next Wednesday", "end of the month") resolve to ISO dates.

## How it works

| Stage | Module | Details |
|---|---|---|
| Preprocess | `preprocess.py` | spaCy sentence split, NER, action verbs (base-form, POS-checked), commitment cues ("will", "need to", imperatives), date expressions (regex + NER, past dates ignored) |
| Embeddings | `embeddings.py` | `all-MiniLM-L6-v2` by default (any sentence-transformers model works) |
| Topics | `topics.py` | k-means over embeddings (k ≈ √(n/2)), labelled with TF-IDF terms |
| Graph | `graph.py` | Sentence–sentence (cosine kNN + local adjacency), sentence–speaker, sentence–topic (weighted by centroid similarity), sentence–entity (entities shared by ≥2 sentences) |
| Ranking | `ranking.py` | Personalized PageRank; restart vector = `(1-mix)·seeds + mix·uniform`, seeds from action verbs (boosted by cues) and dates. MMR (`λ=0.7`) then picks diverse sentences, returned in chronological order |
| Actions | `actions.py`, `dates.py` | Candidates = verb + commitment cue, ranked by PPR. **Owner**: earliest of "NAME will…", "NAME, can you…", "I'll…" → speaker; bare "can you…" → next speaker; "we/let's" → Team; else Unassigned. **Deadline**: first date in the sentence (or an adjacent one), resolved against the meeting date. Near-duplicates are merged |
| Evaluation | `evaluate.py` | ROUGE-1/2/L, BERTScore, action-item P/R/F1 |

### Evaluation

* **ROUGE / BERTScore** compare the generated summary with `reference_summary`.
* **Action-item F1** uses one-to-one (Hungarian) matching. A prediction matches a gold item when task similarity ≥ threshold
  (token-F1 by default, `--semantic-match` for embedding cosine). Three strictness levels are reported: `task`,
  `task+owner`, `task+owner+deadline`, as micro and macro averages, plus owner/deadline accuracy on matched pairs.

Dataset format (JSONL, one meeting per line):

```json
{"id": "m1", "meeting_date": "2026-10-05", "transcript": "Alice: ...\nBob: ...",
 "reference_summary": "...",
 "reference_actions": [{"task": "Send the deck", "owner": "Bob", "deadline": "2026-10-09"}]}
```

`utterances` (list of `{speaker, text}`) can replace `transcript`. Public corpora such as AMI/ICSI/QMSum can be converted to this format; they are not bundled.

## Configuration

All hyperparameters live in `src/meetsum/config.py`; override with `--config configs/default.yaml`.
Key knobs: `rank.uniform_mix` (higher = more general summary, lower = more action-focused),
`rank.mmr_lambda`, `graph.sim_threshold`, `graph.knn`, `summary_sentences`, `max_action_items`.

## Development

```bash
pytest          # 25 tests; use the offline hashing embedder, no model downloads needed
ruff check src tests
```

## Limitations

* Extractive: summary sentences are quoted from the transcript, so they can read as dialogue.
* Owner/deadline extraction is rule-based. Pronoun-heavy or implicit assignments end up "Unassigned" or "Team".
* Action detection is English-only and keyword-driven; tune `lexicons.py` for your domain.
* Bundled sample data is a single synthetic meeting. Treat the demo scores as a smoke test, not a benchmark.

## License

MIT
