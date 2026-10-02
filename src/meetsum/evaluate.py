"""Evaluation: ROUGE, BERTScore and action-item precision/recall/F1."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .dates import resolve_deadline  # noqa: F401  (re-exported for convenience)

MODES = ("task", "task+owner", "task+owner+deadline")
_STOP = frozenset("a an the and or to of in on at for with by is are was be it this that i you we will "
                  "can should need please let's i'll then also so up".split())


# ------------------------------------------------------------------ summaries
def rouge_scores(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    """Mean ROUGE-1/2/L F-measure (stemmed)."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'meetsum[eval]' for ROUGE/BERTScore") from e
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = [sc.score(r, p) for p, r in zip(preds, refs)]
    return {k: float(np.mean([r[k].fmeasure for r in rows])) if rows else 0.0
            for k in ("rouge1", "rouge2", "rougeL")}


def bertscore(preds: Sequence[str], refs: Sequence[str], model_type: Optional[str] = None,
              lang: str = "en") -> Dict[str, float]:
    """Mean BERTScore precision/recall/F1."""
    try:
        from bert_score import score
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'meetsum[eval]' for ROUGE/BERTScore") from e
    P, R, F = score(list(preds), list(refs), lang=lang, model_type=model_type, verbose=False)
    return {"bertscore_p": float(P.mean()), "bertscore_r": float(R.mean()), "bertscore_f1": float(F.mean())}


# ------------------------------------------------------------------ action items
def _as_dict(x: Any) -> Dict[str, Any]:
    return asdict(x) if is_dataclass(x) else dict(x)


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP}


def _token_f1(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    p, r = inter / len(ta), inter / len(tb)
    return 2 * p * r / (p + r)


def _norm_owner(o: Optional[str]) -> str:
    o = (o or "").strip().lower()
    if o in {"", "unassigned", "none", "unknown", "n/a"}:
        return ""
    return o.split()[0]


def _deadline_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    g = (gold.get("deadline") or "").strip().lower()
    cands = {(pred.get("deadline") or "").strip().lower(), (pred.get("deadline_date") or "").strip().lower()}
    cands.discard("")
    if not g and not cands:
        return True
    return g in cands


def _count_matches(valid: np.ndarray, sim: np.ndarray) -> int:
    """Maximum one-to-one matching (Hungarian) over valid pairs; ties broken by similarity."""
    from scipy.optimize import linear_sum_assignment

    if valid.size == 0:
        return 0
    rows, cols = linear_sum_assignment(-(valid.astype(float) + 0.01 * sim))
    return int(sum(valid[r, c] for r, c in zip(rows, cols)))


def action_item_scores(pred: Iterable[Any], gold: Iterable[Any], threshold: float = 0.5,
                       embedder=None) -> Dict[str, Any]:
    """Precision/recall/F1 for action items under three strictness levels.

    A predicted item matches a gold item when task similarity >= threshold (token-F1 by default, or
    embedding cosine if `embedder` is given). Stricter modes additionally require the owner (first-name,
    case-insensitive) and/or the deadline (raw or ISO string) to match. Matching is one-to-one.
    """
    P = [_as_dict(p) for p in pred]
    G = [_as_dict(g) for g in gold]
    n_p, n_g = len(P), len(G)
    sim = np.zeros((n_p, n_g))
    if embedder is not None and n_p and n_g:
        ep = embedder.encode([p.get("task", "") for p in P])
        eg = embedder.encode([g.get("task", "") for g in G])
        sim = ep @ eg.T
    else:
        for i, p in enumerate(P):
            for j, g in enumerate(G):
                sim[i, j] = _token_f1(p.get("task", ""), g.get("task", ""))
    task_ok = sim >= threshold
    owner_ok = np.array([[_norm_owner(p.get("owner")) == _norm_owner(g.get("owner")) for g in G] for p in P],
                        dtype=bool).reshape(n_p, n_g)
    dead_ok = np.array([[_deadline_match(p, g) for g in G] for p in P], dtype=bool).reshape(n_p, n_g)

    valid = {"task": task_ok, "task+owner": task_ok & owner_ok,
             "task+owner+deadline": task_ok & owner_ok & dead_ok}
    out: Dict[str, Any] = {"n_pred": n_p, "n_gold": n_g}
    for mode in MODES:
        tp = _count_matches(valid[mode], sim)
        out[mode] = _prf(tp, n_p, n_g)
    # Field accuracy among task-matched pairs
    from scipy.optimize import linear_sum_assignment
    if n_p and n_g:
        rows, cols = linear_sum_assignment(-(task_ok.astype(float) + 0.01 * sim))
        pairs = [(r, c) for r, c in zip(rows, cols) if task_ok[r, c]]
    else:
        pairs = []
    out["owner_acc"] = float(np.mean([owner_ok[r, c] for r, c in pairs])) if pairs else 0.0
    out["deadline_acc"] = float(np.mean([dead_ok[r, c] for r, c in pairs])) if pairs else 0.0
    out["n_task_matched"] = len(pairs)
    return out


def _prf(tp: int, n_p: int, n_g: int) -> Dict[str, float]:
    p = tp / n_p if n_p else 0.0
    r = tp / n_g if n_g else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "precision": p, "recall": r, "f1": f}


# ------------------------------------------------------------------ dataset-level
def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """JSONL; each line: {id, transcript | utterances, reference_summary, reference_actions, meeting_date}."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evaluate_dataset(records: Sequence[Dict[str, Any]], summarizer, use_bertscore: bool = False,
                     bertscore_model: Optional[str] = None, match_threshold: float = 0.5,
                     semantic_match: bool = False) -> Dict[str, Any]:
    from .data import Transcript

    preds, refs, per_example = [], [], []
    tp = {m: 0 for m in MODES}
    n_pred = n_gold = 0
    for rec in records:
        if "utterances" in rec:
            tr = Transcript.from_json_obj(rec["utterances"])
        else:
            tr = Transcript.from_text(rec["transcript"])
        out = summarizer.summarize(tr, meeting_date=rec.get("meeting_date"))
        gold_actions = rec.get("reference_actions", [])
        scores = action_item_scores(out.action_items, gold_actions, match_threshold,
                                    summarizer.embedder if semantic_match else None)
        preds.append(out.summary_text)
        refs.append(rec.get("reference_summary", ""))
        per_example.append({"id": rec.get("id"), "summary": out.summary_text,
                            "action_items": [a.to_dict() for a in out.action_items], "action_scores": scores})
        for m in MODES:
            tp[m] += scores[m]["tp"]
        n_pred += scores["n_pred"]
        n_gold += scores["n_gold"]

    result: Dict[str, Any] = {"n_examples": len(records)}
    if any(refs):
        result.update(rouge_scores(preds, refs))
        if use_bertscore:
            result.update(bertscore(preds, refs, model_type=bertscore_model))
    result["action_item_micro"] = {m: _prf(tp[m], n_pred, n_gold) for m in MODES}
    result["action_item_macro_f1"] = {
        m: float(np.mean([e["action_scores"][m]["f1"] for e in per_example])) if per_example else 0.0
        for m in MODES}
    return {"metrics": result, "per_example": per_example}
