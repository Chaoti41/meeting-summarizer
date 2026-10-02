import numpy as np

from meetsum.config import Config
from meetsum.data import Transcript
from meetsum.embeddings import HashingEmbedder
from meetsum.graph import build_graph, graph_stats
from meetsum.preprocess import Preprocessor
from meetsum.ranking import mmr_select, personalized_pagerank, seed_weights
from meetsum.topics import build_topics

TEXT = open(__file__.replace("tests/test_graph_ranking.py", "examples/sample_transcript.txt")).read()


def _pipeline(nlp):
    cfg = Config()
    sents = Preprocessor(nlp=nlp).process(Transcript.from_text(TEXT))
    emb = HashingEmbedder().encode([s.text for s in sents])
    topics = build_topics([s.text for s in sents], emb, seed=cfg.seed)
    G = build_graph(sents, emb, topics, cfg.graph)
    return cfg, sents, emb, topics, G


def test_heterogeneous_graph(nlp):
    _, sents, _, topics, G = _pipeline(nlp)
    stats = graph_stats(G)
    assert stats["sentence_nodes"] == len(sents)
    assert stats["speaker_nodes"] == 4
    assert stats["topic_nodes"] == topics.k >= 2
    assert stats["edges"] > len(sents)


def test_ppr_is_distribution_and_boosts_seeds(nlp):
    cfg, sents, _, _, G = _pipeline(nlp)
    scores = personalized_pagerank(G, sents, cfg.rank)
    assert np.all(scores > 0) and scores.sum() <= 1.0 + 1e-6
    seeds = seed_weights(sents, cfg.rank) > 0
    assert scores[seeds].mean() > scores[~seeds].mean()


def test_mmr_prefers_diversity():
    emb = np.array([[1, 0], [0.999, 0.045], [0, 1]], dtype=float)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    scores = np.array([1.0, 0.95, 0.5])
    assert mmr_select(scores, emb, 2, lam=1.0) == [0, 1]       # relevance only
    assert mmr_select(scores, emb, 2, lam=0.5) == [0, 2]       # diversity kicks in
    assert mmr_select(scores, emb, 5, lam=0.5, candidates=[1, 2]) == [1, 2]
