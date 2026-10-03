"""
Reranking inside retrieve_chunks — unit tests.

No Ollama, ChromaDB, or model download: a fake collection returns fixed
chunks, and the cross-encoder is replaced with a lookup of canned scores.
"""

import pytest

import config
from querying import reranker
from querying.query_engine import retrieve_chunks


class FakeCollection:
    """Returns chunks in embedding order, records how many were requested."""

    def __init__(self, chunks):
        # chunks: list of (text, source, distance), already in distance order
        self.chunks = chunks
        self.requested = None

    def query(self, query_embeddings, n_results, include):
        self.requested = n_results
        rows = self.chunks[:n_results]
        return {
            "documents": [[r[0] for r in rows]],
            "metadatas": [[{"source": r[1]} for r in rows]],
            "distances": [[r[2] for r in rows]],
        }


CHUNKS = [
    ("a1", "A.java", 0.10),
    ("a2", "A.java", 0.11),
    ("b1", "B.java", 0.20),
    ("c1", "C.java", 0.30),
    ("d1", "D.java", 0.40),
    ("c2", "C.java", 0.45),
]


@pytest.fixture
def rerank_on(monkeypatch):
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(config, "RERANK_POOL", 50)


def _canned(scores: dict):
    return lambda question, passages: [scores[p] for p in passages]


def test_without_rerank_keeps_embedding_order(monkeypatch):
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    coll = FakeCollection(CHUNKS)
    docs, metas, dists = retrieve_chunks(coll, "q", [0.0], top_k=3, log=lambda _: None)
    assert [m["source"] for m in metas] == ["A.java", "B.java", "C.java"]
    assert coll.requested == 9  # top_k * 3


def test_rerank_reorders_files(monkeypatch, rerank_on):
    monkeypatch.setattr(reranker, "score_pairs",
                        _canned({"a1": 0.1, "a2": 0.2, "b1": 0.3, "c1": 0.9, "d1": 0.8, "c2": 0.0}))
    docs, metas, _ = retrieve_chunks(FakeCollection(CHUNKS), "q", [0.0], top_k=3,
                                     log=lambda _: None)
    assert [m["source"] for m in metas] == ["C.java", "D.java", "B.java"]


def test_rerank_picks_each_files_best_reranked_chunk(monkeypatch, rerank_on):
    """The chunk kept per file is the reranker's favourite, not the embedding's."""
    monkeypatch.setattr(reranker, "score_pairs",
                        _canned({"a1": 0.1, "a2": 0.95, "b1": 0.3, "c1": 0.2, "d1": 0.4, "c2": 0.5}))
    docs, metas, _ = retrieve_chunks(FakeCollection(CHUNKS), "q", [0.0], top_k=2,
                                     log=lambda _: None)
    assert docs == ["a2", "c2"]


def test_rerank_returns_same_file_count(monkeypatch, rerank_on):
    """Reranking must never change how many files come back."""
    monkeypatch.setattr(reranker, "score_pairs", lambda q, ps: [0.0] * len(ps))
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    _, plain, _ = retrieve_chunks(FakeCollection(CHUNKS), "q", [0.0], 3, log=lambda _: None)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    _, reranked, _ = retrieve_chunks(FakeCollection(CHUNKS), "q", [0.0], 3, log=lambda _: None)
    assert len(plain) == len(reranked) == 3


def test_rerank_fetches_the_wider_pool(monkeypatch, rerank_on):
    monkeypatch.setattr(reranker, "score_pairs", lambda q, ps: [0.0] * len(ps))
    coll = FakeCollection(CHUNKS)
    retrieve_chunks(coll, "q", [0.0], top_k=3, log=lambda _: None)
    assert coll.requested == 50


def test_reranked_distances_stay_lower_is_better(monkeypatch, rerank_on):
    monkeypatch.setattr(reranker, "score_pairs",
                        _canned({"a1": 0.1, "a2": 0.2, "b1": 0.3, "c1": 0.9, "d1": 0.8, "c2": 0.0}))
    _, _, dists = retrieve_chunks(FakeCollection(CHUNKS), "q", [0.0], 3, log=lambda _: None)
    assert dists == sorted(dists)
