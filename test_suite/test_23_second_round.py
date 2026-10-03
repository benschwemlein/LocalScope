"""
Second retrieval round — unit tests.

No Ollama or ChromaDB: a fake collection answers substring searches over a
handful of chunks.
"""

import pytest

import config
import querying.second_round as sr


class FakeCollection:
    def __init__(self, chunks):
        # chunk id -> (source, text)
        self.chunks = chunks

    def get(self, ids=None, where_document=None, include=None):
        items = list(self.chunks.items())
        if ids is not None:
            items = [(i, self.chunks[i]) for i in ids]
        if where_document is not None:
            needle = where_document["$contains"]
            items = [(i, c) for i, c in items if needle in c[1]]
        return {
            "ids": [i for i, _ in items],
            "metadatas": [{"source": c[0]} for _, c in items],
            "documents": [c[1] for _, c in items],
        }


# Round 1 finds the controller. The policy class it depends on is reachable
# only by its name, which appears in the controller's code, not the question.
CHUNKS = {
    "c1": ("web/WaiverController.java", "class WaiverController { FineWaiverPolicy policy; }"),
    "c2": ("svc/FineWaiverPolicy.java", "class FineWaiverPolicy { int windowDays = 21; }"),
    "c3": ("svc/BatchJob.java", "class BatchJob { int days = 14; }"),
    "c4": ("util/Strings.java", "class Strings { String trim(String s) { return s; } }"),
    "c5": ("web/HomeController.java", "class HomeController { String home() { return x; } }"),
    "c6": ("web/Other.java", "class Other { }"),
    "c7": ("docs/notes.md", "just some plain notes about the release"),
}


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(sr, "_searches", {})
    monkeypatch.setattr(config, "SECOND_ROUND_SEED_CHUNKS", 1)
    monkeypatch.setattr(config, "SECOND_ROUND_TERMS", 4)
    monkeypatch.setattr(sr, "_MAX_DOC_FRACTION", 0.6)


def _round1(sources):
    return [(CHUNKS[c][1], {"source": CHUNKS[c][0]}, float(i))
            for i, c in enumerate(sources)]


def test_code_identifier_filter():
    assert sr._is_code_identifier("FineWaiverPolicy")
    assert sr._is_code_identifier("window_days")
    assert not sr._is_code_identifier("window")      # plain English
    assert not sr._is_code_identifier("String")      # language noise


def test_prf_terms_prefer_rare_identifiers():
    search = sr._IndexSearch(FakeCollection(CHUNKS))
    terms = sr.prf_terms([CHUNKS["c1"][1]], search, n=4)
    assert "FineWaiverPolicy" in terms


def test_second_round_pulls_in_files_round_one_missed():
    coll = FakeCollection(CHUNKS)
    fused, terms = sr.second_round(coll, "q", _round1(["c1", "c5"]), top_k=3, mode="prf")
    sources = [m["source"] for _, m, _ in fused]
    assert "FineWaiverPolicy" in terms
    assert "svc/FineWaiverPolicy.java" in sources
    assert len(fused) == 3


def test_second_round_never_exceeds_top_k():
    coll = FakeCollection(CHUNKS)
    fused, _ = sr.second_round(coll, "q", _round1(["c1", "c5", "c6"]), top_k=2, mode="prf")
    assert len(fused) == 2


def test_no_terms_returns_round_one_unchanged():
    coll = FakeCollection(CHUNKS)
    round1 = _round1(["c7"])  # prose only, no code identifiers
    fused, terms = sr.second_round(coll, "q", round1, top_k=1, mode="prf")
    assert terms == []
    assert fused == round1


def test_llm_terms_drop_names_not_in_the_index(monkeypatch):
    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": '{"identifiers": ["FineWaiverPolicy", "MadeUpName", "String"]}'}}

    monkeypatch.setattr(sr.requests, "post", lambda *a, **k: Resp())
    search = sr._IndexSearch(FakeCollection(CHUNKS))
    assert sr.llm_terms("q", ["x"], search, n=4) == ["FineWaiverPolicy"]


def test_llm_failure_falls_back_to_round_one(monkeypatch):
    def boom(*a, **k):
        raise sr.requests.ConnectionError("ollama down")

    monkeypatch.setattr(sr.requests, "post", boom)
    coll = FakeCollection(CHUNKS)
    round1 = _round1(["c1", "c5"])
    fused, terms = sr.second_round(coll, "q", round1, top_k=2, mode="llm", log=lambda _: None)
    assert terms == [] and fused == round1
