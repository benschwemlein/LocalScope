"""
Agent search plus whole files (the default answer path) — unit tests.

No Ollama: the index is a stub, the search agent and the chat call are
replaced, and the files are real ones in a temporary repository.
"""

import pytest

import config
import querying.agent_answer as agent_answer
import querying.query_engine as qe
from querying.agent_search import AgentRun, Repo


@pytest.fixture
def repo_root(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "FineService.java").write_text("class FineService { int graceDays = 3; }\n")
    (tmp_path / "src" / "Big.java").write_text("x" * 500)
    (tmp_path / "src" / "Small.java").write_text("class Small {}\n")
    return str(tmp_path)


def test_whole_files_keeps_order_and_skips_what_does_not_fit(repo_root):
    files = ["src/FineService.java", "src/Big.java", "src/missing.java", "src/Small.java"]
    docs, metas, skipped = agent_answer.whole_files(repo_root, files, budget=100)
    assert [m["source"] for m in metas] == ["src/FineService.java", "src/Small.java"]
    assert docs[0].startswith("class FineService")
    assert all(m["chunk_index"] == "whole file" for m in metas)
    assert skipped == ["src/Big.java", "src/missing.java"]


def test_repo_paths_use_forward_slashes(repo_root):
    repo = Repo(repo_root)
    assert "src/FineService.java" in repo.files
    assert repo.resolve("./src/FineService.java") == "src/FineService.java"


class _Collection:
    name = "repo_chunks"


class _Client:
    def __init__(self, *a, **k):
        pass

    def list_collections(self):
        return [_Collection()]

    def get_collection(self, name):
        return _Collection()


@pytest.fixture
def engine(monkeypatch):
    """run_query with the index, agent and chat model faked; records the chat call."""
    seen = {}
    monkeypatch.setattr(qe.chromadb, "PersistentClient", _Client)
    monkeypatch.setattr(config, "ANSWER_MODE", "agent")

    def chat(question, docs, metas, template, log, token_callback=None, cancel_event=None):
        seen["docs"], seen["metas"] = docs, metas
        return "answer"

    monkeypatch.setattr(qe, "_chat_with_context", chat)
    # The snippet path, used only when the agent path can't run.
    monkeypatch.setattr(qe, "_embed_text", lambda text, log: [0.1, 0.2])
    monkeypatch.setattr(qe, "retrieve_chunks",
                        lambda c, q, e, k, log: (["snippet"], [{"source": "src/Small.java"}], [0.3]))
    return seen


def _agent_returns(monkeypatch, files):
    monkeypatch.setattr(agent_answer, "find_files",
                        lambda question, collection, root, log=print: AgentRun(files=files, submitted=True))


def test_agent_mode_answers_from_whole_files(monkeypatch, engine, repo_root):
    _agent_returns(monkeypatch, ["src/FineService.java", "src/Small.java"])
    result = qe.run_query("where is the grace period?", index_dir="unused",
                          repo_root=repo_root, log=lambda _: None)
    assert result["mode"] == "agent"
    assert [m["source"] for m in result["metas"]] == ["src/FineService.java", "src/Small.java"]
    assert engine["docs"][0] == "class FineService { int graceDays = 3; }\n"
    assert result["scores"] == [None, None]


def test_no_repo_root_falls_back_to_snippets(monkeypatch, engine):
    _agent_returns(monkeypatch, ["src/FineService.java"])
    result = qe.run_query("where is the grace period?", index_dir="unused",
                          repo_root="", log=lambda _: None)
    assert result["mode"] == "snippets"
    assert engine["docs"] == ["snippet"]


def test_agent_finding_nothing_falls_back_to_snippets(monkeypatch, engine, repo_root):
    _agent_returns(monkeypatch, [])
    result = qe.run_query("where is the grace period?", index_dir="unused",
                          repo_root=repo_root, log=lambda _: None)
    assert result["mode"] == "snippets"


def test_snippets_mode_ignores_the_agent(monkeypatch, engine, repo_root):
    monkeypatch.setattr(config, "ANSWER_MODE", "snippets")
    _agent_returns(monkeypatch, ["src/FineService.java"])
    result = qe.run_query("where is the grace period?", index_dir="unused",
                          repo_root=repo_root, log=lambda _: None)
    assert result["mode"] == "snippets"
