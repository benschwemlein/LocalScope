"""
Local agent search — unit tests.

No Ollama: a temporary repository on disk, and the chat model replaced with
a scripted sequence of tool calls.
"""

import pytest

import querying.agent_search as agent


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "FineService.java").write_text("class FineService {\n  int graceDays = 3;\n}\n")
    (tmp_path / "src" / "LoanController.java").write_text("class LoanController { FineService fines = new FineService(); }\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/secret-branch\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    return agent.Repo(str(tmp_path))


def test_git_and_binary_files_are_invisible(repo):
    assert set(repo.files) == {"src/FineService.java", "src/LoanController.java"}
    assert "secret-branch" not in repo.grep("secret")


def test_paths_outside_the_repo_cannot_be_read(repo):
    assert repo.read_file("../../etc/passwd").startswith("no such file")
    assert repo.read_file("/etc/passwd").startswith("no such file")


def test_grep_is_case_insensitive_and_reports_lines(repo):
    assert "src/FineService.java:2:" in repo.grep("GRACEDAYS")


def test_bad_regex_falls_back_to_literal(repo):
    assert repo.grep("FineService(") != "no matches"


def test_resolve_accepts_unique_suffix_and_dot_slash(repo):
    assert repo.resolve("./src/FineService.java") == "src/FineService.java"
    assert repo.resolve("FineService.java") == "src/FineService.java"


def _scripted(replies):
    it = iter(replies)
    return lambda model, messages, think, tools: next(it)


def _call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_agent_loop_searches_then_submits(repo, monkeypatch):
    monkeypatch.setattr(agent, "_chat", _scripted([
        _call("grep", pattern="grace"),
        _call("read_file", path="src/FineService.java"),
        _call("submit", files=["src/FineService.java", "LoanController.java", "nope.java"]),
    ]))
    run = agent.run_agent("q", repo, "m", top_k=2)
    assert run.submitted and not run.fallback
    assert run.files == ["src/FineService.java", "src/LoanController.java"]
    assert run.tool_calls == 3


def test_short_submission_is_nudged_once_and_best_kept(repo, monkeypatch):
    monkeypatch.setattr(agent, "_chat", _scripted([
        _call("submit", files=["src/FineService.java"]),
        _call("submit", files=["src/FineService.java", "src/LoanController.java"]),
    ]))
    run = agent.run_agent("q", repo, "m", top_k=10)
    assert run.files == ["src/FineService.java", "src/LoanController.java"]
    assert run.tool_calls == 2 and run.submitted


def test_short_submission_survives_running_out_of_steps(repo, monkeypatch):
    replies = [_call("submit", files=["src/FineService.java"])]
    replies += [{"role": "assistant", "content": "done"}] * 10
    monkeypatch.setattr(agent, "_chat", _scripted(replies))
    run = agent.run_agent("q", repo, "m", top_k=10, max_steps=3)
    assert run.submitted and not run.fallback
    assert run.files == ["src/FineService.java"]


def test_submission_is_capped_at_top_k(repo, monkeypatch):
    monkeypatch.setattr(agent, "_chat", _scripted([
        _call("submit", files=["src/FineService.java", "src/LoanController.java"]),
    ]))
    assert agent.run_agent("q", repo, "m", top_k=1).files == ["src/FineService.java"]


def test_no_submit_falls_back_to_files_read(repo, monkeypatch):
    replies = [_call("read_file", path="src/LoanController.java")]
    replies += [{"role": "assistant", "content": "I think I'm done."}] * 10
    monkeypatch.setattr(agent, "_chat", _scripted(replies))
    run = agent.run_agent("q", repo, "m", top_k=10, max_steps=3)
    assert run.fallback and not run.submitted
    assert run.files == ["src/LoanController.java"]


def _fake_semantic(query):
    return [("src/FineService.java", "class FineService {")]


def test_seed_puts_index_hits_in_the_first_message(repo, monkeypatch):
    seen = {}

    def chat(model, messages, think, tools):
        seen["user"] = messages[1]["content"]
        seen["tools"] = [t["function"]["name"] for t in tools]
        return _call("submit", files=["src/FineService.java"])

    monkeypatch.setattr(agent, "_chat", chat)
    agent.run_agent("where are fines?", repo, "m", top_k=1, semantic=_fake_semantic, seed=True)
    assert "1. src/FineService.java" in seen["user"]
    assert "semantic_search" in seen["tools"]


def test_semantic_tool_is_absent_without_an_index(repo, monkeypatch):
    seen = {}

    def chat(model, messages, think, tools):
        seen["tools"] = [t["function"]["name"] for t in tools]
        return _call("submit", files=["src/FineService.java"])

    monkeypatch.setattr(agent, "_chat", chat)
    agent.run_agent("q", repo, "m", top_k=1)
    assert "semantic_search" not in seen["tools"]


def test_semantic_search_tool_returns_hits(repo, monkeypatch):
    replies = [_call("semantic_search", query="fine rules"),
               _call("submit", files=["src/FineService.java"])]
    captured = []
    it = iter(replies)

    def chat(model, messages, think, tools):
        captured.append(messages[-1])
        return next(it)

    monkeypatch.setattr(agent, "_chat", chat)
    run = agent.run_agent("q", repo, "m", top_k=1, semantic=_fake_semantic)
    assert run.files == ["src/FineService.java"]
    assert "src/FineService.java" in captured[-1]["content"]


def test_trust_seed_uses_the_trust_prompt(repo, monkeypatch):
    seen = {}

    def chat(model, messages, think, tools):
        seen["system"], seen["user"] = messages[0]["content"], messages[1]["content"]
        return _call("submit", files=["src/FineService.java"])

    monkeypatch.setattr(agent, "_chat", chat)
    agent.run_agent("where are fines?", repo, "m", top_k=1, semantic=_fake_semantic, seed="trust")
    assert "usually right" in seen["system"]
    assert "1. src/FineService.java" in seen["user"]
    assert "verify, not as the answer" not in seen["user"]
