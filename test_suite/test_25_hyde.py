"""
HyDE query rewriting — unit tests. No Ollama: the chat call is faked.
"""

import config
import querying.hyde as hyde


class _Resp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self.content}}


def _fake_post(content):
    return lambda *a, **k: _Resp(content)


def test_markdown_fence_is_stripped(monkeypatch):
    monkeypatch.setattr(hyde.requests, "post",
                        _fake_post("```java\nclass FineService {}\n```"))
    assert hyde.hypothetical_snippet("q") == "class FineService {}"


def test_modes(monkeypatch):
    snippet = "class FineService {}"
    assert hyde.text_to_embed("where are fines?", "off", snippet) == "where are fines?"
    assert hyde.text_to_embed("where are fines?", "doc", snippet) == snippet
    assert hyde.text_to_embed("where are fines?", "both", snippet) == \
        "where are fines?\n\nclass FineService {}"


def test_failed_generation_falls_back_to_the_question(monkeypatch):
    def boom(*a, **k):
        raise hyde.requests.ConnectionError("ollama down")

    monkeypatch.setattr(hyde.requests, "post", boom)
    assert hyde.text_to_embed("where are fines?", "doc", log=lambda _: None) == "where are fines?"


def test_default_mode_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "HYDE_MODE", "off")
    assert hyde.text_to_embed("q", snippet="class X {}") == "q"


def test_hint_is_included_when_set(monkeypatch):
    seen = {}

    def post(url, json, timeout):
        seen["prompt"] = json["messages"][0]["content"]
        return _Resp("class X {}")

    monkeypatch.setattr(hyde.requests, "post", post)
    monkeypatch.setattr(config, "HYDE_HINT", "Java/Spring backend")
    hyde.hypothetical_snippet("q")
    assert "Java/Spring backend" in seen["prompt"]
