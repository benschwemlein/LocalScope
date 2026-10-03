"""
Agent search plus whole files, the answer path run_query uses by default.

1. A local search agent (querying/agent_search.py) picks the files. It starts
   from the semantic index's top files for the question, framed as "usually
   right" (seed="trust"), and can grep, find, read and search the index to
   confirm them and fill gaps.
2. The chosen files are read in full, most relevant first, until a character
   budget is used up. A file that doesn't fit is skipped in favour of smaller
   ones further down the list.
3. The chat model answers from those whole files.

On the private benchmark corpus this scored 0.88 answer accuracy against
0.61 for answering from index snippets, with the same chat model.
"""

import os
from typing import Callable

import config
from querying.agent_search import AgentRun, Repo, SemanticSearch, run_agent

LogFn = Callable[[str], object]


def _preview(chunk: str) -> str:
    """First line of real content, skipping the chunker's file-name header."""
    for line in chunk.splitlines():
        line = line.strip()
        if line and not line.startswith("'''"):
            return line[:120]
    return ""


def index_search(collection, top_k: int = 10) -> SemanticSearch:
    """semantic(query) -> [(path, preview)] from the index, as run_query retrieves."""
    from querying.query_engine import _embed_text, retrieve_chunks

    def semantic(query: str) -> list[tuple[str, str]]:
        embedding = _embed_text(query, lambda _: None)
        if embedding is None:
            return []
        docs, metas, _ = retrieve_chunks(collection, query, embedding, top_k, log=lambda _: None)
        return [(m.get("source", ""), _preview(d)) for d, m in zip(docs, metas)]

    return semantic


def whole_files(repo_root: str, files: list[str], budget: int) -> tuple[list[str], list[dict], list[str]]:
    """Read `files` in order until `budget` characters; skip any that don't fit.

    Returns (docs, metas, skipped) in the shape _chat_with_context takes.
    """
    docs, metas, skipped, used = [], [], [], 0
    for rel in files:
        path = os.path.join(repo_root, rel)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            skipped.append(rel)
            continue
        if used + len(text) > budget:
            skipped.append(rel)
            continue
        used += len(text)
        docs.append(text)
        metas.append({"source": rel, "chunk_index": "whole file"})
    return docs, metas, skipped


def find_files(question: str, collection, repo_root: str, log: LogFn = print) -> AgentRun:
    """Run the trust-seeded search agent over `repo_root`."""
    model = config.AGENT_MODEL or config.CHAT_MODEL
    return run_agent(
        question, Repo(repo_root), model,
        top_k=config.AGENT_FILES,
        semantic=index_search(collection),
        seed="trust",
        log=log,
    )
