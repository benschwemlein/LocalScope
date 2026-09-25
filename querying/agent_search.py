"""
Agentic file search with a local model.

Instead of one retrieval pass, a local chat model is given search tools and
runs its own loop: grep for something, read what came back, search again with
the names the code actually uses, until it is ready to name the relevant
files. This is how Claude Code and similar agents find code, run entirely
through Ollama.

Tools (all read-only and confined to the repository root; .git and build
output are invisible):

    grep(pattern)            case-insensitive regex over file contents
    find_files(name)         case-insensitive substring match on file paths
    read_file(path, start)   up to READ_LINES numbered lines of one file
    submit(files)            final answer: file paths, most relevant first

Optionally the agent also gets the semantic index (pass `semantic`):

    semantic_search(query)   files whose code is closest in meaning to query
    seed=True                the index's top files for the question are in
                             the first message, as a starting point to verify

run_agent() returns the submitted files plus a record of what the agent did.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

import config

EXCLUDED_DIRS = {
    ".git", ".idea", ".vscode", "node_modules", "build", "dist", "out",
    "target", ".gradle", ".venv", "venv", "__pycache__",
}
MAX_FILE_BYTES = 500_000
GREP_MAX_LINES = 40
FIND_MAX = 50
READ_LINES = 150
LINE_CHARS = 200
MAX_REPLY_TOKENS = 1024

TOOLS = [
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents for a regular expression (case-insensitive). "
                       f"Returns up to {GREP_MAX_LINES} matching lines as path:line: text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "find_files",
        "description": "List files whose path contains the given text (case-insensitive).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Text to look for in file paths"},
        }, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": f"Read up to {READ_LINES} lines of a file, starting at a line number.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the repository root"},
            "start_line": {"type": "integer", "description": "First line to read (1-based)"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Give the final answer: the files relevant to the question, "
                       "most relevant first. Call this exactly once, when done.",
        "parameters": {"type": "object", "properties": {
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "File paths relative to the repository root"},
        }, "required": ["files"]},
    }},
]

SEMANTIC_TOOL = {"type": "function", "function": {
    "name": "semantic_search",
    "description": "Find files whose code is closest in meaning to a natural-language "
                   "description, even when they share no words with it. Returns up to "
                   "10 file paths, most similar first, each with a one-line preview.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "What the code you want does or is about"},
    }, "required": ["query"]},
}}

# (path, preview) pairs, most similar first
SemanticSearch = Callable[[str], list[tuple[str, str]]]

SYSTEM_PROMPT = """You are finding the files in a codebase that are relevant to a question.
You cannot see the code until you search for it. Use the tools:
grep to search contents, find_files to search paths, read_file to read code{semantic_hint}.
Search in several steps: start from words in the question, read what you find,
then search for the class, method and field names the code actually uses to
find callers, implementations, configuration and the other side of any API
call. When you are confident, call submit with exactly {top_k} distinct file
paths relative to the repository root, most relevant first."""


class Repo:
    """Read-only view of a repository, with the paths the agent may touch."""

    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.files: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(full) > MAX_FILE_BYTES:
                        continue
                    with open(full, encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                if "\x00" in text:
                    continue  # binary
                self.files[os.path.relpath(full, self.root)] = text

    def resolve(self, path: str) -> str | None:
        """A known relative path for what the agent asked for, or None."""
        p = str(path).strip()
        if p.startswith("./"):
            p = p[2:]
        p = os.path.normpath(p)
        if p in self.files:
            return p
        matches = [f for f in self.files if f.endswith("/" + p)]
        return matches[0] if len(matches) == 1 else None

    def grep(self, pattern: str) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        out = []
        for path, text in self.files.items():
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    out.append(f"{path}:{n}: {line.strip()[:LINE_CHARS]}")
                    if len(out) >= GREP_MAX_LINES:
                        return "\n".join(out) + "\n(more matches truncated)"
        return "\n".join(out) or "no matches"

    def find_files(self, name: str) -> str:
        needle = str(name).lower()
        hits = [f for f in self.files if needle in f.lower()][:FIND_MAX]
        return "\n".join(hits) or "no files match"

    def read_file(self, path: str, start_line: int = 1) -> str:
        rel = self.resolve(path)
        if rel is None:
            return f"no such file: {path}"
        lines = self.files[rel].splitlines()
        start = max(1, int(start_line or 1))
        chunk = lines[start - 1: start - 1 + READ_LINES]
        body = "\n".join(f"{start + i}: {l[:LINE_CHARS]}" for i, l in enumerate(chunk))
        more = f"\n(file has {len(lines)} lines)" if start - 1 + READ_LINES < len(lines) else ""
        return f"{rel}\n{body}{more}"


@dataclass
class AgentRun:
    files: list[str] = field(default_factory=list)
    tool_calls: int = 0
    submitted: bool = False
    fallback: bool = False
    seconds: float = 0.0
    error: str = ""


def _chat(model: str, messages: list, think: bool | None, tools: list) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        # num_predict caps one reply. Tool calls and file lists are short; the
        # cap only stops a runaway generation, which otherwise runs to the
        # request timeout (seen once: 600s on a single turn).
        "options": {"temperature": 0.0, "num_ctx": 32768, "num_predict": MAX_REPLY_TOKENS},
    }
    if think is not None:
        payload["think"] = think
    resp = requests.post(f"{config.OLLAMA_URL.rstrip('/')}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]


def _format_hits(hits: list[tuple[str, str]]) -> str:
    return "\n".join(f"{i}. {path}  |  {preview}" for i, (path, preview) in enumerate(hits, 1)) \
        or "no results"


def run_agent(question: str, repo: Repo, model: str, top_k: int = 10,
              max_steps: int = 25, think: bool | None = False,
              semantic: SemanticSearch | None = None, seed: bool = False,
              log=print) -> AgentRun:
    """Let `model` search `repo` for files relevant to `question`.

    With `semantic`, the agent also has a semantic_search tool; with `seed`
    as well, the index's top files for the question open the conversation.
    """
    run = AgentRun()
    read_order: list[str] = []
    nudged = False
    tools = TOOLS + [SEMANTIC_TOOL] if semantic else TOOLS
    hint = ", semantic_search to find code by meaning" if semantic else ""
    user = question
    if semantic and seed:
        user = (f"{question}\n\nA semantic search of the codebase for this question "
                "returned these files, most similar first. Treat them as a starting "
                "point to verify, not as the answer:\n" + _format_hits(semantic(question)))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(top_k=top_k, semantic_hint=hint)},
        {"role": "user", "content": user},
    ]
    start = time.monotonic()
    try:
        for step in range(max_steps + 1):
            if step == max_steps:
                messages.append({"role": "user", "content":
                                 f"Stop searching and call submit now with {top_k} files."})
            msg = _chat(model, messages, think, tools)
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                if step >= max_steps:
                    break
                messages.append({"role": "user", "content":
                                 "Use the tools to search, or call submit when you are done."})
                continue

            for call in calls:
                fn = call.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                run.tool_calls += 1

                if name == "submit":
                    seen = []
                    for f in args.get("files") or []:
                        rel = repo.resolve(f)
                        if rel and rel not in seen:
                            seen.append(rel)
                    if len(seen) > len(run.files):
                        run.files = seen[:top_k]
                    run.submitted = True
                    # One chance to fill a short answer, as the prompt asks for
                    # exactly top_k; the better of the two submissions stands.
                    if len(run.files) < top_k and not nudged and step < max_steps:
                        nudged = True
                        messages.append({"role": "tool", "tool_name": name, "content":
                            f"Only {len(seen)} valid file paths submitted. Search more if "
                            f"needed, then call submit with exactly {top_k} distinct paths."})
                        continue
                    run.seconds = time.monotonic() - start
                    return run

                if name == "grep":
                    result = repo.grep(args.get("pattern", ""))
                elif name == "find_files":
                    result = repo.find_files(args.get("name", ""))
                elif name == "semantic_search" and semantic:
                    result = _format_hits(semantic(str(args.get("query", ""))))
                elif name == "read_file":
                    result = repo.read_file(args.get("path", ""), args.get("start_line", 1))
                    rel = repo.resolve(args.get("path", ""))
                    if rel and rel not in read_order:
                        read_order.append(rel)
                else:
                    result = f"unknown tool: {name}"
                messages.append({"role": "tool", "tool_name": name, "content": result})
    except (requests.RequestException, KeyError, ValueError) as e:
        run.error = str(e)
        log(f"[agent_search] {model} failed: {e}")

    # No submit: fall back to the files it chose to read, in the order read.
    if not run.submitted:
        run.files, run.fallback = read_order[:top_k], True
    run.seconds = time.monotonic() - start
    return run
