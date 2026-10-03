#!/usr/bin/env python3
"""
CLI for querying a local RAG index.

Usage:
    python cli/rag_query.py --index /path/to/index "your question here"
    python cli/rag_query.py --index /path/to/index --top-k 20 "your question"

Reads model settings (embed_model, chat_model, ollama_url) from
~/.local-rag-llm/config.json — the same settings file the GUI uses.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or cli/ directory
_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def _load_settings() -> dict:
    p = Path.home() / ".local-rag-llm" / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def main():
    settings = _load_settings()
    qt = settings.get("query_tab") or {}
    st = settings.get("settings_tab") or {}

    parser = argparse.ArgumentParser(
        description="Query a local RAG index and return ranked code snippets."
    )
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument(
        "--index", "-i",
        default=qt.get("index_dir", ""),
        help="Path to ChromaDB index directory (default: from GUI settings)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=qt.get("top_k", 12),
        help="Number of results to return (default: %(default)s)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=qt.get("max_chars", 4000),
        help="Summarize query if longer than this many chars (default: %(default)s)",
    )
    parser.add_argument(
        "--no-snippets",
        action="store_true",
        help="Print only file paths and scores, not snippet content",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )

    args = parser.parse_args()
    query_text = " ".join(args.query)

    if not args.index:
        print("ERROR: --index is required (or set index_dir in GUI settings)", file=sys.stderr)
        sys.exit(1)

    # Apply same settings the GUI applies before running a query
    import config
    if st.get("ollama_url"):
        config.OLLAMA_URL = st["ollama_url"]
    if st.get("embed_model"):
        config.EMBED_MODEL = st["embed_model"]
    if st.get("chat_model"):
        config.CHAT_MODEL = st["chat_model"]

    summarizer_template = settings.get("prompts_tab", {}).get("summarizer_prompt", "")
    chat_template = settings.get("prompts_tab", {}).get("chat_prompt", "")

    # Use the same run_query function the GUI uses
    from querying.query_engine import run_query

    def log(msg):
        print(msg, file=sys.stderr)

    result = run_query(
        bug_text=query_text,
        index_dir=args.index,
        top_k=args.top_k,
        max_chars=args.max_chars,
        summarizer_template=summarizer_template,
        chat_template=chat_template if not args.no_snippets and not args.json_output else "",
        log=log,
    )

    metas   = result["metas"]
    docs    = result["docs"]
    scores  = result["scores"]
    dists   = result["distances"]

    if args.json_output:
        output = []
        for meta, doc, score, dist in zip(metas, docs, scores, dists):
            entry = {"source": meta.get("source", ""), "score": round(score, 1), "distance": dist}
            if not args.no_snippets:
                entry["snippet"] = doc
            output.append(entry)
        print(json.dumps(output, indent=2))
        return

    print(f"\n=== RAG Results: {len(metas)} files ===\n")
    for i, (meta, doc, score, dist) in enumerate(zip(metas, docs, scores, dists), 1):
        source = meta.get("source", "<unknown>")
        chunk  = meta.get("chunk_index", "?")
        print(f"[{i:02d}] {score:5.1f}%  {source}  (chunk {chunk})")
        if not args.no_snippets:
            print("-" * 60)
            print(doc[:600] + ("..." if len(doc) > 600 else ""))
            print()


if __name__ == "__main__":
    main()
