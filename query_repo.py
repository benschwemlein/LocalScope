#!/usr/bin/env python3
import sys
import os
import argparse
import textwrap
import requests
import chromadb
from chromadb.config import Settings

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"

CHROMA_DIR = "./chroma_repo"
COLLECTION_NAME = "repo_chunks"
DEFAULT_TOP_K = 8


def embed_text(text: str):
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    resp = requests.post(url, json=payload)
    if not resp.ok:
        print(f"[embed_text] Ollama returned {resp.status_code}", file=sys.stderr)
        print(f"[embed_text] Body (first 400 chars): {resp.text[:400]!r}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()["embedding"]


def chat_with_context(question, docs, metas):
    context_parts = []

    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        path = meta.get("path", "<unknown>")
        chunk_idx = meta.get("chunk_index", "?")
        header = f"[Snippet {i} from {path} chunk {chunk_idx}]"
        context_parts.append(header + "\n" + doc)

    context = "\n\n".join(context_parts)

    full_prompt = textwrap.dedent(f"""
    You are a senior engineer analyzing a large proprietary Java and Angular codebase.
    You must answer only using the provided snippets.
    If the answer is not in the snippets, say you do not know.

    Context:
    {context}

    Question:
    {question}

    Answer concisely and point to file paths and classes when possible.
    """)

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "user", "content": full_prompt},
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload)
    if not resp.ok:
        print(f"[chat_with_context] Ollama returned {resp.status_code}", file=sys.stderr)
        print(f"[chat_with_context] Body (first 400 chars): {resp.text[:400]!r}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()["message"]["content"]


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[query_repo] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"[query_repo] Could not decode file as UTF 8: {path}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Query the local code index with a question or a text file (for Jira bugs, logs, etc)."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask about the codebase"
    )
    parser.add_argument(
        "-f", "--file",
        dest="file",
        help="Path to a text file whose contents will be used as the question"
    )
    parser.add_argument(
        "-k", "--top-k",
        dest="top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top snippets to retrieve from the index (default {DEFAULT_TOP_K})"
    )

    args = parser.parse_args()

    if args.file and args.question:
        print("[query_repo] Provide either a question or -f file, not both.", file=sys.stderr)
        sys.exit(1)

    if args.file:
        question = read_file(args.file)
        print(f"[query_repo] Using contents of file as question: {args.file}", file=sys.stderr)
    else:
        if not args.question:
            parser.print_help()
            sys.exit(1)
        question = args.question

    if not os.path.isdir(CHROMA_DIR):
        print(f"[query_repo] Index directory does not exist: {CHROMA_DIR}", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"[query_repo] Could not open collection '{COLLECTION_NAME}': {e}", file=sys.stderr)
        sys.exit(1)

    print("[query_repo] Embedding question...", file=sys.stderr)
    q_embedding = embed_text(question)

    print(f"[query_repo] Querying index for top {args.top_k} snippets...", file=sys.stderr)
    res = collection.query(
        query_embeddings=[q_embedding],
        n_results=args.top_k,
        include=["documents", "metadatas"],
    )

    docs_list = res.get("documents", [[]])
    metas_list = res.get("metadatas", [[]])
    if not docs_list or not docs_list[0]:
        print("[query_repo] No relevant snippets found in the index.", file=sys.stderr)
        sys.exit(0)

    docs = docs_list[0]
    metas = metas_list[0]

    print("Using snippets from:")
    for meta in metas:
        path = meta.get("path", "<unknown>")
        chunk_idx = meta.get("chunk_index", "?")
        print(f"  {path} (chunk {chunk_idx})")

    print()
    answer = chat_with_context(question, docs, metas)
    print(answer)


if __name__ == "__main__":
    main()
