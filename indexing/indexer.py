import os
import requests
import chromadb
from chromadb.config import Settings

import config

INDEX_EXTS = {
    ".java", ".kt",
    ".ts", ".tsx",
    ".js", ".jsx",
    ".html", ".css", ".scss",
    ".md", ".rst", ".adoc", ".txt",
    ".yml", ".yaml", ".json", ".py"
}

MAX_FILE_BYTES = 500_000
CHARS_PER_CHUNK = 1200
CHUNK_OVERLAP = 200


def should_index_file(path: str, index_exts: set[str]) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in index_exts


def chunk_text(text: str, max_len: int, overlap: int):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_len)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def embed_text(text: str):
    url = f"{config.OLLAMA_URL.rstrip('/')}/api/embeddings"
    payload = {"model": config.EMBED_MODEL, "prompt": text}

    try:
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("embedding")
    except Exception:
        return None


def index_repo(
    repo_root: str,
    index_dir: str | None = None,
    collection_name: str | None = None,
    index_exts: set[str] | None = None,
    max_file_bytes: int = MAX_FILE_BYTES,
    chars_per_chunk: int = CHARS_PER_CHUNK,
    chunk_overlap: int = CHUNK_OVERLAP,
    log=print,
):
    repo_root = os.path.abspath(repo_root)
    index_dir = index_dir or config.DEFAULT_INDEX_DIR
    collection_name = collection_name or config.DEFAULT_COLLECTION_NAME
    index_exts = index_exts or INDEX_EXTS

    log(f"Indexing repo at {repo_root}")
    log(f"Using Chroma index directory: {index_dir}")
    log(f"Using collection name: {collection_name}")
    log(f"Using Ollama URL: {config.OLLAMA_URL}")
    log(f"Using embed model: {config.EMBED_MODEL}")
    log(f"Max file size (bytes): {max_file_bytes}")
    log(f"Chars per chunk: {chars_per_chunk}")
    log(f"Chunk overlap: {chunk_overlap}")
    log(f"File types: {', '.join(sorted(index_exts))}")

    client = chromadb.PersistentClient(
        path=index_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Code and documentation chunks"},
    )

    chunk_count = 0
    file_count = 0

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [
            d for d in dirs
            if d not in {
                ".git", ".idea", "node_modules",
                "build", "dist", "out", "target", ".gradle",
            }
        ]

        for fname in files:
            full = os.path.join(root, fname)
            if not should_index_file(full, index_exts):
                continue

            try:
                if os.path.getsize(full) > max_file_bytes:
                    continue
            except OSError:
                continue

            try:
                with open(full, "r", encoding="utf8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue

            if not text.strip():
                continue

            rel = os.path.relpath(full, repo_root)
            chunks = chunk_text(text, chars_per_chunk, chunk_overlap)

            for idx, chunk in enumerate(chunks):
                embedding = embed_text(chunk)
                if embedding is None:
                    log(f"[index_repo] Skipping {rel} chunk {idx} due to embedding error")
                    continue

                collection.add(
                    ids=[f"{rel}::chunk_{idx}"],
                    documents=[chunk],
                    metadatas=[{"path": rel, "chunk_index": idx}],
                    embeddings=[embedding],
                )

                chunk_count += 1
                if chunk_count % 100 == 0:
                    log(f"Indexed {chunk_count} chunks...")

            file_count += 1
            if file_count % 50 == 0:
                log(f"Processed {file_count} files...")

    log("")
    log("DONE.")
    log(f"Files processed: {file_count}")
    log(f"Chunks indexed: {chunk_count}")
