"""
Lexical identifier index — full-text search over source-code identifiers.

Complements the vector (ChromaDB) and structural (graph) retrieval legs with
exact/fuzzy identifier matching. Symbol-anchored queries ("who calls
processInvoice", "InvoiceProcessor") often fail semantic search because an
embedder has no reason to know an unfamiliar identifier is relevant; a plain
inverted index over split identifier tokens catches these directly.

Storage: one SQLite FTS5 virtual table per index_dir, at lexical.db.
Granularity: file-level, matching the file-level graph (graph/graph_store.py)
and ChromaDB's dedup-by-source in querying/query_engine.py.
"""

import os
import re
import sqlite3

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_")

_DB_FILENAME = "lexical.db"

_DEFAULT_EXCLUDED_DIRS = {
    ".git", ".idea", ".vscode",
    "node_modules", "build", "dist", "out", "target", ".gradle",
    ".venv", "venv", "__pycache__",
}


def split_identifier(identifier: str) -> list[str]:
    """Split an identifier into lowercase sub-tokens on camelCase/PascalCase/snake_case boundaries."""
    return [p.lower() for p in _BOUNDARY_RE.split(identifier) if p]


def extract_tokens(text: str) -> str:
    """Extract all identifiers in text and split them into a space-joined token string for FTS indexing."""
    tokens: list[str] = []
    for match in _IDENTIFIER_RE.finditer(text):
        ident = match.group(0)
        if len(ident) < 2:
            continue
        tokens.append(ident.lower())
        sub = split_identifier(ident)
        if len(sub) > 1:
            tokens.extend(sub)
    return " ".join(tokens)


def _db_path(index_dir: str) -> str:
    return os.path.join(index_dir, _DB_FILENAME)


def _connect(index_dir: str) -> sqlite3.Connection:
    os.makedirs(index_dir, exist_ok=True)
    con = sqlite3.connect(_db_path(index_dir))
    con.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5(source UNINDEXED, tokens)"
    )
    return con


def _remove_file(con: sqlite3.Connection, source: str) -> None:
    con.execute("DELETE FROM lexical_fts WHERE source = ?", (source,))


def _index_file(con: sqlite3.Connection, source: str, text: str) -> None:
    """(Re)index one file's identifiers. Caller must remove any prior row first."""
    tokens = extract_tokens(text)
    if tokens:
        con.execute("INSERT INTO lexical_fts (source, tokens) VALUES (?, ?)", (source, tokens))


def build_lexical_index(
    repo_root: str,
    index_dir: str,
    index_exts: set[str] | None = None,
    excluded_dirs: set[str] | None = None,
    log=print,
) -> None:
    """Full (re)build of the lexical index for repo_root into index_dir/lexical.db."""
    from indexing.indexer import DEFAULT_INDEX_EXTS, should_index_file

    index_exts = index_exts or DEFAULT_INDEX_EXTS
    excluded_dirs = excluded_dirs or _DEFAULT_EXCLUDED_DIRS

    con = _connect(index_dir)
    try:
        con.execute("DELETE FROM lexical_fts")
        count = 0
        for root, dirs, fnames in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for fname in fnames:
                full = os.path.join(root, fname)
                if not should_index_file(full, index_exts):
                    continue
                rel = os.path.relpath(full, repo_root)
                try:
                    with open(full, "r", encoding="utf8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                _index_file(con, rel, text)
                count += 1
                if count % 100 == 0:
                    log(f"[lexical_index] Indexed {count} files...")
        con.commit()
        log(f"[lexical_index] Built lexical index: {count} files")
    finally:
        con.close()


def update_lexical_index_incremental(
    index_dir: str,
    changed_files: list[tuple[str, str]],
    deleted_files: list[str] | None = None,
    log=print,
) -> None:
    """
    Incrementally update the lexical index.

    changed_files: [(rel_path, full_path), ...] to (re)index.
    deleted_files: rel_paths to remove.
    """
    con = _connect(index_dir)
    try:
        for rel in deleted_files or []:
            _remove_file(con, rel)
        for rel, full in changed_files:
            _remove_file(con, rel)  # clear stale row before re-adding
            try:
                with open(full, "r", encoding="utf8", errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            _index_file(con, rel, text)
        con.commit()
        log(
            f"[lexical_index] Incremental update: {len(changed_files)} changed, "
            f"{len(deleted_files or [])} deleted"
        )
    finally:
        con.close()


def search_lexical(index_dir: str, query_text: str, limit: int = 16) -> list[tuple[str, float]]:
    """
    Search the lexical index. Returns [(source, bm25_score), ...] ranked best-first
    (SQLite FTS5's bm25() returns negative scores where more negative = better match).
    """
    db_path = _db_path(index_dir)
    if not os.path.exists(db_path):
        return []

    tokens = set(extract_tokens(query_text).split())
    if not tokens:
        return []
    match_query = " OR ".join(f'"{t}"' for t in tokens)

    con = _connect(index_dir)
    try:
        cur = con.execute(
            "SELECT source, bm25(lexical_fts) AS score FROM lexical_fts "
            "WHERE lexical_fts MATCH ? ORDER BY score LIMIT ?",
            (match_query, limit),
        )
        return cur.fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()
