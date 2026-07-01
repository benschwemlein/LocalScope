import os
import json
import hashlib
import logging

from graph.edge import EdgeType
from graph.graph_store import GraphStore
from graph.plugin_registry import default_registry

# Side-effect imports to register all plugins
import graph.plugins.java_plugin       # noqa: F401
import graph.plugins.typescript_plugin # noqa: F401
import graph.plugins.html_plugin       # noqa: F401

log = logging.getLogger(__name__)

_GRAPH_EXTS = {".java", ".ts", ".tsx", ".html"}
_EXCLUDED_DIRS = {
    ".git", ".idea", ".vscode",
    "node_modules", "build", "dist", "out", "target", ".gradle",
    ".venv", "venv", "__pycache__",
}


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def build_incremental(
    repo_root: str,
    graph_path: str,
    changed_files: list[str] | None = None,
    log_fn=None,
) -> GraphStore:
    """
    Build or incrementally update the graph at graph_path.

    If graph_path exists, load it and re-process only changed/new/deleted files.
    If graph_path does not exist, build from scratch.

    changed_files: if provided, only consider these repo-relative paths for update.
    """
    if log_fn is None:
        log_fn = log.info

    hashes_path = os.path.splitext(graph_path)[0] + "_hashes.json"

    # Load existing graph and hashes
    if os.path.exists(graph_path):
        try:
            store = GraphStore.load(graph_path)
            with open(hashes_path, "r") as fh:
                saved_hashes: dict[str, str] = json.load(fh)
        except Exception as e:
            log_fn(f"[graph_builder] Could not load existing graph ({e}), rebuilding from scratch")
            store = GraphStore()
            saved_hashes = {}
    else:
        store = GraphStore()
        saved_hashes = {}

    # Collect current files
    current_files: dict[str, str] = {}  # rel_path → hash
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _GRAPH_EXTS:
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, repo_root)
            h = _file_hash(full)
            if h:
                current_files[rel] = h

    # Determine what to process
    if changed_files is not None:
        to_process = [p for p in changed_files if p in current_files]
        to_delete = [p for p in changed_files if p not in current_files and p in saved_hashes]
    else:
        to_process = [
            p for p, h in current_files.items()
            if h != saved_hashes.get(p)
        ]
        to_delete = [p for p in saved_hashes if p not in current_files]

    log_fn(f"[graph_builder] Files to process: {len(to_process)}, to delete: {len(to_delete)}")

    # Remove deleted/changed files from graph
    for rel in to_delete + to_process:
        store.remove_file(rel)

    # Re-extract edges for changed/new files
    for rel in to_process:
        full = os.path.join(repo_root, rel)
        ext = os.path.splitext(rel)[1].lower()
        plugin = default_registry.get(ext)
        if plugin is None:
            continue
        edges = plugin.extract_edges(full, rel)
        if edges:
            store.add_edges(edges)
            log_fn(f"[graph_builder]   {rel}: {len(edges)} edges")
        saved_hashes[rel] = current_files[rel]

    # Remove hashes for deleted files
    for rel in to_delete:
        saved_hashes.pop(rel, None)

    # Persist
    store.save(graph_path)
    with open(hashes_path, "w") as fh:
        json.dump(saved_hashes, fh)

    log_fn(f"[graph_builder] Graph saved: {store.node_count} nodes, {store.edge_count} edges")
    return store
