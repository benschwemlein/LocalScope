# settings_defaults.py
from config import (
    OLLAMA_URL,
    EMBED_MODEL,
    CHAT_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_REPO_ROOT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_TOP_K,
    DEFAULT_MAX_DIRECT_EMBED_CHARS,
)
from indexing.indexer import CHARS_PER_CHUNK, CHUNK_OVERLAP, MAX_FILE_BYTES
from gui.index_tab import COMMON_FILETYPE_GROUPS

def build_default_settings() -> dict:
    filetypes = {}
    seen = set()
    for _, exts in COMMON_FILETYPE_GROUPS.items():
        for ext in exts:
            if ext in seen:
                continue
            seen.add(ext)
            filetypes[ext] = True

    return {
        "version": 1,
        "settings_tab": {
            "ollama_url": OLLAMA_URL,
            "embed_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
        },
        "index_tab": {
            "repo_root": DEFAULT_REPO_ROOT,
            "index_dir_base": DEFAULT_INDEX_DIR,
            "collection_name": DEFAULT_COLLECTION_NAME,
            "chars_per_chunk": CHARS_PER_CHUNK,
            "chunk_overlap": CHUNK_OVERLAP,
            "max_file_bytes": MAX_FILE_BYTES,
            "exclude_dirs_csv": ".git, .idea, .vscode, node_modules, build, dist, out, target, .gradle, .venv, venv, __pycache__",
            "filetypes": filetypes,
        },
        "query_tab": {
            "index_dir": DEFAULT_INDEX_DIR,
            "repo_root": DEFAULT_REPO_ROOT,
            "top_k": DEFAULT_TOP_K,
            "max_chars": DEFAULT_MAX_DIRECT_EMBED_CHARS,
            "bug_text": "",
        },
        "prompts_tab": {
            "summarizer_prompt": "",
            "chat_prompt": "",
        },
    }
