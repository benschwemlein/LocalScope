import os
from pathlib import Path

APP_NAME = "local_code_query"

BASE_DIR = Path(__file__).resolve().parent


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Human readable window title
APP_TITLE = env("LCQ_APP_TITLE", "Local Code Query")

# Ollama configuration (these will be edited by the Settings tab)
OLLAMA_URL = env("LCQ_OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = env("LCQ_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = env("LCQ_CHAT_MODEL", "llama3.1")

# Index storage
DEFAULT_INDEX_DIR = env(
    "LCQ_INDEX_DIR",
    str(BASE_DIR / "chroma_repo")
)

DEFAULT_COLLECTION_NAME = env(
    "LCQ_COLLECTION_NAME",
    "repo_chunks"
)

# Query behavior defaults
DEFAULT_TOP_K = int(env("LCQ_TOP_K", "16"))
DEFAULT_MAX_DIRECT_EMBED_CHARS = int(
    env("LCQ_MAX_DIRECT_EMBED_CHARS", "4000")
)

# Optional default repo
DEFAULT_REPO_ROOT = env("LCQ_REPO_ROOT", "")
