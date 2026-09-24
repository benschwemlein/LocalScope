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
EMBED_MODEL = env("LCQ_EMBED_MODEL", "mxbai-embed-large")
# Prepended to every chunk before embedding, for models trained with a
# document-side instruction. Changing it requires a full reindex.
EMBED_DOC_PREFIX = env("LCQ_EMBED_DOC_PREFIX", "")
CHAT_MODEL = env("LCQ_CHAT_MODEL", "qwen2.5:7b")

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

# Cross-encoder reranking (off by default). When on, retrieval fetches a
# wider pool of chunks, a cross-encoder scores each (question, chunk) pair,
# and the best-scoring chunk per file decides the final order. The number of
# files returned is unchanged.
RERANK_ENABLED = env("LCQ_RERANK_ENABLED", "0") == "1"
RERANK_MODEL = env("LCQ_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_POOL = int(env("LCQ_RERANK_POOL", "50"))
RERANK_DEVICE = env("LCQ_RERANK_DEVICE", "")  # "" = auto (mps, cuda or cpu)

# Second retrieval round (off by default). Reads the best first-round chunks,
# picks the code identifiers they use, searches the index for those names,
# and fuses that ranking with the first round. The number of files returned
# is unchanged.
#   "off"  one round only
#   "prf"  identifiers chosen by rarity (pseudo-relevance feedback, no LLM)
#   "llm"  identifiers chosen by CHAT_MODEL
SECOND_ROUND = env("LCQ_SECOND_ROUND", "off")
SECOND_ROUND_SEED_CHUNKS = int(env("LCQ_SECOND_ROUND_SEED_CHUNKS", "3"))
SECOND_ROUND_TERMS = int(env("LCQ_SECOND_ROUND_TERMS", "8"))

# Optional default repo
DEFAULT_REPO_ROOT = env("LCQ_REPO_ROOT", "")
