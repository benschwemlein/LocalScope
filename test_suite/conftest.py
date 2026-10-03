"""
Integration test suite for LocalScope indexing and querying.

Requires Ollama to be running locally with nomic-embed-text and llama3.1 (or whatever
is configured in config.py / env vars).

Run with:
    cd /path/to/dev/repos/LocalScope
    python3.13 -m pytest test_suite/ -v

Set SAMPLE_APP_PATH env var to override the default sample app location.
"""

import sys
import os
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

# Project root is one level up from this file (test_suite/)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Clone https://github.com/benschwemlein/library-catalog-app into sample_apps/library-catalog-app
_DEFAULT_SAMPLE_APP = str(PROJECT_ROOT / "sample_apps" / "library-catalog-app")

SAMPLE_APP_PATH = os.environ.get("SAMPLE_APP_PATH", _DEFAULT_SAMPLE_APP)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_app_path():
    p = Path(SAMPLE_APP_PATH)
    if not p.exists():
        pytest.fail(
            f"Sample app not found at {p}.\n"
            "Set the SAMPLE_APP_PATH env var to the correct location."
        )
    return str(p)


@pytest.fixture(scope="session")
def session_index_dir(tmp_path_factory):
    """One shared ChromaDB directory for the entire test session."""
    return str(tmp_path_factory.mktemp("chroma_session"))


@pytest.fixture(scope="session")
def indexed_app(sample_app_path, session_index_dir):
    """
    Index the sample app ONCE for the entire test session.
    All read-only tests share this fixture — no re-embedding.
    """
    from indexing.incremental_indexer import index_repo_incremental

    logs = []
    index_repo_incremental(
        repo_root=sample_app_path,
        index_dir=session_index_dir,
        force_full_reindex=True,
        num_workers=2,  # mxbai-embed-large is stable at 2 workers (4 deadlocks Ollama)
        verbose=False,
        log=logs.append,
    )
    return {
        "index_dir": session_index_dir,
        "sample_app_path": sample_app_path,
        "logs": logs,
    }


@pytest.fixture()
def copied_index_dir(indexed_app, tmp_path):
    """
    Copy the session index into a per-test tmp dir.
    Mutation tests (add/modify/delete) use this so they get an isolated
    starting state without paying to re-embed the whole repo.
    """
    import shutil
    dest = str(tmp_path / "chroma_copy")
    shutil.copytree(indexed_app["index_dir"], dest)
    return dest
