"""
Complete indexer with AST chunking and proper error handling.
"""

import os
import re
import requests
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import time

import config
from indexing.ast_chunker import chunk_code_intelligently, chunk_text

# COMPLETE list of file extensions
DEFAULT_INDEX_EXTS: set[str] = {
    # Microsoft / .NET
    ".cs", ".csx", ".vb",
    ".fs", ".fsi", ".fsx",
    ".xaml", ".cshtml", ".config", ".resx",
    ".sln", ".csproj", ".vbproj", ".fsproj",
    
    # PHP
    ".php", ".phtml", ".twig", ".tpl",
    
    # Java / JVM
    ".java",
    ".kt", ".kts",
    ".groovy", ".gvy",
    ".scala", ".sc",
    ".gradle",
    ".xml",
    ".properties",
    
    # JavaScript / Web
    ".js", ".jsx",
    ".ts", ".tsx",
    ".mjs", ".cjs",
    ".vue", ".svelte", ".astro",
    ".html",
    ".css", ".scss", ".less",
    ".json", ".json5",
    
    # Python / data
    ".py", ".pyi",
    ".ipynb",
    ".toml", ".ini", ".cfg", ".env",
    
    # Mobile / UI
    ".plist", ".storyboard", ".xib",
    
    # C / C++ / embedded
    ".c", ".h",
    ".cpp", ".hpp",
    ".cc", ".cxx", ".hh",
    ".ino", ".mk",
    
    # Rust / Go / Ruby / Swift / ObjC
    ".rs",
    ".go",
    ".rb",
    ".swift",
    ".m", ".mm",
    
    # Cloud / IaC
    ".tf", ".tfvars",
    ".yaml", ".yml",
    
    # SQL
    ".sql", ".psql", ".hql", ".cql",
    
    # Shell
    ".sh", ".bash", ".zsh", ".ksh", ".fish",
    ".bat", ".cmd",
    ".ps1", ".psm1",
    
    # Documentation
    ".md", ".rst", ".adoc",
    ".txt", ".csv",
}

MAX_FILE_BYTES = 500_000
CHARS_PER_CHUNK = 1200
CHUNK_OVERLAP = 200

# Code files use AST chunking
CODE_FILE_EXTS = {
    ".py", ".pyi",
    ".java",
    ".cs", ".csx",
    ".js", ".jsx",
    ".ts", ".tsx",
    ".go",
    ".rs",
    ".cpp", ".cc", ".cxx",
    ".c",
}


def sanitize_chunk(chunk: str) -> str:
    """Sanitize chunk to improve embedding success."""
    chunk = chunk.replace('\x00', '')
    chunk = ''.join(char for char in chunk if ord(char) >= 32 or char in '\n\t\r')
    chunk = re.sub(r'\n{4,}', '\n\n\n', chunk)
    return chunk.strip()


def is_valid_chunk(chunk: str, max_length: int = 8000) -> tuple[bool, str]:
    """Validate if chunk is suitable for embedding."""
    if not chunk or not chunk.strip():
        return False, "empty chunk"
    
    if len(chunk) > max_length:
        return False, f"chunk too long ({len(chunk)} > {max_length})"
    
    null_bytes = chunk.count('\x00')
    if null_bytes > 10:
        return False, f"contains {null_bytes} null bytes"
    
    printable_chars = sum(1 for c in chunk if c.isprintable() or c in '\n\t\r')
    if len(chunk) > 100 and printable_chars / len(chunk) < 0.7:
        return False, f"low printable ratio"
    
    return True, "valid"


def embed_text_with_retry(text: str, max_retries: int = 2, retry_delay: float = 1.0, log=print):
    """Call Ollama embedding with retry logic."""
    text = sanitize_chunk(text)
    
    is_valid, reason = is_valid_chunk(text)
    if not is_valid:
        log(f"[embed] Chunk rejected: {reason}")
        return None
    
    url = f"{config.OLLAMA_URL.rstrip('/')}/api/embeddings"
    payload = {"model": config.EMBED_MODEL, "prompt": text}
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            
            if resp.status_code == 200:
                data = resp.json()
                embedding = data.get("embedding")
                if embedding:
                    return embedding
                else:
                    log(f"[embed] No embedding in response")
                    return None
            
            elif resp.status_code == 500:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('error', 'Unknown')
                    log(f"[embed] Server error (500): {error_msg}")
                except:
                    log(f"[embed] Server error (500)")
                
                log(f"[embed] Chunk size: {len(text)} chars")
                log(f"[embed] Preview: {repr(text[:150])}")
                
                if attempt < max_retries:
                    log(f"[embed] Error 500, retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    log(f"[embed] Failed after {max_retries} retries: 500")
                    return None
            
            else:
                if attempt < max_retries:
                    log(f"[embed] Error {resp.status_code}, retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    log(f"[embed] Failed: {resp.status_code}")
                    return None
        
        except requests.Timeout:
            if attempt < max_retries:
                log(f"[embed] Timeout, retrying...")
                time.sleep(retry_delay)
                continue
            else:
                log(f"[embed] Timeout after {max_retries} retries")
                return None
        
        except Exception as e:
            log(f"[embed] Exception: {type(e).__name__}: {e}")
            return None
    
    return None


def should_index_file(path: str, index_exts: set[str]) -> bool:
    """Check if file should be indexed."""
    _, ext = os.path.splitext(path)
    return ext.lower() in index_exts


def is_code_file(path: str) -> bool:
    """Check if file should use AST chunking."""
    _, ext = os.path.splitext(path)
    return ext.lower() in CODE_FILE_EXTS


def chunk_file_intelligently(text: str, file_path: str, chars_per_chunk: int, chunk_overlap: int) -> list[str]:
    """Chunk file using appropriate strategy."""
    if is_code_file(file_path):
        try:
            chunks = chunk_code_intelligently(text, file_path, max_chunk_size=chars_per_chunk, chunk_overlap=chunk_overlap)
            
            # CRITICAL FIX: Actually split oversized chunks!
            final_chunks = []
            for i, chunk in enumerate(chunks):
                if len(chunk) > chars_per_chunk:
                    print(f"WARNING: Chunk {i} of {file_path} is {len(chunk)} chars (limit: {chars_per_chunk}), splitting...")
                    # Split this oversized chunk
                    split_chunks = chunk_text(chunk, chars_per_chunk, chunk_overlap)
                    final_chunks.extend(split_chunks)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
        except Exception as e:
            print(f"AST chunking failed for {file_path}: {e}")
            return chunk_text(text, chars_per_chunk, chunk_overlap)
    else:
        return chunk_text(text, chars_per_chunk, chunk_overlap)


def index_repo(
    repo_root: str,
    index_dir: str | None = None,
    collection_name: str | None = None,
    index_exts: set[str] | None = None,
    excluded_dirs: set[str] | None = None,
    max_file_bytes: int = MAX_FILE_BYTES,
    chars_per_chunk: int = CHARS_PER_CHUNK,
    chunk_overlap: int = CHUNK_OVERLAP,
    use_ast_chunking: bool = True,
    skip_problematic_files: bool = True,
    log=print,
):
    """Index repository with AST chunking."""
    repo_root = os.path.abspath(repo_root)
    index_dir = index_dir or config.DEFAULT_INDEX_DIR
    collection_name = collection_name or config.DEFAULT_COLLECTION_NAME
    index_exts = index_exts or DEFAULT_INDEX_EXTS

    default_excluded = {
        ".git", ".idea", ".vscode",
        "node_modules",
        "build", "dist", "out", "target", ".gradle",
        ".venv", "venv", "__pycache__",
    }
    excluded_dirs = excluded_dirs or default_excluded

    log(f"Indexing repo at {repo_root}")
    log(f"Using Chroma index directory: {index_dir}")
    log(f"Using collection name: {collection_name}")
    log(f"Using Ollama URL: {config.OLLAMA_URL}")
    log(f"Using embed model: {config.EMBED_MODEL}")
    log(f"Max file size (bytes): {max_file_bytes}")
    log(f"Chars per chunk: {chars_per_chunk}")
    log(f"Chunk overlap: {chunk_overlap}")
    log(f"AST-based chunking: {'ENABLED' if use_ast_chunking else 'DISABLED'}")
    log(f"Skip problematic files: {'YES' if skip_problematic_files else 'NO'}")

    client = chromadb.PersistentClient(
        path=index_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # ============================================================================
    # FIX: Create Ollama embedding function instead of using default
    # ============================================================================
    embedding_function = embedding_functions.OllamaEmbeddingFunction(
        url=config.OLLAMA_URL,
        model_name=config.EMBED_MODEL
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Code and documentation chunks"},
        embedding_function=embedding_function,  # THIS IS THE FIX!
    )

    chunk_count = 0
    file_count = 0
    ast_chunked_count = 0
    char_chunked_count = 0
    failed_chunks_count = 0

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

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
            
            # Chunk the file
            if use_ast_chunking:
                chunks = chunk_file_intelligently(text, full, chars_per_chunk, chunk_overlap)
                if is_code_file(full):
                    ast_chunked_count += 1
                else:
                    char_chunked_count += 1
            else:
                chunks = chunk_text(text, chars_per_chunk, chunk_overlap)
                char_chunked_count += 1
            
            for idx, chunk in enumerate(chunks):
                # NOTE: We're still calling embed_text_with_retry for validation,
                # but ChromaDB will use its OllamaEmbeddingFunction internally
                # when we call collection.add() without explicit embeddings
                
                # Validate the chunk first
                chunk = sanitize_chunk(chunk)
                is_valid, reason = is_valid_chunk(chunk)
                if not is_valid:
                    failed_chunks_count += 1
                    log(f"[index_repo] Skipping {rel} chunk {idx}: {reason}")
                    continue

                # Let ChromaDB handle the embedding via OllamaEmbeddingFunction
                try:
                    collection.add(
                        ids=[f"{rel}::chunk_{idx}"],
                        documents=[chunk],
                        metadatas=[{"source": rel, "chunk_index": idx}],
                        # No embeddings parameter - ChromaDB will call the embedding function
                    )
                    chunk_count += 1
                    
                    if chunk_count % 100 == 0:
                        log(f"Indexed {chunk_count} chunks...")
                        
                except Exception as e:
                    failed_chunks_count += 1
                    log(f"[index_repo] Failed to add {rel} chunk {idx}: {e}")
                    continue

            file_count += 1
            if file_count % 50 == 0:
                log(f"Processed {file_count} files...")

    log("")
    log("DONE.")
    log(f"Files processed: {file_count}")
    log(f"  - AST-chunked (code files): {ast_chunked_count}")
    log(f"  - Char-chunked (docs/configs): {char_chunked_count}")
    log(f"Chunks indexed: {chunk_count}")
    log(f"Chunks failed: {failed_chunks_count}")