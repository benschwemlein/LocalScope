"""
Parallel incremental indexer - can update existing indexes without full rebuild.
Uses thread pool for parallel file processing.
"""

import os
import hashlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue

def get_file_hash(filepath: str) -> str:
    """Get SHA256 hash of file contents."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return ""

def index_repo_incremental(
    repo_root: str,
    index_dir: str | None = None,
    collection_name: str | None = None,
    index_exts: set[str] | None = None,
    excluded_dirs: set[str] | None = None,
    max_file_bytes: int = 500_000,
    chars_per_chunk: int = 800,  # 800 outperforms 1200 in semantic eval
    chunk_overlap: int = 200,
    use_ast_chunking: bool = True,
    skip_problematic_files: bool = True,
    force_full_reindex: bool = False,
    num_workers: int = 4,  # NEW: Number of parallel workers
    verbose: bool = False,  # NEW: Show individual file progress
    log=print,
):
    """
    Index repository incrementally with parallel processing.
    
    Args:
        num_workers: Number of parallel worker threads (default 4, max 16)
    """
    import chromadb
    from chromadb.config import Settings
    import chromadb.utils.embedding_functions as embedding_functions
    import config
    from indexing.ast_chunker import chunk_code_intelligently, chunk_text
    
    # Limit workers to reasonable range
    num_workers = max(1, min(num_workers, 16))
    
    repo_root = os.path.abspath(repo_root)
    index_dir = index_dir or config.DEFAULT_INDEX_DIR
    collection_name = collection_name or config.DEFAULT_COLLECTION_NAME
    
    start_time = time.time()
    
    log(f"Incremental indexing: {repo_root}")
    log(f"Parallel workers: {num_workers}")
    log(f"Force full reindex: {force_full_reindex}")
    
    client = chromadb.PersistentClient(
        path=index_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    
    embedding_function = embedding_functions.OllamaEmbeddingFunction(
        url=config.OLLAMA_URL,
        model_name=config.EMBED_MODEL
    )
    
    # Full reindex: delete and recreate
    if force_full_reindex:
        log("Performing FULL reindex...")
        try:
            client.delete_collection(collection_name)
            log(f"Deleted existing collection: {collection_name}")
        except:
            pass
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Code and documentation chunks"},
        embedding_function=embedding_function,
    )
    
    # Get existing file hashes from metadata
    log("Loading existing index state...")
    existing_chunks = collection.get(include=['metadatas'])
    
    # Build map of file -> hash from existing chunks
    indexed_files = {}
    for meta in existing_chunks.get('metadatas', []):
        if meta:
            source = meta.get('source')
            file_hash = meta.get('file_hash')
            if source and file_hash:
                indexed_files[source] = file_hash
    
    log(f"Found {len(indexed_files)} already-indexed files")
    
    # Scan repository for current files
    current_files = {}
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_root)
            
            if not should_index_file(full_path, index_exts):
                continue
            
            try:
                if os.path.getsize(full_path) > max_file_bytes:
                    continue
            except OSError:
                continue
            
            file_hash = get_file_hash(full_path)
            if file_hash:
                current_files[rel_path] = (full_path, file_hash)
    
    total_repo_files = len(current_files)
    log(f"Found {total_repo_files} indexable files in repository")
    
    # Determine what needs to be done
    files_to_add = []
    files_to_update = []
    files_to_delete = []
    
    for rel_path, (full_path, file_hash) in current_files.items():
        if rel_path not in indexed_files:
            files_to_add.append((rel_path, full_path, file_hash))
        elif indexed_files[rel_path] != file_hash:
            files_to_update.append((rel_path, full_path, file_hash))
    
    for rel_path in indexed_files:
        if rel_path not in current_files:
            files_to_delete.append(rel_path)
    
    log("")
    log(f"Files to add:    {len(files_to_add)}")
    log(f"Files to update: {len(files_to_update)}")
    log(f"Files to delete: {len(files_to_delete)}")
    log(f"Files unchanged: {len(current_files) - len(files_to_add) - len(files_to_update)}")
    
    if not files_to_add and not files_to_update and not files_to_delete:
        log("")
        log("✓ Index is up to date! No changes needed.")
        return
    
    # Delete chunks from removed files
    for rel_path in files_to_delete:
        log(f"Removing: {rel_path}")
        ids_to_delete = []
        for chunk_id, meta in zip(existing_chunks['ids'], existing_chunks['metadatas']):
            if meta and meta.get('source') == rel_path:
                ids_to_delete.append(chunk_id)
        
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            log(f"  Deleted {len(ids_to_delete)} chunks")
    
    # Delete chunks from modified files (will re-add below)
    for rel_path, full_path, file_hash in files_to_update:
        log(f"Updating: {rel_path}")
        ids_to_delete = []
        for chunk_id, meta in zip(existing_chunks['ids'], existing_chunks['metadatas']):
            if meta and meta.get('source') == rel_path:
                ids_to_delete.append(chunk_id)
        
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            log(f"  Deleted {len(ids_to_delete)} old chunks")
    
    # Process new and updated files IN PARALLEL
    files_to_process = files_to_add + files_to_update
    
    if not files_to_process:
        log("")
        log("DONE.")
        return
    
    log("")
    log(f"Processing {len(files_to_process)} files with {num_workers} workers...")
    
    # Thread-safe counters and locks
    total_chunks_added = 0
    total_files_processed = 0
    total_failed = 0
    stats_lock = Lock()
    
    # Worker function to process a single file
    def process_file(file_info):
        rel_path, full_path, file_hash = file_info
        
        try:
            # Read file
            with open(full_path, "r", encoding="utf8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            return {
                'success': False,
                'rel_path': rel_path,
                'error': f"Error reading: {e}",
                'chunks': 0
            }
        
        if not text.strip():
            return {
                'success': True,
                'rel_path': rel_path,
                'chunks': 0
            }
        
        # Chunk the file
        try:
            if use_ast_chunking:
                chunks = chunk_file_intelligently(text, full_path, chars_per_chunk, chunk_overlap)
            else:
                chunks = chunk_text(text, chars_per_chunk, chunk_overlap)
        except Exception as e:
            return {
                'success': False,
                'rel_path': rel_path,
                'error': f"Chunking error: {e}",
                'chunks': 0
            }
        
        # Prepare chunks for batch insertion
        chunk_data = []
        failed_chunks = 0
        
        for idx, chunk in enumerate(chunks):
            chunk = sanitize_chunk(f"// {rel_path}\n{chunk}")
            is_valid, reason = is_valid_chunk(chunk)
            if not is_valid:
                failed_chunks += 1
                continue

            chunk_data.append({
                'id': f"{rel_path}::chunk_{idx}",
                'document': chunk,
                'metadata': {
                    "source": rel_path,
                    "chunk_index": idx,
                    "file_hash": file_hash
                }
            })
        
        return {
            'success': True,
            'rel_path': rel_path,
            'chunk_data': chunk_data,
            'failed_chunks': failed_chunks,
            'chunks': len(chunk_data)
        }
    
    # Process files in parallel with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all files
        future_to_file = {
            executor.submit(process_file, file_info): file_info 
            for file_info in files_to_process
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            result = future.result()
            
            if result['success']:
                # Add chunks to collection (ChromaDB handles threading internally)
                if result.get('chunk_data'):
                    try:
                        collection.add(
                            ids=[c['id'] for c in result['chunk_data']],
                            documents=[c['document'] for c in result['chunk_data']],
                            metadatas=[c['metadata'] for c in result['chunk_data']],
                        )
                        
                        with stats_lock:
                            total_chunks_added += result['chunks']
                            total_files_processed += 1
                            total_failed += result.get('failed_chunks', 0)
                            
                            if verbose:
                                log(f"✓ {result['rel_path']}: {result['chunks']} chunks")
                            elif total_files_processed % 10 == 0:
                                log(f"Progress: {total_files_processed}/{len(files_to_process)} files, {total_chunks_added} chunks")
                    except Exception as e:
                        with stats_lock:
                            total_failed += result['chunks']
                        if verbose:
                            log(f"✗ {result['rel_path']}: Failed to add chunks - {e}")
                else:
                    with stats_lock:
                        total_files_processed += 1
                    if verbose:
                        log(f"○ {result['rel_path']}: Empty/skipped")
            else:
                with stats_lock:
                    total_failed += 1
                if verbose:
                    log(f"✗ {result['rel_path']}: {result.get('error', 'Unknown error')}")
    
    elapsed_time = time.time() - start_time
    
    log("")
    log("DONE.")
    log(f"Files processed: {total_files_processed}")
    log(f"Chunks added: {total_chunks_added}")
    log(f"Chunks failed: {total_failed}")
    log(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")


# Helper functions (copy from original indexer.py)

def sanitize_chunk(chunk: str) -> str:
    """Sanitize chunk to improve embedding success."""
    import re
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

def should_index_file(path: str, index_exts: set[str]) -> bool:
    """Check if file should be indexed."""
    _, ext = os.path.splitext(path)
    return ext.lower() in index_exts

def is_code_file(path: str) -> bool:
    """Check if file should use AST chunking."""
    CODE_FILE_EXTS = {
        ".py", ".pyi", ".java", ".cs", ".csx",
        ".js", ".jsx", ".ts", ".tsx",
        ".go", ".rs", ".cpp", ".cc", ".cxx", ".c",
    }
    _, ext = os.path.splitext(path)
    return ext.lower() in CODE_FILE_EXTS

def chunk_file_intelligently(text: str, file_path: str, chars_per_chunk: int, chunk_overlap: int) -> list[str]:
    """Chunk file using appropriate strategy."""
    from indexing.ast_chunker import chunk_code_intelligently, chunk_text
    
    if is_code_file(file_path):
        try:
            chunks = chunk_code_intelligently(text, file_path, max_chunk_size=chars_per_chunk, chunk_overlap=chunk_overlap)
            
            # Split oversized chunks
            final_chunks = []
            for i, chunk in enumerate(chunks):
                if len(chunk) > chars_per_chunk:
                    split_chunks = chunk_text(chunk, chars_per_chunk, chunk_overlap)
                    final_chunks.extend(split_chunks)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
        except Exception:
            return chunk_text(text, chars_per_chunk, chunk_overlap)
    else:
        return chunk_text(text, chars_per_chunk, chunk_overlap)