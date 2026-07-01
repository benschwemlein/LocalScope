import os
import json
import threading
from typing import Callable, Any

import requests
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

import config  # note: import module, not constants

LogFn = Callable[[str], Any]


def _embed_text(text: str, log: LogFn) -> list[float] | None:
    url = f"{config.OLLAMA_URL.rstrip('/')}/api/embeddings"
    payload = {"model": config.EMBED_MODEL, "prompt": text}

    try:
        resp = requests.post(url, json=payload)
    except requests.RequestException as e:
        log(f"[embed_text] Error calling Ollama: {e}")
        return None

    if not resp.ok:
        log(f"[embed_text] Ollama returned {resp.status_code}")
        try:
            log(f"[embed_text] Body (first 400 chars): {resp.text[:400]!r}")
        except Exception:
            pass
        return None

    try:
        data = resp.json()
    except ValueError as e:
        log(f"[embed_text] Could not parse JSON from Ollama: {e}")
        return None

    embedding = data.get("embedding")
    if embedding is None:
        log(f"[embed_text] No 'embedding' field in response: {data}")
        return None

    return embedding


def _summarize_query(long_text: str, template: str, log: LogFn) -> str:
    if "<<BUG_TEXT>>" in template:
        user_content = template.replace("<<BUG_TEXT>>", long_text)
    else:
        user_content = template + "\n\nBug text:\n" + long_text

    url = f"{config.OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": config.CHAT_MODEL,
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        resp = requests.post(url, json=payload)
    except requests.RequestException as e:
        log(f"[summarize_query] Error calling Ollama: {e}")
        return long_text

    if not resp.ok:
        log(f"[summarize_query] Ollama returned {resp.status_code}")
        log(f"[summarize_query] Body (first 400 chars): {resp.text[:400]!r}")
        return long_text

    try:
        data = resp.json()
    except ValueError as e:
        log(f"[summarize_query] Could not parse JSON: {e}")
        return long_text

    summary = data["message"]["content"].strip()
    log(f"[query_engine] Summarized question to {len(summary)} chars for embedding.")
    return summary


def _chat_with_context(
    question: str,
    docs,
    metas,
    template: str,
    log: LogFn,
    token_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        path = meta.get("source", meta.get("path", "<unknown>"))
        chunk_idx = meta.get("chunk_index", "?")
        header = f"[Snippet {i} from {path} chunk {chunk_idx}]"
        context_parts.append(header + "\n" + doc)

    snippets_text = "\n\n".join(context_parts)

    prompt = template
    if "<<BUG_TEXT>>" in prompt:
        prompt = prompt.replace("<<BUG_TEXT>>", question)
    else:
        prompt = prompt + "\n\nBug description:\n" + question

    if "<<SNIPPETS>>" in prompt:
        prompt = prompt.replace("<<SNIPPETS>>", snippets_text)
    else:
        prompt = prompt + "\n\nRelevant snippets:\n" + snippets_text

    url = f"{config.OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": config.CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": token_callback is not None,
        "options": {"temperature": 0.0},
    }

    resp = requests.post(url, json=payload, stream=token_callback is not None)
    if not resp.ok:
        log(f"[chat_with_context] Ollama returned {resp.status_code}")
        log(f"[chat_with_context] Body (first 400 chars): {resp.text[:400]!r}")
        resp.raise_for_status()

    if token_callback is None:
        data = resp.json()
        return data["message"]["content"].strip()

    # Streaming: yield tokens to callback, check cancel between each
    full_text = []
    try:
        for line in resp.iter_lines():
            if cancel_event and cancel_event.is_set():
                resp.close()
                return "".join(full_text)
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                full_text.append(token)
                token_callback(token)
            if chunk.get("done"):
                break
    except Exception as e:
        log(f"[chat_with_context] Streaming error: {e}")

    return "".join(full_text)


def _compute_relative_scores(distances: list[float]) -> list[float]:
    if not distances:
        return []

    min_d = min(distances)
    max_d = max(distances)

    if max_d == min_d:
        return [100.0 for _ in distances]

    scores: list[float] = []
    for d in distances:
        score = 100.0 * (max_d - d) / (max_d - min_d)
        if score < 0:
            score = 0.0
        if score > 100:
            score = 100.0
        scores.append(score)
    return scores


def run_query(
    bug_text: str,
    index_dir: str | None = None,
    repo_root: str | None = None,
    top_k: int = 12,
    max_chars: int = 4000,
    summarizer_template: str = "",
    chat_template: str = "",
    log: LogFn = print,
    cancel_event: threading.Event | None = None,
    token_callback: Callable[[str], None] | None = None,
    step_callback: Callable[[int, str], None] | None = None,
):
    """
    Run a query against the ChromaDB index using Ollama embeddings and chat.
    """
    index_dir = index_dir or config.DEFAULT_INDEX_DIR
    bug = bug_text.strip()
    if not bug:
        raise ValueError("Bug or question text is required.")

    try:
        client = chromadb.PersistentClient(
            path=index_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        collections = client.list_collections()
        if not collections:
            raise RuntimeError(
                "No collections found in this index directory. "
                "You may need to build an index first."
            )

        if len(collections) > 1:
            log("[query_engine] Multiple collections found in this index directory.")
            log("[query_engine] Available collections:")
            for c in collections:
                log(f"  {c.name}")
            log(f"[query_engine] Using first collection: {collections[0].name}")
        else:
            log(f"[query_engine] Using collection: {collections[0].name}")

        # Get collection without embedding function since we do manual embedding
        collection = client.get_collection(collections[0].name)

    except Exception as e:
        raise RuntimeError(f"Could not open collection from index directory: {e}") from e

    log(f"[query_engine] Using index directory: {index_dir}")
    log(f"[query_engine] Using embed model: {config.EMBED_MODEL}")
    log(f"[query_engine] Using chat model: {config.CHAT_MODEL}")
    log(f"[query_engine] Using Ollama URL: {config.OLLAMA_URL}")

    def _cancelled():
        return cancel_event is not None and cancel_event.is_set()

    def _step(n: int, label: str):
        if step_callback:
            step_callback(n, label)

    query_for_embedding = bug
    if len(bug) > max_chars:
        if _cancelled():
            raise RuntimeError("Cancelled.")
        log(f"[query_engine] Bug text is {len(bug)} chars, summarizing before embedding...")
        _step(1, "Summarizing...")
        query_for_embedding = _summarize_query(bug, summarizer_template, log)

    if _cancelled():
        raise RuntimeError("Cancelled.")

    _step(1 if len(bug) <= max_chars else 2, "Embedding...")
    log("[query_engine] Embedding query text...")
    q_embedding = _embed_text(query_for_embedding, log)
    if q_embedding is None:
        raise RuntimeError("Failed to obtain embedding from Ollama.")

    _step(2, "Searching index...")
    log(f"[query_engine] Querying index for top {top_k} snippets...")

    # Fetch more candidates so deduplication still yields top_k results
    res = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k * 3,
        include=["documents", "metadatas", "distances"],
    )

    docs_list = res.get("documents", [[]])
    metas_list = res.get("metadatas", [[]])
    dist_list = res.get("distances", [[]])

    if not docs_list or not docs_list[0]:
        log("[query_engine] No relevant snippets found in the index.")
        raise RuntimeError("No relevant snippets found in the index.")

    # Deduplicate by source file — build comprehensive map first, then slice top_k.
    # The full map is used as the seed pool for graph fusion so graph-adjacent
    # files that rank 11–30 in the vector pass can still get cosine scores.
    ext_pool: dict[str, tuple] = {}  # source → (doc, meta, dist)
    for doc, meta, dist in zip(docs_list[0], metas_list[0], dist_list[0]):
        source = meta.get("source", "")
        if source and source not in ext_pool:
            ext_pool[source] = (doc, meta, dist)

    # Top-k slice for the non-fusion path and as initial ordering
    top_pool = list(ext_pool.values())[:top_k]
    docs  = [r[0] for r in top_pool]
    metas = [r[1] for r in top_pool]
    dists = [r[2] for r in top_pool]

    # Graph-hybrid fusion: expand and re-rank seeds via structural graph traversal
    if config.GRAPH_ENABLED:
        graph_path = os.path.join(index_dir, "graph.json")
        if os.path.exists(graph_path):
            try:
                from graph.graph_store import GraphStore, GraphLoadError
                from graph.fusion import SeedResult, expand_and_rerank

                graph_store = GraphStore.load(graph_path)

                # Augment ext_pool with 1-hop graph neighbors not yet retrieved.
                # Files like concrete strategy classes referenced by the seed file
                # may never appear in ChromaDB's top-N results, yet are structurally
                # adjacent. Fetch them specifically so they get real cosine scores
                # and the fusion re-ranker can surface them.
                try:
                    import networkx as _nx
                    _g_undir = graph_store._g.to_undirected()
                    _orig_pool: set[str] = set(ext_pool.keys())
                    _missing: set[str] = set()
                    for _fp in _orig_pool:
                        if _fp in _g_undir:
                            for _nb in _g_undir.neighbors(_fp):
                                if _nb not in ext_pool:
                                    _missing.add(_nb)
                    if _missing:
                        _adj = collection.query(
                            query_embeddings=[q_embedding],
                            n_results=min(len(_missing) * 3, 90),
                            where={"source": {"$in": list(_missing)}},
                            include=["documents", "metadatas", "distances"],
                        )
                        _before = len(ext_pool)
                        for _doc, _meta, _dist in zip(
                            _adj.get("documents", [[]])[0],
                            _adj.get("metadatas", [[]])[0],
                            _adj.get("distances", [[]])[0],
                        ):
                            _src = _meta.get("source", "")
                            if _src and _src not in ext_pool:
                                ext_pool[_src] = (_doc, _meta, _dist)
                        log(
                            f"[query_engine] Graph expansion: added "
                            f"{len(ext_pool) - _before} adjacent files to pool"
                        )

                    # Graph-proximity boost: reduce the effective cosine distance
                    # of expanded files (graph neighbors not in the vector top-N)
                    # that are structurally adjacent via REFERENCES or INHERITS to
                    # one of the top-3 vector seeds.
                    #
                    # Only EXPANDED files are boosted — orig_pool files keep their
                    # natural vector distances so P@5 and MRR are not degraded for
                    # queries where vector search already performs well.
                    #
                    # Only the top-3 seeds may propagate the boost. This prevents
                    # tangentially-related seeds (e.g. OverdueFineContext appearing
                    # in a loan-checkout query pool) from pulling in fine-domain
                    # files and displacing genuinely relevant checkout files.
                    #
                    # IMPORTS edges are excluded: they link to loosely-related
                    # utility/entity classes across package boundaries.
                    _beta = config.GRAPH_BETA
                    _BOOST_EDGE_TYPES = frozenset({"REFERENCES", "INHERITS"})
                    _dg = graph_store._g  # directed graph for edge-type lookup
                    _orig_dists = sorted(ext_pool[fp][2] for fp in _orig_pool)
                    _N_SEEDS = 3
                    _idx = min(_N_SEEDS - 1, len(_orig_dists) - 1)
                    _quality_threshold = _orig_dists[_idx] if _orig_dists else 1.0
                    for _af in list(ext_pool.keys()):
                        if _af in _orig_pool:
                            continue  # only boost newly-expanded files
                        if _af not in _g_undir:
                            continue
                        _best_seed_dist: float | None = None
                        for _nb in _g_undir.neighbors(_af):
                            if _nb == _af:
                                continue
                            if _nb not in _orig_pool:
                                continue  # only boost from original vector seeds
                            if ext_pool[_nb][2] > _quality_threshold:
                                continue  # only top-3 seeds may propagate
                            _ed = _dg.get_edge_data(_nb, _af) or _dg.get_edge_data(_af, _nb)
                            if _ed and _ed.get("edge_type") in _BOOST_EDGE_TYPES:
                                _sd = ext_pool[_nb][2]
                                if _best_seed_dist is None or _sd < _best_seed_dist:
                                    _best_seed_dist = _sd
                        if _best_seed_dist is None:
                            continue
                        _raw_dist = ext_pool[_af][2]
                        _boosted_dist = min(_raw_dist, _best_seed_dist * _beta)
                        if _boosted_dist < _raw_dist:
                            _doc_af, _meta_af, _ = ext_pool[_af]
                            ext_pool[_af] = (_doc_af, _meta_af, _boosted_dist)
                except Exception as _adj_exc:
                    log(f"[query_engine] Adjacent-file expansion failed (non-fatal): {_adj_exc}")

                # Use the full extended + adjacent pool as seeds
                seeds = [
                    SeedResult(
                        file_path=fp,
                        cos_distance=d,
                        content_snippet=doc[:500],
                    )
                    for fp, (doc, _meta, d) in ext_pool.items()
                ]
                fused = expand_and_rerank(
                    seeds,
                    query_for_embedding,
                    graph_store,
                    alpha=config.GRAPH_ALPHA,
                    beta=config.GRAPH_BETA,
                )
                if fused:
                    fused_paths = [r.file_path for r in fused[:top_k]]
                    # Rebuild docs/metas/dists from the comprehensive pool
                    new_docs, new_metas, new_dists = [], [], []
                    for fp in fused_paths:
                        if fp in ext_pool:
                            doc, meta, d = ext_pool[fp]
                            new_docs.append(doc)
                            new_metas.append(meta)
                            new_dists.append(d)
                    if new_docs:
                        docs, metas, dists = new_docs, new_metas, new_dists
                        log(f"[query_engine] Graph fusion reranked {len(docs)} results")
            except Exception as exc:
                log(f"[query_engine] Graph fusion failed, falling back to vector-only: {exc}")

    scores = _compute_relative_scores(dists)

    count = len(metas)
    log(f"Retrieved {count} snippet chunks.")

    log("Using snippets from:")
    for idx, (meta, dist, score) in enumerate(zip(metas, dists, scores), start=1):
        # FIXED: Changed from "path" to "source" to match indexer metadata
        path = meta.get("source", meta.get("path", "<unknown>"))
        chunk_idx = meta.get("chunk_index", "?")
        log(
            f"  [{idx:02d}] {score:5.1f}%  {path} (chunk {chunk_idx}, distance {dist:.4f})"
        )

    best_score = max(scores) if scores else 0.0
    if best_score < 15.0:
        log("")
        log("[query_engine] WARNING: All retrieved snippets have very low relative scores.")
        log("[query_engine] The answer may rely mostly on the bug text and not on code context.")

    _step(3, "Searching index...")

    if _cancelled():
        raise RuntimeError("Cancelled.")

    _step(3, "Generating answer...")
    log("")
    log("[query_engine] Asking LLM with retrieved context...")
    if token_callback:
        log("\n=== ANSWER ===\n")

    answer = _chat_with_context(
        bug, docs, metas, chat_template, log,
        token_callback=token_callback,
        cancel_event=cancel_event,
    )

    return {
        "answer": answer,
        "docs": docs,
        "metas": metas,
        "distances": dists,
        "scores": scores,
    }