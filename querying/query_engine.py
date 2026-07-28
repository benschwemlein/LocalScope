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

    # Symbol-anchored routing: "who calls X", "what implements Y" etc. skip
    # vector search entirely and dispatch straight to a precanned graph query
    # (querying/router.py). Falls through to the conceptual (vector) path
    # below if graph retrieval is disabled, unavailable, the question doesn't
    # match a known pattern, or the resolved files aren't in this index.
    routed = None
    if config.GRAPH_ENABLED:
        graph_path = os.path.join(index_dir, "graph.json")
        if os.path.exists(graph_path):
            try:
                from graph.graph_store import GraphStore, GraphLoadError
                from querying.router import route_query

                graph_store = GraphStore.load(graph_path)
                routed = route_query(bug, graph_store)
                if routed:
                    log(
                        f"[query_engine] Routed to graph query '{routed.operation}': "
                        f"{len(routed.files)} file(s)"
                    )
            except GraphLoadError as e:
                log(f"[query_engine] Graph unavailable, falling back to vector search: {e}")
            except Exception as e:
                log(f"[query_engine] Routing failed (non-fatal), falling back to vector search: {e}")

    if routed is not None:
        if _cancelled():
            raise RuntimeError("Cancelled.")
        _step(1, "Querying graph...")

        fetched = collection.get(
            where={"source": {"$in": routed.files}},
            include=["documents", "metadatas"],
        )
        by_source: dict[str, tuple] = {}
        for doc, meta in zip(fetched.get("documents", []), fetched.get("metadatas", [])):
            source = meta.get("source", "")
            idx = meta.get("chunk_index", 0)
            if source and (source not in by_source or idx < by_source[source][1].get("chunk_index", 0)):
                by_source[source] = (doc, meta)

        docs, metas = [], []
        for f in routed.files:
            if f in by_source:
                doc, meta = by_source[f]
                docs.append(doc)
                metas.append(meta)

        if not docs:
            log(
                "[query_engine] Graph query resolved files but none are in this "
                "index; falling back to vector search."
            )
            routed = None
        else:
            # Graph hits are exact structural matches, not similarity-ranked —
            # a uniform distance keeps the relative score flat rather than
            # implying a ranking that doesn't exist.
            dists = [0.0] * len(docs)

    if routed is None:
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

        # Deduplicate by source file — keep only the best-scoring chunk per file.
        # This prevents one large file from flooding all top-k slots.
        seen_sources: set[str] = set()
        deduped: list[tuple] = []
        for doc, meta, dist in zip(docs_list[0], metas_list[0], dist_list[0]):
            source = meta.get("source", "")
            if source and source not in seen_sources:
                seen_sources.add(source)
                deduped.append((doc, meta, dist))

        top_pool = deduped[:top_k]
        docs  = [r[0] for r in top_pool]
        metas = [r[1] for r in top_pool]
        dists = [r[2] for r in top_pool]

        # Graph expansion: pull in files structurally adjacent (any edge type,
        # either direction) to the top-3 vector hits, so files the embedding
        # missed but that are directly connected still make it into context.
        # Appended after the ranked vector hits, not score-blended with them —
        # no published system fuses graph and vector scores into one ranking;
        # sequential composition (vector picks seeds, graph expands them) is
        # what the peer-reviewed literature converges on.
        if config.GRAPH_ENABLED:
            graph_path = os.path.join(index_dir, "graph.json")
            if os.path.exists(graph_path):
                try:
                    from graph.graph_store import GraphStore
                    from graph.queries import one_hop_neighbors

                    graph_store = GraphStore.load(graph_path)
                    neighbor_candidates: set[str] = set()
                    for meta in metas[:3]:
                        seed_source = meta.get("source", "")
                        if seed_source:
                            neighbor_candidates.update(one_hop_neighbors(graph_store, seed_source))
                    neighbor_candidates -= seen_sources

                    if neighbor_candidates:
                        fetched = collection.get(
                            where={"source": {"$in": list(neighbor_candidates)}},
                            include=["documents", "metadatas"],
                        )
                        worst_dist = max(dists) if dists else 1.0
                        added = 0
                        for doc, meta in zip(fetched.get("documents", []), fetched.get("metadatas", [])):
                            source = meta.get("source", "")
                            if source and source not in seen_sources and len(docs) < top_k * 2:
                                seen_sources.add(source)
                                docs.append(doc)
                                metas.append(meta)
                                dists.append(worst_dist)
                                added += 1
                        if added:
                            log(f"[query_engine] Graph expansion added {added} structurally adjacent file(s)")
                except Exception as e:
                    log(f"[query_engine] Graph expansion failed (non-fatal): {e}")

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