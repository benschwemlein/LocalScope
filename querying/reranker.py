"""
Cross-encoder reranking of retrieved chunks.

The embedding model scores the question and each chunk separately and
compares the two vectors. A cross-encoder reads the question and the chunk
together in one pass, which is slower but judges relevance much better. So
the embedding search casts a wide net cheaply and the cross-encoder orders
the catch.

A reranker can only reorder what the first pass retrieved. It cannot raise
recall above what the candidate pool contains.
"""

import threading

import config

_model = None
_model_key: tuple[str, str] | None = None
_lock = threading.Lock()


def _get_model():
    """Load the cross-encoder once per (model, device) and reuse it."""
    global _model, _model_key
    key = (config.RERANK_MODEL, config.RERANK_DEVICE)
    with _lock:
        if _model is None or _model_key != key:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(
                config.RERANK_MODEL,
                device=config.RERANK_DEVICE or None,
                max_length=512,
            )
            _model_key = key
    return _model


def score_pairs(question: str, passages: list[str]) -> list[float]:
    """Relevance score for each passage against the question; higher is better."""
    if not passages:
        return []
    model = _get_model()
    scores = model.predict([(question, p) for p in passages], batch_size=16)
    return [float(s) for s in scores]
