"""
The embedding function the indexers hand to ChromaDB.

Some embedding models are trained with an instruction on the document side
as well as the query side (for example jina-code-embeddings expects
"Candidate code snippet:\\n" before every passage). Without it they embed
documents in a different space from the one they were trained to search.
config.EMBED_DOC_PREFIX supplies it; the default is no prefix, which is what
most models expect.

The prefix changes only what is embedded, never the stored document text.
"""

import chromadb.utils.embedding_functions as embedding_functions

import config


class _PrefixedOllamaEmbeddingFunction(embedding_functions.OllamaEmbeddingFunction):
    def __init__(self, prefix: str, **kwargs):
        super().__init__(**kwargs)
        self._prefix = prefix

    def __call__(self, input):
        return super().__call__([self._prefix + doc for doc in input])


def make_embedding_function():
    kwargs = {"url": config.OLLAMA_URL, "model_name": config.EMBED_MODEL}
    if config.EMBED_DOC_PREFIX:
        return _PrefixedOllamaEmbeddingFunction(config.EMBED_DOC_PREFIX, **kwargs)
    return embedding_functions.OllamaEmbeddingFunction(**kwargs)
