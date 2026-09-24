"""
A second retrieval round driven by the code's own vocabulary.

A question is written in the asker's words; the code uses its own names. The
first round (embedding search) usually lands near the answer without landing
on all of it. Reading what it found reveals the names the codebase actually
uses, and searching for those names finds files the question's wording could
not reach. That is one iteration of the loop a search agent runs, scripted:

    round 1   embedding search, top files as usual
    terms     identifiers from the best round-1 chunks
    round 2   files in the index that contain those identifiers
    fuse      reciprocal rank fusion of the two rankings, cut to top_k

Terms are chosen one of two ways:

    prf   no model. Identifiers in the seed chunks, scored by how often they
          appear there times how rare they are across the index (classic
          pseudo-relevance feedback).
    llm   the chat model reads the question and the seed chunks and names
          the identifiers worth searching for. Names that don't occur in the
          index are discarded, so the model can only point at real code.

Round 2 is a substring search over indexed chunk text, so it needs no access
to the repository on disk and sees exactly the files the index holds.
"""

import json
import math
import re

import requests

import config

RRF_K = 60

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# Identifiers so common in Java/TypeScript that they carry no topic. Rarity
# weighting would sink most of these anyway; this keeps them out of the
# candidate list for the LLM path too.
_LANGUAGE_NOISE = {
    "String", "Integer", "Long", "Boolean", "Double", "Object", "List", "Map",
    "Set", "Optional", "Override", "Autowired", "Component", "Service",
    "Injectable", "Input", "Output", "Observable", "Promise", "BigDecimal",
    "LocalDate", "LocalDateTime", "ArrayList", "HashMap", "Collectors",
    "Exception", "RuntimeException", "Transactional", "RequestMapping",
    "GetMapping", "PostMapping", "PutMapping", "DeleteMapping", "PathVariable",
    "RequestBody", "RequestParam", "ResponseEntity", "Test", "BeforeEach",
    "Assertions", "Mockito", "NgModule", "OnInit", "OnDestroy",
    "HttpClient", "EventEmitter", "ViewChild", "HostListener", "TODO", "NOTE",
}

_MAX_DOC_FRACTION = 0.2  # terms in more than this share of files are noise


def _is_code_identifier(token: str) -> bool:
    """Names that look like code rather than English: an uppercase letter or
    an inner underscore. Plain lowercase words are mostly prose in comments."""
    if token in _LANGUAGE_NOISE:
        return False
    return any(c.isupper() for c in token) or "_" in token.strip("_")


class _IndexSearch:
    """Substring search over one collection's chunk text, with file counts."""

    def __init__(self, collection):
        self.collection = collection
        self._term_files: dict[str, dict[str, list[str]]] = {}
        metas = collection.get(include=["metadatas"])["metadatas"]
        self.n_files = len({m.get("source", "") for m in metas}) or 1

    def files_with(self, term: str) -> dict[str, list[str]]:
        """source file -> ids of its chunks containing term (case-sensitive)."""
        if term not in self._term_files:
            res = self.collection.get(
                where_document={"$contains": term}, include=["metadatas"]
            )
            files: dict[str, list[str]] = {}
            for cid, meta in zip(res["ids"], res["metadatas"]):
                files.setdefault(meta.get("source", ""), []).append(cid)
            self._term_files[term] = files
        return self._term_files[term]

    def idf(self, term: str) -> float:
        df = len(self.files_with(term))
        return math.log(self.n_files / df) if df else 0.0

    def usable(self, term: str) -> bool:
        df = len(self.files_with(term))
        return 0 < df <= max(1, _MAX_DOC_FRACTION * self.n_files)


# One searcher per collection object, so the file count and term lookups are
# computed once per index rather than once per question. The collection is
# kept alongside so its id() can't be reused by another object.
_searches: dict[int, tuple[object, _IndexSearch]] = {}


def _search_for(collection) -> _IndexSearch:
    entry = _searches.get(id(collection))
    if entry is None or entry[0] is not collection:
        entry = (collection, _IndexSearch(collection))
        _searches[id(collection)] = entry
    return entry[1]


def prf_terms(seed_docs: list[str], search: _IndexSearch, n: int) -> list[str]:
    counts: dict[str, int] = {}
    for doc in seed_docs:
        for tok in _IDENT.findall(doc):
            if _is_code_identifier(tok):
                counts[tok] = counts.get(tok, 0) + 1
    scored = [
        (count * search.idf(tok), tok)
        for tok, count in counts.items()
        if search.usable(tok)
    ]
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [tok for _, tok in scored[:n]]


_LLM_SCHEMA = {
    "type": "object",
    "properties": {"identifiers": {"type": "array", "items": {"type": "string"}}},
    "required": ["identifiers"],
}


def llm_terms(question: str, seed_docs: list[str], search: _IndexSearch, n: int,
              log=print) -> list[str]:
    snippets = "\n\n".join(f"--- snippet {i} ---\n{d}" for i, d in enumerate(seed_docs, 1))
    prompt = (
        "You are helping search a codebase for everything relevant to a question.\n\n"
        f"Question:\n{question}\n\n"
        f"Code found so far:\n{snippets}\n\n"
        f"Name up to {n} identifiers (class, interface, method, field or constant "
        "names, spelled exactly as in code) that should be searched for next to "
        "find the rest of the code involved: callers, implementations, "
        "configuration, the other side of an API call. Prefer names that appear "
        "in or are referenced by the snippets. Do not include generic framework "
        "or language names."
    )
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": config.CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": _LLM_SCHEMA,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        names = json.loads(resp.json()["message"]["content"]).get("identifiers", [])
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        log(f"[second_round] LLM term selection failed, using round 1 only: {e}")
        return []

    terms, seen = [], set()
    for name in names:
        name = str(name).strip()
        if name and name not in seen and name not in _LANGUAGE_NOISE and search.usable(name):
            seen.add(name)
            terms.append(name)
    return terms[:n]


def second_round(collection, question: str, round1: list[tuple], top_k: int,
                 mode: str, log=print) -> tuple[list[tuple], list[str]]:
    """
    Fuse a vocabulary-driven second round into round1.

    round1 is the first round's (doc, meta, dist) list, best first, one per
    file. Returns the fused list of at most top_k entries in the same shape,
    and the terms searched. Distances in the result are rank-based and
    lower-is-better.
    """
    search = _search_for(collection)
    seed_docs = [r[0] for r in round1[: config.SECOND_ROUND_SEED_CHUNKS]]
    n = config.SECOND_ROUND_TERMS

    if mode == "llm":
        terms = llm_terms(question, seed_docs, search, n, log)
    else:
        terms = prf_terms(seed_docs, search, n)
    if not terms:
        return round1[:top_k], []

    # Round 2: rank files by the summed rarity of the terms they contain.
    file_score: dict[str, float] = {}
    file_chunks: dict[str, dict[str, int]] = {}
    for term in terms:
        w = search.idf(term)
        for source, chunk_ids in search.files_with(term).items():
            file_score[source] = file_score.get(source, 0.0) + w
            hits = file_chunks.setdefault(source, {})
            for cid in chunk_ids:
                hits[cid] = hits.get(cid, 0) + 1
    round2 = sorted(file_score, key=lambda s: (-file_score[s], s))[: top_k * 2]

    round1_sources = [r[1].get("source", "") for r in round1]
    fused: dict[str, float] = {}
    for ranking in (round1_sources, round2):
        for rank, source in enumerate(ranking):
            fused[source] = fused.get(source, 0.0) + 1.0 / (RRF_K + rank + 1)
    order = sorted(fused, key=lambda s: -fused[s])[:top_k]

    # Round-1 files keep their chunk; new files get the chunk matching most terms.
    by_source = {r[1].get("source", ""): r for r in round1}
    missing = [s for s in order if s not in by_source]
    if missing:
        best_ids = [max(file_chunks[s], key=file_chunks[s].get) for s in missing]
        got = collection.get(ids=best_ids, include=["documents", "metadatas"])
        for doc, meta in zip(got["documents"], got["metadatas"]):
            by_source[meta.get("source", "")] = (doc, meta, 0.0)

    result = [
        (by_source[s][0], by_source[s][1], float(rank))
        for rank, s in enumerate(order)
        if s in by_source
    ]
    return result, terms
