"""
Cross-language REST edges: HTTP client calls -> server route handlers.

Frontend and backend in a full-stack repo are connected by HTTP, not by
imports, so no per-file parser can see the link. An Angular service issuing
`GET /api/patrons/{id}` and the Spring controller annotated to serve that
route share no symbol, no import, and no file reference. Without this pass
the two halves of the graph are entirely disconnected components.

Unlike the language plugins this is a WHOLE-REPO pass: routes are collected
from every server file and every client file, then joined on the normalized
(verb, path) pair. It cannot be expressed as LanguagePlugin.extract_edges,
which sees one file at a time.

Matching is deliberately conservative. A wrong edge is worse than a missing
one, since it silently corrupts every traversal that crosses it.

Route normalization handles the ways real projects differ:
  - path parameters      /patrons/{id}  and  /patrons/${id}   -> /patrons/*
  - server context path  server.servlet.context-path prepended to routes
  - class-level prefix   @RequestMapping on the class + @GetMapping on method
  - client base URLs     private api = 'http://host:8080/api/loans'
  - absolute URLs        scheme://host:port stripped
  - query strings        ?page=1&size=20 stripped

Because that list is necessarily incomplete (JAX-RS, fetch/axios, generated
OpenAPI clients, GraphQL), extract_rest_edges reports how many routes it
found on each side and how many matched. A silent zero is the failure mode
worth guarding against: it looks exactly like "this repo has no frontend".
"""

import os
import re
import logging
from dataclasses import dataclass

from graph.edge import Edge, EdgeType

log = logging.getLogger(__name__)

_SERVER_EXTS = {".java"}
_CLIENT_EXTS = {".ts", ".tsx", ".js", ".jsx"}

_EXCLUDED_DIRS = {
    ".git", ".idea", ".vscode",
    "node_modules", "build", "dist", "out", "target", ".gradle",
    ".venv", "venv", "__pycache__",
}

# --- server-side annotations ------------------------------------------------

_SPRING_METHOD_ANN = re.compile(
    r"@(Get|Post|Put|Delete|Patch|Request)Mapping\s*(?:\(([^)]*)\))?", re.S
)
_CLASS_LEVEL = re.compile(
    r"@RequestMapping\s*\(\s*(?:value\s*=\s*)?\"([^\"]*)\"[^)]*\)"
    r"(?:[^;{]*?)\bclass\b",
    re.S,
)
_ANN_PATH = re.compile(r"(?:value|path)\s*=\s*\"([^\"]*)\"")
_ANN_BARE_PATH = re.compile(r"^\s*\"([^\"]*)\"")
_ANN_METHOD = re.compile(r"RequestMethod\.(\w+)")

# JAX-RS
_JAXRS_PATH = re.compile(r"@Path\s*\(\s*\"([^\"]*)\"\s*\)")
_JAXRS_VERB = re.compile(r"@(GET|POST|PUT|DELETE|PATCH)\b")

# --- client-side calls ------------------------------------------------------

# .get<T>('...')  .post(`...`)  http.delete("...")
_HTTP_CALL = re.compile(
    r"\.\s*(get|post|put|delete|patch)\s*(?:<[^>()]*>)?\s*\(\s*([`'\"])(.*?)\2",
    re.S,
)
# private apiUrl = 'http://localhost:8080/api/loans';   readonly base = `/holds`;
_FIELD_ASSIGN = re.compile(
    r"(?:private|public|protected|readonly|const|let|var)\s+(?:readonly\s+)?"
    r"(\w+)\s*(?::\s*[\w<>\[\]|\s]+)?\s*=\s*[`'\"]([^`'\"]*)[`'\"]"
)

_SCHEME_HOST = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/]+")
_TEMPLATE_VAR = re.compile(r"\$\{([^}]*)\}")
_PATH_PARAM = re.compile(r"\{[^}]*\}")


@dataclass
class RestDiagnostics:
    server_routes: int = 0
    client_calls: int = 0
    matched_routes: int = 0
    edges: int = 0
    context_path: str = ""

    def summary(self) -> str:
        ctx = f", context-path={self.context_path}" if self.context_path else ""
        return (
            f"server routes={self.server_routes}, client calls={self.client_calls}, "
            f"matched={self.matched_routes}, edges={self.edges}{ctx}"
        )


def _normalize_path(raw: str) -> str | None:
    """Reduce a route to a comparable form, or None if it isn't a usable path."""
    p = raw.strip()
    if not p:
        return None
    p = _SCHEME_HOST.sub("", p)          # http://host:8080/api/x -> /api/x
    p = p.split("?", 1)[0].split("#", 1)[0]
    p = _TEMPLATE_VAR.sub("*", p)        # ${id}  -> *
    p = _PATH_PARAM.sub("*", p)          # {id}   -> *
    p = re.sub(r"/+", "/", p)
    if not p.startswith("/"):
        return None                      # header lookups like .get('Authorization')
    p = p.rstrip("/")
    return p or "/"


def _find_context_path(repo_root: str) -> str:
    """Read server.servlet.context-path, which the client must include but the
    server's annotations omit."""
    candidates = []
    for base in ("src/main/resources", "src/main/resources/config", "config", "."):
        for name in ("application.yml", "application.yaml", "application.properties"):
            candidates.append(os.path.join(repo_root, base, name))

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue

        m = re.search(r"^\s*(?:server\.)?(?:servlet\.)?context-path\s*[:=]\s*(\S+)",
                      text, re.M)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val.startswith("/"):
                return val.rstrip("/")
    return ""


def _walk(repo_root: str, exts: set[str], excluded: set[str]):
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in excluded]
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                full = os.path.join(root, fname)
                yield full, os.path.relpath(full, repo_root)


def _extract_server_routes(repo_root: str, context_path: str, excluded: set[str]):
    """(verb, normalized_path) -> {relative file paths}"""
    routes: dict[tuple[str, str], set[str]] = {}

    def add(verb: str, path: str, rel: str):
        norm = _normalize_path(context_path + path if path.startswith("/") else
                               context_path + "/" + path if path else context_path or "/")
        if norm:
            routes.setdefault((verb, norm), set()).add(rel)

    for full, rel in _walk(repo_root, _SERVER_EXTS, excluded):
        try:
            text = open(full, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if "Mapping" not in text and "@Path" not in text:
            continue

        cls = _CLASS_LEVEL.search(text)
        prefix = cls.group(1) if cls else ""

        for m in _SPRING_METHOD_ANN.finditer(text):
            kind, body = m.group(1), m.group(2) or ""
            if cls and m.start() < cls.end() and kind == "Request":
                continue  # the class-level annotation itself, not a handler

            pm = _ANN_PATH.search(body) or _ANN_BARE_PATH.search(body)
            path = pm.group(1) if pm else ""

            if kind == "Request":
                verbs = [v.upper() for v in _ANN_METHOD.findall(body)] or ["GET"]
            else:
                verbs = [kind.upper()]

            for v in verbs:
                add(v, prefix + path, rel)

        # JAX-RS: class @Path + method @Path/@GET
        jax_cls = _JAXRS_PATH.search(text)
        if jax_cls:
            for vm in _JAXRS_VERB.finditer(text):
                seg = text[vm.end(): vm.end() + 400]
                pm = _JAXRS_PATH.search(seg)
                add(vm.group(1).upper(), jax_cls.group(1) + (pm.group(1) if pm else ""), rel)

    return routes


def _extract_client_calls(repo_root: str, excluded: set[str]):
    """(verb, normalized_path) -> {relative file paths}"""
    calls: dict[tuple[str, str], set[str]] = {}

    for full, rel in _walk(repo_root, _CLIENT_EXTS, excluded):
        try:
            text = open(full, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if "http" not in text.lower():
            continue

        # Base URLs held in fields, so `${this.apiUrl}/x` resolves to a real path
        consts = {name: val for name, val in _FIELD_ASSIGN.findall(text)}

        for m in _HTTP_CALL.finditer(text):
            verb, raw = m.group(1).upper(), m.group(3)

            def resolve(match):
                # `${this.apiUrl}` is a base URL worth substituting;
                # `${id}` is a genuine path parameter -> wildcard.
                key = match.group(1).strip().split(".")[-1]
                return consts.get(key, "*")

            norm = _normalize_path(_TEMPLATE_VAR.sub(resolve, raw))
            if norm:
                calls.setdefault((verb, norm), set()).add(rel)

    return calls


def _suffix_match(client_path: str, server_path: str, min_segments: int = 2) -> bool:
    """Fallback when an unresolved prefix keeps two equivalent routes apart.
    Requires enough trailing segments to make a coincidental match unlikely."""
    c = [s for s in client_path.split("/") if s]
    s = [s for s in server_path.split("/") if s]
    n = min(len(c), len(s))
    if n < min_segments:
        return False
    return c[-n:] == s[-n:]


def extract_rest_edges(
    repo_root: str,
    excluded_dirs: set[str] | None = None,
    log_fn=None,
) -> tuple[list[Edge], RestDiagnostics]:
    """
    Join client HTTP calls to server route handlers across the whole repo.

    Returns the edges plus diagnostics. Callers should surface the diagnostics:
    zero matches in a repo that clearly has both halves means the idioms here
    don't cover that project, and that must not fail silently.
    """
    if log_fn is None:
        log_fn = log.info
    excluded = excluded_dirs or _EXCLUDED_DIRS

    context_path = _find_context_path(repo_root)
    server = _extract_server_routes(repo_root, context_path, excluded)
    client = _extract_client_calls(repo_root, excluded)

    diag = RestDiagnostics(
        server_routes=len(server), client_calls=len(client), context_path=context_path
    )

    pairs: set[tuple[str, str]] = set()
    matched_keys = set()

    for ckey, cfiles in client.items():
        targets = server.get(ckey)
        if targets is None:
            verb, cpath = ckey
            hits = [
                sfiles for (sverb, spath), sfiles in server.items()
                if sverb == verb and _suffix_match(cpath, spath)
            ]
            if len(hits) != 1:
                continue  # ambiguous or absent: refuse to guess
            targets = hits[0]

        matched_keys.add(ckey)
        for cf in cfiles:
            for sf in targets:
                if cf != sf:
                    pairs.add((cf, sf))

    diag.matched_routes = len(matched_keys)
    diag.edges = len(pairs)

    edges = [Edge(c, s, EdgeType.CALLS_ENDPOINT) for c, s in sorted(pairs)]

    if client and not matched_keys:
        log_fn(
            f"[rest_edges] WARNING: found {len(client)} client HTTP calls and "
            f"{len(server)} server routes but matched NONE. The REST idioms in "
            f"this repo are not covered; the frontend and backend will remain "
            f"disconnected in the graph."
        )
    else:
        log_fn(f"[rest_edges] {diag.summary()}")

    return edges, diag
