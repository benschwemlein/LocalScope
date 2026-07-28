import os
import re
import logging

from graph.edge import Edge, EdgeType
from graph.plugin_registry import LanguagePlugin, default_registry

log = logging.getLogger(__name__)

# <%@ include file="header.jsp" %>  — static (compile-time) include directive
_DIRECTIVE_INCLUDE = re.compile(
    r"<%@\s*include\s+file\s*=\s*[\"']([^\"']+)[\"']", re.I
)

# <jsp:include page="header.jsp" />  — dynamic (runtime) include action
_ACTION_INCLUDE = re.compile(
    r"<jsp:include\s+[^>]*page\s*=\s*[\"']([^\"']+)[\"']", re.I
)

# Regex rather than tree-sitter: there is no tree-sitter-jsp grammar, and a
# JSP is HTML with arbitrary Java/EL interleaved, so the HTML grammar chokes
# on it. Include directives are a small, well-defined surface — matching them
# directly is more reliable here than parsing the whole document.


def _resolve_include(raw_path: str, source_dir: str, webapp_roots: list[str]) -> str | None:
    """
    Resolve a JSP include path to an absolute file path.

    Paths starting with "/" are context-relative (rooted at the webapp dir);
    everything else is relative to the including page.
    """
    if raw_path.startswith("/"):
        for root in webapp_roots:
            candidate = os.path.normpath(os.path.join(root, raw_path.lstrip("/")))
            if os.path.exists(candidate):
                return candidate
        return None

    candidate = os.path.normpath(os.path.join(source_dir, raw_path))
    return candidate if os.path.exists(candidate) else None


def _webapp_roots(abs_file: str, repo_root: str) -> list[str]:
    """
    Walk up from the JSP toward repo_root collecting plausible webapp roots
    for context-relative ("/foo.jsp") includes. Handles src/main/webapp,
    WebContent, and similar layouts without hardcoding any of them.
    """
    roots: list[str] = []
    current = os.path.dirname(abs_file)
    while current.startswith(repo_root) and current != repo_root:
        roots.append(current)
        current = os.path.dirname(current)
    roots.append(repo_root)
    return roots


class JspPlugin(LanguagePlugin):
    extensions = [".jsp", ".jspf", ".tag", ".tagx"]

    def extract_edges(self, file_path: str, source: str) -> list[Edge]:
        try:
            return self._extract(file_path, source)
        except Exception as exc:
            log.warning("JspPlugin: parse error in %s: %s", file_path, exc)
            return []

    def _extract(self, file_path: str, source: str) -> list[Edge]:
        abs_file = os.path.abspath(file_path)
        norm_src = os.path.normpath(source)
        repo_root: str | None = None
        if abs_file.endswith(os.sep + norm_src):
            repo_root = abs_file[: -(len(norm_src) + 1)]
        elif abs_file.endswith(norm_src):
            repo_root = abs_file[: -len(norm_src)].rstrip(os.sep)

        if repo_root is None:
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            return []

        source_dir = os.path.dirname(abs_file)
        roots = _webapp_roots(abs_file, repo_root)

        edges: list[Edge] = []
        seen: set[str] = set()

        for pattern in (_DIRECTIVE_INCLUDE, _ACTION_INCLUDE):
            for match in pattern.finditer(text):
                raw = match.group(1)
                resolved = _resolve_include(raw, source_dir, roots)
                if resolved is None:
                    continue
                if not resolved.startswith(repo_root + os.sep):
                    continue
                target = resolved[len(repo_root) + 1:]
                if target == source or target in seen:
                    continue
                seen.add(target)
                edges.append(Edge(source, target, EdgeType.IMPORTS))

        return edges


default_registry.register(JspPlugin())
