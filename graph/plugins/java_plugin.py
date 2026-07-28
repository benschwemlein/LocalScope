import os
import logging

import tree_sitter_java as tsjava
import tree_sitter

from graph.edge import Edge, EdgeType
from graph.plugin_registry import LanguagePlugin, default_registry

log = logging.getLogger(__name__)

_JAVA_LANG = tree_sitter.Language(tsjava.language())

# Discovered source roots per repo_root — the walk is repo-wide, and
# extract_edges runs once per file, so cache it rather than re-walking.
_SRC_ROOT_CACHE: dict[str, list[str]] = {}

_EXCLUDED_DIRS = {
    ".git", ".idea", ".vscode",
    "node_modules", "build", "dist", "out", "target", ".gradle",
    ".venv", "venv", "__pycache__",
}

# Java standard library type names we should not create edges for
_JAVA_BUILTINS = frozenset({
    "String", "Integer", "Long", "Double", "Float", "Boolean", "Byte", "Short", "Character",
    "Object", "Class", "Enum", "Void", "Number", "Math",
    "List", "Map", "Set", "Collection", "Queue", "Deque", "Stack", "Iterator",
    "ArrayList", "LinkedList", "HashMap", "HashSet", "TreeMap", "TreeSet",
    "Optional", "Stream", "Arrays", "Collections", "Objects",
    "StringBuilder", "StringBuffer",
    "Exception", "RuntimeException", "Error", "Throwable",
    "Thread", "Runnable", "Callable",
    "BigDecimal", "BigInteger", "LocalDate", "LocalTime", "LocalDateTime",
    "Duration", "Period", "Instant", "ZonedDateTime",
    "EnumMap", "EnumSet", "LinkedHashMap", "LinkedHashSet",
    "InputStream", "OutputStream", "Reader", "Writer",
    "Path", "File", "URI", "URL",
    "Override", "Deprecated", "SuppressWarnings", "FunctionalInterface",
    "Component", "Service", "Repository", "Controller", "Bean", "Autowired",
    "Slf4j", "Data", "Getter", "Setter", "Builder", "NoArgsConstructor",
    "AllArgsConstructor", "RequiredArgsConstructor",
})


def _repo_root(file_path: str, source: str) -> str | None:
    abs_file = os.path.abspath(file_path)
    norm_src = os.path.normpath(source)
    if abs_file.endswith(os.sep + norm_src):
        return abs_file[: -(len(norm_src) + 1)]
    if abs_file.endswith(norm_src):
        return abs_file[: -len(norm_src)].rstrip(os.sep)
    return None


def _discover_src_roots(repo_root: str) -> list[str]:
    """
    Find every Java source root in the repo, repo-relative, nearest-first.

    Multi-module builds (Gradle/Maven subprojects) put their sources at
    e.g. discovery-service/src/main/java, not just src/main/java at the top
    level, so a fixed list of top-level candidates silently resolves nothing
    for every module but the root one. Walk once and find them all.
    """
    cached = _SRC_ROOT_CACHE.get(repo_root)
    if cached is not None:
        return cached

    roots: list[str] = []
    for dirpath, dirnames, _files in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
        norm = dirpath.replace(os.sep, "/")
        if norm.endswith("/src/main/java") or norm.endswith("/src/test/java"):
            roots.append(os.path.relpath(dirpath, repo_root))

    # Shallowest first, so the root module wins ties over nested modules.
    roots.sort(key=lambda p: (p.count(os.sep), p))
    # Keep the legacy bare "src" fallback last for layouts without src/main/java.
    if os.path.isdir(os.path.join(repo_root, "src")):
        roots.append("src")

    _SRC_ROOT_CACHE[repo_root] = roots
    return roots


def _resolve_java(qualified: str, repo_root: str) -> str | None:
    rel = qualified.replace(".", "/") + ".java"
    for root in _discover_src_roots(repo_root):
        candidate = os.path.join(repo_root, root, rel)
        if os.path.exists(candidate):
            return os.path.join(root, rel)
    return None


def _resolve_same_package(simple_name: str, source_dir: str, repo_root: str, source: str) -> str | None:
    """Resolve a type name to a file in the same package directory."""
    candidate = os.path.join(source_dir, simple_name + ".java")
    if os.path.exists(candidate):
        rel = os.path.relpath(candidate, repo_root)
        if rel != source:
            return rel
    return None


def _txt(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _collect_type_identifiers(node, result: set[str]):
    if node.type == "type_identifier":
        name = _txt(node)
        if name and name not in _JAVA_BUILTINS:
            result.add(name)
    for child in node.children:
        _collect_type_identifiers(child, result)


def _walk_invokes(node, import_map: dict, source_dir: str, repo_root: str, source: str, edges: list[Edge]):
    if node.type == "object_creation_expression":
        t = node.child_by_field_name("type")
        if t:
            name = _txt(t).split("<")[0]
            if name in import_map:
                target = _resolve_java(import_map[name], repo_root)
                if target and target != source:
                    edges.append(Edge(source, target, EdgeType.INVOKES))
            elif name not in _JAVA_BUILTINS:
                target = _resolve_same_package(name, source_dir, repo_root, source)
                if target:
                    edges.append(Edge(source, target, EdgeType.INVOKES))
    for child in node.children:
        _walk_invokes(child, import_map, source_dir, repo_root, source, edges)


class JavaPlugin(LanguagePlugin):
    extensions = [".java"]

    def extract_edges(self, file_path: str, source: str) -> list[Edge]:
        try:
            return self._extract(file_path, source)
        except Exception as exc:
            log.warning("JavaPlugin: parse error in %s: %s", file_path, exc)
            return []

    def _extract(self, file_path: str, source: str) -> list[Edge]:
        root_dir = _repo_root(file_path, source)
        if root_dir is None:
            return []

        source_dir = os.path.dirname(os.path.abspath(file_path))

        with open(file_path, "rb") as fh:
            code = fh.read()

        parser = tree_sitter.Parser(_JAVA_LANG)
        tree = parser.parse(code)
        root = tree.root_node

        # Build import map: simple name → qualified name
        import_map: dict[str, str] = {}
        for node in root.children:
            if node.type == "import_declaration":
                for child in node.children:
                    if child.type in ("scoped_identifier", "identifier"):
                        qualified = _txt(child)
                        simple = qualified.split(".")[-1]
                        import_map[simple] = qualified
                        break

        edges: list[Edge] = []

        # IMPORTS edges
        for simple, qualified in import_map.items():
            target = _resolve_java(qualified, root_dir)
            if target and target != source:
                edges.append(Edge(source, target, EdgeType.IMPORTS))

        # INHERITS from extends/implements
        for node in root.children:
            if node.type == "class_declaration":
                sc = node.child_by_field_name("superclass")
                if sc:
                    for child in sc.children:
                        if child.type == "type_identifier":
                            name = _txt(child)
                            if name in import_map:
                                target = _resolve_java(import_map[name], root_dir)
                            else:
                                target = _resolve_same_package(name, source_dir, root_dir, source)
                            if target and target != source:
                                edges.append(Edge(source, target, EdgeType.INHERITS))

                ifaces = node.child_by_field_name("interfaces")
                if ifaces:
                    for tlist in ifaces.children:
                        if tlist.type == "type_list":
                            for t in tlist.children:
                                if t.type == "type_identifier":
                                    name = _txt(t)
                                    if name in import_map:
                                        target = _resolve_java(import_map[name], root_dir)
                                    else:
                                        target = _resolve_same_package(name, source_dir, root_dir, source)
                                    if target and target != source:
                                        edges.append(Edge(source, target, EdgeType.INHERITS))

        # INVOKES from object_creation_expression
        _walk_invokes(root, import_map, source_dir, root_dir, source, edges)

        # REFERENCES: same-package type uses in field/param/variable declarations
        # Collect all type_identifier nodes; resolve those in same package
        all_types: set[str] = set()
        _collect_type_identifiers(root, all_types)
        for name in all_types:
            if name in import_map:
                continue  # already handled as IMPORTS
            target = _resolve_same_package(name, source_dir, root_dir, source)
            if target:
                edges.append(Edge(source, target, EdgeType.REFERENCES))

        # Deduplicate
        seen: set[tuple] = set()
        result: list[Edge] = []
        for e in edges:
            key = (e.source, e.target, e.edge_type)
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result


default_registry.register(JavaPlugin())
