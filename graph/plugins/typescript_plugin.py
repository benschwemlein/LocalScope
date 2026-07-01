import os
import logging

import tree_sitter_typescript
import tree_sitter

from graph.edge import Edge, EdgeType
from graph.plugin_registry import LanguagePlugin, default_registry

log = logging.getLogger(__name__)

_TS_LANG = tree_sitter.Language(tree_sitter_typescript.language_typescript())
_TSX_LANG = tree_sitter.Language(tree_sitter_typescript.language_tsx())

_TS_EXTS = {".ts", ".tsx"}

# Candidate extensions when resolving a bare import path
_RESOLVE_EXTS = [".ts", ".tsx", ".js", ".jsx"]


def _resolve_ts_import(raw_path: str, source_dir: str) -> str | None:
    """Resolve a relative TS import path to a repo-relative file path."""
    if not raw_path.startswith("."):
        return None  # skip node_modules / bare module names

    base = os.path.normpath(os.path.join(source_dir, raw_path))

    # Try exact path first (in case extension was included)
    for ext in _RESOLVE_EXTS:
        candidate = base if base.endswith(ext) else base + ext
        if os.path.exists(candidate):
            return candidate
        # Also check index file
        index = os.path.join(base, f"index{ext}")
        if os.path.exists(index):
            return index
    return None


def _txt(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _find(node, node_type: str):
    if node.type == node_type:
        return node
    for child in node.children:
        r = _find(child, node_type)
        if r:
            return r
    return None


def _walk_invokes(node, import_paths: dict[str, str], source: str, edges: list[Edge]):
    if node.type == "new_expression":
        # new Foo() — constructor is the first child after "new"
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                name = _txt(child)
                if name in import_paths:
                    target = import_paths[name]
                    if target != source:
                        edges.append(Edge(source, target, EdgeType.INVOKES))
                break
    for child in node.children:
        _walk_invokes(child, import_paths, source, edges)


class TypeScriptPlugin(LanguagePlugin):
    extensions = [".ts", ".tsx"]

    def extract_edges(self, file_path: str, source: str) -> list[Edge]:
        try:
            return self._extract(file_path, source)
        except Exception as exc:
            log.warning("TypeScriptPlugin: parse error in %s: %s", file_path, exc)
            return []

    def _extract(self, file_path: str, source: str) -> list[Edge]:
        ext = os.path.splitext(file_path)[1].lower()
        lang = _TSX_LANG if ext == ".tsx" else _TS_LANG
        source_dir = os.path.dirname(os.path.abspath(file_path))

        with open(file_path, "rb") as fh:
            code = fh.read()

        parser = tree_sitter.Parser(lang)
        tree = parser.parse(code)
        root = tree.root_node

        # Build import map: symbol name → resolved absolute path → repo-relative path
        # import_paths: simple_name → repo-relative target path
        import_paths: dict[str, str] = {}
        edges: list[Edge] = []

        for node in root.children:
            stmt = node if node.type == "import_statement" else _find(node, "import_statement")
            if stmt is None:
                continue

            # Extract the module specifier (string literal)
            raw_path: str | None = None
            for child in stmt.children:
                if child.type == "string":
                    frag = _find(child, "string_fragment")
                    if frag:
                        raw_path = _txt(frag)
                    break

            if raw_path is None:
                continue

            resolved_abs = _resolve_ts_import(raw_path, source_dir)
            if resolved_abs is None:
                continue

            # Make repo-relative by stripping source_dir back to repo root
            # Use same trick as Java: derive repo root from file_path and source
            abs_file = os.path.abspath(file_path)
            norm_src = os.path.normpath(source)
            repo_root: str | None = None
            if abs_file.endswith(os.sep + norm_src):
                repo_root = abs_file[: -(len(norm_src) + 1)]
            elif abs_file.endswith(norm_src):
                repo_root = abs_file[: -len(norm_src)].rstrip(os.sep)

            if repo_root is None:
                continue

            if resolved_abs.startswith(repo_root + os.sep):
                target_rel = resolved_abs[len(repo_root) + 1:]
            else:
                continue

            if target_rel == source:
                continue

            edges.append(Edge(source, target_rel, EdgeType.IMPORTS))

            # Map imported symbol names for INHERITS/INVOKES resolution
            import_clause = _find(stmt, "import_clause")
            if import_clause:
                named = _find(import_clause, "named_imports")
                if named:
                    for spec in named.children:
                        if spec.type == "import_specifier":
                            sym = _txt(spec).split(" as ")[0].strip()
                            import_paths[sym] = target_rel
                default_id = import_clause.child_by_field_name("name")
                if default_id:
                    import_paths[_txt(default_id)] = target_rel

        # INHERITS from class_heritage
        def walk_classes(node):
            if node.type == "class_declaration":
                heritage = _find(node, "class_heritage")
                if heritage:
                    for clause in heritage.children:
                        if clause.type in ("extends_clause", "implements_clause"):
                            for child in clause.children:
                                if child.type in ("identifier", "type_identifier"):
                                    name = _txt(child)
                                    if name in import_paths:
                                        target = import_paths[name]
                                        if target != source:
                                            edges.append(Edge(source, target, EdgeType.INHERITS))
            for child in node.children:
                walk_classes(child)

        walk_classes(root)

        # INVOKES from new_expression
        _walk_invokes(root, import_paths, source, edges)

        # Deduplicate
        seen: set[tuple] = set()
        result: list[Edge] = []
        for e in edges:
            key = (e.source, e.target, e.edge_type)
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result


default_registry.register(TypeScriptPlugin())
