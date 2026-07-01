import os
import logging

import tree_sitter_java as tsjava
import tree_sitter

from graph.edge import Edge, EdgeType
from graph.plugin_registry import LanguagePlugin, default_registry

log = logging.getLogger(__name__)

_JAVA_LANG = tree_sitter.Language(tsjava.language())

_JAVA_SRC_ROOTS = ["src/main/java", "src/test/java", "src"]


def _repo_root(file_path: str, source: str) -> str | None:
    abs_file = os.path.abspath(file_path)
    norm_src = os.path.normpath(source)
    if abs_file.endswith(os.sep + norm_src):
        return abs_file[: -(len(norm_src) + 1)]
    if abs_file.endswith(norm_src):
        return abs_file[: -len(norm_src)].rstrip(os.sep)
    return None


def _resolve_java(qualified: str, repo_root: str) -> str | None:
    rel = qualified.replace(".", "/") + ".java"
    for root in _JAVA_SRC_ROOTS:
        candidate = os.path.join(repo_root, root, rel)
        if os.path.exists(candidate):
            return os.path.join(root, rel)
    return None


def _txt(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _walk_invokes(node, import_map: dict, repo_root: str, source: str, edges: list[Edge]):
    if node.type == "object_creation_expression":
        t = node.child_by_field_name("type")
        if t:
            name = _txt(t).split("<")[0]
            if name in import_map:
                target = _resolve_java(import_map[name], repo_root)
                if target and target != source:
                    edges.append(Edge(source, target, EdgeType.INVOKES))
    for child in node.children:
        _walk_invokes(child, import_map, repo_root, source, edges)


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

        # Walk class declarations for INHERITS
        for node in root.children:
            if node.type == "class_declaration":
                sc = node.child_by_field_name("superclass")
                if sc:
                    for child in sc.children:
                        if child.type == "type_identifier":
                            name = _txt(child)
                            if name in import_map:
                                target = _resolve_java(import_map[name], root_dir)
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
                                        if target and target != source:
                                            edges.append(Edge(source, target, EdgeType.INHERITS))

        # INVOKES edges from object_creation_expression
        _walk_invokes(root, import_map, root_dir, source, edges)

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
