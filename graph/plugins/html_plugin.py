import os
import logging

import tree_sitter_html
import tree_sitter

from graph.edge import Edge, EdgeType
from graph.plugin_registry import LanguagePlugin, default_registry

log = logging.getLogger(__name__)

_HTML_LANG = tree_sitter.Language(tree_sitter_html.language())

_COMPONENT_EXTS = [".ts", ".tsx"]


def _selector_to_file(selector: str, source_dir: str) -> str | None:
    """Convert an Angular kebab-case selector to a component file path candidate."""
    # app-book-list → book-list.component.ts or book-list.component.tsx
    name = selector
    if name.startswith("app-"):
        name = name[4:]
    candidates = [
        f"{name}.component",
        f"{name}",
    ]
    for cand in candidates:
        for ext in _COMPONENT_EXTS:
            for search_dir in [source_dir] + _parent_dirs(source_dir, 4):
                path = os.path.join(search_dir, cand + ext)
                if os.path.exists(path):
                    return path
    return None


def _parent_dirs(start: str, depth: int) -> list[str]:
    dirs = []
    current = start
    for _ in range(depth):
        parent = os.path.dirname(current)
        if parent == current:
            break
        dirs.append(parent)
        current = parent
    return dirs


def _txt(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _is_angular_component(tag: str) -> bool:
    return "-" in tag and not tag.startswith("ng-")


class HtmlPlugin(LanguagePlugin):
    extensions = [".html"]

    def extract_edges(self, file_path: str, source: str) -> list[Edge]:
        try:
            return self._extract(file_path, source)
        except Exception as exc:
            log.warning("HtmlPlugin: parse error in %s: %s", file_path, exc)
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

        with open(file_path, "rb") as fh:
            code = fh.read()

        parser = tree_sitter.Parser(_HTML_LANG)
        tree = parser.parse(code)
        root = tree.root_node

        source_dir = os.path.dirname(abs_file)
        edges: list[Edge] = []
        seen: set[str] = set()

        def walk(node):
            if node.type == "start_tag":
                tag_name_node = node.child_by_field_name("name")
                if tag_name_node is None:
                    for c in node.children:
                        if c.type == "tag_name":
                            tag_name_node = c
                            break
                if tag_name_node:
                    tag = _txt(tag_name_node)
                    if _is_angular_component(tag) and tag not in seen:
                        seen.add(tag)
                        resolved = _selector_to_file(tag, source_dir)
                        if resolved and resolved.startswith(repo_root + os.sep):
                            target_rel = resolved[len(repo_root) + 1:]
                            if target_rel != source:
                                edges.append(
                                    Edge(source, target_rel, EdgeType.REFERENCES, weight=0.7)
                                )
            for child in node.children:
                walk(child)

        walk(root)
        return edges


default_registry.register(HtmlPlugin())
