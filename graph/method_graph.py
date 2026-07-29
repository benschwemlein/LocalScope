"""
Method-level graph extraction for Java.

The file-level graph measured at roughly zero value: stripping the 12
cross-language REST edges left the other ~1000 edges contributing +0.001,
and slightly negative on Java. The diagnosis is precision. A file-level
IMPORTS edge says "this file mentions that file", and a Java file imports
about fifteen things, so expanding along one drags in whole files of
non-evidence. RepoGraph, the strongest peer-reviewed result in this area,
is line-level, where a def-to-ref edge says "this exact function is used
here". This module is the step toward that granularity.

Node identity is "relative/path/File.java::methodName". Overloads collapse
onto one node deliberately: distinguishing them needs full type resolution
of every argument, and retrieval works at chunk granularity where overloads
usually share a chunk anyway.

Call resolution is best-effort and explicitly approximate:

  receiver.method(...)  ->  resolve `receiver` to a type via field, parameter,
                            or local-variable declarations in the enclosing
                            file, then map the type to a file via imports or
                            the same package, then attach to that file's
                            method of the matching name.
  this.method(...)      ->  same file.
  method(...)           ->  same file.

What it deliberately does NOT do: interface dispatch (a call through an
interface attaches to the interface, not implementations), generics, static
imports, lambdas, or inherited methods declared in a superclass. Each would
need a real type checker. The extractor reports its resolution rate so the
cost of that approximation stays visible rather than being assumed.
"""

import os
import re
import logging
from dataclasses import dataclass, field

import tree_sitter
import tree_sitter_java as tsjava

log = logging.getLogger(__name__)

_JAVA_LANG = tree_sitter.Language(tsjava.language())

_JAVA_BUILTIN_METHODS = frozenset({
    "toString", "equals", "hashCode", "clone", "getClass", "notify", "notifyAll",
    "wait", "format", "valueOf", "of", "get", "set", "add", "put", "size",
    "isEmpty", "contains", "stream", "map", "filter", "collect", "forEach",
    "builder", "build", "length", "charAt", "substring", "split", "trim",
    "info", "debug", "warn", "error", "trace", "println", "printf",
})


@dataclass
class MethodNode:
    file: str            # repo-relative path
    name: str            # method name
    start_line: int
    end_line: int

    @property
    def id(self) -> str:
        return f"{self.file}::{self.name}"


@dataclass
class MethodExtraction:
    methods: list[MethodNode] = field(default_factory=list)
    # (caller_id, callee_id)
    calls: list[tuple[str, str]] = field(default_factory=list)
    attempted_calls: int = 0
    resolved_calls: int = 0

    @property
    def resolution_rate(self) -> float:
        return self.resolved_calls / self.attempted_calls if self.attempted_calls else 0.0


def _txt(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _walk(node, node_type: str):
    if node.type == node_type:
        yield node
    for child in node.children:
        yield from _walk(child, node_type)


def _enclosing_method(methods: list[MethodNode], line: int) -> MethodNode | None:
    """Innermost method containing a line; methods are small so a scan is fine."""
    best = None
    for m in methods:
        if m.start_line <= line <= m.end_line:
            if best is None or (m.end_line - m.start_line) < (best.end_line - best.start_line):
                best = m
    return best


class JavaMethodExtractor:
    """Extracts method nodes and best-effort call edges from one Java file."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self._parser = tree_sitter.Parser(_JAVA_LANG)
        # simple type name -> repo-relative file, per file being processed
        self._type_index: dict[str, str] = {}

    def build_type_index(self, files: list[str]) -> None:
        """Map simple class names to files once, so call resolution can look up
        a receiver's declared type without re-walking the tree."""
        for rel in files:
            base = os.path.splitext(os.path.basename(rel))[0]
            # last writer wins only when the name is genuinely ambiguous; that
            # ambiguity is reported rather than silently resolved
            self._type_index.setdefault(base, rel)

    def extract(self, rel_path: str) -> MethodExtraction:
        full = os.path.join(self.repo_root, rel_path)
        try:
            code = open(full, "rb").read()
        except OSError:
            return MethodExtraction()

        tree = self._parser.parse(code)
        root = tree.root_node
        out = MethodExtraction()

        # --- method declarations ---
        for node in _walk(root, "method_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            out.methods.append(MethodNode(
                file=rel_path,
                name=_txt(name_node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))
        for node in _walk(root, "constructor_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            out.methods.append(MethodNode(
                file=rel_path,
                name=_txt(name_node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

        if not out.methods:
            return out

        # --- receiver name -> declared type, from fields, params, locals ---
        receiver_types: dict[str, str] = {}
        for decl_type in ("field_declaration", "formal_parameter", "local_variable_declaration"):
            for node in _walk(root, decl_type):
                tnode = node.child_by_field_name("type")
                if tnode is None:
                    continue
                type_name = _txt(tnode).split("<")[0].strip()
                if not type_name or not type_name[0].isupper():
                    continue
                if decl_type == "formal_parameter":
                    nnode = node.child_by_field_name("name")
                    if nnode is not None:
                        receiver_types[_txt(nnode)] = type_name
                else:
                    for d in _walk(node, "variable_declarator"):
                        nnode = d.child_by_field_name("name")
                        if nnode is not None:
                            receiver_types[_txt(nnode)] = type_name

        # --- call sites ---
        for node in _walk(root, "method_invocation"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            callee_name = _txt(name_node)
            if callee_name in _JAVA_BUILTIN_METHODS:
                continue

            line = node.start_point[0] + 1
            caller = _enclosing_method(out.methods, line)
            if caller is None:
                continue

            out.attempted_calls += 1

            obj_node = node.child_by_field_name("object")
            if obj_node is None or _txt(obj_node) == "this":
                target_file = rel_path                      # unqualified or this.x()
            else:
                obj = _txt(obj_node)
                type_name = receiver_types.get(obj)
                if type_name is None and obj and obj[0].isupper():
                    type_name = obj                          # static call: Foo.bar()
                if type_name is None:
                    continue
                target_file = self._type_index.get(type_name)
                if target_file is None:
                    continue

            out.resolved_calls += 1
            out.calls.append((caller.id, f"{target_file}::{callee_name}"))

        return out


def extract_method_graph(
    repo_root: str,
    excluded_dirs: set[str] | None = None,
    log_fn=None,
) -> tuple[list[MethodNode], list[tuple[str, str]], float]:
    """
    Walk the repo and extract method nodes plus resolved call edges.

    Returns (methods, call_edges, resolution_rate). The resolution rate is
    reported because the approximations above are the whole risk here: a low
    rate means the call graph is mostly guesswork and should not be trusted
    for traversal.
    """
    if log_fn is None:
        log_fn = log.info
    excluded = excluded_dirs or {
        ".git", ".idea", ".vscode", "node_modules", "build", "dist", "out",
        "target", ".gradle", ".venv", "venv", "__pycache__",
    }

    java_files: list[str] = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in excluded]
        for f in files:
            if f.endswith(".java"):
                java_files.append(os.path.relpath(os.path.join(root, f), repo_root))

    extractor = JavaMethodExtractor(repo_root)
    extractor.build_type_index(java_files)

    all_methods: list[MethodNode] = []
    all_calls: list[tuple[str, str]] = []
    attempted = resolved = 0

    for rel in java_files:
        res = extractor.extract(rel)
        all_methods.extend(res.methods)
        all_calls.extend(res.calls)
        attempted += res.attempted_calls
        resolved += res.resolved_calls

    # Drop calls whose target method was never declared anywhere: usually an
    # inherited or interface method, which this extractor cannot see.
    declared = {m.id for m in all_methods}
    kept = [(c, t) for c, t in all_calls if t in declared]

    rate = resolved / attempted if attempted else 0.0
    log_fn(
        f"[method_graph] {len(java_files)} files, {len(all_methods)} methods, "
        f"{attempted} call sites, {resolved} resolved ({rate:.0%}), "
        f"{len(kept)} edges to declared methods"
    )
    return all_methods, kept, rate
