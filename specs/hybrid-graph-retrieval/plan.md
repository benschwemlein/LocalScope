# Implementation Plan: Hybrid Graph + Vector Retrieval

**Branch**: `hybrid-graph-retrieval` | **Date**: 2026-06-30 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/hybrid-graph-retrieval/spec.md`

## Summary

Add a structural code knowledge graph alongside the existing ChromaDB vector index.
At index time, tree-sitter parses each supported file via a language plugin and extracts
typed structural edges (IMPORTS, INVOKES, INHERITS, REFERENCES, CONTAINS) into a networkx
DiGraph persisted to disk. At query time, ChromaDB seeds are used as entry points for
Dijkstra expansion across the graph; the KGCompass formula (`S(f) = β^l(f) · (α·cos_norm +
(1−α)·lev)`) re-ranks all candidates — seeds plus graph-discovered files — into a single
fused result list. The feature ships behind `GRAPH_ENABLED=false` until test_10 validates
the success criteria (mean R@10 ≥ 0.70, fine_calculation_strategy ≥ 0.80). Academic basis:
KGCompass (arXiv 2503.21710), RepoGraph (arXiv 2410.14684), CodeCompass (arXiv 2602.20048).

## Technical Context

**Language/Version**: Python 3.14. No new runtime dependencies beyond tree-sitter grammars.

**Primary Dependencies**:
- `networkx` — in-memory DiGraph, Dijkstra, node-link JSON serialization
- `tree-sitter` + `tree-sitter-java`, `tree-sitter-typescript`, `tree-sitter-html` — AST parsing per language
- `python-Levenshtein` — lexical similarity for the KGCompass lev() term
- Existing: ChromaDB, mxbai-embed-large via Ollama, incremental_indexer.py, query_engine.py

**Storage**: networkx DiGraph serialized to `{index_dir}/graph.json` (node-link format).
No external database. Co-located with the ChromaDB index directory.

**Testing**: Existing pytest suite (test_04, test_09) as regression baseline.
New test_10_graph_retrieval.py as the primary validation gate. Unit tests per plugin.

**Target Platform**: Local developer laptop (macOS/Linux). Same runtime as existing LocalScope.

**Project Type**: Python library — no UI changes in this feature. Graph is a retrieval backend.

**Performance Goals**: Full graph build ≤ 60s for library-catalog-app (SC-007).
Incremental update ≤ 5s per changed file (SC-008). Query fusion overhead ≤ 500ms added latency.

**Constraints**:
- `GRAPH_ENABLED` defaults to `false` until test_10 passes (FR-017)
- No regressions on the 9 currently-passing test_04 recall floor cases (SC-006)
- No external graph database — networkx only (operational simplicity for a local tool)
- KGCompass α/β defaults (0.3/0.6) are Python/SWE-bench tuned; test_10 validates and tunes for Java/Angular

**Scale/Scope**: library-catalog-app corpus (~600 files, estimated 5k-15k edges). networkx
is sufficient at this scale. Function-level granularity (RepoGraph-style) is a future upgrade.

## Project Structure

### Documentation (this feature)

```
specs/hybrid-graph-retrieval/
├── plan.md        # This file
└── spec.md        # Feature specification
```

### Source Code

```
LocalScope/
  graph/
    __init__.py
    edge.py              # Edge dataclass + EdgeType enum
    graph_store.py       # networkx DiGraph wrapper — build, persist, load, incremental update
    graph_builder.py     # Orchestrates plugin registry + graph_store per file
    plugin_registry.py   # LanguagePlugin ABC + PluginRegistry
    plugins/
      __init__.py
      java_plugin.py     # tree-sitter-java: IMPORTS, INHERITS, INVOKES, CONTAINS
      typescript_plugin.py  # tree-sitter-typescript: IMPORTS, INHERITS, INVOKES, CONTAINS
      html_plugin.py     # tree-sitter-html: REFERENCES (Angular selectors/directives)
    fusion.py            # KGCompass scoring + Dijkstra expansion
  indexing/
    incremental_indexer.py   # MODIFIED — calls graph_builder after embedding pass
  querying/
    query_engine.py          # MODIFIED — calls fusion.py after ChromaDB retrieval
  config.py                  # MODIFIED — GRAPH_ENABLED, GRAPH_ALPHA, GRAPH_BETA
  test_suite/
    test_10_graph_retrieval.py   # NEW — graph retrieval evaluation + tuning loop
```

## Tasks

Each task ships as its own commit. Tasks 1–6 are pure additions with no modifications to
existing files. Tasks 7–9 modify existing files. Task 9 is the validation gate.

---

### Task 1 — Edge schema and data model
**Commit**: `Add graph edge schema: EdgeType enum and Edge dataclass`
**Depends on**: nothing

- Create `graph/edge.py`
  - `EdgeType` enum: `IMPORTS`, `INVOKES`, `INHERITS`, `REFERENCES`, `CONTAINS`
  - `Edge` dataclass: `source: str`, `target: str`, `edge_type: EdgeType`, `weight: float = 1.0`
- Create `graph/__init__.py` (empty, package marker)
- Unit tests: Edge construction, EdgeType values, default weight

**Satisfies**: FR-015, FR-016

---

### Task 2 — Language plugin interface and registry
**Commit**: `Add language plugin registry and LanguagePlugin interface`
**Depends on**: Task 1

- Create `graph/plugin_registry.py`
  - `LanguagePlugin` abstract base class
    - `extensions: list[str]` (class attribute)
    - `extract_edges(file_path: str, source: str) -> list[Edge]` (abstract method)
  - `PluginRegistry` class: `register(plugin)`, `get(ext: str) -> LanguagePlugin | None`
  - Module-level `default_registry` instance
- Create `graph/plugins/__init__.py` (empty)
- Unit tests: register/lookup by extension, None for unregistered, multiple extensions per plugin

**Satisfies**: FR-011, FR-013, FR-014

---

### Task 3 — Java language plugin
**Commit**: `Add Java graph plugin: imports, inheritance, invocations, containment`
**Depends on**: Task 2

- Create `graph/plugins/java_plugin.py`
  - Uses `tree_sitter_java` grammar
  - Extracts:
    - `IMPORTS` — `import_declaration` nodes → source file to imported class file
    - `INHERITS` — `superclass` and `super_interfaces` in class declarations
    - `INVOKES` — `method_invocation` nodes where target is resolvable to a repo file
    - `CONTAINS` — class → method containment hierarchy
  - Registers for `.java` in `default_registry` on module load
  - Graceful degradation: syntax errors log a warning and return `[]`
- Unit tests against sample Java snippets covering each edge type; syntax error case

**Satisfies**: FR-012, FR-015

---

### Task 4 — TypeScript language plugin
**Commit**: `Add TypeScript graph plugin: ES module imports, inheritance, calls`
**Depends on**: Task 2

- Create `graph/plugins/typescript_plugin.py`
  - Uses `tree_sitter_typescript` grammar (`.ts` and `.tsx`)
  - Extracts:
    - `IMPORTS` — `import_statement` nodes (static imports; dynamic imports best-effort)
    - `INHERITS` — `class_heritage` extends/implements clauses
    - `INVOKES` — `call_expression` nodes where callee resolves to a repo file
    - `CONTAINS` — class → method containment
  - Registers for `.ts`, `.tsx` in `default_registry`
- Unit tests against sample TypeScript snippets; `.tsx` extension registration

**Satisfies**: FR-012, FR-015

---

### Task 5 — HTML language plugin
**Commit**: `Add HTML graph plugin: Angular component and directive references`
**Depends on**: Task 2

- Create `graph/plugins/html_plugin.py`
  - Uses `tree_sitter_html` grammar
  - Extracts `REFERENCES` edges for Angular component selectors and `*ngDirective` attributes
  - Edge weight set to `0.7` (lower confidence — inferred from selector matching, not direct symbol resolution)
  - Registers for `.html` in `default_registry`
- Unit tests against sample Angular template HTML

**Satisfies**: FR-012, FR-015, FR-016

---

### Task 6 — Graph store and builder
**Commit**: `Add networkx graph store with incremental build and disk persistence`
**Depends on**: Tasks 1–5

- Create `graph/graph_store.py`
  - Wraps `networkx.DiGraph`
  - `add_edges(edges: list[Edge])` — adds nodes and edges to the graph
  - `remove_file(path: str)` — removes all edges where source == path
  - `shortest_path_length(source: str, target: str) -> float` — Dijkstra via `nx.single_source_dijkstra_path_length`; returns `inf` if unreachable
  - `save(path: str)` — serializes to JSON via `nx.node_link_data`
  - `load(path: str) -> GraphStore` — deserializes; raises `GraphLoadError` on corrupt file
- Create `graph/graph_builder.py`
  - Iterates repo files, resolves plugin by extension from `default_registry`
  - Tracks file content hashes; only re-extracts files whose hash changed since last build
  - Calls `graph_store.remove_file` then re-adds edges for modified files
  - Calls `graph_store.remove_file` for deleted files
- Unit tests: incremental update (only changed files processed), delete propagation, persist/load round-trip, corrupt file fallback

**Satisfies**: FR-006, FR-007, FR-008, FR-009, FR-010

---

### Task 7 — Wire graph builder into incremental indexer
**Commit**: `Wire graph builder into indexer alongside vector embedding pass`
**Depends on**: Task 6

- Modify `indexing/incremental_indexer.py`
  - After embedding pass completes, call `graph_builder.build_incremental(repo_root, graph_path, changed_files)`
  - `graph_path` = `{index_dir}/graph.json`
  - Skip if `config.GRAPH_ENABLED` is false
- Modify `config.py`
  - Add `GRAPH_ENABLED: bool = env("LCQ_GRAPH_ENABLED", "false").lower() == "true"`
  - Add `GRAPH_ALPHA: float = float(env("LCQ_GRAPH_ALPHA", "0.3"))`
  - Add `GRAPH_BETA: float = float(env("LCQ_GRAPH_BETA", "0.6"))`

**Satisfies**: FR-004, FR-017, FR-018

---

### Task 8 — KGCompass fusion layer
**Commit**: `Add KGCompass hybrid fusion: Dijkstra expansion and re-ranking`
**Depends on**: Tasks 6, 7

- Create `graph/fusion.py`
  - `expand_and_rerank(seeds, query_text, graph_store, alpha, beta, max_hops=3) -> list[RankedResult]`
  - For each seed: run Dijkstra to collect all nodes within `max_hops`
  - For each candidate (seeds + graph-discovered):
    - `cos_norm` from ChromaDB distance score (passed in with seeds)
    - `lev` from `Levenshtein.ratio(query_text, file_content_snippet)`
    - `l` = shortest path length from nearest seed
    - `score = beta ** l * (alpha * cos_norm + (1 - alpha) * lev)`
  - Skip candidates with no cosine score (not in vector index)
  - Return sorted by score descending
- Modify `querying/query_engine.py`
  - After ChromaDB retrieval, if `GRAPH_ENABLED` and graph file exists, call `fusion.expand_and_rerank`
  - Fall back to vector-only if graph unavailable or raises `GraphLoadError`
- Unit tests: formula correctness, seed-only case (no graph expansion), fallback on missing graph

**Satisfies**: FR-001, FR-002, FR-003, FR-004, FR-005

---

### Task 9 — Test suite: graph retrieval evaluation
**Commit**: `Add test_10: graph retrieval evaluation and regression guard`
**Depends on**: Tasks 7, 8

- Create `test_suite/test_10_graph_retrieval.py`
  - Session fixture enables `GRAPH_ENABLED=true` for the test run
  - Runs same 10 ground truth queries as test_04 with graph enabled
  - Reports P@5, R@10, MRR with delta column vs test_04 vector-only baseline
  - Hard assertions:
    - `fine_calculation_strategy` R@10 ≥ 0.80 (SC-001)
    - `loan_eligibility_chain` R@10 ≥ 0.80 (SC-002)
    - Mean R@10 ≥ 0.70 (SC-003)
    - Mean P@5 ≥ 0.50 (SC-004)
    - Mean MRR ≥ 0.88 (SC-005)
  - Regression guard: all 9 cases currently passing test_04 recall floor must still pass (SC-006)
  - Timing assertions: graph build ≤ 60s, incremental update ≤ 5s (SC-007, SC-008)
  - α/β tuning loop: optional parametrized test that sweeps α ∈ {0.1, 0.3, 0.5, 0.7} and
    β ∈ {0.4, 0.6, 0.8} and reports best combination by mean R@10

**Satisfies**: SC-001 through SC-010

## Complexity Tracking

No constitution violations — no entries.
