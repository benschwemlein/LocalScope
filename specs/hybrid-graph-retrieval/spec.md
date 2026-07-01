# Feature Specification: Hybrid Graph + Vector Retrieval

**Feature Branch**: `hybrid-graph-retrieval`

**Created**: 2026-06-30

**Status**: Draft

**Input**: Extend LocalScope's retrieval pipeline with a structural code knowledge graph built alongside the existing ChromaDB vector index, fused using the KGCompass scoring formula. Vector search handles semantic recall; graph traversal handles structural precision and multi-hop discovery. Language-specific edge extraction uses a plugin architecture so new languages can be added without touching core graph logic.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Retrieve structurally-connected files the embedder misses (Priority: P1)

A developer queries LocalScope about a pattern or flow whose implementation is spread across multiple files that are structurally connected (via calls, inheritance, or containment) but not semantically similar to the query text. The hybrid retrieval finds the entry-point file via vector search and then traverses the graph to surface all connected files, returning a complete and ranked result set.

**Why this priority**: This is the core failure mode the feature exists to fix. Benchmark results (test_04, test_09) show mean R@10=0.56 with the best embedding model. The `fine_calculation_strategy` query gets R@10=0.20 across all four embedding models — the concrete strategy classes are structurally reachable from `OverdueFineContext` in one hop but are not semantically similar to the query. Graph traversal fixes this directly. Everything else (incremental build, plugin architecture, config) exists to support this capability in production.

**Independent Test**: Run the `fine_calculation_strategy` ground truth query with graph enabled. Confirm that `StandardFineStrategy.java`, `PremiumFineStrategy.java`, and `StudentFineStrategy.java` appear in the top-10 results alongside `OverdueFineContext.java`. R@10 must be ≥ 0.80. No manual inspection of the graph required — test_10 asserts this automatically.

**Acceptance Scenarios**:

1. **Given** graph retrieval is enabled, **When** a query is run whose answer files are reachable from a vector-seed node via ≤3 graph hops, **Then** those files appear in the top-10 results ranked higher than unrelated files.
2. **Given** the vector search returns a seed file with structural edges in the graph, **When** fusion runs, **Then** Dijkstra expands from that seed and candidates discovered via graph traversal receive a score incorporating their path distance decay.
3. **Given** a query where vector search already returns perfect recall (e.g. `recommendation_engine` R@10=1.00), **When** graph fusion runs, **Then** the result set is not degraded — no previously-returned file drops out of top-10.
4. **Given** graph retrieval is disabled via config (`GRAPH_ENABLED=false`), **When** a query runs, **Then** results are identical to the vector-only baseline.

---

### User Story 2 — Graph is built incrementally alongside the vector index (Priority: P2)

When a developer re-indexes a repository after editing files, the graph is updated only for changed files — not rebuilt from scratch. The full index time for a fresh repo and the incremental update time after a typical edit are both within acceptable bounds for an interactive tool.

**Why this priority**: The academic literature (GraphRAG) explicitly calls out high re-indexing cost as a production pitfall. LocalScope's incremental indexer is already a core feature; the graph must match that contract or it becomes unusable in practice. This is a prerequisite for shipping — a graph that forces a full rebuild on every file change is not a usable local tool.

**Independent Test**: Index the library-catalog-app corpus from scratch and record full build time. Edit one Java file. Re-index and confirm only that file's edges are re-extracted (log output shows one file processed, not 397). Measure total incremental time.

**Acceptance Scenarios**:

1. **Given** a fresh repository with no existing graph, **When** a full index is triggered, **Then** the graph is built for all supported file types and persisted to disk alongside the ChromaDB index within 60 seconds for the library-catalog-app corpus.
2. **Given** an existing graph index, **When** a single file is modified and re-indexed, **Then** only that file's edges are re-extracted and the graph is updated; unchanged files are not re-processed.
3. **Given** a file is deleted from the repository, **When** re-indexing runs, **Then** all edges sourced from that file are removed from the graph.
4. **Given** a new file is added to the repository, **When** re-indexing runs, **Then** edges for that file are extracted and added to the graph without affecting existing edges.
5. **Given** re-indexing completes, **When** the graph is loaded for the next query, **Then** it reflects the current state of the repository with no stale edges from deleted or modified files.

---

### User Story 3 — Language plugins extract edges for Java, TypeScript, and HTML (Priority: P2)

Each supported language has its own plugin that extracts structural edges using tree-sitter. Plugins register themselves for specific file extensions. Adding a new language in the future requires only writing a new plugin — no changes to the graph builder, graph store, or fusion layer.

**Why this priority**: The library-catalog-app corpus has 397 Java files, 143 TypeScript files, and Angular HTML templates — all three must be covered for the graph to provide meaningful retrieval on the full corpus. The plugin architecture is what makes this maintainable; hardcoding per-language logic would block future language additions.

**Independent Test**: Index the library-catalog-app. Inspect the graph and confirm: (a) Java import and inheritance edges exist between known classes (e.g. `StandardFineStrategy` → `FineCalculationStrategy`), (b) TypeScript import edges exist between Angular component files, (c) HTML template edges reference their component classes. Run the plugin registry lookup for `.java`, `.ts`, `.html` and confirm each returns the correct plugin. Run the same lookup for `.css` and confirm it returns None.

**Acceptance Scenarios**:

1. **Given** a `.java` file is indexed, **When** the Java plugin runs, **Then** it extracts IMPORTS, INHERITS, INVOKES, and CONTAINS edges using the tree-sitter-java grammar and returns them as typed Edge objects.
2. **Given** a `.ts` or `.tsx` file is indexed, **When** the TypeScript plugin runs, **Then** it extracts IMPORTS, INHERITS, INVOKES, and CONTAINS edges using the tree-sitter-typescript grammar.
3. **Given** a `.html` file is indexed, **When** the HTML plugin runs, **Then** it extracts REFERENCES edges for Angular component selectors and directive attributes.
4. **Given** a `.css` or `.scss` file is encountered, **When** the plugin registry is queried, **Then** it returns None and the file is skipped for graph extraction (but still embedded by the vector indexer).
5. **Given** a new language plugin is written and registered, **When** a file with its extension is indexed, **Then** the graph builder calls that plugin with no changes to graph_builder.py, graph_store.py, or fusion.py.

---

### User Story 4 — α and β fusion weights are tunable (Priority: P3)

The KGCompass scoring formula uses two weights — α (embedding vs lexical balance) and β (path distance decay) — that are published for Python/SWE-bench and need tuning for Java/Angular. A developer can adjust these values via config without code changes and measure the effect on retrieval metrics using the existing test suite.

**Why this priority**: The hybrid formula only delivers its potential if the weights are tuned to the corpus. The published defaults (α=0.3, β=0.6) are a starting point, not a guarantee. Exposing them as config is low effort and necessary for the benchmarking loop that validates the feature.

**Independent Test**: Set `GRAPH_ALPHA=0.5` and `GRAPH_BETA=0.8` via environment variables. Run test_10 and confirm the scores differ from the defaults, proving the config values are read and applied. Set them back to defaults and confirm scores match the baseline.

**Acceptance Scenarios**:

1. **Given** `LCQ_GRAPH_ALPHA` and `LCQ_GRAPH_BETA` are set as environment variables, **When** the query engine runs, **Then** the fusion layer uses those values rather than the hardcoded defaults.
2. **Given** no environment variables are set, **When** the query engine runs, **Then** α defaults to 0.3 and β defaults to 0.6.
3. **Given** different α/β values are tested against test_10, **When** results are compared, **Then** the metrics differ in a way that reflects the weight change (tuning has a measurable effect).

---

## Edge Cases

- A Java file with a syntax error that tree-sitter cannot parse: the plugin logs a warning and returns an empty edge list for that file; the rest of the graph builds normally.
- A TypeScript file that uses dynamic imports (`import()`): treated as a best-effort IMPORTS edge if the target can be statically resolved; skipped silently if not.
- A circular import (A imports B, B imports A): the graph stores both edges normally; Dijkstra handles cycles via visited-node tracking.
- A query whose vector search returns zero results above the similarity threshold: graph expansion has no seeds; falls back to vector-only results (empty if vector also returned nothing).
- A file referenced as a graph edge target that does not exist in the index (e.g. an external library): the edge is stored but the target node has no vector embedding; it is excluded from re-ranked results since no cosine score can be computed.
- The graph file on disk is corrupted or from an incompatible version: graph load fails gracefully, logs a warning, and falls back to vector-only mode; does not crash the query engine.
- A repository with no supported files (all `.css`, `.md`, images): graph is empty; system falls back to vector-only without error.
- α=0 (pure lexical) or α=1 (pure embedding): both are valid edge cases the formula handles arithmetically; no divide-by-zero or special-casing required.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Hybrid retrieval

- **FR-001**: The system MUST fuse vector search results with graph traversal results using the KGCompass scoring formula: `S(f) = β^l(f) · (α · cos_norm(e_query, e_f) + (1−α) · lev(query, f))`.
- **FR-002**: The system MUST use Dijkstra shortest-path to compute `l(f)` — the path distance from the nearest vector-seed node to each graph-discovered candidate.
- **FR-003**: The system MUST NOT return graph-discovered candidates that have no vector embedding (no cosine score available).
- **FR-004**: The system MUST fall back to vector-only results when `GRAPH_ENABLED=false` or when the graph index is unavailable, without error.
- **FR-005**: The system MUST NOT regress mean R@10 on queries that already achieve R@10=1.00 in vector-only mode.

#### Graph construction

- **FR-006**: The system MUST build the graph incrementally — only files whose content hash has changed since the last index are re-processed.
- **FR-007**: The system MUST persist the graph to disk as a JSON file (networkx node-link format) co-located with the ChromaDB index directory.
- **FR-008**: The system MUST remove all edges sourced from a file when that file is deleted or modified, before adding new edges for modified files.
- **FR-009**: Graph construction MUST complete within 60 seconds for the library-catalog-app corpus (397 Java + 143 TypeScript + HTML files) on a full rebuild.
- **FR-010**: Incremental graph update for a single changed file MUST complete within 5 seconds.

#### Language plugin architecture

- **FR-011**: The system MUST use a plugin registry pattern where each plugin declares the file extensions it handles and implements `extract_edges(file_path: str, source: str) -> list[Edge]`.
- **FR-012**: The system MUST ship plugins for Java (`.java`), TypeScript (`.ts`, `.tsx`), and HTML (`.html`) at launch.
- **FR-013**: The system MUST skip files with no registered plugin silently — no error, no warning for known non-code extensions (`.css`, `.scss`, `.md`, `.json`).
- **FR-014**: Adding a new language plugin MUST require no changes to `graph_builder.py`, `graph_store.py`, or `fusion.py`.

#### Edge types

- **FR-015**: Each plugin MUST express edges using the standard EdgeType enum: `IMPORTS`, `INVOKES`, `INHERITS`, `REFERENCES`, `CONTAINS`.
- **FR-016**: Edge weight MUST default to 1.0 and MAY be set to values < 1.0 by plugins for lower-confidence edges (e.g. HTML template inferences).

#### Configuration

- **FR-017**: `GRAPH_ENABLED` MUST default to `false`; the graph layer is opt-in until validated by test_10.
- **FR-018**: `GRAPH_ALPHA` (default 0.3) and `GRAPH_BETA` (default 0.6) MUST be readable from environment variables `LCQ_GRAPH_ALPHA` and `LCQ_GRAPH_BETA`.

### Key Entities

- **Edge**: A directed structural relationship between two files. Fields: `source` (file path), `target` (file path), `edge_type` (EdgeType enum), `weight` (float, default 1.0).
- **EdgeType**: Enum of standard relationship types: `IMPORTS`, `INVOKES`, `INHERITS`, `REFERENCES`, `CONTAINS`.
- **GraphStore**: The networkx DiGraph wrapper. Owns persistence (load/save JSON), incremental update (add/remove edges by file), and shortest-path queries (Dijkstra).
- **LanguagePlugin**: Abstract base class. Declares `extensions: list[str]` and `extract_edges(file_path, source) -> list[Edge]`. Registered in the PluginRegistry at module load.
- **PluginRegistry**: Maps file extensions to plugin instances. Resolved at index time per file. Returns None for unregistered extensions.
- **FusionResult**: A re-ranked retrieval result combining vector cosine score, lexical similarity, and graph path distance via the KGCompass formula.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `fine_calculation_strategy` R@10 ≥ 0.80 with graph enabled (up from 0.20 vector-only), as asserted by test_10.
- **SC-002**: `loan_eligibility_chain` R@10 ≥ 0.80 with graph enabled (up from 0.20-0.40 vector-only), as asserted by test_10.
- **SC-003**: Mean R@10 across all 10 ground truth queries ≥ 0.70 with graph enabled (up from 0.56 vector-only).
- **SC-004**: Mean P@5 ≥ 0.50 with graph enabled (up from 0.38 vector-only).
- **SC-005**: Mean MRR ≥ 0.88 with graph enabled (up from 0.83 vector-only).
- **SC-006**: All 9 ground truth queries that currently pass test_04's recall floor continue to pass with graph enabled — zero regressions.
- **SC-007**: Full graph build for library-catalog-app completes in ≤ 60 seconds on a developer laptop (measured during test_10 session setup).
- **SC-008**: Incremental graph update for a single changed file completes in ≤ 5 seconds (measured by a dedicated incremental timing test in test_10).
- **SC-009**: Plugin registry returns the correct plugin for `.java`, `.ts`, `.tsx`, `.html` and returns None for `.css` in 100% of unit test cases.
- **SC-010**: The system falls back to vector-only results without error when `GRAPH_ENABLED=false` or graph file is missing, in 100% of test cases.

---

## Assumptions

- **Graph granularity is file-level (v1)**: Nodes represent files, not functions or lines. RepoGraph's line-level granularity is the gold standard but comes with higher build and maintenance cost. File-level is the right starting point; function-level is a future upgrade if results plateau.
- **tree-sitter grammars are available via pip**: `tree-sitter-java`, `tree-sitter-typescript`, `tree-sitter-html` are installable and compatible with the project's Python 3.14 environment.
- **KGCompass defaults need tuning**: Published α=0.3 and β=0.6 are optimized for Python/SWE-bench. The Java/Angular corpus will likely require different values. test_10 provides the tuning loop.
- **Graph is disabled by default until test_10 validates it**: `GRAPH_ENABLED=false` in config until SC-001 through SC-006 are confirmed. This protects the existing passing test suite during development.
- **External library edges are not resolved**: If a Java file imports `org.springframework.cache.annotation.Cacheable`, the edge is stored but the target node has no vector embedding. These edges are useful for graph traversal within the repo but do not produce retrievable results for external symbols.
- **HTML plugin confidence is lower than Java/TypeScript**: Angular template-to-component edges are inferred from selector matching, not direct symbol resolution. Edge weight may be set < 1.0 by the HTML plugin to reflect this.
- **CSS and SCSS are excluded from graph extraction**: Stylesheets have no structural code relationships relevant to code navigation queries. They remain indexed by the vector embedder.
- **networkx is sufficient at library-catalog-app scale**: The corpus (~600 files, estimated ~5k-15k edges) fits comfortably in memory. If the tool is applied to much larger monorepos in the future, a graph database backend may be warranted.
