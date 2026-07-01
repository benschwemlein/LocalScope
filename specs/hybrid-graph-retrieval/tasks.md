# Tasks: Hybrid Graph + Vector Retrieval

**Input**: Design documents from `specs/hybrid-graph-retrieval/`

**Organization**: Tasks are grouped by user story to enable independent implementation
and testing of each story. Each task ships as its own git commit per plan.md.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no shared dependencies)
- **[Story]**: Which user story this task belongs to (US1–US4)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the `graph/` package skeleton so all subsequent tasks have a home.

- [x] T001 Create `graph/__init__.py` and `graph/plugins/__init__.py` package markers
- [x] T002 Add `networkx`, `tree-sitter`, `tree-sitter-java`, `tree-sitter-typescript`, `tree-sitter-html`, `python-Levenshtein` to project dependencies (requirements.txt or pyproject.toml)

**Checkpoint**: `graph/` and `graph/plugins/` directories exist and are importable.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Edge schema and plugin registry — shared by all three language plugins and the
graph store. No user story work can begin until these two modules exist.

**⚠️ CRITICAL**: US1, US2, US3, and US4 all depend on this phase.

- [x] T003 Create `graph/edge.py` — `EdgeType` enum (IMPORTS, INVOKES, INHERITS, REFERENCES, CONTAINS) and `Edge` dataclass (source, target, edge_type, weight=1.0) per `data-model.md`
- [x] T004 Create `graph/plugin_registry.py` — `LanguagePlugin` ABC with `extensions: list[str]` and `extract_edges(file_path, source) -> list[Edge]`; `PluginRegistry` with `register()` and `get(ext)`; module-level `default_registry` instance; per `contracts/plugin-interface.md`

**Checkpoint**: `from graph.edge import Edge, EdgeType` and `from graph.plugin_registry import default_registry` both import without error.

---

## Phase 3: User Story 1 — Retrieve structurally-connected files (Priority: P1) 🎯 MVP

**Goal**: A query whose answer files are structurally connected but not semantically similar
to the query text surfaces all connected files via graph traversal and KGCompass fusion.

**Independent Test**: Run `fine_calculation_strategy` ground truth query with graph enabled.
`StandardFineStrategy.java`, `PremiumFineStrategy.java`, `StudentFineStrategy.java`, and
`OverdueFineContext.java` must all appear in top-10. R@10 ≥ 0.80 (up from 0.20 vector-only).

### Implementation for User Story 1

- [x] T005 [P] [US1] Create `graph/plugins/java_plugin.py` — tree-sitter-java parser; extract IMPORTS (import_declaration), INHERITS (superclass/super_interfaces), INVOKES (method_invocation resolvable to repo file), CONTAINS (class→method); register `.java` in `default_registry`; return `[]` on parse error; per `contracts/plugin-interface.md`
- [x] T006 [P] [US1] Create `graph/plugins/typescript_plugin.py` — tree-sitter-typescript parser; extract IMPORTS (import_statement, static only; dynamic best-effort), INHERITS (class_heritage extends/implements), INVOKES (call_expression resolvable to repo file), CONTAINS (class→method); register `.ts`, `.tsx` in `default_registry`
- [x] T007 [P] [US1] Create `graph/plugins/html_plugin.py` — tree-sitter-html parser; extract REFERENCES edges for Angular component selectors and `*ngDirective` attributes; set `edge.weight = 0.7`; register `.html` in `default_registry`
- [x] T008 [US1] Create `graph/graph_store.py` — wraps `networkx.DiGraph`; implement `add_edges(edges)`, `remove_file(path)`, `shortest_path_length(source, target) -> float` (Dijkstra via `nx.single_source_dijkstra_path_length`; returns `math.inf` if unreachable), `save(path)` (node-link JSON), `load(path) -> GraphStore` (raises `GraphLoadError` on corrupt file); per `data-model.md`
- [x] T009 [US1] Create `graph/graph_builder.py` — iterate repo files; resolve plugin by extension from `default_registry`; track file content hashes in `graph_hashes.json`; skip unchanged files; call `graph_store.remove_file` then re-add edges for changed files; call `graph_store.remove_file` for deleted files; per `data-model.md` and `contracts/plugin-interface.md`
- [x] T010 [US1] Create `graph/fusion.py` — implement `expand_and_rerank(seeds, query_text, graph_store, alpha=0.3, beta=0.6, max_hops=3) -> list[FusionResult]`; Dijkstra expansion from each seed up to `max_hops`; score each candidate using `beta^l * (alpha * cos_norm + (1-alpha) * lev)`; exclude candidates with no cosine score; return sorted descending; per `contracts/fusion-api.md`
- [x] T011 [US1] Modify `config.py` — add `GRAPH_ENABLED = env("LCQ_GRAPH_ENABLED", "false").lower() == "true"`, `GRAPH_ALPHA = float(env("LCQ_GRAPH_ALPHA", "0.3"))`, `GRAPH_BETA = float(env("LCQ_GRAPH_BETA", "0.6"))`
- [x] T012 [US1] Modify `querying/query_engine.py` — after ChromaDB retrieval, if `config.GRAPH_ENABLED` and graph file exists, load `GraphStore` and call `fusion.expand_and_rerank`; catch `GraphLoadError` and any exception and fall back to vector-only results; per `contracts/fusion-api.md`

**Checkpoint**: With `LCQ_GRAPH_ENABLED=true`, query "how does the system calculate overdue fines for different member types" returns strategy files in top-10. Strategy files absent with `LCQ_GRAPH_ENABLED=false`.

---

## Phase 4: User Story 2 — Graph built incrementally (Priority: P2)

**Goal**: Re-indexing after editing one file updates only that file's edges. Full build and
incremental update both complete within the timing targets.

**Independent Test**: Index library-catalog-app from scratch, edit one Java file, re-index.
Log output shows exactly one file processed. Incremental update completes in ≤ 5 seconds.

### Implementation for User Story 2

- [x] T013 [US2] Modify `indexing/incremental_indexer.py` — after embedding pass completes, if `config.GRAPH_ENABLED`, call `graph_builder.build_incremental(repo_root, graph_path, changed_files)` where `graph_path = {index_dir}/graph.json`; skip graph build if `GRAPH_ENABLED=false`; depends on T009, T011

**Checkpoint**: Run `python cli/index.py --repo /path/to/library-catalog-app` twice. Second run log shows 0 graph files re-processed (all hashes match). Edit one file; third run shows exactly 1 file re-processed and graph saved.

---

## Phase 5: User Story 3 — Language plugins (Priority: P2)

**Goal**: Each supported file extension resolves to the correct plugin. Unsupported extensions
return None without error. Adding a new plugin requires no changes to core graph logic.

**Independent Test**: Import all three plugin modules; verify `default_registry.get(".java")`,
`default_registry.get(".ts")`, `default_registry.get(".tsx")`, `default_registry.get(".html")`
each return the correct plugin class; verify `default_registry.get(".css")` returns None.

### Implementation for User Story 3

User Story 3 is fully satisfied by T005–T007 (the three plugin implementations in Phase 3).
No additional tasks required — the plugin architecture (T004) and all three plugins (T005–T007)
already fulfill FR-011 through FR-016 and the US3 acceptance scenarios. The independent test
above can be run as soon as T004–T007 are complete.

**Checkpoint**: Running the plugin registration verification script from `quickstart.md`
shows JavaPlugin, TypeScriptPlugin, TypeScriptPlugin, HtmlPlugin, None for the five extensions.

---

## Phase 6: User Story 4 — Tunable α and β weights (Priority: P3)

**Goal**: Setting `LCQ_GRAPH_ALPHA` and `LCQ_GRAPH_BETA` env vars changes fusion scores.
test_10 α/β sweep identifies best weights for the Java/Angular corpus.

**Independent Test**: Set `LCQ_GRAPH_ALPHA=0.5 LCQ_GRAPH_BETA=0.8`, run test_10, confirm
scores differ from α=0.3/β=0.6 defaults. Set back to defaults; scores match baseline.

### Implementation for User Story 4

- [x] T014 [US4] Create `test_suite/test_10_graph_retrieval.py` — session fixture sets `LCQ_GRAPH_ENABLED=true`; runs all 10 ground truth queries from test_04 with graph enabled; reports P@5, R@10, MRR with delta vs test_04 baseline; hard assertions: fine_calculation_strategy R@10 ≥ 0.80 (SC-001), loan_eligibility_chain R@10 ≥ 0.80 (SC-002), mean R@10 ≥ 0.70 (SC-003), mean P@5 ≥ 0.50 (SC-004), mean MRR ≥ 0.88 (SC-005); regression guard for 9 passing test_04 cases (SC-006); timing assertions for full build ≤ 60s (SC-007) and incremental ≤ 5s (SC-008); parametrized α/β sweep: α ∈ {0.1, 0.3, 0.5, 0.7} × β ∈ {0.4, 0.6, 0.8} reporting best mean R@10

**Checkpoint**: `LCQ_GRAPH_ENABLED=true pytest test_suite/test_10_graph_retrieval.py -v` runs all assertions. Parametrized sweep outputs α/β combination rankings.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Fallback safety, graceful degradation, and regression guard.

- [ ] T015 [P] Verify `LCQ_GRAPH_ENABLED=false` leaves all existing test_04 and test_07 results identical to the pre-feature baseline — run `pytest test_suite/test_04_semantic_eval.py test_suite/test_07_llm_keywords.py` and confirm no regressions
- [ ] T016 [P] Verify graceful fallback: corrupt `graph.json`, run a query with `LCQ_GRAPH_ENABLED=true`, confirm warning logged and results match vector-only baseline with no crash (per `quickstart.md` step 7)
- [ ] T017 Flip `LCQ_GRAPH_ENABLED` default to `true` in `config.py` once test_10 benchmark gate passes SC-001 through SC-006; cite benchmark results in commit message (Constitution Principle IV gate)

**Checkpoint**: All existing tests pass. Fallback works. Feature is default-on.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — blocks all user stories
- **Phase 3 (US1 — Retrieval)**: Depends on Phase 2 — core feature, MVP
- **Phase 4 (US2 — Incremental)**: Depends on Phase 3 (T009, T011)
- **Phase 5 (US3 — Plugins)**: Satisfied by Phase 3 tasks T004–T007 — no additional work
- **Phase 6 (US4 — Tuning)**: Depends on Phase 3 (all US1 tasks complete)
- **Phase 7 (Polish)**: Depends on Phase 6 (test_10 must pass before flag flip)

### User Story Dependencies

- **US1 (P1)**: Can start after Phase 2 — no story dependencies
- **US2 (P2)**: Depends on US1 (T009, T011 must exist before T013)
- **US3 (P2)**: Fully satisfied by US1 tasks T004–T007 — runs in parallel with US2
- **US4 (P3)**: Depends on US1 (all graph + fusion modules must exist for test_10)

### Within US1

- T005, T006, T007 are [P] — all three plugins can be written simultaneously
- T008 (graph_store) must precede T009 (graph_builder)
- T009 (graph_builder) and T010 (fusion) can be written in parallel after T008
- T011 (config) is independent — can be written any time after Phase 2
- T012 (query_engine wiring) depends on T010 and T011

---

## Parallel Opportunities

```bash
# Phase 2 — foundational (sequential, T003 before T004):
T003 → T004

# Phase 3 — US1 plugins (all three in parallel):
T005 (java_plugin.py) || T006 (typescript_plugin.py) || T007 (html_plugin.py)

# After T008 (graph_store) completes:
T009 (graph_builder.py) || T010 (fusion.py) || T011 (config.py)

# Phase 7 — polish (parallel):
T015 (regression check) || T016 (fallback verification)
```

---

## Implementation Strategy

### MVP First (US1 Only — Phases 1–3)

1. Phase 1: Setup — create package structure (T001–T002)
2. Phase 2: Foundational — edge schema + plugin registry (T003–T004)
3. Phase 3: US1 — all three plugins + graph store + builder + fusion + config + query wiring (T005–T012)
4. **STOP and VALIDATE**: Run `fine_calculation_strategy` query manually (quickstart.md step 3). Confirm strategy files appear in top-10.
5. Write test_10 (T014) to lock in the improvement with assertions.

### Incremental Delivery

1. MVP (Phases 1–3) → US1 working end-to-end → validate manually
2. Add T013 (incremental indexer wiring) → US2 complete → validate via quickstart.md step 5
3. US3 already done — validate plugin registration via quickstart.md step 7
4. Add T014 (test_10) → US4 complete → run full benchmark gate
5. Phase 7 polish → flip flag default → feature shipped

---

## Notes

- [P] tasks touch different files with no shared incomplete dependencies
- Each task is a single git commit (per plan.md and Constitution Principle II)
- US3 (plugin architecture) has no dedicated implementation tasks — it is structurally
  delivered by T004 (registry) + T005/T006/T007 (the three plugin implementations)
- T017 (flag flip) should only run after `pytest test_suite/test_10_graph_retrieval.py`
  passes all hard assertions (SC-001 through SC-006)
- The α/β tuning sweep in T014 may update `LCQ_GRAPH_ALPHA`/`LCQ_GRAPH_BETA` defaults
  in `config.py` as a follow-on commit if published defaults (0.3/0.6) are not optimal
