"""
Cross-language REST edge extraction.

These are the only edges that cross a language boundary, and the only ones
derived from matching route strings rather than resolved symbols, so they
are the easiest to get subtly wrong. A false edge is worse than a missing
one: it silently corrupts every traversal that crosses it.

Fixtures below are synthetic and cover the idiom variations that differ
between real projects, each of which was found by running the extractor
against two genuinely different codebases rather than invented here.
"""

import pytest

from graph.edge import EdgeType
from graph.rest_edges import extract_rest_edges, _normalize_path


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------

def test_path_params_normalize_to_wildcard():
    assert _normalize_path("/patrons/{id}") == "/patrons/*"
    assert _normalize_path("/patrons/${id}") == "/patrons/*"


def test_absolute_url_is_reduced_to_path():
    assert _normalize_path("http://localhost:8080/api/loans") == "/api/loans"


def test_query_string_is_stripped():
    assert _normalize_path("/books?page=1&size=20") == "/books"


def test_non_path_strings_are_rejected():
    """Header lookups like .get('Authorization') must not become routes."""
    assert _normalize_path("Authorization") is None
    assert _normalize_path("Cache-Control") is None
    assert _normalize_path("") is None


# ---------------------------------------------------------------------------
# Repo fixtures
# ---------------------------------------------------------------------------

def _write(base, rel, text):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def simple_repo(tmp_path):
    """Literal absolute paths, no context path, no base URL."""
    _write(tmp_path, "src/main/java/HoldController.java", """
@RestController
public class HoldController {
    @RequestMapping(value = "/holds/{id}", method = RequestMethod.GET)
    public Hold get(long id) { return null; }

    @RequestMapping(value = "/holds", method = RequestMethod.POST)
    public long place() { return 0; }
}
""")
    _write(tmp_path, "ui/src/holds-api.service.ts", """
@Injectable()
export class HoldsApiService {
  constructor(private http: HttpClient) {}
  get(id: number) { return this.http.get<Hold>(`/holds/${id}`); }
  place() { return this.http.post<number>('/holds', null); }
}
""")
    return tmp_path


@pytest.fixture
def hard_repo(tmp_path):
    """Context path, class-level prefix, and a base URL held in a field."""
    _write(tmp_path, "src/main/resources/application.yml",
           "server:\n  port: 8080\n  servlet:\n    context-path: /api\n")
    _write(tmp_path, "src/main/java/AcquisitionController.java", """
@RestController
@RequestMapping("/library/acquisitions")
public class AcquisitionController {
    @PostMapping("/requests/member")
    public Req create() { return null; }

    @GetMapping("/requests/{id}")
    public Req find(long id) { return null; }
}
""")
    _write(tmp_path, "ui/src/acquisition.service.ts", """
export class AcquisitionService {
  private base = 'http://localhost:8080/api/library/acquisitions';
  create(body: any) { return this.http.post<Req>(`${this.base}/requests/member`, body); }
  find(id: number) { return this.http.get<Req>(`${this.base}/requests/${id}`); }
}
""")
    return tmp_path


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------

def test_simple_repo_links_client_to_controller(simple_repo):
    edges, diag = extract_rest_edges(str(simple_repo), log_fn=lambda _: None)

    assert diag.matched_routes == 2
    pairs = {(e.source, e.target) for e in edges}
    assert ("ui/src/holds-api.service.ts", "src/main/java/HoldController.java") in pairs
    assert all(e.edge_type is EdgeType.CALLS_ENDPOINT for e in edges)


def test_context_path_and_base_url_are_resolved(hard_repo):
    """The client says /api/library/acquisitions/requests/member; the server
    annotations say /library/acquisitions + /requests/member and get /api
    only from application.yml. Matching requires reconciling all three."""
    edges, diag = extract_rest_edges(str(hard_repo), log_fn=lambda _: None)

    assert diag.context_path == "/api"
    assert diag.matched_routes == 2
    pairs = {(e.source, e.target) for e in edges}
    assert ("ui/src/acquisition.service.ts",
            "src/main/java/AcquisitionController.java") in pairs


def test_header_lookups_do_not_become_edges(tmp_path):
    _write(tmp_path, "src/main/java/C.java",
           '@RestController public class C { @GetMapping("/x") void x() {} }')
    _write(tmp_path, "ui/a.ts", """
    const auth = headers.get('Authorization');
    const cc = res.headers.get('Cache-Control');
    """)
    edges, diag = extract_rest_edges(str(tmp_path), log_fn=lambda _: None)
    assert diag.client_calls == 0
    assert edges == []


def test_unmatched_routes_produce_no_edges(tmp_path):
    """A client calling a route no server declares must yield nothing rather
    than guessing at the nearest handler."""
    _write(tmp_path, "src/main/java/C.java",
           '@RestController public class C { @GetMapping("/alpha") void a() {} }')
    _write(tmp_path, "ui/a.ts",
           "export class S { go() { return this.http.get('/completely/unrelated/route'); } }")

    edges, diag = extract_rest_edges(str(tmp_path), log_fn=lambda _: None)
    assert diag.server_routes == 1
    assert diag.client_calls == 1
    assert diag.matched_routes == 0
    assert edges == []


def test_total_mismatch_is_reported_loudly(tmp_path):
    """Silently returning zero edges looks identical to 'this repo has no
    frontend', which is how the multi-module resolver bug hid. Warn instead."""
    _write(tmp_path, "src/main/java/C.java",
           '@RestController public class C { @GetMapping("/alpha") void a() {} }')
    _write(tmp_path, "ui/a.ts",
           "export class S { go() { return this.http.get('/zzz/nope'); } }")

    logs = []
    extract_rest_edges(str(tmp_path), log_fn=logs.append)
    assert any("WARNING" in line and "matched NONE" in line for line in logs), logs


def test_diagnostics_summary_is_populated(simple_repo):
    _, diag = extract_rest_edges(str(simple_repo), log_fn=lambda _: None)
    summary = diag.summary()
    assert "server routes=" in summary and "matched=" in summary
