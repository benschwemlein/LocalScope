"""
JSP include-edge extraction.

Legacy JSP web layers were previously invisible to every retrieval leg:
.jsp wasn't in DEFAULT_INDEX_EXTS (so neither vector nor lexical saw
them) and no graph plugin handled them.

Only edge type extracted is IMPORTS, from the two include forms:
  <%@ include file="..." %>   static / compile-time directive
  <jsp:include page="..." />  dynamic / runtime action

Deliberately NOT extracted: controller -> view-name edges. Resolving a
Spring view name to a JSP requires a view resolver's prefix/suffix
config, and inventing that mapping would fabricate edges the codebase
doesn't actually declare.
"""

import os

import pytest

from graph.plugins.jsp_plugin import JspPlugin


@pytest.fixture
def webapp(tmp_path):
    repo = tmp_path / "repo"
    jsp_dir = repo / "src/main/webapp/WEB-INF/jsp"
    jsp_dir.mkdir(parents=True)

    (jsp_dir / "header.jsp").write_text("<html><body>\n")
    (jsp_dir / "footer.jsp").write_text("</body></html>\n")
    (jsp_dir / "nav.jspf").write_text("<nav>...</nav>\n")

    (jsp_dir / "patronList.jsp").write_text(
        '<%@ page contentType="text/html;charset=UTF-8" language="java" %>\n'
        '<%@ include file="header.jsp" %>\n'
        '<jsp:include page="nav.jspf" />\n'
        "<table></table>\n"
        '<%@ include file="footer.jsp" %>\n'
    )

    (jsp_dir / "orphan.jsp").write_text("<p>nothing included</p>\n")

    return repo


def _rel(*parts: str) -> str:
    return os.path.join("src", "main", "webapp", "WEB-INF", "jsp", *parts)


def test_extracts_static_include_directives(webapp):
    source = _rel("patronList.jsp")
    edges = JspPlugin().extract_edges(str(webapp / source), source)
    targets = {e.target for e in edges}

    assert _rel("header.jsp") in targets
    assert _rel("footer.jsp") in targets


def test_extracts_dynamic_include_actions(webapp):
    source = _rel("patronList.jsp")
    edges = JspPlugin().extract_edges(str(webapp / source), source)

    assert _rel("nav.jspf") in {e.target for e in edges}


def test_jsp_with_no_includes_yields_no_edges(webapp):
    source = _rel("orphan.jsp")
    assert JspPlugin().extract_edges(str(webapp / source), source) == []


def test_unresolvable_include_is_skipped_not_invented(webapp):
    """An include pointing at a file that doesn't exist must produce no edge
    rather than a phantom node."""
    jsp_dir = webapp / "src/main/webapp/WEB-INF/jsp"
    (jsp_dir / "broken.jsp").write_text('<%@ include file="doesNotExist.jsp" %>\n')

    source = _rel("broken.jsp")
    assert JspPlugin().extract_edges(str(webapp / source), source) == []


def test_context_relative_include_resolves_from_webapp_root(webapp):
    """A leading-slash include is context-relative, not page-relative."""
    jsp_dir = webapp / "src/main/webapp/WEB-INF/jsp"
    (jsp_dir / "ctx.jsp").write_text(
        '<%@ include file="/WEB-INF/jsp/header.jsp" %>\n'
    )

    source = _rel("ctx.jsp")
    edges = JspPlugin().extract_edges(str(webapp / source), source)

    assert _rel("header.jsp") in {e.target for e in edges}


def test_duplicate_includes_are_deduplicated(webapp):
    jsp_dir = webapp / "src/main/webapp/WEB-INF/jsp"
    (jsp_dir / "dupe.jsp").write_text(
        '<%@ include file="header.jsp" %>\n'
        '<%@ include file="header.jsp" %>\n'
    )

    source = _rel("dupe.jsp")
    edges = JspPlugin().extract_edges(str(webapp / source), source)

    assert len(edges) == 1


def test_jsp_extensions_are_indexed(webapp):
    """The indexer must actually pick up .jsp files, or none of the above matters."""
    from indexing.indexer import DEFAULT_INDEX_EXTS

    for ext in (".jsp", ".jspf", ".tag", ".tagx"):
        assert ext in DEFAULT_INDEX_EXTS
