"""
Multi-module Java source-root resolution.

Regression coverage for a bug that produced silent, total edge loss:
_resolve_java() only ever tried a fixed list of source roots directly
under repo_root ("src/main/java", "src/test/java", "src"), so in a
multi-module Gradle/Maven build every import *within* a subproject
(e.g. discovery-service/src/main/java/...) resolved to nothing. No
error, no warning — those modules just silently got zero graph edges,
which reads downstream as "nothing implements this" rather than "the
resolver never looked there".

Fixed by walking the repo once to discover every src/main/java and
src/test/java root at any depth.
"""

import os

import pytest

from graph.plugins.java_plugin import _discover_src_roots, _SRC_ROOT_CACHE


@pytest.fixture
def multimodule_repo(tmp_path):
    """A miniature multi-module Gradle layout with a subproject-internal import."""
    repo = tmp_path / "repo"

    root_pkg = repo / "src/main/java/com/example/app"
    root_pkg.mkdir(parents=True)
    (root_pkg / "RootService.java").write_text(
        "package com.example.app;\n"
        "public class RootService {}\n"
    )

    sub_pkg = repo / "sub-service/src/main/java/com/example/sub"
    sub_pkg.mkdir(parents=True)
    (sub_pkg / "SubHelper.java").write_text(
        "package com.example.sub;\n"
        "public class SubHelper {}\n"
    )
    (sub_pkg / "SubController.java").write_text(
        "package com.example.sub;\n"
        "import com.example.sub.SubHelper;\n"
        "public class SubController {\n"
        "    private SubHelper helper;\n"
        "}\n"
    )

    # Build output that must not be mistaken for a source root
    stale = repo / "sub-service/build/classes/java/main/com/example/sub"
    stale.mkdir(parents=True)
    (stale / "SubHelper.class").write_text("")

    _SRC_ROOT_CACHE.clear()
    return repo


def test_discovers_source_roots_in_every_module(multimodule_repo):
    roots = _discover_src_roots(str(multimodule_repo))
    assert "src/main/java" in roots
    assert os.path.join("sub-service", "src", "main", "java") in roots


def test_discovery_skips_build_output_dirs(multimodule_repo):
    roots = _discover_src_roots(str(multimodule_repo))
    assert not any("build" in r.split(os.sep) for r in roots)


def test_root_module_sorts_before_nested_modules(multimodule_repo):
    """Shallowest-first ordering keeps the root module winning ties."""
    roots = _discover_src_roots(str(multimodule_repo))
    assert roots.index("src/main/java") < roots.index(
        os.path.join("sub-service", "src", "main", "java")
    )


def test_subproject_internal_import_resolves_to_an_edge(multimodule_repo):
    """The actual payoff: an import inside a subproject must produce a real
    edge, not silently resolve to nothing."""
    from graph.plugins.java_plugin import JavaPlugin

    source = os.path.join("sub-service", "src", "main", "java", "com", "example", "sub",
                          "SubController.java")
    edges = JavaPlugin().extract_edges(str(multimodule_repo / source), source)

    targets = {e.target for e in edges}
    expected = os.path.join("sub-service", "src", "main", "java", "com", "example", "sub",
                            "SubHelper.java")
    assert expected in targets


def test_single_module_repo_still_resolves(tmp_path):
    """The common single-module case must keep working unchanged."""
    from graph.plugins.java_plugin import JavaPlugin

    repo = tmp_path / "simple"
    pkg = repo / "src/main/java/com/example"
    pkg.mkdir(parents=True)
    (pkg / "Helper.java").write_text("package com.example;\npublic class Helper {}\n")
    (pkg / "Main.java").write_text(
        "package com.example;\n"
        "import com.example.Helper;\n"
        "public class Main { private Helper h; }\n"
    )
    _SRC_ROOT_CACHE.clear()

    source = os.path.join("src", "main", "java", "com", "example", "Main.java")
    edges = JavaPlugin().extract_edges(str(repo / source), source)

    assert os.path.join("src", "main", "java", "com", "example", "Helper.java") in {
        e.target for e in edges
    }
