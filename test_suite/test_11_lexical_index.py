"""
Lexical identifier index — unit tests.

No Ollama or ChromaDB dependency: pure file I/O + SQLite FTS5, so these run
fast and in isolation from the rest of the suite's embedding-backed fixtures.
"""

import os

import pytest

from indexing.lexical_index import (
    build_lexical_index,
    update_lexical_index_incremental,
    search_lexical,
    extract_tokens,
    split_identifier,
)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def test_split_identifier_camel_case():
    assert split_identifier("processInvoicePayment") == ["process", "invoice", "payment"]


def test_split_identifier_pascal_case():
    assert split_identifier("InvoiceProcessor") == ["invoice", "processor"]


def test_split_identifier_snake_case():
    assert split_identifier("process_invoice_payment") == ["process", "invoice", "payment"]


def test_extract_tokens_includes_whole_and_split_forms():
    tokens = extract_tokens("class InvoiceProcessor { void processInvoice() {} }").split()
    assert "invoiceprocessor" in tokens
    assert "invoice" in tokens
    assert "processor" in tokens
    assert "processinvoice" in tokens
    assert "process" in tokens


# ---------------------------------------------------------------------------
# Fixtures — a tiny synthetic repo, no Ollama needed
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "InvoiceProcessor.java").write_text(
        "public class InvoiceProcessor {\n"
        "    public void processInvoice(Invoice invoice) { invoice.pay(); }\n"
        "}\n"
    )
    (repo / "OverdueFineCalculator.java").write_text(
        "public class OverdueFineCalculator {\n"
        "    public double calculateFine(Loan loan) { return loan.daysOverdue() * 0.25; }\n"
        "}\n"
    )
    (repo / "Unrelated.java").write_text(
        "public class Unrelated {\n"
        "    public void doNothing() {}\n"
        "}\n"
    )
    return repo


@pytest.fixture
def index_dir(tmp_path):
    return str(tmp_path / "index")


# ---------------------------------------------------------------------------
# Build + search
# ---------------------------------------------------------------------------

def test_search_finds_file_by_partial_identifier(tiny_repo, index_dir):
    build_lexical_index(str(tiny_repo), index_dir, log=lambda _: None)

    results = search_lexical(index_dir, "invoice", limit=10)
    sources = [r[0] for r in results]

    assert "InvoiceProcessor.java" in sources
    assert "OverdueFineCalculator.java" not in sources
    assert "Unrelated.java" not in sources


def test_search_matches_symbol_never_typed_as_whole_word(tiny_repo, index_dir):
    """The identifier 'processInvoice' never appears as the standalone word
    'invoice' in the source — only as a camelCase sub-token. Confirms the
    split-token indexing, not literal substring matching, is what finds it."""
    build_lexical_index(str(tiny_repo), index_dir, log=lambda _: None)

    results = search_lexical(index_dir, "fine calculation", limit=10)
    sources = [r[0] for r in results]

    assert "OverdueFineCalculator.java" in sources


def test_search_empty_before_any_index_exists(index_dir):
    assert search_lexical(index_dir, "invoice") == []


def test_search_empty_query_returns_empty(tiny_repo, index_dir):
    build_lexical_index(str(tiny_repo), index_dir, log=lambda _: None)
    assert search_lexical(index_dir, "   ") == []


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def test_incremental_add_and_delete(tiny_repo, index_dir):
    build_lexical_index(str(tiny_repo), index_dir, log=lambda _: None)

    new_file = tiny_repo / "PaymentRetryService.java"
    new_file.write_text(
        "public class PaymentRetryService {\n"
        "    public void retryPayment() {}\n"
        "}\n"
    )

    update_lexical_index_incremental(
        index_dir,
        changed_files=[("PaymentRetryService.java", str(new_file))],
        deleted_files=["Unrelated.java"],
        log=lambda _: None,
    )

    added = [r[0] for r in search_lexical(index_dir, "retry payment", limit=10)]
    assert "PaymentRetryService.java" in added

    removed = [r[0] for r in search_lexical(index_dir, "nothing", limit=10)]
    assert "Unrelated.java" not in removed


def test_incremental_reindex_replaces_stale_tokens(tiny_repo, index_dir):
    build_lexical_index(str(tiny_repo), index_dir, log=lambda _: None)

    invoice_file = tiny_repo / "InvoiceProcessor.java"
    invoice_file.write_text(
        "public class InvoiceProcessor {\n"
        "    public void archiveInvoice(Invoice invoice) {}\n"
        "}\n"
    )

    update_lexical_index_incremental(
        index_dir,
        changed_files=[("InvoiceProcessor.java", str(invoice_file))],
        log=lambda _: None,
    )

    # Old identifier "process" (from processInvoice) should no longer be the
    # only thing matching this file more than once — new "archive" token present.
    results = [r[0] for r in search_lexical(index_dir, "archive", limit=10)]
    assert "InvoiceProcessor.java" in results
