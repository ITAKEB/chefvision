"""Property-based tests for PDF text extraction.

# Feature: recipe-chunking, Property 1: Extraction filters empty pages
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.model.schema import PageText
from backend.services.ingestion import extract_text_from_pdf

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for page text: either empty/whitespace-only or meaningful content
_empty_text = st.sampled_from(["", " ", "  \n\t  ", "\n", "\t"])
_non_empty_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
).filter(lambda t: t.strip() != "")

_page_text = st.one_of(_empty_text, _non_empty_text)

# A list of page texts representing a multi-page PDF (at least 1 page)
_page_list = st.lists(_page_text, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mock_reader(page_texts: list[str]) -> MagicMock:
    """Create a mock pypdf.PdfReader whose pages return the given texts."""
    reader = MagicMock()
    mock_pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)
    reader.pages = mock_pages
    return reader


# ---------------------------------------------------------------------------
# Property-based test – Property 1: Extraction filters empty pages
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 1: Extraction filters empty pages
@settings(max_examples=100)
@given(page_texts=_page_list)
def test_extraction_filters_empty_pages(page_texts: list[str]) -> None:
    """**Validates: Requirements 1.1, 1.2**

    For any PDF input containing a mix of pages with text and pages with
    empty/whitespace-only content, the PDF extractor should return only pages
    with non-empty text, each paired with its correct 1-based page number,
    and the count of returned pairs should equal the count of non-empty pages
    in the input.
    """
    mock_reader = _build_mock_reader(page_texts)

    with patch("backend.services.ingestion.pypdf.PdfReader", return_value=mock_reader):
        result = extract_text_from_pdf(Path("recipe.pdf"))

    # Compute expected non-empty pages (1-based page numbers)
    expected = [
        PageText(page_number=i + 1, text=text)
        for i, text in enumerate(page_texts)
        if text.strip()
    ]

    # Count must match
    assert len(result) == len(expected), (
        f"Expected {len(expected)} non-empty pages, got {len(result)}"
    )

    # Each returned page must match expected page number and text
    for actual, exp in zip(result, expected):
        assert actual.page_number == exp.page_number, (
            f"Page number mismatch: got {actual.page_number}, expected {exp.page_number}"
        )
        assert actual.text == exp.text, (
            f"Text mismatch on page {exp.page_number}"
        )

# ---------------------------------------------------------------------------
# Imports for unit tests
# ---------------------------------------------------------------------------

import pytest

from backend.services.error import PdfFormatError, PdfReadError


# ---------------------------------------------------------------------------
# Unit tests – PDF extraction edge cases
# ---------------------------------------------------------------------------


def test_non_pdf_file_raises_pdf_format_error() -> None:
    """**Validates: Requirement 1.3**

    A file that does not have a .pdf extension should raise PdfFormatError.
    """
    with pytest.raises(PdfFormatError, match="Invalid file format"):
        extract_text_from_pdf(Path("recipe.txt"))


def test_corrupt_pdf_raises_pdf_read_error() -> None:
    """**Validates: Requirement 1.4**

    A corrupt/unreadable PDF should raise PdfReadError.
    """
    with patch(
        "backend.services.ingestion.pypdf.PdfReader",
        side_effect=Exception("file is corrupted"),
    ):
        with pytest.raises(PdfReadError, match="Failed to read PDF"):
            extract_text_from_pdf(Path("corrupt.pdf"))


def test_single_page_pdf_returns_one_page_text() -> None:
    """**Validates: Requirements 1.1, 1.2**

    A single-page PDF with non-empty text should return exactly one PageText
    with page_number=1.
    """
    mock_reader = _build_mock_reader(["Hello recipe world"])

    with patch("backend.services.ingestion.pypdf.PdfReader", return_value=mock_reader):
        result = extract_text_from_pdf(Path("single.pdf"))

    assert len(result) == 1
    assert result[0].page_number == 1
    assert result[0].text == "Hello recipe world"
    assert isinstance(result[0], PageText)
