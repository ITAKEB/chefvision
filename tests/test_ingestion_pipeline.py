"""Property-based tests for the ingestion pipeline.

# Feature: recipe-chunking, Property 8: Duplicate ingestion is a no-op
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from backend.model.schema import ChunkingConfig, EmbeddingStatus, IngestionResult, PageText
from backend.services.ingestion import ingest_pdf


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_filenames = st.from_regex(r"[a-zA-Z0-9]{1,12}\.pdf", fullmatch=True)

# Recipe-like text that will score above the default threshold of 5.
_recipe_texts = st.sampled_from(
    [
        "Mix 100g flour. Cook for 20 minutes. Add 2 cups water. Bake at 350. Stir well. Boil the mixture.",
        "Fry 200g chicken. Heat the oil. Stir 3 tbsp sauce. Bake for 30 minutes. Add 1 cup broth. Mix thoroughly.",
        "Boil 500ml water. Add 50g pasta. Cook for 10 minutes. Stir occasionally. Heat 2 tbsp butter. Bake until golden.",
    ]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ChromaDB at a temporary directory for full test isolation."""
    monkeypatch.setattr("backend.services.vector_store.settings.CHROMA_DIR", tmp_path)


# ---------------------------------------------------------------------------
# Property-based test – Property 8: Duplicate ingestion is a no-op
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 8: Duplicate ingestion is a no-op
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    filename=_filenames,
    recipe_text=_recipe_texts,
)
def test_duplicate_ingestion_is_noop(
    tmp_path: Path,
    filename: str,
    recipe_text: str,
) -> None:
    """**Validates: Requirements 6.2**

    For any file that already has embeddings in the vector store, calling the
    ingestion pipeline again should skip processing and return a result
    indicating the file is already embedded, without modifying existing
    embeddings.
    """
    from backend.services.ingestion import ingest_pdf
    from backend.services.vector_store import _get_collection

    # Create a fake PDF path (we mock extraction so the file doesn't need to exist)
    pdf_path = tmp_path / filename

    config = ChunkingConfig()

    mock_pages = [PageText(page_number=1, text=recipe_text)]

    with patch("backend.services.ingestion.extract_text_from_pdf", return_value=mock_pages):
        # First ingestion — should embed successfully
        result1 = ingest_pdf(pdf_path, config)

    assert result1.status == EmbeddingStatus.EMBEDDED
    assert result1.chunks_processed > 0

    # Capture the embeddings after first ingestion
    collection = _get_collection()
    first_results = collection.get(where={"source_filename": filename})
    first_ids = sorted(first_results["ids"])
    first_docs = [
        first_results["documents"][first_results["ids"].index(id_)]
        for id_ in first_ids
    ]

    with patch("backend.services.ingestion.extract_text_from_pdf", return_value=mock_pages):
        # Second ingestion — should be a no-op
        result2 = ingest_pdf(pdf_path, config)

    # The second call should report EMBEDDED with zero chunks processed
    assert result2.status == EmbeddingStatus.EMBEDDED
    assert result2.chunks_processed == 0

    # Existing embeddings should be unchanged
    second_results = collection.get(where={"source_filename": filename})
    second_ids = sorted(second_results["ids"])
    second_docs = [
        second_results["documents"][second_results["ids"].index(id_)]
        for id_ in second_ids
    ]

    assert first_ids == second_ids, "Embedding IDs changed after duplicate ingestion"
    assert first_docs == second_docs, "Embedding documents changed after duplicate ingestion"


# ---------------------------------------------------------------------------
# Unit tests – Ingestion pipeline orchestration
# ---------------------------------------------------------------------------

from backend.services.error import PdfReadError


class TestIngestPdfFullPipelineSuccess:
    """Full pipeline executes steps in sequence and returns EMBEDDED status.

    Validates: Requirements 5.1
    """

    def test_full_pipeline_success(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "recipes.pdf"

        recipe_text = (
            "Mix 100g flour. Cook for 20 minutes. Add 2 cups water. "
            "Bake at 350. Stir well. Boil the mixture."
        )
        mock_pages = [PageText(page_number=1, text=recipe_text)]

        with patch(
            "backend.services.ingestion.extract_text_from_pdf",
            return_value=mock_pages,
        ):
            result = ingest_pdf(pdf_path, ChunkingConfig())

        assert result.status == EmbeddingStatus.EMBEDDED
        assert result.chunks_processed > 0
        assert result.error_message is None


class TestIngestPdfExtractionFailure:
    """Failure at PDF extraction step sets status to NOT_EMBEDDED.

    Validates: Requirements 5.4
    """

    def test_pdf_read_error_sets_not_embedded(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "corrupt.pdf"

        with patch(
            "backend.services.ingestion.extract_text_from_pdf",
            side_effect=PdfReadError("Corrupt PDF file"),
        ):
            result = ingest_pdf(pdf_path, ChunkingConfig())

        assert result.status == EmbeddingStatus.NOT_EMBEDDED
        assert result.error_message is not None
        assert "PDF extraction failed" in result.error_message


class TestIngestPdfNoRecipeContent:
    """Pages with non-recipe text result in NOT_EMBEDDED status.

    Validates: Requirements 5.1, 5.4
    """

    def test_no_recipe_content_sets_not_embedded(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "boring.pdf"

        non_recipe_text = "This is a plain document about software engineering best practices."
        mock_pages = [PageText(page_number=1, text=non_recipe_text)]

        with patch(
            "backend.services.ingestion.extract_text_from_pdf",
            return_value=mock_pages,
        ):
            result = ingest_pdf(pdf_path, ChunkingConfig())

        assert result.status == EmbeddingStatus.NOT_EMBEDDED
        assert result.error_message is not None
        assert "No recipe content" in result.error_message


class TestIngestPdfInvalidConfig:
    """Invalid ChunkingConfig (chunk_size <= overlap) sets NOT_EMBEDDED.

    Validates: Requirements 5.4, 8.3
    """

    def test_invalid_config_sets_not_embedded(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "recipes.pdf"
        bad_config = ChunkingConfig(chunk_size=100, overlap=100)

        result = ingest_pdf(pdf_path, bad_config)

        assert result.status == EmbeddingStatus.NOT_EMBEDDED
        assert result.error_message is not None
        assert "configuration" in result.error_message.lower()


class TestIngestPdfDuplicateSkip:
    """Duplicate file triggers skip and returns EMBEDDED with 0 chunks.

    Validates: Requirements 6.1, 6.2
    """

    def test_duplicate_file_skips_processing(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "recipes.pdf"

        recipe_text = (
            "Mix 100g flour. Cook for 20 minutes. Add 2 cups water. "
            "Bake at 350. Stir well. Boil the mixture."
        )
        mock_pages = [PageText(page_number=1, text=recipe_text)]

        with patch(
            "backend.services.ingestion.extract_text_from_pdf",
            return_value=mock_pages,
        ):
            first_result = ingest_pdf(pdf_path, ChunkingConfig())

        assert first_result.status == EmbeddingStatus.EMBEDDED

        # Second call should skip processing
        with patch(
            "backend.services.ingestion.extract_text_from_pdf",
            return_value=mock_pages,
        ) as mock_extract:
            second_result = ingest_pdf(pdf_path, ChunkingConfig())

        assert second_result.status == EmbeddingStatus.EMBEDDED
        assert second_result.chunks_processed == 0
        # extract_text_from_pdf should NOT have been called for the duplicate
        mock_extract.assert_not_called()
