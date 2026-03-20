"""Property-based tests for the vector store service.

# Feature: recipe-chunking, Property 7: Embedding round-trip
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.model.schema import Chunk


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_filenames = st.from_regex(r"[a-zA-Z0-9]{1,15}\.pdf", fullmatch=True)

_chunk_texts = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
).filter(lambda t: t.strip() != "")

_chunks_list = st.lists(
    st.tuples(
        _chunk_texts,
        st.integers(min_value=1, max_value=100),  # page_number
    ),
    min_size=1,
    max_size=5,
)


def _build_chunks(items: list[tuple[str, int]], filename: str) -> list[Chunk]:
    """Build a list of Chunk objects with sequential chunk indices."""
    return [
        Chunk(
            text=text,
            source_filename=filename,
            page_number=page,
            chunk_index=idx,
        )
        for idx, (text, page) in enumerate(items)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ChromaDB at a temporary directory for full test isolation."""
    monkeypatch.setattr("backend.services.vector_store.settings.CHROMA_DIR", tmp_path)


# ---------------------------------------------------------------------------
# Property-based test – Property 7: Embedding round-trip
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 7: Embedding round-trip
@settings(max_examples=20, deadline=None)
@given(
    filename=_filenames,
    items=_chunks_list,
)
def test_embedding_round_trip(
    filename: str,
    items: list[tuple[str, int]],
) -> None:
    """**Validates: Requirements 4.1**

    For any list of Chunks with valid metadata, after calling embed_chunks,
    querying ChromaDB by each chunk's deterministic ID should return the
    original chunk text and metadata.
    """
    from backend.services.vector_store import embed_chunks, _get_collection

    chunks = _build_chunks(items, filename)

    result = embed_chunks(chunks, filename)

    # All chunks should embed successfully
    assert result.success_count == len(chunks)
    assert result.error_count == 0

    # Query each chunk by its deterministic ID and verify round-trip
    collection = _get_collection()
    for chunk in chunks:
        fetched = collection.get(ids=[chunk.chunk_id], include=["documents", "metadatas"])

        assert len(fetched["ids"]) == 1, (
            f"Expected 1 result for {chunk.chunk_id}, got {len(fetched['ids'])}"
        )
        assert fetched["documents"][0] == chunk.text, (
            f"Document text mismatch for {chunk.chunk_id}"
        )

        meta = fetched["metadatas"][0]
        assert meta["source_filename"] == chunk.source_filename
        assert meta["page_number"] == chunk.page_number
        assert meta["chunk_index"] == chunk.chunk_index


# ---------------------------------------------------------------------------
# Property-based test – Property 9: Deletion removes all embeddings
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 9: Deletion removes all embeddings
@settings(max_examples=20, deadline=None)
@given(
    filename=_filenames,
    items=_chunks_list,
)
def test_deletion_removes_all_embeddings(
    filename: str,
    items: list[tuple[str, int]],
) -> None:
    """**Validates: Requirements 7.1**

    For any filename with existing embeddings in ChromaDB, after calling
    delete_embeddings(filename), querying the vector store for that filename
    should return zero results.
    """
    from backend.services.vector_store import (
        embed_chunks,
        delete_embeddings,
        has_embeddings,
        _get_collection,
    )

    chunks = _build_chunks(items, filename)

    # Embed chunks first
    result = embed_chunks(chunks, filename)
    assert result.success_count == len(chunks)

    # Verify embeddings exist before deletion
    assert has_embeddings(filename) is True

    # Delete all embeddings for the filename
    delete_embeddings(filename)

    # Verify has_embeddings returns False
    assert has_embeddings(filename) is False

    # Verify querying by filename returns zero results
    collection = _get_collection()
    results = collection.get(where={"source_filename": filename})
    assert len(results["ids"]) == 0, (
        f"Expected 0 results after deletion, got {len(results['ids'])}"
    )


# ---------------------------------------------------------------------------
# Unit tests – Vector store error handling
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock
from backend.services.error import EmbeddingDeletionError


class TestEmbedChunksErrorHandling:
    """Unit tests for embed_chunks error handling (Requirements 4.3, 4.4)."""

    def test_failed_embedding_logs_error_and_continues(self) -> None:
        """**Validates: Requirements 4.3**

        When upsert fails for the first chunk, embed_chunks should log the
        error and continue processing remaining chunks successfully.
        """
        from backend.services.vector_store import embed_chunks

        chunks = _build_chunks(
            [("Recipe text one", 1), ("Recipe text two", 2), ("Recipe text three", 3)],
            "fail_test.pdf",
        )

        call_count = 0

        def _upsert_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("ChromaDB upsert failed")

        with patch("backend.services.vector_store._get_collection") as mock_get:
            mock_collection = MagicMock()
            mock_collection.upsert.side_effect = _upsert_side_effect
            mock_get.return_value = mock_collection

            result = embed_chunks(chunks, "fail_test.pdf")

        assert result.success_count == 2
        assert result.error_count == 1
        assert len(result.errors) == 1
        assert "fail_test.pdf_0" in result.errors[0]

    def test_successful_full_embedding(self) -> None:
        """**Validates: Requirements 4.4**

        When all chunks embed successfully, EmbeddingResult should report
        all successes and zero errors.
        """
        from backend.services.vector_store import embed_chunks

        chunks = _build_chunks(
            [("Bake the cake", 1), ("Stir the soup", 2)],
            "success_test.pdf",
        )

        result = embed_chunks(chunks, "success_test.pdf")

        assert result.success_count == len(chunks)
        assert result.error_count == 0
        assert result.errors == []


class TestDeleteEmbeddingsErrorHandling:
    """Unit tests for delete_embeddings error handling (Requirement 7.3)."""

    def test_failed_deletion_raises_embedding_deletion_error(self) -> None:
        """**Validates: Requirements 7.3**

        When the underlying ChromaDB operation fails during deletion,
        delete_embeddings should raise EmbeddingDeletionError.
        """
        from backend.services.vector_store import delete_embeddings

        with patch("backend.services.vector_store._get_collection") as mock_get:
            mock_collection = MagicMock()
            mock_collection.get.side_effect = RuntimeError("ChromaDB unavailable")
            mock_get.return_value = mock_collection

            with pytest.raises(EmbeddingDeletionError, match="Failed to delete embeddings"):
                delete_embeddings("some_file.pdf")
