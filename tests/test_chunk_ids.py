"""Property-based tests for deterministic chunk IDs.

# Feature: recipe-chunking, Property 6: Deterministic chunk IDs
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.model.schema import Chunk


# Feature: recipe-chunking, Property 6: Deterministic chunk IDs
@settings(max_examples=100)
@given(
    source_filename=st.from_regex(r"[a-zA-Z0-9_\-]{1,30}\.[a-z]{1,5}", fullmatch=True),
    chunk_index=st.integers(min_value=0, max_value=100_000),
)
def test_chunk_id_is_deterministic(source_filename: str, chunk_index: int) -> None:
    """**Validates: Requirements 4.2**

    For any Chunk with a given source_filename and chunk_index, the chunk_id
    property should always return the string '{source_filename}_{chunk_index}'.
    """
    chunk = Chunk(
        text="some text",
        source_filename=source_filename,
        page_number=1,
        chunk_index=chunk_index,
    )

    expected_id = f"{source_filename}_{chunk_index}"
    assert chunk.chunk_id == expected_id, (
        f"chunk_id {chunk.chunk_id!r} != expected {expected_id!r}"
    )
    # Calling it again should return the exact same value (determinism).
    assert chunk.chunk_id == expected_id


# Feature: recipe-chunking, Property 6: Deterministic chunk IDs
@settings(max_examples=100)
@given(
    filename_a=st.from_regex(r"[a-zA-Z0-9_\-]{1,30}\.[a-z]{1,5}", fullmatch=True),
    index_a=st.integers(min_value=0, max_value=100_000),
    filename_b=st.from_regex(r"[a-zA-Z0-9_\-]{1,30}\.[a-z]{1,5}", fullmatch=True),
    index_b=st.integers(min_value=0, max_value=100_000),
)
def test_different_pairs_produce_different_ids(
    filename_a: str, index_a: int, filename_b: str, index_b: int
) -> None:
    """**Validates: Requirements 4.2**

    Two Chunks with different (source_filename, chunk_index) pairs should
    produce different chunk_id values.
    """
    assume((filename_a, index_a) != (filename_b, index_b))

    chunk_a = Chunk(text="a", source_filename=filename_a, page_number=1, chunk_index=index_a)
    chunk_b = Chunk(text="b", source_filename=filename_b, page_number=1, chunk_index=index_b)

    assert chunk_a.chunk_id != chunk_b.chunk_id, (
        f"Different pairs ({filename_a!r}, {index_a}) and ({filename_b!r}, {index_b}) "
        f"produced the same chunk_id: {chunk_a.chunk_id!r}"
    )
