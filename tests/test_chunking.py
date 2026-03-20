"""Property-based and unit tests for the chunking engine.

# Feature: recipe-chunking, Property 2: Chunk size and content invariants
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.services.chunking import chunk_text


# ---------------------------------------------------------------------------
# Property-based test – Property 2: Chunk size and content invariants
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 2: Chunk size and content invariants
@settings(max_examples=100)
@given(
    text=st.text(min_size=1, max_size=5000),
    chunk_size=st.integers(min_value=10, max_value=2000),
    overlap=st.integers(min_value=1, max_value=1999),
)
def test_chunk_size_and_content_invariants(
    text: str, chunk_size: int, overlap: int
) -> None:
    """**Validates: Requirements 2.1, 2.5**

    For any non-empty text string and any valid (chunk_size, overlap) pair
    where chunk_size > overlap > 0, every chunk produced by chunk_text should
    be at most chunk_size characters long, non-empty, and equal to its
    whitespace-trimmed form.
    """
    assume(chunk_size > overlap)
    # Only test when text has non-whitespace content (otherwise empty list is valid)
    assume(text.strip())

    chunks = chunk_text(text, "test.pdf", 1, chunk_size, overlap)

    for chunk in chunks:
        assert len(chunk.text) <= chunk_size, (
            f"Chunk length {len(chunk.text)} exceeds chunk_size {chunk_size}"
        )
        assert chunk.text != "", "Chunk text must not be empty"
        assert chunk.text == chunk.text.strip(), (
            f"Chunk text is not trimmed: {chunk.text!r}"
        )


# ---------------------------------------------------------------------------
# Property-based test – Property 3: Chunk overlap correctness
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 3: Chunk overlap correctness
@settings(max_examples=100)
@given(
    # Generate plain word-based text (no recipe-like patterns) long enough
    # to produce multiple chunks via the character-based splitting path.
    words=st.lists(
        st.from_regex(r"[a-z]{3,8}", fullmatch=True),
        min_size=80,
        max_size=300,
    ),
    chunk_size=st.integers(min_value=60, max_value=300),
    overlap=st.integers(min_value=10, max_value=100),
)
def test_chunk_overlap_correctness(
    words: list[str], chunk_size: int, overlap: int
) -> None:
    """**Validates: Requirements 2.2**

    For any text string that produces more than one chunk via the
    character-based splitting path, the tail of chunk N (last `overlap`
    characters worth of content) should appear as a prefix of chunk N+1,
    preserving context across chunk boundaries.
    """
    assume(chunk_size > overlap)

    text = " ".join(words)
    # Ensure text is long enough to actually produce multiple chunks.
    assume(len(text) > chunk_size)

    chunks = chunk_text(text, "test.pdf", 1, chunk_size, overlap)

    # We need at least 2 chunks to verify overlap.
    assume(len(chunks) >= 2)

    for i in range(len(chunks) - 1):
        current_text = chunks[i].text
        next_text = chunks[i + 1].text

        # The tail of the current chunk (up to `overlap` chars) should appear
        # at the start of the next chunk.  Because chunks are word-boundary
        # aligned, the exact overlap length may be shorter than `overlap`, but
        # the shared content must be present.
        tail = current_text[-overlap:]
        # Word-boundary alignment may trim the tail; use the last complete
        # words from the tail as the overlap token.
        tail_words = tail.strip().split()
        if not tail_words:
            continue

        # At least the last word of the current chunk should appear at the
        # beginning of the next chunk (word-boundary alignment means the
        # overlap region shares complete words).
        overlap_token = tail_words[-1]
        # The next chunk's prefix (first `overlap` chars + some slack for
        # word-boundary alignment) should contain the overlap token.
        prefix = next_text[: overlap + len(overlap_token) + 10]
        assert overlap_token in prefix, (
            f"Overlap token {overlap_token!r} from tail of chunk {i} "
            f"not found in prefix of chunk {i+1}.\n"
            f"  chunk[{i}] tail ({overlap} chars): {tail!r}\n"
            f"  chunk[{i+1}] prefix: {prefix!r}"
        )


# ---------------------------------------------------------------------------
# Property-based test – Property 4: Chunk metadata completeness
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 4: Chunk metadata completeness
@settings(max_examples=100)
@given(
    # Simple alphanumeric filename ending in .pdf
    filename=st.from_regex(r"[a-zA-Z0-9]{1,20}\.pdf", fullmatch=True),
    page_number=st.integers(min_value=1, max_value=10000),
    text=st.text(min_size=1, max_size=5000),
)
def test_chunk_metadata_completeness(
    filename: str, page_number: int, text: str
) -> None:
    """**Validates: Requirements 2.3**

    For any chunking operation given a source filename, page number, and
    input text, every returned chunk should carry the provided
    source_filename, page_number, and a sequential chunk_index starting
    from 0.
    """
    # Only meaningful when text has non-whitespace content
    assume(text.strip())

    chunks = chunk_text(text, filename, page_number)

    for i, chunk in enumerate(chunks):
        assert chunk.source_filename == filename, (
            f"Chunk {i} source_filename {chunk.source_filename!r} != {filename!r}"
        )
        assert chunk.page_number == page_number, (
            f"Chunk {i} page_number {chunk.page_number} != {page_number}"
        )
        assert chunk.chunk_index == i, (
            f"Chunk {i} chunk_index {chunk.chunk_index} != expected {i}"
        )
