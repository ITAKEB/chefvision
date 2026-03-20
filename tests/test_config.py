"""Tests for ChunkingConfig validation.

# Feature: recipe-chunking, Property 10: Invalid config rejection
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.model.schema import ChunkingConfig
from backend.services.error import ConfigValidationError


# ---------------------------------------------------------------------------
# Property-based test – Property 10: Invalid config rejection
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 10: Invalid config rejection
@settings(max_examples=100)
@given(
    chunk_size=st.integers(min_value=1, max_value=10_000),
    overlap=st.integers(min_value=1, max_value=10_000),
)
def test_invalid_config_raises_when_chunk_size_lte_overlap(
    chunk_size: int, overlap: int
) -> None:
    """**Validates: Requirements 8.3**

    For any (chunk_size, overlap) pair where chunk_size <= overlap,
    ChunkingConfig.validate() must raise ConfigValidationError.
    """
    # Only test the invalid region
    from hypothesis import assume

    assume(chunk_size <= overlap)

    config = ChunkingConfig(chunk_size=chunk_size, overlap=overlap)
    with pytest.raises(ConfigValidationError):
        config.validate()


# ---------------------------------------------------------------------------
# Unit test – valid config does NOT raise
# ---------------------------------------------------------------------------

def test_valid_config_does_not_raise() -> None:
    """A config where chunk_size > overlap should validate without error."""
    config = ChunkingConfig(chunk_size=1000, overlap=200)
    config.validate()  # should not raise
