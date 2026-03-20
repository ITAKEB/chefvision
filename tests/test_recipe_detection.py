"""Property-based and unit tests for recipe block detection.

# Feature: recipe-chunking, Property 5: Recipe scoring consistency
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.services.chunking import score_recipe_block, is_recipe_block, COOKING_VERBS


# ---------------------------------------------------------------------------
# Helpers – strategies for generating recipe-like text
# ---------------------------------------------------------------------------

# Words that are NOT cooking verbs (plain filler words)
_FILLER_WORDS = st.from_regex(r"[a-z]{4,10}", fullmatch=True)

# Quantity patterns like "100g", "2 cup", "15 tsp", or bare numbers like "42"
_QUANTITY_PATTERN = st.from_regex(
    r"\d{1,4}\s?(g|ml|cup|tbsp|tsp)?", fullmatch=True
)

# Strategy that picks a random cooking verb from the predefined list
_COOKING_VERB = st.sampled_from(COOKING_VERBS)

# A single token is either a filler word, a cooking verb, or a quantity pattern
_TOKEN = st.one_of(_FILLER_WORDS, _COOKING_VERB, _QUANTITY_PATTERN)


# ---------------------------------------------------------------------------
# Property-based test – Property 5: Recipe scoring consistency
# ---------------------------------------------------------------------------

# Feature: recipe-chunking, Property 5: Recipe scoring consistency
@settings(max_examples=100)
@given(
    tokens=st.lists(_TOKEN, min_size=0, max_size=40),
    threshold=st.integers(min_value=0, max_value=20),
)
def test_recipe_scoring_consistency(tokens: list[str], threshold: int) -> None:
    """**Validates: Requirements 3.1, 3.2, 3.4**

    For any text string and any integer threshold, `is_recipe_block(text, threshold)`
    should return True if and only if `score_recipe_block(text) > threshold`.
    """
    text = " ".join(tokens)

    score = score_recipe_block(text)
    result = is_recipe_block(text, threshold)

    assert result == (score > threshold), (
        f"is_recipe_block returned {result} but score={score} and threshold={threshold}. "
        f"Expected {score > threshold}."
    )
