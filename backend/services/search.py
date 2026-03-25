"""Semantic search service for querying recipe chunks in ChromaDB.

Uses a hybrid approach:
1. ChromaDB vector similarity for semantic relevance
2. Ingredient-match re-ranking to boost chunks that actually contain
   the queried ingredients
"""

from __future__ import annotations

import logging
import random

from backend.config import settings
from backend.model.schema import SearchResult
from backend.services.vector_store import _get_collection

logger = logging.getLogger(__name__)

# Weight for ingredient matching vs semantic similarity in final score.
# 0.6 means 60% ingredient match + 40% semantic similarity.
INGREDIENT_WEIGHT = 0.6
SEMANTIC_WEIGHT = 0.4


def _ingredient_match_ratio(text: str, ingredients: list[str]) -> float:
    """Return fraction of *ingredients* found in *text* (case-insensitive).

    Returns 0.0 if no ingredients match, 1.0 if all match.
    """
    if not ingredients:
        return 0.0
    lower = text.lower()
    matches = sum(1 for ing in ingredients if ing.lower() in lower)
    return matches / len(ingredients)


def search_recipes(
    query_terms: list[str],
    top_k: int | None = None,
    min_relevance: float | None = None,
) -> list[SearchResult]:
    """Search ChromaDB for recipe chunks matching the given query terms.

    Strategy:
    1. Fetch a large pool of candidates via semantic search.
    2. Re-rank using a hybrid score that combines semantic similarity
       with an ingredient-match ratio (how many queried ingredients
       actually appear in the chunk text).
    3. Add light randomization among top candidates for variety.

    Args:
        query_terms: List of ingredient names (translated) to search for.
        top_k: Maximum number of results to return.
        min_relevance: Minimum *semantic* relevance threshold.

    Returns:
        A list of SearchResult ordered by hybrid score (descending).
    """
    if not query_terms:
        return []

    if top_k is None:
        top_k = settings.TOP_K
    if min_relevance is None:
        min_relevance = settings.MIN_RELEVANCE_SCORE

    query_string = f"receta con {', '.join(query_terms)}"

    # Fetch a large candidate pool — we'll re-rank aggressively.
    fetch_k = min(top_k * 5, 100)

    collection = _get_collection()
    results = collection.query(
        query_texts=[query_string],
        n_results=fetch_k,
    )

    search_results: list[SearchResult] = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, metadata, distance in zip(documents, metadatas, distances):
        semantic_score = 1.0 / (1.0 + distance)

        if semantic_score < min_relevance:
            continue

        # Hybrid score: ingredient match + semantic similarity
        match_ratio = _ingredient_match_ratio(doc, query_terms)
        hybrid_score = (
            INGREDIENT_WEIGHT * match_ratio
            + SEMANTIC_WEIGHT * semantic_score
        )

        search_results.append(
            SearchResult(
                text=doc,
                source_filename=metadata.get("source_filename", ""),
                page_number=metadata.get("page_number", 0),
                chunk_index=metadata.get("chunk_index", 0),
                relevance_score=round(hybrid_score, 4),
            )
        )

    # Sort by hybrid score descending
    search_results.sort(key=lambda r: r.relevance_score, reverse=True)

    # Light shuffle among top candidates for variety
    if len(search_results) > top_k:
        top_fixed = search_results[:3]
        rest = search_results[3 : top_k * 2]
        random.shuffle(rest)
        search_results = top_fixed + rest

    search_results = search_results[:top_k]
    search_results.sort(key=lambda r: r.relevance_score, reverse=True)

    return search_results
