"""Chunking engine for splitting text into overlapping chunks with metadata.

Strategy:
1. Try to detect recipe boundaries using general heuristics (title-like lines,
   metadata patterns, ingredient transitions, section breaks).
2. Merge small adjacent sections that together fit within chunk_size.
3. For sections still larger than chunk_size, fall back to word-boundary
   character splitting with overlap.
4. If no meaningful boundaries are detected, fall back entirely to
   character-based chunking.
5. All chunks are stripped and non-empty, with sequential chunk_index from 0.
"""

from __future__ import annotations

import re

from backend.model.schema import Chunk
from backend.services.error import ConfigValidationError

# ---------------------------------------------------------------------------
# Recipe-boundary heuristics (general-purpose, not book-specific)
# ---------------------------------------------------------------------------

# A "title-like" line: short (≤80 chars), not ending with common sentence
# punctuation that would indicate a continuation paragraph.  Captures things
# like "ARROZ CON POLLO", "Arroz con Pollo [Chocó]", "## Soup", etc.
_TITLE_LINE_RE = re.compile(
    r"^(?!.*[,;:]$)"           # must NOT end with , ; :
    r".{1,80}$",               # at most 80 characters
    re.MULTILINE,
)

# Common recipe-metadata patterns (language-agnostic where possible):
# difficulty, time, servings, prep, cook, yield, portions, raciones, etc.
_METADATA_RE = re.compile(
    r"(?i)\b(?:dificultad|difficulty|tiempo|time|raciones|servings|serves"
    r"|prep|cook|yield|portions?|porciones)\b"
)

# Ingredient-list signals: lines starting with ■, •, -, *, or a number
# followed by a unit-like word.
_INGREDIENT_SIGNAL_RE = re.compile(
    r"^[\s]*[■•\-\*]"                           # bullet-like starts
    r"|^\s*\d+[\s./)]+\s*\w"                     # numbered list items
    r"|\b\d+\s?(?:g|kg|ml|l|oz|lb|cup|cups"
    r"|tbsp|tsp|cucharada|cucharadita|taza|libra)\b",  # quantity+unit
    re.IGNORECASE | re.MULTILINE,
)


def _find_recipe_boundaries(text: str) -> list[int]:
    """Return character offsets where a new recipe section likely begins.

    Heuristic: scan for double-newline breaks where the *next* non-blank line
    looks like a title (short, not a continuation sentence) or is followed
    within a few lines by metadata / ingredient signals.
    """
    boundaries: list[int] = []
    # Split on runs of 2+ newlines (paragraph breaks).
    gap_iter = re.finditer(r"\n\s*\n", text)

    for match in gap_iter:
        after_pos = match.end()
        # Grab the next ~300 chars to inspect.
        lookahead = text[after_pos: after_pos + 300]
        if not lookahead.strip():
            continue

        first_line = lookahead.split("\n", 1)[0].strip()

        # Check if the first line after the gap looks title-like.
        is_title = (
            len(first_line) <= 80
            and not first_line.endswith((",", ";", ":"))
            and len(first_line.split()) <= 12
        )

        # Check if nearby text has metadata or ingredient signals.
        has_metadata = bool(_METADATA_RE.search(lookahead[:200]))
        has_ingredients = bool(_INGREDIENT_SIGNAL_RE.search(lookahead[:300]))

        if is_title and (has_metadata or has_ingredients):
            boundaries.append(after_pos)

    return boundaries


def _split_by_recipe_boundaries(text: str) -> list[str] | None:
    """Try to split *text* on detected recipe boundaries.

    Returns a list of sections if meaningful boundaries were found, or
    ``None`` if boundary detection didn't produce useful splits.
    """
    boundaries = _find_recipe_boundaries(text)
    if not boundaries:
        return None

    # Build sections from boundary offsets.
    starts = [0] + boundaries
    sections: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    # If we only got one section, boundaries weren't useful.
    if len(sections) <= 1:
        return None

    return sections


# ---------------------------------------------------------------------------
# Paragraph-based fallback boundary splitting
# ---------------------------------------------------------------------------

_PARAGRAPH_BREAK_RE = re.compile(r"\n{2,}")


def _split_on_paragraphs(text: str) -> list[str]:
    """Split *text* on double-newline paragraph breaks."""
    sections = _PARAGRAPH_BREAK_RE.split(text)
    return [s.strip() for s in sections if s.strip()]


# ---------------------------------------------------------------------------
# Word-boundary character splitting (overlap-aware)
# ---------------------------------------------------------------------------


def _word_boundary_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* characters on word
    boundaries, with *overlap* characters of shared context between consecutive
    chunks."""
    results: list[str] = []
    pos = 0
    length = len(text)

    while pos < length:
        end = min(pos + chunk_size, length)

        # Back up to a word boundary if we're not at the end of the text.
        if end < length:
            space_idx = text.rfind(" ", pos, end + 1)
            if space_idx > pos:
                end = space_idx

        chunk_str = text[pos:end].strip()
        if chunk_str:
            results.append(chunk_str)

        if end >= length:
            break

        # Compute next start position with overlap.
        next_pos = max(end - overlap, pos + 1)
        if next_pos > 0 and next_pos < length and text[next_pos] != " ":
            space_before = text.rfind(" ", pos, next_pos)
            if space_before > pos:
                next_pos = space_before + 1

        pos = next_pos

    return results


# ---------------------------------------------------------------------------
# Merge helper: combine small adjacent sections that fit within chunk_size
# ---------------------------------------------------------------------------


def _merge_small_sections(sections: list[str], chunk_size: int) -> list[str]:
    """Merge consecutive small sections that together fit within *chunk_size*."""
    merged: list[str] = []
    buffer = ""
    for section in sections:
        candidate = (buffer + "\n\n" + section).strip() if buffer else section
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = section
    if buffer:
        merged.append(buffer)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    source_filename: str,
    page_number: int,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters.

    The function uses a two-tier strategy:

    1. **Recipe-boundary splitting** — heuristics look for title-like lines
       followed by metadata or ingredient signals.  If meaningful boundaries
       are found, the text is split there first.
    2. **Character-based fallback** — when no recipe boundaries are detected
       (or a boundary-derived section exceeds *chunk_size*), the text is split
       on word boundaries with *overlap* characters of shared context.

    Parameters
    ----------
    text:
        Raw text to chunk.
    source_filename:
        Name of the source file (attached as metadata).
    page_number:
        Page number within the source file (attached as metadata).
    chunk_size:
        Maximum characters per chunk (default 1000).
    overlap:
        Characters of overlap between consecutive chunks (default 200).

    Returns
    -------
    list[Chunk]
        Possibly-empty list of ``Chunk`` objects with sequential
        ``chunk_index`` starting from 0.

    Raises
    ------
    ConfigValidationError
        If *chunk_size* ≤ *overlap*.
    """
    if chunk_size <= overlap:
        raise ConfigValidationError(
            "chunk_size must be greater than overlap"
        )

    stripped = text.strip()
    if not stripped:
        return []

    # Fast path: entire text fits in one chunk.
    if len(stripped) <= chunk_size:
        return [
            Chunk(
                text=stripped,
                source_filename=source_filename,
                page_number=page_number,
                chunk_index=0,
            )
        ]

    # --- Phase 1: try recipe-boundary splitting ---
    sections = _split_by_recipe_boundaries(stripped)

    # --- Phase 2: fall back to paragraph splitting if no recipe boundaries ---
    if sections is None:
        sections = _split_on_paragraphs(stripped)

    # If paragraph splitting produced only one section (or none), treat the
    # whole text as a single section for character-based splitting.
    if len(sections) <= 1:
        sections = [stripped]

    # --- Phase 3: merge small adjacent sections ---
    merged = _merge_small_sections(sections, chunk_size)

    # --- Phase 4: sub-chunk oversized sections with word-boundary splitting ---
    raw_chunks: list[str] = []
    for section in merged:
        if len(section) <= chunk_size:
            raw_chunks.append(section)
        else:
            raw_chunks.extend(
                _word_boundary_chunks(section, chunk_size, overlap)
            )

    # --- Phase 5: build Chunk objects with sequential indices ---
    return [
        Chunk(
            text=c,
            source_filename=source_filename,
            page_number=page_number,
            chunk_index=i,
        )
        for i, c in enumerate(raw_chunks)
        if c
    ]

# ---------------------------------------------------------------------------
# Recipe block detection
# ---------------------------------------------------------------------------

COOKING_VERBS = [
    # Spanish
    "mezclar", "cocinar", "hornear", "hervir", "freír", "revolver",
    "calentar", "añadir", "agregar", "picar", "cortar", "pelar",
    "batir", "amasar", "sazonar", "servir", "preparar", "cocer",
    "saltear", "dorar", "rallar", "licuar", "colar", "escurrir",
    "incorporar", "verter", "untar", "espolvorear", "rehogar",
    # English fallback
    "mix", "cook", "bake", "boil", "fry", "stir", "heat", "add",
]

# Recipe section headers (Spanish)
_RECIPE_SECTION_WORDS = [
    "ingredientes", "preparación", "modo de hacerlo", "elaboración",
    "procedimiento", "instrucciones", "para el relleno", "para la salsa",
    "para la masa", "modo de preparación",
]

# Quantity pattern: number followed by a cooking unit (strict)
_QUANTITY_PATTERN = re.compile(
    r"\b\d+\s*"
    r"(?:g|kg|ml|l|oz|lb|cups?|tbsp|tsp"
    r"|gramos?|kilos?|litros?|mililitros?"
    r"|cucharadas?|cucharaditas?|tazas?|libras?"
    r"|unidades?|pizca|dientes?|ramitas?|hojas?)\b",
    re.IGNORECASE,
)

# Negative signals: table of contents, indices, page listings
_TOC_PATTERN = re.compile(
    r"(?:\.{3,}|…)\s*\d+",  # "........... 123" or "… 123"
)


def score_recipe_block(text: str) -> int:
    """Return a recipe relevance score based on cooking verb and quantity pattern matches.

    Penalizes table-of-contents and index-like content.
    """
    lower = text.lower()
    verb_score = sum(1 for verb in COOKING_VERBS if verb in lower)
    quantity_score = len(_QUANTITY_PATTERN.findall(lower))
    section_score = sum(2 for word in _RECIPE_SECTION_WORDS if word in lower)

    # Penalize TOC/index pages
    toc_matches = len(_TOC_PATTERN.findall(text))
    toc_penalty = min(toc_matches * 2, 10)

    return max(0, verb_score + quantity_score + section_score - toc_penalty)


def is_recipe_block(text: str, threshold: int = 5) -> bool:
    """Return True if the chunk's recipe score exceeds the threshold."""
    return score_recipe_block(text) > threshold
