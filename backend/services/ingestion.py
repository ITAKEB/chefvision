from pathlib import Path

import pypdf

from backend.model.schema import (
    ChunkingConfig,
    EmbeddingStatus,
    IngestionResult,
    PageText,
)
from backend.services.chunking import chunk_text, is_recipe_block
from backend.services.error import PdfFormatError, PdfReadError, RecipeChunkingError
from backend.services.vector_store import embed_chunks, has_embeddings


def extract_text_from_pdf(pdf_path: Path) -> list[PageText]:
    """Extract text from each page of a PDF.

    Returns a list of PageText for pages with non-empty text.
    Raises PdfFormatError for non-PDF files and PdfReadError for corrupt PDFs.
    """
    if pdf_path.suffix.lower() != ".pdf":
        raise PdfFormatError(
            f"Invalid file format: '{pdf_path.suffix}'. Only .pdf files are accepted."
        )

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:
        raise PdfReadError(f"Failed to read PDF '{pdf_path.name}': {exc}") from exc

    pages: list[PageText] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        pages.append(PageText(page_number=i + 1, text=text))

    return pages

def ingest_pdf(pdf_path: Path, config: ChunkingConfig) -> IngestionResult:
    """Full pipeline: extract text, chunk, filter recipe blocks, embed.

    Updates embedding status throughout.
    Returns IngestionResult with status and any error messages.
    """
    filename = pdf_path.name

    # Step 0: Check for duplicate embeddings — skip if already embedded.
    try:
        if has_embeddings(filename):
            return IngestionResult(
                filename=filename,
                status=EmbeddingStatus.EMBEDDED,
                chunks_processed=0,
            )
    except Exception as exc:
        return IngestionResult(
            filename=filename,
            status=EmbeddingStatus.NOT_EMBEDDED,
            chunks_processed=0,
            error_message=f"Failed to check existing embeddings: {exc}",
        )

    # Mark as in progress.
    result = IngestionResult(
        filename=filename,
        status=EmbeddingStatus.IN_PROGRESS,
        chunks_processed=0,
    )

    # Step 1: Validate config.
    try:
        config.validate()
    except RecipeChunkingError as exc:
        result.status = EmbeddingStatus.NOT_EMBEDDED
        result.error_message = f"Invalid configuration: {exc}"
        return result

    # Step 2: Extract text from PDF.
    try:
        pages = extract_text_from_pdf(pdf_path)
    except RecipeChunkingError as exc:
        result.status = EmbeddingStatus.NOT_EMBEDDED
        result.error_message = f"PDF extraction failed: {exc}"
        return result

    if not pages:
        result.status = EmbeddingStatus.NOT_EMBEDDED
        result.error_message = "No text content found in PDF."
        return result

    # Step 3: Concatenate all pages into one text, then chunk.
    # This prevents recipes spanning multiple pages from being split.
    full_text = "\n\n".join(page.text for page in pages)

    all_chunks = chunk_text(
        text=full_text,
        source_filename=filename,
        page_number=1,  # page_number is approximate for concatenated text
        chunk_size=config.chunk_size,
        overlap=config.overlap,
    )

    # Filter for recipe blocks only.
    recipe_chunks = [
        chunk for chunk in all_chunks
        if is_recipe_block(chunk.text, config.recipe_threshold)
    ]

    if not recipe_chunks:
        result.status = EmbeddingStatus.NOT_EMBEDDED
        result.error_message = "No recipe content detected in PDF."
        return result

    # Step 4: Embed the filtered recipe chunks.
    try:
        embedding_result = embed_chunks(recipe_chunks, filename)
    except Exception as exc:
        result.status = EmbeddingStatus.NOT_EMBEDDED
        result.error_message = f"Embedding failed: {exc}"
        return result

    result.chunks_processed = embedding_result.success_count

    if embedding_result.success_count == 0:
        result.status = EmbeddingStatus.NOT_EMBEDDED
        result.error_message = (
            f"All {embedding_result.error_count} chunks failed to embed: "
            + "; ".join(embedding_result.errors)
        )
        return result

    if embedding_result.error_count > 0:
        result.status = EmbeddingStatus.EMBEDDED
        result.error_message = (
            f"{embedding_result.error_count} chunks failed to embed: "
            + "; ".join(embedding_result.errors)
        )
        return result

    result.status = EmbeddingStatus.EMBEDDED
    return result

