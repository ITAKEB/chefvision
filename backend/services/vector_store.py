"""Vector store service for managing ChromaDB embeddings."""

from __future__ import annotations

import logging

import chromadb

from backend.config import settings
from backend.model.schema import Chunk, EmbeddingResult
from backend.services.error import EmbeddingDeletionError

logger = logging.getLogger(__name__)

COLLECTION_NAME = "recipe_chunks"


def _get_collection() -> chromadb.Collection:
    """Return the recipe_chunks ChromaDB collection (persistent client)."""
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_chunks(chunks: list[Chunk], filename: str) -> EmbeddingResult:
    """Generate embeddings and upsert chunks into ChromaDB.

    Uses deterministic IDs: {filename}_{chunk_index}.
    Logs per-chunk errors and continues processing remaining chunks.
    Returns EmbeddingResult with success count and errors.
    """
    collection = _get_collection()
    success_count = 0
    errors: list[str] = []

    for chunk in chunks:
        chunk_id = chunk.chunk_id
        try:
            collection.upsert(
                ids=[chunk_id],
                documents=[chunk.text],
                metadatas=[
                    {
                        "source_filename": chunk.source_filename,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    }
                ],
            )
            success_count += 1
        except Exception as exc:
            error_msg = f"Failed to embed chunk {chunk_id}: {exc}"
            logger.error(error_msg)
            errors.append(error_msg)

    return EmbeddingResult(
        success_count=success_count,
        error_count=len(errors),
        errors=errors,
    )


def delete_embeddings(filename: str) -> None:
    """Delete all chunk embeddings associated with a filename.

    Queries by metadata filter to find all chunk IDs for the file,
    then deletes them. Raises EmbeddingDeletionError on failure.
    """
    try:
        collection = _get_collection()
        results = collection.get(
            where={"source_filename": filename},
        )
        ids_to_delete = results["ids"]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
    except Exception as exc:
        raise EmbeddingDeletionError(
            f"Failed to delete embeddings for {filename}: {exc}"
        ) from exc


def has_embeddings(filename: str) -> bool:
    """Check if any embeddings exist for the given filename."""
    collection = _get_collection()
    results = collection.get(
        where={"source_filename": filename},
    )
    return len(results["ids"]) > 0
