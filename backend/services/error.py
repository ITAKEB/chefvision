"""Custom error types for the recipe chunking pipeline."""


class RecipeChunkingError(Exception):
    """Base error for recipe chunking pipeline."""


class PdfFormatError(RecipeChunkingError):
    """Raised when a file is not a valid PDF (wrong extension or magic bytes)."""


class PdfReadError(RecipeChunkingError):
    """Raised when a PDF is corrupted or unreadable."""


class ConfigValidationError(RecipeChunkingError):
    """Raised when chunking configuration is invalid (e.g. chunk_size <= overlap)."""


class EmbeddingError(RecipeChunkingError):
    """Raised when a ChromaDB embedding operation fails for a chunk."""


class EmbeddingDeletionError(RecipeChunkingError):
    """Raised when ChromaDB deletion of embeddings fails."""
