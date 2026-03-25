"""Data models for the recipe chunking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from backend.services.error import ConfigValidationError


class EmbeddingStatus(str, Enum):
    """Per-file indicator of embedding state in the vector store."""

    EMBEDDED = "embedded"
    NOT_EMBEDDED = "not embedded"
    IN_PROGRESS = "in progress"


@dataclass
class PageText:
    """A single page's extracted text paired with its page number."""

    page_number: int
    text: str


@dataclass
class Chunk:
    """A text segment produced by the chunking engine with source metadata."""

    text: str
    source_filename: str
    page_number: int
    chunk_index: int

    @property
    def chunk_id(self) -> str:
        """Deterministic identifier derived from source filename and chunk index."""
        return f"{self.source_filename}_{self.chunk_index}"


@dataclass
class ChunkingConfig:
    """Configuration for the chunking engine with sensible defaults."""

    chunk_size: int = 1000
    overlap: int = 200
    recipe_threshold: int = 5

    def validate(self) -> None:
        """Raise ConfigValidationError if chunk_size <= overlap."""
        if self.chunk_size <= self.overlap:
            raise ConfigValidationError(
                "chunk_size must be greater than overlap"
            )


@dataclass
class EmbeddingResult:
    """Result of an embedding operation with success/error counts."""

    success_count: int
    error_count: int
    errors: list[str]


@dataclass
class IngestionResult:
    """Result of the full ingestion pipeline for a single file."""

    filename: str
    status: EmbeddingStatus
    chunks_processed: int
    error_message: str | None = None


@dataclass
class DetectedItem:
    """A single ingredient detected by the YOLO model."""

    name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class TranslatedIngredient:
    """An ingredient with both English and Spanish names."""

    name_en: str
    name_es: str
    translated: bool


@dataclass
class SearchResult:
    """A recipe chunk returned from semantic search."""

    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    relevance_score: float

