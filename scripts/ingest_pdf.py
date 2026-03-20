"""Ingest a PDF file through the recipe chunking pipeline."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so backend imports resolve.
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from backend.config import settings
from backend.model.schema import ChunkingConfig
from backend.services.ingestion import ingest_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_pdf.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    config = ChunkingConfig(
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP,
        recipe_threshold=settings.RECIPE_THRESHOLD,
    )

    print(f"Processing {pdf_path.name}...")
    result = ingest_pdf(pdf_path, config)
    print(f"Status: {result.status.value}")
    print(f"Chunks processed: {result.chunks_processed}")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    main()
