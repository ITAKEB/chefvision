from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from backend.api.models import (
    DeleteResponse,
    FileInfo,
    IngestionResponse,
    StatusResponse,
)
from backend.config import settings
from backend.model.schema import ChunkingConfig, EmbeddingStatus
from backend.services.error import EmbeddingDeletionError, PdfFormatError
from backend.services.ingestion import ingest_pdf
from backend.services.vector_store import delete_embeddings, has_embeddings

router = APIRouter()


@router.post("/files/upload")
async def upload_file(file: UploadFile) -> IngestionResponse:
    """Accept PDF upload, save to data dir, trigger ingestion pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    saved_path = settings.UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        saved_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {exc}"
        ) from exc

    config = ChunkingConfig(
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP,
        recipe_threshold=settings.RECIPE_THRESHOLD,
    )

    try:
        result = ingest_pdf(saved_path, config)
    except PdfFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Ingestion failed: {exc}"
        ) from exc

    if (
        result.status == EmbeddingStatus.EMBEDDED
        and result.chunks_processed == 0
    ):
        raise HTTPException(
            status_code=409,
            detail=f"File '{file.filename}' is already embedded.",
        )

    return IngestionResponse(
        filename=result.filename,
        status=result.status.value,
        chunks_processed=result.chunks_processed,
        error_message=result.error_message,
    )


@router.get("/files")
async def list_files() -> list[FileInfo]:
    """Return all uploaded files with metadata and embedding status."""
    files: list[FileInfo] = []
    upload_dir = settings.UPLOAD_DIR

    if not upload_dir.exists():
        return files

    for path in sorted(upload_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue

        stat = path.stat()
        size_kb = round(stat.st_size / 1024, 2)
        uploaded = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()
        embedded = has_embeddings(path.name)

        files.append(
            FileInfo(
                name=path.name,
                size_kb=size_kb,
                uploaded=uploaded,
                embedded=embedded,
            )
        )

    return files


@router.delete("/files/{filename}")
async def delete_file(filename: str) -> DeleteResponse:
    """Remove file from disk and delete embeddings from ChromaDB."""
    file_path = settings.UPLOAD_DIR / filename

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    error_message: str | None = None
    try:
        delete_embeddings(filename)
    except EmbeddingDeletionError as exc:
        error_message = str(exc)

    try:
        file_path.unlink()
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file from disk: {exc}",
        ) from exc

    if error_message:
        raise HTTPException(status_code=500, detail=error_message)

    return DeleteResponse(filename=filename, deleted=True)


@router.get("/files/{filename}/status")
async def file_status(filename: str) -> StatusResponse:
    """Return current embedding status for a specific file."""
    file_path = settings.UPLOAD_DIR / filename

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    embedded = has_embeddings(filename)
    status = "embedded" if embedded else "not embedded"

    return StatusResponse(filename=filename, status=status)
