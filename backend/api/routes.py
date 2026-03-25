import io
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image

from backend.api.models import (
    DeleteResponse,
    DetectedIngredient,
    DetectionResponse,
    FileInfo,
    GenerateRecipeRequest,
    GenerateRecipeResponse,
    IngestionResponse,
    RecipeResult,
    RecipeSearchRequest,
    RecipeSearchResponse,
    StatusResponse,
)
from backend.config import settings
from backend.model.schema import ChunkingConfig, EmbeddingStatus
from backend.services.error import (
    DetectionError,
    EmbeddingDeletionError,
    ModelLoadError,
    PdfFormatError,
)
from backend.services.ingredient_translator import IngredientTranslator
from backend.services.ingestion import ingest_pdf
from backend.services.recipe_generator import generate_recipe
from backend.services.search import search_recipes as search_recipes_service
from backend.services.vector_store import delete_embeddings, has_embeddings
from backend.services.yolo_detector import YOLODetector

router = APIRouter()

_translator = IngredientTranslator()
_detector = YOLODetector()


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

@router.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile) -> DetectionResponse:
    """Accept an image, run YOLO detection, return identified ingredients."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Please upload an image file.",
        )

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        ) from exc

    try:
        detected_items = _detector.detect(image)
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DetectionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ingredients: list[DetectedIngredient] = []
    for item in detected_items:
        name_es = _translator.translate(item.name)
        ingredients.append(
            DetectedIngredient(
                name_en=item.name,
                name_es=name_es,
                confidence=item.confidence,
            )
        )

    return DetectionResponse(ingredients=ingredients)


@router.post("/search-recipes")
async def search_recipes(request: RecipeSearchRequest) -> RecipeSearchResponse:
    """Accept ingredient names, translate to Spanish, search ChromaDB."""
    translated = _translator.translate_batch(request.ingredients)
    query_terms = [t.name_es for t in translated]

    results = search_recipes_service(
        query_terms=query_terms,
        top_k=request.top_k,
    )

    recipe_results = [
        RecipeResult(
            text=r.text,
            source_filename=r.source_filename,
            page_number=r.page_number,
            relevance_score=r.relevance_score,
        )
        for r in results
    ]

    return RecipeSearchResponse(results=recipe_results, query_terms=query_terms)


@router.post("/generate-recipe")
async def generate_recipe_endpoint(request: GenerateRecipeRequest) -> GenerateRecipeResponse:
    """Search for relevant recipe chunks, then generate a recipe with the LLM."""
    translated = _translator.translate_batch(request.ingredients_en)
    query_terms = [t.name_es for t in translated]

    results = search_recipes_service(
        query_terms=query_terms,
        top_k=request.top_k,
    )

    chunk_texts = [r.text for r in results]

    # Use provided Spanish names or fall back to translated ones
    ingredients_es = request.ingredients_es if request.ingredients_es else query_terms

    recipe_text = generate_recipe(
        ingredients_es=ingredients_es,
        ingredients_en=request.ingredients_en,
        meal_type=request.meal_type,
        recipe_chunks=chunk_texts,
    )

    recipe_results = [
        RecipeResult(
            text=r.text,
            source_filename=r.source_filename,
            page_number=r.page_number,
            relevance_score=r.relevance_score,
        )
        for r in results[:3]
    ]

    # Include raw chunks in debug mode
    debug_chunks = chunk_texts[:5] if settings.DEBUG_MODE else None

    return GenerateRecipeResponse(
        recipe=recipe_text,
        source_chunks=recipe_results,
        debug_chunks=debug_chunks,
    )

