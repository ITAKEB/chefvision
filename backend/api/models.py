from pydantic import BaseModel


# --- File management models ---


class FileInfo(BaseModel):
    name: str
    size_kb: float
    uploaded: str
    embedded: bool


class IngestionResponse(BaseModel):
    filename: str
    status: str
    chunks_processed: int
    error_message: str | None = None


class DeleteResponse(BaseModel):
    filename: str
    deleted: bool
    error_message: str | None = None


class StatusResponse(BaseModel):
    filename: str
    status: str


# --- Ingredient detection models ---


class DetectedIngredient(BaseModel):
    name_en: str
    name_es: str
    confidence: float


class DetectionResponse(BaseModel):
    ingredients: list[DetectedIngredient]
    error_message: str | None = None


class RecipeSearchRequest(BaseModel):
    ingredients: list[str]
    meal_type: str | None = None
    top_k: int = 20


class RecipeResult(BaseModel):
    text: str
    source_filename: str
    page_number: int
    relevance_score: float


class RecipeSearchResponse(BaseModel):
    results: list[RecipeResult]
    query_terms: list[str]


class GenerateRecipeRequest(BaseModel):
    ingredients_en: list[str]
    ingredients_es: list[str]
    meal_type: str = "Almuerzo"
    top_k: int = 5


class GenerateRecipeResponse(BaseModel):
    recipe: str
    source_chunks: list[RecipeResult]
    debug_chunks: list[str] | None = None
