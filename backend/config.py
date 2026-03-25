"""
Modulo de configuracion de RAG
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Setting(BaseSettings):
  BASE_DIR: Path = Path(__file__).resolve().parents[1]
  DATA_DIR: Path = BASE_DIR / "data"
  CHROMA_DIR: Path = DATA_DIR / "chroma"
  UPLOAD_DIR: Path = DATA_DIR / "uploads"
  
  OPEN_API_KEY: str = "ollama"

  LLM_BASE_URL: str = "http://localhost:11434/v1"
  LLM_MODEL: str = "gemma3:4b"

  EMBEDDING_MODEL: str = "mxbai-embed-large"

  TOP_K: int = 10
  MIN_RELEVANCE_SCORE: float = 0.3

  CHUNK_SIZE: int = 2500
  CHUNK_OVERLAP: int = 300
  RECIPE_THRESHOLD: int = 3

  # YOLO model settings
  YOLO_MODEL_REPO: str = "darien-or/chefVision"
  YOLO_MODEL_FILENAME: str = "best.pt"
  YOLO_CONFIDENCE_THRESHOLD: float = 0.2

  # Translation settings
  INGREDIENT_DICT_PATH: Path | None = None

  # Debug mode — shows raw chunks in the UI
  DEBUG_MODE: bool = True

  class Config:
    env_file = ".env"

settings = Setting()

# Ensure required directories exist on import
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)