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
  
  OPEN_API_KEY: str | None = None

  EMBEDDING_MODEL: str = "text-embedding3-small"
  LLM_MODEL: str = "gpt-4-o-mini"

  TOP_K: int = 20
  MIN_RELEVANCE_SCORE: float = 0.4

  CHUNK_SIZE: int = 1000
  CHUNK_OVERLAP: int = 200
  RECIPE_THRESHOLD: int = 5

  class Config:
    env_file = ".env"

settings = Setting()

# Ensure required directories exist on import
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)