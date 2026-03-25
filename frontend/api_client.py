"""
API client helper for communicating with the ChefVision backend.

All frontend-to-backend communication goes through these functions,
using the requests library over HTTP.
"""

import os
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def is_debug_mode() -> bool:
    """Check if backend debug mode is enabled."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/config/debug", timeout=3)
        return r.json().get("debug_mode", False)
    except Exception:
        return False


def upload_pdf(file) -> dict:
    """Upload a PDF file to the backend via POST /api/files/upload.

    Args:
        file: A Streamlit UploadedFile object with .name and .getbuffer() methods.

    Returns:
        Response JSON dict with upload/ingestion result.

    Raises:
        requests.HTTPError: If the backend returns an error status code.
    """
    response = requests.post(
        f"{BACKEND_URL}/api/files/upload",
        files={"file": (file.name, file.getbuffer(), "application/pdf")},
        timeout=300,  # Embedding can take a while for large PDFs
    )
    response.raise_for_status()
    return response.json()


def list_files() -> list[dict]:
    """Fetch all uploaded files with metadata and embedding status.

    Returns:
        List of file info dicts from GET /api/files.

    Raises:
        requests.HTTPError: If the backend returns an error status code.
    """
    response = requests.get(f"{BACKEND_URL}/api/files")
    response.raise_for_status()
    return response.json()


def delete_file(filename: str) -> dict:
    """Delete a file and its embeddings via DELETE /api/files/{filename}.

    Args:
        filename: Name of the file to delete.

    Returns:
        Response JSON dict with deletion result.

    Raises:
        requests.HTTPError: If the backend returns an error status code.
    """
    response = requests.delete(f"{BACKEND_URL}/api/files/{filename}")
    response.raise_for_status()
    return response.json()


def get_file_status(filename: str) -> dict:
    """Get the embedding status of a specific file.

    Args:
        filename: Name of the file to check.

    Returns:
        Response JSON dict with status info from GET /api/files/{filename}/status.

    Raises:
        requests.HTTPError: If the backend returns an error status code.
    """
    response = requests.get(f"{BACKEND_URL}/api/files/{filename}/status")
    response.raise_for_status()
    return response.json()


def detect_ingredients(image_file) -> dict:
    """Send an image to the backend for YOLO ingredient detection.

    Args:
        image_file: A Streamlit UploadedFile object with .name, .getbuffer(), and .type.

    Returns:
        Response JSON dict with detected ingredients.

    Raises:
        requests.HTTPError: If the backend returns an error status code.
    """
    response = requests.post(
        f"{BACKEND_URL}/api/detect-ingredients",
        files={"file": (image_file.name, image_file.getbuffer(), image_file.type)},
    )
    response.raise_for_status()
    return response.json()


def search_recipes(ingredients: list[str], meal_type: str = None) -> dict:
    """Search for recipes matching the given ingredients.

    Args:
        ingredients: List of ingredient names to search for.
        meal_type: Optional meal type filter.

    Returns:
        Response JSON dict with recipe search results.

    Raises:
        requests.HTTPError: If the backend returns an error status code.
    """
    payload = {"ingredients": ingredients}
    if meal_type:
        payload["meal_type"] = meal_type
    response = requests.post(f"{BACKEND_URL}/api/search-recipes", json=payload)
    response.raise_for_status()
    return response.json()


def generate_recipe(
    ingredients_en: list[str],
    ingredients_es: list[str],
    meal_type: str = "Almuerzo",
) -> dict:
    """Generate a recipe using LLM based on detected ingredients and meal type.

    Args:
        ingredients_en: English ingredient names.
        ingredients_es: Spanish ingredient names.
        meal_type: Meal type (Desayuno, Almuerzo, Cena).

    Returns:
        Response JSON dict with generated recipe and source chunks.

    Raises:
        requests.HTTPError: If the backend returns an error status code.
    """
    payload = {
        "ingredients_en": ingredients_en,
        "ingredients_es": ingredients_es,
        "meal_type": meal_type,
    }
    response = requests.post(
        f"{BACKEND_URL}/api/generate-recipe",
        json=payload,
        timeout=120,  # LLM generation can take a while
    )
    response.raise_for_status()
    return response.json()
