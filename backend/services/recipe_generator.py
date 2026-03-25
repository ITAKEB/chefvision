"""Recipe generation service using Ollama LLM.

Takes detected ingredients, relevant recipe chunks from the vector store,
and the meal type to generate a personalized recipe suggestion.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

MEAL_TYPE_MAP = {
    "Desayuno": "desayuno (breakfast) — ligero y energético",
    "Almuerzo": "almuerzo (lunch) — plato principal, contundente",
    "Cena": "cena (dinner) — ligero y reconfortante",
}


def generate_recipe(
    ingredients_es: list[str],
    ingredients_en: list[str],
    meal_type: str,
    recipe_chunks: list[str],
) -> str:
    """Generate a recipe using Ollama based on ingredients and reference chunks.

    Args:
        ingredients_es: Spanish ingredient names.
        ingredients_en: English ingredient names.
        meal_type: Meal type (Desayuno, Almuerzo, Cena).
        recipe_chunks: Relevant recipe text chunks from the vector store.

    Returns:
        Generated recipe text in Spanish.
    """
    meal_desc = MEAL_TYPE_MAP.get(meal_type, meal_type)
    ingredients_list = ", ".join(
        f"{es} ({en})" for es, en in zip(ingredients_es, ingredients_en)
    )

    # Build context from top recipe chunks (limit to avoid token overflow)
    context_chunks = recipe_chunks[:5]
    context_text = "\n\n---\n\n".join(context_chunks)
    if len(context_text) > 4000:
        context_text = context_text[:4000] + "…"

    prompt = f"""Eres un chef experto que vas a instruir al usuario a preparar una comida con los alimentos que tiene. El usuario tiene estos ingredientes disponibles:
{ingredients_list}

Momento del día: {meal_desc}

Aquí hay algunas recetas de referencia de libros de cocina que podrían servir de inspiración:

{context_text}

---

Basándote en los ingredientes disponibles y las recetas de referencia, genera UNA receta completa y práctica que el usuario pueda preparar. La receta debe:

1. Usar principalmente los ingredientes disponibles (puedes asumir que tienen ingredientes básicos como sal, aceite, agua)
2. Ser apropiada para {meal_desc}
3. Estar inspirada en las recetas de referencia pero adaptada a los ingredientes disponibles
4. Incluir: nombre del plato, ingredientes con cantidades, y pasos de preparación claros

Responde SOLO con la receta en español, sin explicaciones adicionales."""

    try:
        client = OpenAI(
            api_key=settings.OPEN_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "Eres un chef experto que crea recetas prácticas y deliciosas basadas en los ingredientes disponibles del usuario."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1500,
        )

        return response.choices[0].message.content or "No se pudo generar la receta."

    except Exception as exc:
        logger.error("Recipe generation failed: %s", exc)
        return f"Error al generar la receta: {exc}"
