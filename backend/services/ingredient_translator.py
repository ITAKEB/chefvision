"""Ingredient translation service using a static dictionary.

Translates detected ingredient names from English to Spanish using a
built-in dictionary. Falls back to the original English name for unknown entries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.model.schema import TranslatedIngredient

logger = logging.getLogger(__name__)

# Static dictionary for common ingredients
_FALLBACK_DICTIONARY: dict[str, str] = {
    "apple": "manzana", "banana": "banano", "orange": "naranja",
    "lemon": "limón", "strawberry": "fresa", "grape": "uva",
    "pineapple": "piña", "watermelon": "sandía", "mango": "mango",
    "peach": "melocotón", "pear": "pera", "cherry": "cereza",
    "avocado": "aguacate", "coconut": "coco", "lime": "lima",
    "tomato": "tomate", "onion": "cebolla", "garlic": "ajo",
    "potato": "papa", "carrot": "zanahoria", "pepper": "pimiento",
    "lettuce": "lechuga", "cucumber": "pepino", "broccoli": "brócoli",
    "spinach": "espinaca", "corn": "maíz", "mushroom": "champiñón",
    "eggplant": "berenjena", "zucchini": "calabacín", "cabbage": "repollo",
    "chicken": "pollo", "beef": "carne de res", "pork": "cerdo",
    "fish": "pescado", "egg": "huevo", "rice": "arroz",
    "cheese": "queso", "milk": "leche", "butter": "mantequilla",
    "oil": "aceite", "salt": "sal", "sugar": "azúcar",
    "flour": "harina", "pasta": "pasta", "bread": "pan",
    "sausage": "salchicha", "ham": "jamón", "bacon": "tocino",
    "shrimp": "camarón", "tuna": "atún", "salmon": "salmón",
    "celery": "apio", "parsley": "perejil", "cilantro": "cilantro",
    "ginger": "jengibre", "cinnamon": "canela", "oregano": "orégano",
    "basil": "albahaca", "thyme": "tomillo", "rosemary": "romero",
    "honey": "miel", "vinegar": "vinagre", "cream": "crema",
    "yogurt": "yogur", "chocolate": "chocolate", "vanilla": "vainilla",
    "almond": "almendra", "walnut": "nuez", "peanut": "cacahuate",
    "bean": "frijol", "lentil": "lenteja", "chickpea": "garbanzo",
    "pea": "guisante", "olive": "aceituna", "radish": "rábano",
    "beet": "remolacha", "sweet potato": "camote", "squash": "calabaza",
}


class IngredientTranslator:
    """Translates ingredient names from English to Spanish using a static dictionary."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = dict(_FALLBACK_DICTIONARY)

    def translate(self, name_en: str) -> str:
        """Translate a single ingredient name from English to Spanish."""
        key = name_en.lower().strip()
        return self._cache.get(key, name_en)

    def translate_batch(self, names_en: list[str]) -> list[TranslatedIngredient]:
        """Translate a list of English ingredient names to Spanish."""
        if not names_en:
            return []

        results: list[TranslatedIngredient] = []
        for name in names_en:
            key = name.lower().strip()
            name_es = self._cache.get(key, name)
            results.append(TranslatedIngredient(
                name_en=name,
                name_es=name_es,
                translated=key in self._cache,
            ))
        return results
