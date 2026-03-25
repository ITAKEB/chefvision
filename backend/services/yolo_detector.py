"""YOLO-based ingredient detection service.

Loads a YOLO model from HuggingFace Hub and runs inference on images
to detect groceries, fruits, and vegetables.
"""

from __future__ import annotations

import logging

from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from backend.config import settings
from backend.model.schema import DetectedItem
from backend.services.error import DetectionError, ModelLoadError

logger = logging.getLogger(__name__)


class YOLODetector:
    """Singleton service that loads a YOLO model from HuggingFace and detects
    groceries, fruits, and vegetables in images."""

    _instance: YOLODetector | None = None

    def __new__(cls, model_repo: str | None = None) -> YOLODetector:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_repo = model_repo or settings.YOLO_MODEL_REPO
            cls._instance._model = None
        return cls._instance

    def __init__(self, model_repo: str | None = None) -> None:
        # Only set on first init; singleton reuses existing state.
        if not hasattr(self, "_initialized"):
            self._model_repo: str = model_repo or settings.YOLO_MODEL_REPO
            self._model: YOLO | None = None
            self._initialized = True

    def _load_model(self) -> None:
        """Download and cache the YOLO model from HuggingFace, then load it.

        Raises:
            ModelLoadError: If the model cannot be downloaded or loaded.
        """
        try:
            model_path = hf_hub_download(
                repo_id=self._model_repo,
                filename=settings.YOLO_MODEL_FILENAME,
            )
            self._model = YOLO(model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load YOLO model from '{self._model_repo}': {exc}"
            ) from exc

    def detect(self, image: Image.Image) -> list[DetectedItem]:
        """Run inference on an image and return detected items.

        Args:
            image: A PIL Image to run detection on.

        Returns:
            A list of DetectedItem with name, confidence, and bbox.
            Returns an empty list if no ingredients are detected.

        Raises:
            DetectionError: If inference fails.
        """
        if self._model is None:
            self._load_model()

        try:
            results = self._model(image)  # type: ignore[misc]
        except Exception as exc:
            raise DetectionError(
                f"YOLO inference failed: {exc}"
            ) from exc

        detected: list[DetectedItem] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                if confidence < settings.YOLO_CONFIDENCE_THRESHOLD:
                    continue
                cls_idx = int(boxes.cls[i])
                name = self._model.names.get(cls_idx, f"class_{cls_idx}")  # type: ignore[union-attr]
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detected.append(
                    DetectedItem(
                        name=name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detected

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return self._model is not None
