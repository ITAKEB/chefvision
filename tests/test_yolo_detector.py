"""Tests for YOLO ingredient detection service.

# Feature: recipe-chunking, Property 14: YOLO detector returns valid output structure
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from backend.model.schema import DetectedItem
from backend.services.yolo_detector import YOLODetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the YOLODetector singleton before each test."""
    YOLODetector._instance = None
    yield
    YOLODetector._instance = None


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ingredient_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "")

confidence_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

bbox_coords = st.tuples(
    st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)

detection_strategy = st.lists(
    st.tuples(ingredient_names, confidence_values, bbox_coords),
    min_size=1,
    max_size=10,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mock_result(detections: list[tuple[str, float, tuple]]) -> MagicMock:
    """Build a mock YOLO result object from a list of (name, confidence, bbox)."""
    boxes = MagicMock()
    confs = [d[1] for d in detections]
    classes = list(range(len(detections)))
    bboxes = [list(d[2]) for d in detections]

    boxes.conf = torch.tensor(confs)
    boxes.cls = torch.tensor(classes, dtype=torch.int64)
    boxes.xyxy = torch.tensor(bboxes)
    boxes.__len__ = lambda self: len(confs)

    result = MagicMock()
    result.boxes = boxes
    return result


# ---------------------------------------------------------------------------
# Property-based test – Property 14: YOLO detector returns valid output structure
# ---------------------------------------------------------------------------


# Feature: recipe-chunking, Property 14: YOLO detector returns valid output structure
@settings(max_examples=100)
@given(detections=detection_strategy)
def test_yolo_detector_returns_valid_output_structure(
    detections: list[tuple[str, float, tuple]],
) -> None:
    """**Validates: Requirements 12.1**

    For any valid PIL Image input, verify returned DetectedItem objects have
    non-empty name and confidence in [0.0, 1.0].
    """
    # Build name mapping and mock result
    names = {i: d[0] for i, d in enumerate(detections)}
    mock_result = _build_mock_result(detections)

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    mock_model.names = names

    with patch.object(YOLODetector, "_load_model") as mock_load:
        # Reset singleton for each hypothesis example
        YOLODetector._instance = None
        detector = YOLODetector(model_repo="test-repo")
        detector._model = mock_model

        image = Image.new("RGB", (100, 100), color="red")
        items = detector.detect(image)

    # All returned items must be valid DetectedItem instances
    for item in items:
        assert isinstance(item, DetectedItem)
        assert isinstance(item.name, str)
        assert len(item.name) > 0, "name must be non-empty"
        assert 0.0 <= item.confidence <= 1.0, (
            f"confidence {item.confidence} not in [0.0, 1.0]"
        )
        assert isinstance(item.bbox, tuple)
        assert len(item.bbox) == 4
        for coord in item.bbox:
            assert isinstance(coord, float)


# ---------------------------------------------------------------------------
# Unit test – no detections returns empty list
# ---------------------------------------------------------------------------


def test_detect_returns_empty_list_when_no_detections() -> None:
    """When the model returns no detections, detect() returns an empty list."""
    boxes = MagicMock()
    boxes.conf = torch.tensor([])
    boxes.cls = torch.tensor([], dtype=torch.int64)
    boxes.xyxy = torch.tensor([]).reshape(0, 4)
    boxes.__len__ = lambda self: 0

    result = MagicMock()
    result.boxes = boxes

    mock_model = MagicMock()
    mock_model.return_value = [result]
    mock_model.names = {}

    with patch.object(YOLODetector, "_load_model"):
        detector = YOLODetector(model_repo="test-repo")
        detector._model = mock_model

        image = Image.new("RGB", (100, 100), color="blue")
        items = detector.detect(image)

    assert items == []
