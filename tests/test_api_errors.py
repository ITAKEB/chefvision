"""Property test for API error response mapping.

# Feature: recipe-chunking, Property 12: API error response mapping
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from backend.main import app
from backend.services.error import (
    EmbeddingDeletionError,
    PdfFormatError,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ERROR_MAP: list[tuple[str, type[Exception], int]] = [
    ("PdfFormatError", PdfFormatError, 400),
    ("EmbeddingDeletionError", EmbeddingDeletionError, 500),
]

_error_type_strategy = st.sampled_from(_ERROR_MAP)
_message_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


# ---------------------------------------------------------------------------
# Property-based test – Property 12: API error response mapping
# ---------------------------------------------------------------------------


# Feature: recipe-chunking, Property 12: API error response mapping
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    error_entry=_error_type_strategy,
    message=_message_strategy,
)
def test_api_error_response_mapping(
    error_entry: tuple[str, type[Exception], int],
    message: str,
) -> None:
    """**Validates: Requirements 9.5**

    For each backend service error type, verify the API returns the correct
    HTTP status code and a JSON body containing a ``detail`` field.
    """
    error_name, error_cls, expected_status = error_entry

    if error_cls is PdfFormatError:
        # PdfFormatError is raised during ingestion after the file is saved.
        with (
            patch("backend.api.routes.ingest_pdf", side_effect=error_cls(message)),
            patch("backend.api.routes.settings") as mock_settings,
        ):
            mock_settings.UPLOAD_DIR = Path("/tmp/chefvision_test_uploads")
            mock_settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            mock_settings.CHUNK_SIZE = 1000
            mock_settings.CHUNK_OVERLAP = 200
            mock_settings.RECIPE_THRESHOLD = 5

            response = client.post(
                "/api/files/upload",
                files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )

    elif error_cls is EmbeddingDeletionError:
        # EmbeddingDeletionError is raised during delete_embeddings.
        with (
            patch("backend.api.routes.delete_embeddings", side_effect=error_cls(message)),
            patch("backend.api.routes.settings") as mock_settings,
        ):
            # Create a temporary file so the 404 check passes.
            tmp_dir = Path("/tmp/chefvision_test_uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_file = tmp_dir / "test.pdf"
            tmp_file.write_bytes(b"%PDF-1.4 fake")
            mock_settings.UPLOAD_DIR = tmp_dir

            response = client.delete("/api/files/test.pdf")

            # Cleanup
            if tmp_file.exists():
                tmp_file.unlink()
    else:
        return  # pragma: no cover

    assert response.status_code == expected_status, (
        f"{error_name} should map to HTTP {expected_status}, "
        f"got {response.status_code}"
    )
    body = response.json()
    assert "detail" in body, f"Response body must contain 'detail' key: {body}"
    assert isinstance(body["detail"], str)
    assert len(body["detail"]) > 0


# ---------------------------------------------------------------------------
# Property-based test – file-not-found returns 404
# ---------------------------------------------------------------------------


# Feature: recipe-chunking, Property 12: API error response mapping
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    filename=st.from_regex(r"[a-z][a-z0-9_]{1,20}\.pdf", fullmatch=True),
)
def test_file_not_found_returns_404(filename: str) -> None:
    """**Validates: Requirements 9.5**

    For any filename that does not exist on disk, the DELETE and GET status
    endpoints should return HTTP 404 with a JSON body containing ``detail``.
    """
    with patch("backend.api.routes.settings") as mock_settings:
        tmp_dir = Path("/tmp/chefvision_test_empty")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        mock_settings.UPLOAD_DIR = tmp_dir

        # DELETE endpoint
        resp_delete = client.delete(f"/api/files/{filename}")
        assert resp_delete.status_code == 404
        body_delete = resp_delete.json()
        assert "detail" in body_delete
        assert isinstance(body_delete["detail"], str)

        # GET status endpoint
        resp_status = client.get(f"/api/files/{filename}/status")
        assert resp_status.status_code == 404
        body_status = resp_status.json()
        assert "detail" in body_status
        assert isinstance(body_status["detail"], str)


# ---------------------------------------------------------------------------
# Property-based test – invalid file format returns 400
# ---------------------------------------------------------------------------


# Feature: recipe-chunking, Property 12: API error response mapping
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    ext=st.from_regex(r"\.[a-z]{1,5}", fullmatch=True).filter(
        lambda e: e.lower() != ".pdf"
    ),
)
def test_non_pdf_upload_returns_400(ext: str) -> None:
    """**Validates: Requirements 9.5**

    For any file whose extension is not .pdf, the upload endpoint should
    return HTTP 400 with a JSON body containing ``detail``.
    """
    response = client.post(
        "/api/files/upload",
        files={"file": (f"file{ext}", b"not a pdf", "application/octet-stream")},
    )
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], str)
