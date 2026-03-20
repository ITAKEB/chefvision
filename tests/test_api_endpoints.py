"""Unit tests for API endpoints using FastAPI TestClient.

Tests upload, list, delete, and status endpoints, including error responses
for invalid file format, missing file, and duplicate upload.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.model.schema import EmbeddingStatus, IngestionResult
from backend.services.error import EmbeddingDeletionError, PdfFormatError

client = TestClient(app)


@pytest.fixture()
def upload_dir(tmp_path: Path):
    """Provide a temporary upload directory and patch settings to use it."""
    d = tmp_path / "uploads"
    d.mkdir()
    with patch("backend.api.routes.settings") as mock_settings:
        mock_settings.UPLOAD_DIR = d
        mock_settings.CHUNK_SIZE = 1000
        mock_settings.CHUNK_OVERLAP = 200
        mock_settings.RECIPE_THRESHOLD = 5
        yield d


# ---------------------------------------------------------------------------
# POST /api/files/upload
# ---------------------------------------------------------------------------


class TestUploadFile:
    """Requirement 9.1 – POST /api/files/upload"""

    def test_successful_upload(self, upload_dir: Path) -> None:
        result = IngestionResult(
            filename="recipe.pdf",
            status=EmbeddingStatus.EMBEDDED,
            chunks_processed=5,
        )
        with patch("backend.api.routes.ingest_pdf", return_value=result):
            resp = client.post(
                "/api/files/upload",
                files={"file": ("recipe.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "recipe.pdf"
        assert body["status"] == "embedded"
        assert body["chunks_processed"] == 5
        assert body["error_message"] is None

    def test_upload_saves_file_to_disk(self, upload_dir: Path) -> None:
        result = IngestionResult(
            filename="saved.pdf",
            status=EmbeddingStatus.EMBEDDED,
            chunks_processed=1,
        )
        with patch("backend.api.routes.ingest_pdf", return_value=result):
            client.post(
                "/api/files/upload",
                files={"file": ("saved.pdf", b"%PDF-content", "application/pdf")},
            )

        assert (upload_dir / "saved.pdf").exists()
        assert (upload_dir / "saved.pdf").read_bytes() == b"%PDF-content"

    def test_upload_non_pdf_returns_400(self, upload_dir: Path) -> None:
        resp = client.post(
            "/api/files/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400
        assert "detail" in resp.json()

    def test_upload_no_extension_returns_400(self, upload_dir: Path) -> None:
        resp = client.post(
            "/api/files/upload",
            files={"file": ("noext", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_duplicate_returns_409(self, upload_dir: Path) -> None:
        result = IngestionResult(
            filename="dup.pdf",
            status=EmbeddingStatus.EMBEDDED,
            chunks_processed=0,
        )
        with patch("backend.api.routes.ingest_pdf", return_value=result):
            resp = client.post(
                "/api/files/upload",
                files={"file": ("dup.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )

        assert resp.status_code == 409
        assert "already embedded" in resp.json()["detail"]

    def test_upload_pdf_format_error_returns_400(self, upload_dir: Path) -> None:
        with patch(
            "backend.api.routes.ingest_pdf",
            side_effect=PdfFormatError("bad format"),
        ):
            resp = client.post(
                "/api/files/upload",
                files={"file": ("bad.pdf", b"%PDF-broken", "application/pdf")},
            )

        assert resp.status_code == 400
        assert "bad format" in resp.json()["detail"]

    def test_upload_unexpected_error_returns_500(self, upload_dir: Path) -> None:
        with patch(
            "backend.api.routes.ingest_pdf",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.post(
                "/api/files/upload",
                files={"file": ("err.pdf", b"%PDF-1.4", "application/pdf")},
            )

        assert resp.status_code == 500
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# GET /api/files
# ---------------------------------------------------------------------------


class TestListFiles:
    """Requirement 9.2 – GET /api/files"""

    def test_list_empty_directory(self, upload_dir: Path) -> None:
        with patch("backend.api.routes.has_embeddings", return_value=False):
            resp = client.get("/api/files")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_pdf_files(self, upload_dir: Path) -> None:
        (upload_dir / "a.pdf").write_bytes(b"%PDF-1.4 content")
        (upload_dir / "b.pdf").write_bytes(b"%PDF-1.4 more content")
        # Non-PDF should be excluded
        (upload_dir / "readme.txt").write_text("ignore me")

        with patch("backend.api.routes.has_embeddings", return_value=True):
            resp = client.get("/api/files")

        assert resp.status_code == 200
        body = resp.json()
        names = [f["name"] for f in body]
        assert "a.pdf" in names
        assert "b.pdf" in names
        assert "readme.txt" not in names
        for f in body:
            assert "size_kb" in f
            assert "uploaded" in f
            assert f["embedded"] is True

    def test_list_nonexistent_directory(self, upload_dir: Path) -> None:
        # Point to a directory that doesn't exist
        with patch("backend.api.routes.settings") as mock_settings:
            mock_settings.UPLOAD_DIR = upload_dir / "nonexistent"
            resp = client.get("/api/files")

        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# DELETE /api/files/{filename}
# ---------------------------------------------------------------------------


class TestDeleteFile:
    """Requirement 9.3 – DELETE /api/files/{filename}"""

    def test_delete_existing_file(self, upload_dir: Path) -> None:
        (upload_dir / "remove.pdf").write_bytes(b"%PDF-1.4")

        with patch("backend.api.routes.delete_embeddings"):
            resp = client.delete("/api/files/remove.pdf")

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "remove.pdf"
        assert body["deleted"] is True
        assert not (upload_dir / "remove.pdf").exists()

    def test_delete_missing_file_returns_404(self, upload_dir: Path) -> None:
        resp = client.delete("/api/files/ghost.pdf")

        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_delete_embedding_failure_returns_500(self, upload_dir: Path) -> None:
        (upload_dir / "fail.pdf").write_bytes(b"%PDF-1.4")

        with patch(
            "backend.api.routes.delete_embeddings",
            side_effect=EmbeddingDeletionError("chroma down"),
        ):
            resp = client.delete("/api/files/fail.pdf")

        assert resp.status_code == 500
        assert "chroma down" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/files/{filename}/status
# ---------------------------------------------------------------------------


class TestFileStatus:
    """Requirement 9.4 – GET /api/files/{filename}/status"""

    def test_status_embedded(self, upload_dir: Path) -> None:
        (upload_dir / "emb.pdf").write_bytes(b"%PDF-1.4")

        with patch("backend.api.routes.has_embeddings", return_value=True):
            resp = client.get("/api/files/emb.pdf/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "emb.pdf"
        assert body["status"] == "embedded"

    def test_status_not_embedded(self, upload_dir: Path) -> None:
        (upload_dir / "raw.pdf").write_bytes(b"%PDF-1.4")

        with patch("backend.api.routes.has_embeddings", return_value=False):
            resp = client.get("/api/files/raw.pdf/status")

        assert resp.status_code == 200
        assert resp.json()["status"] == "not embedded"

    def test_status_missing_file_returns_404(self, upload_dir: Path) -> None:
        resp = client.get("/api/files/nope.pdf/status")

        assert resp.status_code == 404
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
