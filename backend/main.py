from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(title="ChefVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/config/debug")
def debug_config():
    from backend.config import settings
    return {
        "debug_bbox": settings.DEBUG_BBOX,
        "debug_chunks": settings.DEBUG_CHUNKS,
    }


@app.post("/api/config/debug")
def set_debug_config(body: dict):
    from backend.config import settings
    if "debug_bbox" in body:
        settings.DEBUG_BBOX = bool(body["debug_bbox"])
    if "debug_chunks" in body:
        settings.DEBUG_CHUNKS = bool(body["debug_chunks"])
    return {
        "debug_bbox": settings.DEBUG_BBOX,
        "debug_chunks": settings.DEBUG_CHUNKS,
    }
