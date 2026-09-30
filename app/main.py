"""FastAPI entry point: API routes plus the static web UI."""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.retrieval.embedder import get_embedder
from app.retrieval.hybrid import get_reranker
from app.service import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def _warm_up_models() -> None:
    """Download/load local models in the background so the first request isn't slow."""
    settings = get_settings()
    try:
        if settings.embedding_provider == "fastembed":
            get_embedder(settings.embedding_provider, settings.embedding_model)
        if settings.rerank_enabled:
            get_reranker(settings.reranker_model)
        logger.info("Local models ready")
    except Exception:
        logger.exception("Model warm-up failed; models will load on first use")


def create_app(service: RAGService | None = None, warm_up: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or RAGService(get_settings())
        if warm_up:
            threading.Thread(target=_warm_up_models, daemon=True).start()
        yield

    app = FastAPI(
        title="RAG Generator",
        description="Upload documents at runtime and get a grounded, cited Q&A app over them.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    def health() -> dict:
        settings = app.state.service.settings
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        }

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/c/{collection_id}", include_in_schema=False)
    def chat_page(collection_id: str):
        return FileResponse(STATIC_DIR / "chat.html")

    return app


app = create_app()
