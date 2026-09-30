"""FastAPI entry point."""

from fastapi import FastAPI

from app.config import get_settings

app = FastAPI(
    title="RAG Generator",
    description="Upload documents at runtime and get a grounded, cited Q&A app over them.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }
