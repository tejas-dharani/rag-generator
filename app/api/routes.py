"""REST API: collections, documents and question answering."""

import json
import logging
from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.generation.llm import LLMError
from app.service import NotFoundError, RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def get_service(request: Request) -> RAGService:
    return request.app.state.service


class CreateCollection(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    chunk_size: int | None = Field(default=None, ge=100, le=2000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=500)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    stream: bool = False


def _not_found(exc: NotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


# ---- collections ----


@router.post("/collections", status_code=201)
def create_collection(body: CreateCollection, service: RAGService = Depends(get_service)):
    try:
        collection = service.create_collection(
            body.name, body.description, chunk_size=body.chunk_size, chunk_overlap=body.chunk_overlap, top_k=body.top_k
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return asdict(collection)


@router.get("/collections")
def list_collections(service: RAGService = Depends(get_service)):
    return [asdict(c) for c in service.list_collections()]


@router.get("/collections/{collection_id}")
def get_collection(collection_id: str, service: RAGService = Depends(get_service)):
    try:
        return asdict(service.get_collection(collection_id))
    except NotFoundError as exc:
        raise _not_found(exc) from exc


@router.delete("/collections/{collection_id}", status_code=204)
def delete_collection(collection_id: str, service: RAGService = Depends(get_service)):
    try:
        service.delete_collection(collection_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc


# ---- documents ----


@router.post("/collections/{collection_id}/documents", status_code=202)
async def upload_documents(
    collection_id: str,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    service: RAGService = Depends(get_service),
):
    """Accept files immediately; parsing and indexing run in the background (poll the documents list)."""
    try:
        service.get_collection(collection_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc

    results = []
    for upload in files:
        data = await upload.read()
        try:
            doc, is_new = service.register_upload(collection_id, upload.filename or "upload", data)
        except ValueError as exc:
            results.append({"filename": upload.filename, "status": "rejected", "error": str(exc)})
            continue
        if is_new:
            background.add_task(service.ingest_document, doc.id)
        results.append({**asdict(doc), "duplicate": not is_new})
    return {"documents": results}


@router.get("/collections/{collection_id}/documents")
def list_documents(collection_id: str, service: RAGService = Depends(get_service)):
    try:
        return [asdict(d) for d in service.list_documents(collection_id)]
    except NotFoundError as exc:
        raise _not_found(exc) from exc


@router.delete("/collections/{collection_id}/documents/{doc_id}", status_code=204)
def delete_document(collection_id: str, doc_id: str, service: RAGService = Depends(get_service)):
    try:
        service.delete_document(collection_id, doc_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc


# ---- query ----


@router.post("/collections/{collection_id}/query")
def query(collection_id: str, body: Query, service: RAGService = Depends(get_service)):
    """Ask a question. With `stream: true` the response is Server-Sent Events:
    `sources`, then `token`s, then `done` (or `error`)."""
    try:
        service.get_collection(collection_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    history = [turn.model_dump() for turn in body.history]

    if body.stream:
        def events():
            try:
                for event in service.answer_stream(collection_id, body.question, history):
                    yield f"data: {json.dumps(event)}\n\n"
            except LLMError as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            except Exception:
                logger.exception("Query failed")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Internal error while answering'})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    try:
        return service.answer(collection_id, body.question, history)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
