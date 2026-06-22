from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from rag_api.dependencies.services import get_rag_service
from rag_api.schemas.chat import ChatRequest, ChatResponse, SourceResponse
from rag_app.models import RetrievedDocument
from rag_app.rag import parse_focused_answer
from rag_app.services import RagService


router = APIRouter(prefix="/v1", tags=["chat"])


def _source_to_response(source: RetrievedDocument) -> SourceResponse:
    metadata = dict(source.metadata or {})
    title = metadata.get("title")
    knowledge_id = metadata.get("knowledge_id")
    preview = source.text[:500]
    if len(source.text) > 500:
        preview = f"{preview}..."

    return SourceResponse(
        id=source.id,
        title=str(title) if title not in (None, "") else None,
        knowledge_id=knowledge_id if knowledge_id not in ("", None) else None,
        score=source.score,
        text_preview=preview,
        metadata=metadata,
    )


def _dump_model(model: SourceResponse) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    response = service.answer(
        question=request.question.strip(),
        mode=request.mode,
        top_k=request.top_k,
        fetch_k=max(request.fetch_k, request.top_k),
        mmr_lambda=request.mmr_lambda,
    )
    return ChatResponse(
        answer=response.answer,
        mode=response.mode,
        sources=[_source_to_response(source) for source in response.sources],
        diagnostics=response.diagnostics,
    )


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    service: RagService = Depends(get_rag_service),
) -> StreamingResponse:
    response = service.answer_stream(
        question=request.question.strip(),
        mode=request.mode,
        top_k=request.top_k,
        fetch_k=max(request.fetch_k, request.top_k),
        mmr_lambda=request.mmr_lambda,
    )
    sources = [_source_to_response(source) for source in response.sources]

    def events():
        yield _sse_event(
            "metadata",
            {
                "mode": response.mode,
                "sources": [_dump_model(source) for source in sources],
                "diagnostics": response.diagnostics,
            },
        )

        answer_parts: list[str] = []
        try:
            for chunk in response.chunks:
                if not chunk:
                    continue
                answer_parts.append(chunk)
                yield _sse_event("token", {"token": chunk})
            yield _sse_event(
                "done",
                {"answer": parse_focused_answer("".join(answer_parts))},
            )
        except Exception as exc:
            yield _sse_event(
                "error",
                {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
