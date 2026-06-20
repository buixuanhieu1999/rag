from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_api.dependencies.services import get_rag_service
from rag_api.schemas.chat import ChatRequest, ChatResponse, SourceResponse
from rag_app.models import RetrievedDocument
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
