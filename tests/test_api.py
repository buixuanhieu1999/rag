from __future__ import annotations

import logging
from io import StringIO

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_api.dependencies.services import get_job_service, get_rag_service
from rag_api.main import create_app
from rag_api.middleware.request_id import RequestIDMiddleware
from rag_api.middleware.request_logging import RequestLoggingMiddleware
from rag_api.services.job_service import JobService
from rag_app.models import RagResponse, RetrievedDocument
from rag_app.services import IngestResult


class FakeRagService:
    def __init__(self) -> None:
        self.answer_calls = []
        self.ingest_calls = []

    def count(self) -> int:
        return 7

    def answer(self, **kwargs):
        self.answer_calls.append(kwargs)
        return RagResponse(
            answer="Renew the license file.",
            mode=kwargs["mode"],
            diagnostics={"test": True},
            sources=[
                RetrievedDocument(
                    id="KNOW-1::0",
                    text="Knowledge text preview",
                    metadata={"title": "License error", "knowledge_id": 1},
                    score=0.91,
                )
            ],
        )

    def ingest_knowledge(self, **kwargs):
        self.ingest_calls.append(kwargs)
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback({"event": "documents_loaded", "documents": 2})
            progress_callback({"event": "batch_done", "start": 0, "end": 3, "total": 3})
        return IngestResult(documents_loaded=2, chunks_indexed=3, collection_count=9)


def test_health_returns_collection_count():
    app = create_app()
    service = FakeRagService()
    app.dependency_overrides[get_rag_service] = lambda: service
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["chroma_count"] == 7


def test_chat_endpoint_returns_answer_and_sources():
    app = create_app()
    service = FakeRagService()
    app.dependency_overrides[get_rag_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        json={"question": "How to fix license?", "mode": "Semantic", "top_k": 3, "fetch_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Renew the license file."
    assert body["mode"] == "Semantic"
    assert body["sources"][0]["id"] == "KNOW-1::0"
    assert body["sources"][0]["title"] == "License error"
    assert service.answer_calls[0]["fetch_k"] == 3


def test_ingest_endpoint_returns_job_id_and_status():
    app = create_app()
    service = FakeRagService()
    job_service = JobService()
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_job_service] = lambda: job_service
    client = TestClient(app)

    response = client.post("/v1/ingest", json={"reset": True, "max_pages": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"]

    status_response = client.get(f"/v1/ingest/{body['job_id']}")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["status"] == "completed"
    assert status["documents_loaded"] == 2
    assert status["chunks_indexed"] == 3
    assert status["collection_count"] == 9
    assert service.ingest_calls[0]["reset"] is True
    assert service.ingest_calls[0]["max_pages"] == 1


def test_ingest_endpoint_rejects_second_active_job():
    app = create_app()
    service = FakeRagService()
    job_service = JobService()
    active_job = job_service.create_ingest_job()
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_job_service] = lambda: job_service
    client = TestClient(app)

    response = client.post("/v1/ingest", json={"reset": True})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["message"] == "An ingest job is already active."
    assert detail["job_id"] == active_job.job_id
    assert detail["status"] == "queued"
    assert service.ingest_calls == []


def test_request_id_and_logging_middleware():
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, logger_name="rag_api.middleware_test")
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger("rag_api.middleware_test")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        response = client.get("/ping", headers={"X-Request-ID": "req-123"})
    finally:
        logger.removeHandler(handler)

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-123"
    assert "GET /ping 200" in log_stream.getvalue()
    assert "request_id=req-123" in log_stream.getvalue()
