from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from rag_api.dependencies.services import get_job_service, get_rag_service
from rag_api.schemas.ingest import IngestJobResponse, IngestJobStatusResponse, IngestRequest
from rag_api.services.job_service import ActiveIngestJobError, IngestJobState, JobService
from rag_app.services import RagService


router = APIRouter(prefix="/v1", tags=["ingest"])
logger = logging.getLogger("rag_api")


def _job_to_status_response(job: IngestJobState) -> IngestJobStatusResponse:
    return IngestJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        documents_loaded=job.documents_loaded,
        chunks_indexed=job.chunks_indexed,
        collection_count=job.collection_count,
        error=job.error,
    )


def _progress_message(event: dict[str, object]) -> str | None:
    event_name = event.get("event")
    if event_name == "fetch_start":
        return f"Fetching knowledge-export page {event.get('page')}."
    if event_name == "fetch_done":
        return f"Fetched page {event.get('page')}: {event.get('records')} raw records."
    if event_name == "page_done":
        return (
            f"Cleaned page {event.get('page')}: total documents "
            f"{event.get('total_documents')}."
        )
    if event_name == "documents_loaded":
        return f"Loaded {event.get('documents')} source knowledge records."
    if event_name == "chunks_prepared":
        return f"Prepared {event.get('chunks')} chunks."
    if event_name == "index_wait_start":
        return "Waiting for Chroma index lock."
    if event_name == "index_start":
        return "Chroma indexing started."
    if event_name == "reset_start":
        return "Resetting Chroma collection."
    if event_name == "reset_done":
        return "Chroma collection reset completed."
    if event_name == "batch_start":
        return (
            f"Indexing chunk batch {event.get('start')}-{event.get('end')} "
            f"of {event.get('total')}."
        )
    if event_name == "batch_done":
        return (
            f"Indexed chunk batch {event.get('start')}-{event.get('end')} "
            f"of {event.get('total')}."
        )
    return None


def _run_ingest_job(
    *,
    job_id: str,
    request: IngestRequest,
    service: RagService,
    job_service: JobService,
) -> None:
    job_service.update(job_id, status="running", message="Ingest job started.")

    def progress_callback(event: dict[str, object]) -> None:
        updates: dict[str, object] = {}
        message = _progress_message(event)
        if message is not None:
            updates["message"] = message

        event_name = event.get("event")
        if event_name == "page_done":
            updates["documents_loaded"] = event.get("total_documents")
        elif event_name == "documents_loaded":
            updates["documents_loaded"] = event.get("documents")
        elif event_name == "batch_done":
            updates["chunks_indexed"] = event.get("end")

        if updates:
            job_service.update(job_id, **updates)

    try:
        result = service.ingest_knowledge(
            reset=request.reset,
            max_pages=request.max_pages,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.exception("Ingest job %s failed.", job_id)
        job_service.update(
            job_id,
            status="failed",
            message="Ingest job failed.",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return

    job_service.update(
        job_id,
        status="completed",
        message="Ingest job completed.",
        documents_loaded=result.documents_loaded,
        chunks_indexed=result.chunks_indexed,
        collection_count=result.collection_count,
        error=None,
    )


@router.post("/ingest", response_model=IngestJobResponse)
def start_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    service: RagService = Depends(get_rag_service),
    job_service: JobService = Depends(get_job_service),
) -> IngestJobResponse:
    try:
        job = job_service.create_ingest_job()
    except ActiveIngestJobError as exc:
        active_job = exc.active_job
        raise HTTPException(
            status_code=409,
            detail={
                "message": "An ingest job is already active.",
                "job_id": active_job.job_id,
                "status": active_job.status,
            },
        ) from exc

    background_tasks.add_task(
        _run_ingest_job,
        job_id=job.job_id,
        request=request,
        service=service,
        job_service=job_service,
    )
    return IngestJobResponse(job_id=job.job_id, status=job.status)


@router.get("/ingest/{job_id}", response_model=IngestJobStatusResponse)
def get_ingest_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> IngestJobStatusResponse:
    job = job_service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found.")
    return _job_to_status_response(job)
