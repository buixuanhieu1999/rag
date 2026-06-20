from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    reset: bool = False
    max_pages: int | None = Field(default=None, ge=1)


class IngestJobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str = ""
    documents_loaded: int | None = None
    chunks_indexed: int | None = None
    collection_count: int | None = None
    error: str | None = None
