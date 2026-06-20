from __future__ import annotations

from functools import lru_cache

from rag_app.services import RagService

from rag_api.core.settings import ApiSettings
from rag_api.services.job_service import JobService


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return ApiSettings()


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(get_api_settings().app_config)


@lru_cache(maxsize=1)
def get_job_service() -> JobService:
    return JobService()
