from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from .config import AppConfig
from .data_loader import chunk_documents, load_knowledge_export_documents
from .models import RagResponse, RagStreamResponse
from .rag import answer_question, stream_answer_question
from .retrievers import BM25Index
from .vector_store import ChromaKnowledgeStore


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class IngestResult:
    documents_loaded: int
    chunks_indexed: int
    collection_count: int


class RagService:
    """Framework-free owner for shared RAG resources and workflows."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._store: ChromaKnowledgeStore | None = None
        self._bm25: BM25Index | None = None
        self._lock = RLock()

    @property
    def store(self) -> ChromaKnowledgeStore:
        with self._lock:
            if self._store is None:
                self._store = self._create_store()
            return self._store

    def _create_store(self, *, reset: bool = False) -> ChromaKnowledgeStore:
        return ChromaKnowledgeStore(
            chroma_dir=self.config.chroma_dir,
            collection_name=self.config.collection_name,
            embedding_provider=self.config.embedding_provider,
            embedding_model=self.config.embedding_model,
            embedding_host=self.config.local_ollama_host,
            keep_alive=self.config.ollama_keep_alive,
            reset=reset,
        )

    def reset_store(self) -> None:
        with self._lock:
            if self._store is not None:
                self._store.close()
            self._store = self._create_store(reset=True)
            self._bm25 = None

    @property
    def bm25(self) -> BM25Index:
        with self._lock:
            if self._bm25 is None:
                self.refresh_retrieval_cache()
            if self._bm25 is None:
                raise RuntimeError("BM25 cache could not be initialized.")
            return self._bm25

    def count(self) -> int:
        with self._lock:
            return self.store.count()

    def refresh_retrieval_cache(self) -> None:
        with self._lock:
            self._bm25 = BM25Index.from_store(self.store)

    def ingest_knowledge(
        self,
        *,
        reset: bool = False,
        max_pages: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestResult:
        documents = load_knowledge_export_documents(
            url=self.config.knowledge_export_url,
            token=self.config.knowledge_export_token,
            limit=self.config.knowledge_export_limit,
            max_pages=max_pages,
            timeout=self.config.knowledge_export_timeout,
            access_os=self.config.knowledge_export_os,
            access_version=self.config.knowledge_export_version,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback({"event": "documents_loaded", "documents": len(documents)})

        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if progress_callback is not None:
            progress_callback({"event": "chunks_prepared", "chunks": len(chunks)})
            progress_callback({"event": "index_wait_start"})

        with self._lock:
            if progress_callback is not None:
                progress_callback({"event": "index_start"})

            if reset:
                if progress_callback is not None:
                    progress_callback({"event": "reset_start"})
                self.reset_store()
                if progress_callback is not None:
                    progress_callback({"event": "reset_done"})

            collection_count = self.store.ingest(chunks, progress_callback=progress_callback)
            self.refresh_retrieval_cache()
            return IngestResult(
                documents_loaded=len(documents),
                chunks_indexed=len(chunks),
                collection_count=collection_count,
            )

    def answer(
        self,
        *,
        question: str,
        mode: str = "Auto Router",
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.5,
    ) -> RagResponse:
        with self._lock:
            return answer_question(
                config=self.config,
                store=self.store,
                bm25=self.bm25,
                question=question,
                mode=mode,
                k=top_k,
                fetch_k=fetch_k,
                lambda_mult=mmr_lambda,
            )

    def answer_stream(
        self,
        *,
        question: str,
        mode: str = "Auto Router",
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.5,
    ) -> RagStreamResponse:
        with self._lock:
            return stream_answer_question(
                config=self.config,
                store=self.store,
                bm25=self.bm25,
                question=question,
                mode=mode,
                k=top_k,
                fetch_k=fetch_k,
                lambda_mult=mmr_lambda,
            )
