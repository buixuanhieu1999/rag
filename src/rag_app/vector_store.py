from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .data_loader import sanitize_metadata
from .models import KnowledgeDocument, RetrievedDocument


class OllamaEmbeddingFunction:
    """Chroma-compatible embedding function backed by local Ollama."""

    def __init__(
        self,
        model_name: str = "embeddinggemma:latest",
        host: str = "http://localhost:11434",
        batch_size: int = 32,
        keep_alive: str = "10s",
    ) -> None:
        from ollama import Client

        self.model_name = model_name
        self.host = host
        self.batch_size = batch_size
        self.keep_alive = keep_alive
        self.client = Client(host=host)

    def __call__(self, input: list[str]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        embeddings: list[list[float]] = []
        for start in range(0, len(input), self.batch_size):
            batch = input[start : start + self.batch_size]
            response = self.client.embed(
                model=self.model_name,
                input=batch,
                keep_alive=self.keep_alive,
            )
            batch_embeddings = (
                response["embeddings"]
                if isinstance(response, dict)
                else response.embeddings
            )
            embeddings.extend([list(embedding) for embedding in batch_embeddings])
        return embeddings

    def name(self) -> str:
        return f"ollama-{self.model_name}"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


def build_embedding_function(
    provider: str,
    model_name: str,
    host: str = "http://localhost:11434",
    keep_alive: str = "10s",
):
    from chromadb.utils import embedding_functions

    provider = (provider or "default").strip().lower()
    if provider == "ollama":
        return OllamaEmbeddingFunction(
            model_name=model_name,
            host=host,
            keep_alive=keep_alive,
        )
    if provider == "sentence-transformers":
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        except Exception as exc:
            raise RuntimeError(
                "SentenceTransformer embeddings need the optional dependency set: "
                "run `uv sync --extra multilingual`, then rebuild the index."
            ) from exc
    if provider == "default":
        return embedding_functions.DefaultEmbeddingFunction()
    raise ValueError(
        f"Unsupported embedding provider '{provider}'. Use 'default' or "
        "'sentence-transformers' or 'ollama'."
    )


class ChromaKnowledgeStore:
    def __init__(
        self,
        chroma_dir: str | Path,
        collection_name: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_host: str = "http://localhost:11434",
        keep_alive: str = "10s",
        reset: bool = False,
    ) -> None:
        import chromadb

        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.embedding_function = build_embedding_function(
            embedding_provider,
            embedding_model,
            embedding_host,
            keep_alive,
        )
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
        self.collection = self._open_collection()

    def _open_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        return self.collection

    def _refresh_collection(self) -> None:
        self.collection = self._open_collection()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._refresh_collection()

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as exc:
            if exc.__class__.__name__ != "NotFoundError":
                raise
            self._refresh_collection()
            return self.collection.count()

    def ingest(
        self,
        documents: list[KnowledgeDocument],
        batch_size: int = 100,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> int:
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "batch_start",
                        "start": start,
                        "end": start + len(batch),
                        "total": len(documents),
                    }
                )
            self.collection.upsert(
                ids=[doc.id for doc in batch],
                documents=[doc.text for doc in batch],
                metadatas=[sanitize_metadata(doc.metadata) for doc in batch],
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "batch_done",
                        "start": start,
                        "end": start + len(batch),
                        "total": len(documents),
                    }
                )
        return self.count()

    def query(
        self,
        query_text: str,
        k: int = 5,
        include_embeddings: bool = False,
    ) -> list[RetrievedDocument]:
        include: list[str] = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        try:
            result = self.collection.query(
                query_texts=[query_text],
                n_results=max(1, k),
                include=include,
            )
        except Exception as exc:
            if exc.__class__.__name__ != "NotFoundError":
                raise
            self._refresh_collection()
            result = self.collection.query(
                query_texts=[query_text],
                n_results=max(1, k),
                include=include,
            )
        return _query_result_to_documents(result, include_embeddings=include_embeddings)

    def get_all_documents(self) -> list[RetrievedDocument]:
        try:
            result = self.collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            if exc.__class__.__name__ != "NotFoundError":
                raise
            self._refresh_collection()
            result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            RetrievedDocument(
                id=str(doc_id),
                text=str(text or ""),
                metadata=dict(metadata or {}),
            )
            for doc_id, text, metadata in zip(ids, docs, metadatas)
        ]

    def get_by_knowledge_id(self, knowledge_id: int) -> list[RetrievedDocument]:
        try:
            result = self.collection.get(
                where={"knowledge_id": knowledge_id},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            if exc.__class__.__name__ != "NotFoundError":
                raise
            self._refresh_collection()
            result = self.collection.get(
                where={"knowledge_id": knowledge_id},
                include=["documents", "metadatas"],
            )

        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            RetrievedDocument(
                id=str(doc_id),
                text=str(text or ""),
                metadata=dict(metadata or {}),
                score=1.0,
            )
            for doc_id, text, metadata in zip(ids, docs, metadatas)
        ]


def _query_result_to_documents(
    result: dict[str, Any],
    include_embeddings: bool = False,
) -> list[RetrievedDocument]:
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    embeddings = (result.get("embeddings") or [[]])[0] if include_embeddings else []

    retrieved: list[RetrievedDocument] = []
    for index, doc_id in enumerate(ids):
        distance = distances[index] if index < len(distances) else None
        embedding = embeddings[index] if index < len(embeddings) else None
        retrieved.append(
            RetrievedDocument(
                id=str(doc_id),
                text=str(docs[index] or ""),
                metadata=dict(metadatas[index] or {}),
                distance=float(distance) if distance is not None else None,
                score=(1.0 - float(distance)) if distance is not None else None,
                embedding=list(embedding) if embedding is not None else None,
            )
        )
    return retrieved
