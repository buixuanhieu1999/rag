from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Iterable

import numpy as np
from ollama import Client
from rank_bm25 import BM25Okapi

from .models import RetrievedDocument
from .vector_store import ChromaKnowledgeStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


class BM25Index:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents
        self._tokenized = [tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized) if documents else None

    @classmethod
    def from_store(cls, store: ChromaKnowledgeStore) -> "BM25Index":
        return cls(store.get_all_documents())

    def search(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        if not self.documents or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][: max(1, k)]
        results: list[RetrievedDocument] = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0 and results:
                continue
            results.append(replace(self.documents[int(idx)], score=score))
        return results


def semantic_search(store: ChromaKnowledgeStore, query: str, k: int = 5) -> list[RetrievedDocument]:
    return store.query(query, k=k)


def bm25_search(index: BM25Index, query: str, k: int = 5) -> list[RetrievedDocument]:
    return index.search(query, k=k)


class OllamaEmbeddingReranker:
    """Rerank candidates with a local Ollama reranker served via embed."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qllama/bge-reranker-v2-m3:latest",
        batch_size: int = 16,
        bm25_weight: float = 0.65,
        keep_alive: str = "10s",
    ) -> None:
        self.host = host
        self.model = model
        self.batch_size = batch_size
        self.bm25_weight = bm25_weight
        self.keep_alive = keep_alive
        self.client = Client(host=host)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embed(
                model=self.model,
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

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        limit: int = 5,
    ) -> list[RetrievedDocument]:
        if not documents:
            return []

        embeddings = self._embed([query] + [doc.text for doc in documents])
        expected = len(documents) + 1
        if len(embeddings) != expected:
            raise RuntimeError(
                f"Ollama reranker returned {len(embeddings)} embeddings for {expected} inputs."
            )

        query_embedding = np.array(embeddings[0], dtype=np.float32)
        bm25_scores = [float(doc.score or 0.0) for doc in documents]
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        reranker_weight = 1.0 - self.bm25_weight
        scored: list[RetrievedDocument] = []
        for doc, embedding, bm25_score in zip(documents, embeddings[1:], bm25_scores):
            reranker_score = _cosine(query_embedding, np.array(embedding, dtype=np.float32))
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            score = self.bm25_weight * normalized_bm25 + reranker_weight * reranker_score
            metadata = dict(doc.metadata)
            metadata["bm25_score"] = bm25_score
            metadata["reranker_similarity"] = reranker_score
            metadata["reranker_model"] = self.model
            scored.append(replace(doc, score=score, metadata=metadata))

        return sorted(scored, key=lambda doc: doc.score or 0.0, reverse=True)[:limit]


def bm25_ollama_rerank_search(
    index: BM25Index,
    reranker: OllamaEmbeddingReranker,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
) -> list[RetrievedDocument]:
    candidates = bm25_search(index, query, k=max(fetch_k, k))
    return reranker.rerank(query, candidates, limit=k)


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[RetrievedDocument]],
    limit: int = 5,
    rrf_k: int = 60,
) -> list[RetrievedDocument]:
    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDocument] = {}
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)
            docs[doc.id] = doc

    ordered_ids = sorted(scores, key=scores.get, reverse=True)
    return [replace(docs[doc_id], score=scores[doc_id]) for doc_id in ordered_ids[:limit]]


def hybrid_rrf_search(
    store: ChromaKnowledgeStore,
    bm25: BM25Index,
    query: str,
    k: int = 5,
    fetch_k: int = 10,
    rrf_k: int = 60,
) -> list[RetrievedDocument]:
    semantic = semantic_search(store, query, k=fetch_k)
    sparse = bm25_search(bm25, query, k=fetch_k)
    return reciprocal_rank_fusion([semantic, sparse], limit=k, rrf_k=rrf_k)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0 or math.isnan(denominator):
        return 0.0
    return float(np.dot(a, b) / denominator)


def mmr_search(
    store: ChromaKnowledgeStore,
    query: str,
    k: int = 5,
    fetch_k: int = 12,
    lambda_mult: float = 0.5,
) -> list[RetrievedDocument]:
    candidates = store.query(query, k=max(fetch_k, k), include_embeddings=True)
    candidates = [doc for doc in candidates if doc.embedding]
    if not candidates:
        return semantic_search(store, query, k=k)

    query_embedding = np.array(store.embedding_function([query])[0], dtype=np.float32)
    doc_embeddings = [np.array(doc.embedding, dtype=np.float32) for doc in candidates]
    query_scores = [_cosine(query_embedding, emb) for emb in doc_embeddings]

    selected: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < k:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _cosine(doc_embeddings[idx], doc_embeddings[chosen])
                    for chosen in selected
                )
            score = lambda_mult * query_scores[idx] - (1.0 - lambda_mult) * diversity_penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [
        replace(candidates[idx], score=float(query_scores[idx]))
        for idx in selected
    ]
