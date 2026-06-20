from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


DEFAULT_KNOWLEDGE_EXPORT_URL = (
    "https://oms.diginet.com.vn/api-dev/v8.0.0/ai/knowledge-export"
)


@dataclass(frozen=True)
class AppConfig:
    chroma_dir: Path = Path(os.getenv("RAG_CHROMA_DIR", ".chroma"))
    collection_name: str = os.getenv("RAG_COLLECTION_NAME", "oms_knowledge_base")
    knowledge_export_url: str = os.getenv(
        "RAG_KNOWLEDGE_EXPORT_URL",
        DEFAULT_KNOWLEDGE_EXPORT_URL,
    )
    knowledge_export_token: str = os.getenv("RAG_KNOWLEDGE_EXPORT_TOKEN", "")
    knowledge_export_os: str = os.getenv("RAG_KNOWLEDGE_EXPORT_OS", "web")
    knowledge_export_version: str = os.getenv("RAG_KNOWLEDGE_EXPORT_VERSION", "8.0.0")
    knowledge_export_limit: int = int(os.getenv("RAG_KNOWLEDGE_EXPORT_LIMIT", "30"))
    knowledge_export_timeout: float = float(os.getenv("RAG_KNOWLEDGE_EXPORT_TIMEOUT", "30"))
    embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "ollama")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma:latest")
    local_ollama_host: str = os.getenv("RAG_LOCAL_OLLAMA_HOST", "http://localhost:11434")

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10s")
    router_model: str = os.getenv("RAG_ROUTER_MODEL", "qwen3:1.7b")
    reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL",
        "qllama/bge-reranker-v2-m3:latest",
    )

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "128"))
    default_k: int = int(os.getenv("RAG_TOP_K", "5"))
    fetch_k: int = int(os.getenv("RAG_FETCH_K", "20"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
    mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))

    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    @property
    def uses_direct_ollama_cloud(self) -> bool:
        host = self.ollama_host.rstrip("/").lower()
        return host in {"https://ollama.com", "https://www.ollama.com"}

    def with_overrides(self, **kwargs: object) -> "AppConfig":
        values = self.__dict__.copy()
        values.update(kwargs)
        if "chroma_dir" in values:
            values["chroma_dir"] = Path(str(values["chroma_dir"]))
        return AppConfig(**values)
