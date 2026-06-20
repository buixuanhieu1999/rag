from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag_app.config import AppConfig
from rag_app.ollama_client import OllamaConfigurationError
from rag_app.services import RagService


st.set_page_config(page_title="OMS Knowledge RAG", layout="wide")


@st.cache_resource(show_spinner=False)
def get_rag_service(config: AppConfig) -> RagService:
    return RagService(config)


def build_index(config: AppConfig, reset: bool) -> int:
    service = get_rag_service(config)
    result = service.ingest_knowledge(reset=reset)
    return result.collection_count


base_config = AppConfig()

st.title("OMS Knowledge RAG")

with st.sidebar:
    st.subheader("Index")
    knowledge_url = st.text_input("Knowledge export API", value=base_config.knowledge_export_url)
    knowledge_token = st.text_input(
        "Knowledge export token",
        value=base_config.knowledge_export_token,
        type="password",
    )
    knowledge_limit = st.slider(
        "API page size",
        min_value=1,
        max_value=30,
        value=min(max(base_config.knowledge_export_limit, 1), 30),
    )
    chroma_dir = st.text_input("Chroma directory", value=str(base_config.chroma_dir))
    collection_name = st.text_input("Collection", value=base_config.collection_name)
    embedding_provider = st.selectbox(
        "Embedding provider",
        ["ollama", "default", "sentence-transformers"],
        index=["ollama", "default", "sentence-transformers"].index(base_config.embedding_provider)
        if base_config.embedding_provider in {"ollama", "default", "sentence-transformers"}
        else 0,
    )
    embedding_model = st.text_input("Embedding model", value=base_config.embedding_model)

    st.subheader("Ollama")
    ollama_host = st.text_input("Answer model host", value=base_config.ollama_host)
    ollama_model = st.text_input("Answer model", value=base_config.ollama_model)
    local_ollama_host = st.text_input("Local embedding host", value=base_config.local_ollama_host)
    router_model = st.text_input("Local router", value=base_config.router_model)

    st.subheader("Retrieval")
    mode = st.selectbox(
        "Mode",
        [
            "Auto Router",
            "BM25",
            "Hybrid RRF",
            "Semantic",
            "MMR",
            "HyDE",
            "Decomposition",
        ],
        index=0,
    )
    top_k = st.slider("Top K", min_value=1, max_value=12, value=base_config.default_k)
    fetch_k = st.slider("Fetch K", min_value=top_k, max_value=30, value=max(base_config.fetch_k, top_k))
    mmr_lambda = st.slider("MMR lambda", min_value=0.0, max_value=1.0, value=base_config.mmr_lambda, step=0.05)

config = base_config.with_overrides(
    knowledge_export_url=knowledge_url,
    knowledge_export_token=knowledge_token,
    knowledge_export_limit=knowledge_limit,
    chroma_dir=chroma_dir,
    collection_name=collection_name,
    embedding_provider=embedding_provider,
    embedding_model=embedding_model,
    local_ollama_host=local_ollama_host,
    ollama_host=ollama_host,
    ollama_model=ollama_model,
    router_model=router_model,
)

service = get_rag_service(config)
count = service.count()

left, right = st.columns([2, 1])
with left:
    st.caption(f"Chroma chunks indexed: {count}")
with right:
    reset_index = st.checkbox("Reset before indexing", value=count == 0)
    if st.button("Build index", use_container_width=True):
        try:
            with st.spinner("Indexing knowledge articles into Chroma..."):
                count = build_index(config, reset=reset_index)
            st.success(f"Indexed {count} chunks.")
        except Exception as exc:
            st.error(str(exc))

question = st.text_area(
    "Question",
    value="D05F1602 thêm mới đơn đặt hàng kế thừa D05F1600 như thế nào?",
    height=110,
)

ask = st.button("Ask", type="primary")

if ask:
    if not question.strip():
        st.warning("Enter a question first.")
    elif service.count() == 0:
        st.warning("Build the Chroma index first.")
    else:
        try:
            with st.spinner("Retrieving and asking Ollama..."):
                response = service.answer(
                    question=question.strip(),
                    mode=mode,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    mmr_lambda=mmr_lambda,
                )
            st.subheader("Answer")
            st.write(response.answer)

            if response.diagnostics:
                with st.expander("Query transform"):
                    st.json(response.diagnostics)

            st.subheader("Sources")
            for index, source in enumerate(response.sources, start=1):
                metadata = source.metadata
                title = metadata.get("title") or "Untitled knowledge"
                source_doc = metadata.get("source_doc_id") or metadata.get("doc_id") or source.id
                score = f"{source.score:.4f}" if source.score is not None else "n/a"
                with st.expander(f"{index}. {source_doc} - {title} - score {score}"):
                    st.write(source.text)
                    st.json(metadata)
        except OllamaConfigurationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"RAG request failed: {exc}")
