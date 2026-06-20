# OMS Knowledge RAG

Local RAG service for OMS knowledge-export data.

The project has three entrypoints:

- `rag_api`: FastAPI backend for JS apps and API clients
- `streamlit_app.py`: optional local/admin UI
- `scripts/`: CLI tools for export and ingestion

Core RAG logic stays in `rag_app`, so the API, Streamlit UI, and CLI reuse the same Chroma, BM25, reranker, and answer workflow.

## Features

- Pulls OMS knowledge articles from the knowledge-export API
- Cleans HTML/article content into readable text
- Chunks articles for retrieval
- Stores embeddings in persistent ChromaDB
- Supports semantic search, BM25, hybrid RRF, MMR, HyDE, decomposition, and auto routing
- Uses Ollama for embeddings, reranking, routing, and answer generation
- Exposes FastAPI endpoints with Swagger UI
- Runs background ingestion jobs with `job_id` status tracking
- Logs request method, path, status code, duration, and request ID

## Setup

```powershell
cd C:\Users\AD\Downloads\RAG
uv sync
copy .env.example .env
```

Set the OMS API token in `.env`:

```env
RAG_KNOWLEDGE_EXPORT_TOKEN=your_knowledge_export_token
```

The OMS token is used only for pulling source data from:

```text
https://oms.diginet.com.vn/api-dev/v8.0.0/ai/knowledge-export
```

## Ollama Models

Start Ollama, then pull the local models:

```powershell
ollama pull qwen3:4b-instruct
ollama pull embeddinggemma
ollama pull qwen3:1.7b
ollama pull qllama/bge-reranker-v2-m3
```

Default `.env` model settings:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
RAG_LOCAL_OLLAMA_HOST=http://localhost:11434
RAG_EMBEDDING_PROVIDER=ollama
RAG_EMBEDDING_MODEL=embeddinggemma:latest
RAG_ROUTER_MODEL=qwen3:1.7b
RAG_RERANKER_MODEL=qllama/bge-reranker-v2-m3:latest
```

## Run With Docker Compose

Docker Compose can run the API with either CPU Ollama or NVIDIA GPU Ollama.
Use one profile at a time because both profiles publish the same local ports.

CPU:

```powershell
docker compose --profile cpu up -d --build
```

NVIDIA GPU:

```powershell
docker compose --profile gpu up -d --build
```

The GPU profile requires a working NVIDIA driver plus NVIDIA Container Toolkit
or Docker Desktop GPU support.

Compose starts:

```text
api / api-gpu                 FastAPI RAG service on port 8000
ollama / ollama-gpu           local Ollama server on port 11434
ollama-pull / ollama-pull-gpu one-shot model pull helper
```

Open:

```text
Swagger UI: http://127.0.0.1:8000/docs
Health:     http://127.0.0.1:8000/health
```

Watch logs:

```powershell
docker compose logs -f api
docker compose logs -f ollama-pull
docker compose logs -f api-gpu
docker compose logs -f ollama-pull-gpu
```

Run ingestion inside the API container:

```powershell
docker compose run --rm api python scripts/ingest.py --reset
# or, with the GPU profile:
docker compose run --rm api-gpu python scripts/ingest.py --reset
```

Stop containers:

```powershell
docker compose down
```

Persisted Docker data:

```text
./docker-data/chroma  ChromaDB data
./docker-data/ollama  Ollama models
./logs                app logs
./exports             exported JSON files
```

Rebuilding the image does not delete Chroma data or Ollama models because those folders are mounted from the host.

## Run FastAPI

```powershell
uv run uvicorn rag_api.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
Swagger UI: http://127.0.0.1:8000/docs
Health:     http://127.0.0.1:8000/health
```

FastAPI is the recommended integration layer for another JS app.

## API Endpoints

### Health

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "app": "OMS Knowledge RAG API",
  "version": "0.1.0",
  "collection_name": "oms_knowledge_base",
  "chroma_count": 7382,
  "error": null
}
```

### Chat

```http
POST /v1/chat
Content-Type: application/json
```

Request:

```json
{
  "question": "Lỗi tập tin License 0x2008 xử lý thế nào?",
  "mode": "Auto Router",
  "top_k": 5,
  "fetch_k": 20,
  "mmr_lambda": 0.5
}
```

Response:

```json
{
  "answer": "...",
  "mode": "BM25 + Ollama Reranker",
  "sources": [
    {
      "id": "KNOW-1::0",
      "title": "Lỗi tập tin License 0x2008",
      "knowledge_id": 1,
      "score": 0.91,
      "text_preview": "...",
      "metadata": {}
    }
  ],
  "diagnostics": {}
}
```

### Start Ingest

```http
POST /v1/ingest
Content-Type: application/json
```

Request:

```json
{
  "reset": false,
  "max_pages": null
}
```

Response:

```json
{
  "job_id": "8c2a1f3e-3ef9-4d1c-9c5e-91df91df6721",
  "status": "queued"
}
```

### Check Ingest Job

```http
GET /v1/ingest/{job_id}
```

Example:

```http
GET /v1/ingest/8c2a1f3e-3ef9-4d1c-9c5e-91df91df6721
```

Response:

```json
{
  "job_id": "8c2a1f3e-3ef9-4d1c-9c5e-91df91df6721",
  "status": "completed",
  "message": "Ingest job completed.",
  "documents_loaded": 2144,
  "chunks_indexed": 7382,
  "collection_count": 7382,
  "error": null
}
```

## JS App Example

```js
const response = await fetch("http://127.0.0.1:8000/v1/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    question: "Lỗi tập tin License 0x2008 xử lý thế nào?",
    mode: "Auto Router",
    top_k: 5,
    fetch_k: 20
  })
});

const data = await response.json();
console.log(data.answer);
```

This version has no API auth. Keep it on localhost or behind a trusted internal gateway before exposing it to other machines.

## Export Clean Raw JSON

Small sample:

```powershell
uv run python scripts\export_knowledge.py --knowledge-limit 5 --knowledge-max-pages 1
```

Full export:

```powershell
uv run python scripts\export_knowledge.py --all --knowledge-limit 30 --output exports\knowledge_export_all.json --log-file logs\export_knowledge.log
```

The export script writes cleaned JSON only. It does not embed, index, or touch Chroma.

## Build Or Refresh Chroma

Small test ingestion:

```powershell
uv run python scripts\ingest.py --knowledge-limit 5 --knowledge-max-pages 1 --collection-name knowledge_export_test --chroma-dir .chroma_knowledge_test --reset
```

Full ingestion:

```powershell
uv run python scripts\ingest.py --knowledge-limit 30 --reset --log-file logs\ingest.log
```

Without `--reset`, ingest uses Chroma `upsert`:

```text
New article IDs are added.
Existing article chunk IDs are updated.
Removed API articles may remain in Chroma.
```

Use `--reset` when you want Chroma to mirror the current API data cleanly.

## Run Streamlit Admin UI

```powershell
uv run streamlit run streamlit_app.py
```

Streamlit is useful for local testing and manual indexing, but JS apps should call FastAPI instead.

## Project Structure

```text
src/
  rag_app/
    config.py
    data_loader.py
    models.py
    rag.py
    retrievers.py
    services.py
    vector_store.py

  rag_api/
    main.py
    core/
    dependencies/
    middleware/
    routes/
    schemas/
    services/

scripts/
  export_knowledge.py
  ingest.py
  cli_logging.py

tests/
```

Important split:

```text
rag_app = reusable RAG engine
rag_api = HTTP/API wrapper
scripts = CLI operations
Streamlit = optional local UI
```

## Configuration

Common `.env` values:

```env
RAG_CHROMA_DIR=.chroma
RAG_COLLECTION_NAME=oms_knowledge_base

RAG_KNOWLEDGE_EXPORT_URL=https://oms.diginet.com.vn/api-dev/v8.0.0/ai/knowledge-export
RAG_KNOWLEDGE_EXPORT_TOKEN=
RAG_KNOWLEDGE_EXPORT_OS=web
RAG_KNOWLEDGE_EXPORT_VERSION=8.0.0
RAG_KNOWLEDGE_EXPORT_LIMIT=30
RAG_KNOWLEDGE_EXPORT_TIMEOUT=30

RAG_TOP_K=5
RAG_FETCH_K=20
```

Changing embedding provider/model usually requires rebuilding Chroma with `--reset`.

## Test

```powershell
uv run pytest
```

Compile check:

```powershell
uv run python -m compileall src scripts tests streamlit_app.py
```
