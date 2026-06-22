# OMS Knowledge RAG

Dịch vụ RAG cục bộ cho dữ liệu xuất từ OMS knowledge.

Dự án có ba điểm chạy chính:

- `rag_api`: backend FastAPI cho ứng dụng JS và các client gọi API
- `streamlit_app.py`: giao diện cục bộ/quản trị tùy chọn
- `scripts/`: công cụ CLI để xuất và nạp dữ liệu

Phần logic RAG lõi nằm trong `rag_app`, vì vậy API, giao diện Streamlit và CLI đều dùng chung Chroma, BM25, router và luồng tạo câu trả lời.

## Tính năng

- Lấy bài viết tri thức OMS từ API knowledge-export
- Làm sạch nội dung HTML/bài viết thành văn bản dễ đọc
- Chia nhỏ bài viết để truy xuất
- Lưu embedding vào ChromaDB bền vững
- Hỗ trợ semantic search, BM25, hybrid RRF, MMR, HyDE, decomposition và tự định tuyến
- Dùng Ollama cho embedding, routing và sinh câu trả lời
- Cung cấp các endpoint FastAPI kèm Swagger UI
- Chạy các tác vụ nạp dữ liệu nền với trạng thái `job_id`
- Ghi log phương thức request, path, mã trạng thái, thời lượng và request ID

## Thiết lập

```powershell
cd C:\Users\AD\Downloads\RAG
uv sync
copy .env.example .env
```

Thiết lập token OMS API trong `.env`:

```env
RAG_KNOWLEDGE_EXPORT_TOKEN=your_knowledge_export_token
```

Token OMS chỉ được dùng để lấy dữ liệu nguồn từ:

```text
https://oms.diginet.com.vn/api-dev/v8.0.0/ai/knowledge-export
```

## Ollama Models

Khởi động Ollama, sau đó kéo các model cục bộ:

```powershell
ollama pull qwen3:4b-instruct
ollama pull embeddinggemma
ollama pull qwen3:1.7b
```

Cấu hình model mặc định trong `.env`:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
RAG_LOCAL_OLLAMA_HOST=http://localhost:11434
RAG_EMBEDDING_PROVIDER=ollama
RAG_EMBEDDING_MODEL=embeddinggemma:latest
RAG_ROUTER_MODEL=qwen3:1.7b
```

## Chạy bằng Docker Compose

Docker Compose có thể chạy API với Ollama CPU, GPU 0, GPU 1 hoặc cả hai GPU.
Mỗi profile dùng cổng và thư mục dữ liệu riêng.

CPU:

```powershell
docker compose --profile cpu up -d --build
```

NVIDIA GPU:

```powershell
docker compose --profile gpu0 up -d --build
docker compose --profile gpu1 up -d --build

# Chạy cả GPU 0 và GPU 1:
docker compose --profile gpu up -d --build
```

Profile GPU yêu cầu driver NVIDIA hoạt động cùng NVIDIA Container Toolkit
hoặc hỗ trợ GPU của Docker Desktop.

Compose khởi động:

```text
api-cpu / ollama-cpu / ollama-pull-cpu           ports 8000 / 11434
api-gpu-0 / ollama-gpu-0 / ollama-pull-gpu-0     ports 8001 / 11435
api-gpu-1 / ollama-gpu-1 / ollama-pull-gpu-1     ports 8002 / 11436
```

Mở:

```text
Swagger UI: http://127.0.0.1:8000/docs
Health:     http://127.0.0.1:8000/health
```

Xem log:

```powershell
docker compose logs -f api-cpu
docker compose logs -f ollama-pull-cpu
docker compose logs -f api-gpu-0
docker compose logs -f ollama-pull-gpu-0
docker compose logs -f api-gpu-1
docker compose logs -f ollama-pull-gpu-1
```

Chạy nạp dữ liệu bên trong container API:

```powershell
docker compose run --rm api-cpu python scripts/ingest.py --reset
# hoặc với profile GPU:
docker compose run --rm api-gpu-0 python scripts/ingest.py --reset
docker compose run --rm api-gpu-1 python scripts/ingest.py --reset
```

Dừng container:

```powershell
docker compose down
```

Dữ liệu Docker được lưu lại:

```text
./docker-data/chroma  Dữ liệu ChromaDB
./docker-data/ollama  Model Ollama
./logs                log ứng dụng
./exports             file JSON đã xuất
```

Việc rebuild image không xóa dữ liệu Chroma hay model Ollama vì các thư mục này được mount từ máy chủ.

## Chạy FastAPI

```powershell
uv run uvicorn rag_api.main:app --host 127.0.0.1 --port 8000
```

Mở:

```text
Swagger UI: http://127.0.0.1:8000/docs
Health:     http://127.0.0.1:8000/health
```

FastAPI là lớp tích hợp được khuyến nghị cho ứng dụng JS khác.

## API Endpoints

### Health

```http
GET /health
```

Ví dụ response:

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
  "mode": "BM25",
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

Ví dụ:

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

## Ví dụ cho JS App

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

Phiên bản này không có xác thực API. Chỉ nên chạy trên localhost hoặc sau một gateway nội bộ đáng tin cậy trước khi mở cho máy khác.

## Xuất JSON gốc đã làm sạch

Mẫu nhỏ:

```powershell
uv run python scripts\export_knowledge.py --knowledge-limit 5 --knowledge-max-pages 1
```

Xuất toàn bộ:

```powershell
uv run python scripts\export_knowledge.py --all --knowledge-limit 30 --output exports\knowledge_export_all.json --log-file logs\export_knowledge.log
```

Script export chỉ ghi JSON đã làm sạch. Nó không embed, không index, và không tác động đến Chroma.

## Tạo lại Chroma hoặc làm mới

Nạp dữ liệu thử nhỏ:

```powershell
uv run python scripts\ingest.py --knowledge-limit 5 --knowledge-max-pages 1 --collection-name knowledge_export_test --chroma-dir .chroma_knowledge_test --reset
```

Nạp dữ liệu đầy đủ:

```powershell
uv run python scripts\ingest.py --knowledge-limit 30 --reset --log-file logs\ingest.log
```

Không dùng `--reset`, ingest sẽ dùng Chroma `upsert`:

```text
ID bài viết mới sẽ được thêm vào.
Chunk ID của bài viết hiện có sẽ được cập nhật.
Bài viết đã bị xóa khỏi API vẫn có thể còn trong Chroma.
```

Dùng `--reset` khi muốn Chroma phản ánh dữ liệu API hiện tại một cách sạch sẽ.

## Chạy Streamlit Admin UI

```powershell
uv run streamlit run streamlit_app.py
```

Streamlit hữu ích cho kiểm thử cục bộ và index thủ công, nhưng ứng dụng JS nên gọi FastAPI thay vì dùng trực tiếp.

## Cấu trúc dự án

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

Phân tách quan trọng:

```text
rag_app = engine RAG có thể tái sử dụng
rag_api = lớp bọc HTTP/API
scripts = thao tác CLI
Streamlit = giao diện cục bộ tùy chọn
```

## Cấu hình

Các giá trị phổ biến trong `.env`:

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

Khi thay đổi embedding provider/model, thường cần build lại Chroma với `--reset`.

## Kiểm thử

```powershell
uv run pytest
```

Kiểm tra biên dịch:

```powershell
uv run python -m compileall src scripts tests streamlit_app.py
```
