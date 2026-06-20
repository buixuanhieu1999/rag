from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Callable

from .models import KnowledgeDocument


class _HTMLTextExtractor(HTMLParser):
    block_tags = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "li":
            self._append_break()
            self._parts.append("- ")
        elif tag.lower() in self.block_tags:
            self._append_break()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.block_tags:
            self._append_break()

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def _append_break(self) -> None:
        if self._parts and self._parts[-1] != "\n":
            self._parts.append("\n")

    def text(self) -> str:
        return "".join(self._parts)


def clean_vietnamese_text(text: str) -> str:
    """Normalize whitespace while preserving Vietnamese accents and field layout."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_html_text(text: str) -> str:
    """Convert simple article HTML to plain text suitable for retrieval."""

    text = re.sub(r"<img[^>]*>", "", str(text), flags=re.IGNORECASE)
    if "<" not in text or ">" not in text:
        return clean_vietnamese_text(html.unescape(text))

    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    extracted = parser.text()
    extracted = re.sub(r"[ \t]*\n[ \t]*", "\n", extracted)
    extracted = re.sub(r"\n{3,}", "\n\n", extracted)
    return clean_vietnamese_text(extracted)


def _scalar_metadata(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        scalar = _scalar_metadata(value)
        if scalar is not None:
            clean[str(key)] = scalar
    return clean


def clean_knowledge_export_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one OMS knowledge-export API row for review or ingestion."""

    knowledge_id = record.get("id")
    instruction = clean_vietnamese_text(str(record.get("instruction") or ""))
    keywords = clean_vietnamese_text(str(record.get("input") or ""))
    output = clean_html_text(str(record.get("output") or ""))

    return {
        "id": knowledge_id,
        "instruction": instruction,
        "input": keywords,
        "output": output,
    }


def build_page_content_from_knowledge_record(record: dict[str, Any]) -> str:
    clean_record = clean_knowledge_export_record(record)
    parts: list[str] = []
    knowledge_id = clean_record.get("id")
    instruction = clean_record.get("instruction")
    keywords = clean_record.get("input")
    output = clean_record.get("output")

    if knowledge_id not in (None, ""):
        parts.append(f"Knowledge ID: {knowledge_id}")
    if instruction not in (None, ""):
        parts.append(f"Title: {instruction}")
    if keywords not in (None, ""):
        parts.append(f"Keywords: {keywords}")
    if output:
        parts.append(f"Content: {output}")
    return clean_vietnamese_text("\n".join(parts))


def knowledge_record_to_document(
    record: dict[str, Any],
    source_url: str,
    index: int,
) -> KnowledgeDocument | None:
    text = build_page_content_from_knowledge_record(record)
    if not text:
        return None

    knowledge_id = record.get("id")
    if knowledge_id in (None, ""):
        knowledge_id = index
    doc_id = f"KNOW-{knowledge_id}"
    clean_record = clean_knowledge_export_record(record)
    metadata = {
        "doc_id": doc_id,
        "knowledge_id": knowledge_id,
        "title": clean_record.get("instruction") or "",
        "keywords": clean_record.get("input") or "",
        "source": "knowledge_export_api",
        "source_url": source_url,
        "row_index": index,
    }
    return KnowledgeDocument(
        id=doc_id,
        text=text,
        metadata=sanitize_metadata(metadata),
    )


def fetch_knowledge_export_page(
    *,
    url: str,
    token: str,
    limit: int = 30,
    skip: int = 0,
    timeout: float = 30,
    access_os: str = "web",
    access_version: str = "8.0.0",
) -> list[dict[str, Any]]:
    if not token:
        raise ValueError("RAG_KNOWLEDGE_EXPORT_TOKEN is required for knowledge export ingestion.")

    limit = min(max(1, int(limit)), 30)
    body = urllib.parse.urlencode({"limit": limit, "skip": int(skip)}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Access-Token": token,
            "X-Access-OS": access_os,
            "X-Access-Version": access_version,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Knowledge export API failed with HTTP {exc.code}: {detail}") from exc

    code = payload.get("code")
    if code not in (None, 200):
        message = payload.get("message") or payload.get("error") or payload
        raise RuntimeError(f"Knowledge export API returned code {code}: {message}")

    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("Knowledge export API response field 'data' must be a list.")
    return [item for item in data if isinstance(item, dict)]


KnowledgePageFetcher = Callable[..., list[dict[str, Any]]]
KnowledgeProgressCallback = Callable[[dict[str, Any]], None]


def load_knowledge_export_documents(
    *,
    url: str,
    token: str,
    limit: int = 30,
    max_pages: int | None = None,
    timeout: float = 30,
    access_os: str = "web",
    access_version: str = "8.0.0",
    page_fetcher: KnowledgePageFetcher | None = None,
    progress_callback: KnowledgeProgressCallback | None = None,
) -> list[KnowledgeDocument]:
    limit = min(max(1, int(limit)), 30)
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided.")

    fetcher = page_fetcher or fetch_knowledge_export_page
    documents: list[KnowledgeDocument] = []
    skip = 0

    while max_pages is None or skip < max_pages:
        if progress_callback is not None:
            progress_callback({"event": "fetch_start", "page": skip, "limit": limit})
        page = fetcher(
            url=url,
            token=token,
            limit=limit,
            skip=skip,
            timeout=timeout,
            access_os=access_os,
            access_version=access_version,
        )
        if progress_callback is not None:
            progress_callback({"event": "fetch_done", "page": skip, "records": len(page)})
        if not page:
            if progress_callback is not None:
                progress_callback({"event": "stop_empty", "page": skip})
            break

        before_count = len(documents)
        for item in page:
            document = knowledge_record_to_document(item, source_url=url, index=len(documents))
            if document is not None:
                documents.append(document)

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "page_done",
                    "page": skip,
                    "records": len(page),
                    "documents_added": len(documents) - before_count,
                    "total_documents": len(documents),
                }
            )

        if len(page) < limit:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "stop_short_page",
                        "page": skip,
                        "records": len(page),
                        "limit": limit,
                    }
                )
            break
        skip += 1

    return documents


def _find_chunk_end(text: str, start: int, max_end: int, min_end: int) -> int:
    if max_end >= len(text):
        return len(text)
    candidates = [
        text.rfind("\n\n", start, max_end),
        text.rfind("\n", start, max_end),
        text.rfind(". ", start, max_end),
        text.rfind("; ", start, max_end),
    ]
    best = max(candidates)
    if best >= min_end:
        return best + 1
    return max_end


def chunk_documents(
    documents: list[KnowledgeDocument],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> list[KnowledgeDocument]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[KnowledgeDocument] = []
    for document in documents:
        text = document.text
        if len(text) <= chunk_size:
            metadata = dict(document.metadata)
            metadata.update({"source_doc_id": document.id, "chunk_index": 0, "chunk_count": 1})
            chunks.append(KnowledgeDocument(id=f"{document.id}::0", text=text, metadata=metadata))
            continue

        doc_chunks: list[KnowledgeDocument] = []
        start = 0
        chunk_index = 0
        while start < len(text):
            min_end = start + max(1, chunk_size // 2)
            end = _find_chunk_end(text, start, min(start + chunk_size, len(text)), min_end)
            chunk_text = clean_vietnamese_text(text[start:end])
            if chunk_text:
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "source_doc_id": document.id,
                        "chunk_index": chunk_index,
                    }
                )
                doc_chunks.append(
                    KnowledgeDocument(
                        id=f"{document.id}::{chunk_index}",
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)

        for chunk in doc_chunks:
            chunk.metadata["chunk_count"] = len(doc_chunks)
        chunks.extend(doc_chunks)

    return chunks
