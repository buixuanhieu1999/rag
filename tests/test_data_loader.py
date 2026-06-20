from __future__ import annotations

from rag_app.data_loader import (
    chunk_documents,
    clean_knowledge_export_record,
    clean_html_text,
    load_knowledge_export_documents,
)


def test_chunk_documents_keeps_short_article_whole():
    from rag_app.models import KnowledgeDocument

    document = KnowledgeDocument(id="KNOW-7", text="Short knowledge", metadata={"doc_id": "KNOW-7"})
    chunks = chunk_documents([document], chunk_size=1000, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].id == "KNOW-7::0"
    assert chunks[0].metadata["chunk_count"] == 1


def test_chunk_documents_splits_long_text():
    from rag_app.models import KnowledgeDocument

    document = KnowledgeDocument(id="KNOW-1", text="A" * 2500, metadata={"doc_id": "KNOW-1"})
    chunks = chunk_documents([document], chunk_size=1000, chunk_overlap=100)

    assert len(chunks) >= 3
    assert chunks[0].metadata["chunk_count"] == len(chunks)
    assert chunks[0].id == "KNOW-1::0"


def test_clean_html_text_removes_article_markup():
    text = clean_html_text(
        "<p><strong>Mô tả:</strong> Lỗi đăng nhập</p>"
        "<p><img src='x.png' />"
        "<strong>Cách xử lý:</strong> Cấp lại license</p>"
    )

    assert "Mô tả: Lỗi đăng nhập" in text
    assert "Cách xử lý: Cấp lại license" in text
    assert "<p>" not in text
    assert "<img" not in text


def test_clean_knowledge_export_record_keeps_review_shape():
    record = clean_knowledge_export_record(
        {
            "id": 7,
            "instruction": "  License error  ",
            "input": " License ",
            "output": "<p><strong>Fix:</strong> Renew license</p>",
        }
    )

    assert record == {
        "id": 7,
        "instruction": "License error",
        "input": "License",
        "output": "Fix: Renew license",
    }


def test_load_knowledge_export_documents_pages_until_short_page():
    pages = {
        0: [
            {
                "id": 1,
                "instruction": "Lỗi tập tin License 0x2008",
                "input": "License",
                "output": "<p><strong>Mô tả:</strong> Hết hạn license</p>",
            },
            {
                "id": 2,
                "instruction": "Unable to connect to FTPServer",
                "input": "Error",
                "output": "<p>Kiểm tra FTP service</p>",
            },
        ],
        1: [
            {
                "id": 3,
                "instruction": "Định dạng hệ thống không đúng chuẩn",
                "input": "Error",
                "output": "<p>Đổi định dạng ngày tháng</p>",
            }
        ],
    }
    calls = []
    events = []

    def fake_fetcher(**kwargs):
        calls.append((kwargs["limit"], kwargs["skip"]))
        return pages.get(kwargs["skip"], [])

    docs = load_knowledge_export_documents(
        url="https://example.test/knowledge-export",
        token="test-token",
        limit=2,
        page_fetcher=fake_fetcher,
        progress_callback=events.append,
    )

    assert calls == [(2, 0), (2, 1)]
    assert [event["event"] for event in events] == [
        "fetch_start",
        "fetch_done",
        "page_done",
        "fetch_start",
        "fetch_done",
        "page_done",
        "stop_short_page",
    ]
    assert len(docs) == 3
    assert docs[0].id == "KNOW-1"
    assert "Title: Lỗi tập tin License 0x2008" in docs[0].text
    assert "Content: Mô tả: Hết hạn license" in docs[0].text
    assert docs[0].metadata["source"] == "knowledge_export_api"
    assert docs[0].metadata["title"] == "Lỗi tập tin License 0x2008"


def test_load_knowledge_export_documents_requires_positive_max_pages():
    import pytest

    with pytest.raises(ValueError, match="max_pages"):
        load_knowledge_export_documents(
            url="https://example.test/knowledge-export",
            token="test-token",
            max_pages=0,
            page_fetcher=lambda **kwargs: [],
        )
