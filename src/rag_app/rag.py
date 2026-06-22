from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .config import AppConfig
from .models import PreparedRagRequest, RagResponse, RagStreamResponse, RetrievedDocument
from .ollama_client import chat, chat_stream
from .retrievers import (
    BM25Index,
    bm25_search,
    hybrid_rrf_search,
    mmr_search,
    reciprocal_rank_fusion,
    semantic_search,
)
from .vector_store import ChromaKnowledgeStore


ANSWER_SYSTEM_PROMPT = (
    "Bạn là trợ lý RAG cho cơ sở tri thức OMS/LEMON/DIGINET. "
    "Ngữ cảnh có thể là bài knowledge, hướng dẫn thao tác màn hình/form, xử lý lỗi, "
    "cấu hình, báo cáo, phân quyền, quy trình ERP/CRM/HR/Web6/APP. "
    "Chỉ dùng ngữ cảnh được cung cấp, trả lời bằng tiếng Việt rõ ràng, "
    "và nói 'Không có thông tin' nếu ngữ cảnh không đủ. "
    "Giữ nguyên mã module/form như D05, W29, D05F1602, W89F1000 và các thuật ngữ "
    "OMS, LEMON, ERP, CRM, Web6, APP, SQL, report, License, FTP, HardwareID, user."
)

ANSWER_TEMPLATE = """[TÀI LIỆU]
{context}

[CÂU HỎI]
{question}

Hãy trả lời dựa trên tài liệu.
Yêu cầu:
- Ưu tiên đúng bài knowledge, màn hình/form, module, keyword và nội dung trong nguồn.
- Giữ nguyên mã form/module, tên menu, tab, nút, report, file, SQL object và lỗi kỹ thuật.
- Nếu tài liệu có mô tả/nguyên nhân/cách xử lý/quy trình/thao tác, trình bày theo đúng các ý đó.
- Nếu có nhiều nguồn liên quan, gộp câu trả lời ngắn gọn và nêu khác biệt quan trọng.
- Không suy đoán ngoài tài liệu.
[TRẢ LỜI]:"""

HYDE_TEMPLATE = """Hãy viết một đoạn văn ngắn 100-150 từ bằng tiếng Việt như một bài knowledge OMS/LEMON có khả năng trả lời câu hỏi này.
Đoạn văn nên dùng từ khóa gần với module, form, màn hình, báo cáo, cấu hình, phân quyền, lỗi hoặc quy trình nếu câu hỏi có gợi ý.
Không nói rằng đây là giả định.

Câu hỏi: {question}

Đoạn văn:"""

DECOMPOSE_TEMPLATE = """Tách câu hỏi sau thành tối đa 4 câu hỏi con ngắn bằng tiếng Việt để tìm kiếm trong cơ sở tri thức OMS/LEMON.
Giữ nguyên mã module/form, tên report, menu, tab, lỗi, file hoặc SQL object.
Chỉ trả về danh sách, mỗi dòng một câu hỏi.

Câu hỏi: {question}"""

ROUTER_SYSTEM_PROMPT = """You are a deterministic router for an OMS/LEMON knowledge-base RAG app.
Choose exactly one retrieval mode. Output only valid JSON.

Mode priority:
1. BM25: explicit knowledge/article IDs, exact module/form codes like D05F1602 or W89F1000, exact screen/report/menu/tab/button names, quoted UI labels, error codes, file paths, SQL/object names, customer/product keywords, or highly specific keywords.
2. Decomposition: the user asks multiple independent questions, compares items, or asks about several forms/modules/aspects at once.
3. MMR: the user asks for diverse examples, groups, categories, overviews, common cases, or different types of knowledge articles.
4. Hybrid RRF: both exact keywords and semantic meaning matter. Use this as the safest default when unsure.
5. Semantic: vague paraphrase, symptom similarity, or natural-language search with few exact terms.
6. HyDE: conceptual process/behavior search with very little vocabulary overlap.

Never invent a mode. Never include markdown or explanation outside JSON."""

ROUTER_TEMPLATE = """Return JSON:
{{"mode":"one allowed mode","reason":"short reason"}}

Allowed modes:
- BM25
- Hybrid RRF
- Semantic
- MMR
- Decomposition
- HyDE

Examples:
Q: "knowledge 51493"
A: {{"mode":"BM25","reason":"knowledge id"}}

Q: "D05F1602 them moi don dat hang ke thua D05F1600 nhu the nao?"
A: {{"mode":"BM25","reason":"exact form codes"}}

Q: "Tim cac bai lien quan den phan quyen user tren Web6 nhung khong thay menu"
A: {{"mode":"Hybrid RRF","reason":"keyword plus semantic intent"}}

Q: "Nguoi dung cap nhat du lieu nhung he thong khong ghi nhan, co bai nao tuong tu khong?"
A: {{"mode":"Semantic","reason":"paraphrased similarity search"}}

Q: "Liet ke cac nhom huong dan thuong gap trong module D54"
A: {{"mode":"MMR","reason":"diverse groups requested"}}

Q: "So sanh D05F1602 va D06F1011: muc dich, thao tac, luu y"
A: {{"mode":"Decomposition","reason":"comparison with multiple aspects"}}

Q: "Khi quy trinh duyet khong chay dung theo cau hinh thi nen tim dang bai nao?"
A: {{"mode":"HyDE","reason":"conceptual low-overlap search"}}

Detected query signals: {signals}

Question: {question}"""

NO_CONTEXT_ANSWER = (
    "Kh\u00f4ng c\u00f3 th\u00f4ng tin trong c\u01a1 s\u1edf tri th\u1ee9c "
    "\u0111\u00e3 l\u1eadp ch\u1ec9 m\u1ee5c."
)


ALLOWED_ROUTER_MODES = {
    "BM25",
    "Hybrid RRF",
    "Semantic",
    "MMR",
    "Decomposition",
    "HyDE",
}

ROUTER_MODE_ALIASES = {
    "bm25": "BM25",
    "hybrid": "Hybrid RRF",
    "hybrid rrf": "Hybrid RRF",
    "rrf": "Hybrid RRF",
    "semantic": "Semantic",
    "dense": "Semantic",
    "mmr": "MMR",
    "decompose": "Decomposition",
    "decomposition": "Decomposition",
    "hyde": "HyDE",
}

KNOWLEDGE_ID_PATTERNS = [
    re.compile(
        r"\b(?:knowledge|know|kb|article|record|bai|ma bai|ban ghi|id)\s*(?:id)?\s*[:#-]?\s*(\d{1,6})\b",
        flags=re.IGNORECASE,
    ),
]

FORM_CODE_PATTERN = re.compile(r"\b[dw]\d{2}f\d{3,5}\b", flags=re.IGNORECASE)
MODULE_CODE_PATTERN = re.compile(r"\b[dw]\d{2}\b", flags=re.IGNORECASE)
ERROR_CODE_PATTERN = re.compile(r"\b0x[0-9a-f]{3,8}\b", flags=re.IGNORECASE)
QUOTED_TEXT_PATTERN = re.compile(r"[\"'`].+?[\"'`]")
PATH_PATTERN = re.compile(r"([a-z]:\\|\\\\|/[a-z0-9_.-]+/)", flags=re.IGNORECASE)

TECHNICAL_TERMS = {
    "app",
    "bao cao",
    "button",
    "cai dat",
    "cap nhat",
    "cau hinh",
    "column",
    "cot",
    "crm",
    "database",
    "dll",
    "duong dan",
    "email",
    "error",
    "erp",
    "exe",
    "file",
    "form",
    "ftp",
    "grid",
    "hardwareid",
    "hoa don",
    "invalid object name",
    "label",
    "license",
    "luu",
    "man hinh",
    "mau in",
    "menu",
    "module",
    "nut",
    "object",
    "not found",
    "object name",
    "path",
    "phan quyen",
    "phieu",
    "popup",
    "quyen",
    "report",
    "sql",
    "tab",
    "table",
    "user",
    "web6",
    "workflow",
}

SEMANTIC_TERMS = {
    "bai nao",
    "co loi nao",
    "co tai lieu nao",
    "gan giong",
    "giong vay",
    "giao dien",
    "khong chay",
    "khong hien",
    "khong ghi nhan",
    "khong mo duoc",
    "khong phan hoi",
    "lien quan",
    "mong doi",
    "nguoi dung",
    "nhu vay",
    "thao tac",
    "tim bai",
    "tim cac bai",
    "trai nghiem",
    "tuong tu",
}

DIVERSITY_TERMS = {
    "cac nhom",
    "cac loai",
    "cac truong hop",
    "da dang",
    "different",
    "group",
    "huong dan thuong gap",
    "khac nhau",
    "loai loi",
    "nhieu vi du",
    "overview",
    "tong quan",
    "thuong gap",
}

CONCEPTUAL_TERMS = {
    "bat dau tu dau",
    "cach trinh bay",
    "can doi",
    "can tim gi",
    "hanh vi thuc te",
    "khac cau hinh",
    "khac ky vong",
    "khai niem",
    "nghiep vu",
    "nen tim",
    "quy trinh",
    "quy trinh duyet",
    "the hien",
    "thiet ke",
    "truong hop nao",
    "tu nhien",
}

ASPECT_TERMS = {
    "buoc",
    "cach lam",
    "cach thuc hien",
    "cach xu ly",
    "huong dan",
    "ket qua test",
    "luu y",
    "muc dich",
    "mo ta",
    "nguyen nhan",
    "quy trinh",
    "solution",
    "thao tac",
    "test result",
}


def normalize_query(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").casefold())
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.replace("\u0111", "d").replace("\u00c4\u2018", "d")


def count_terms(query: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in query)


def canonicalize_router_mode(mode: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", (mode or "").strip())
    if cleaned in ALLOWED_ROUTER_MODES:
        return cleaned
    return ROUTER_MODE_ALIASES.get(normalize_query(cleaned))


def extract_knowledge_ids(question: str) -> list[int]:
    ids: list[int] = []
    normalized = normalize_query(question)
    for pattern in KNOWLEDGE_ID_PATTERNS:
        for match in pattern.finditer(normalized):
            knowledge_id = int(match.group(1))
            if knowledge_id not in ids:
                ids.append(knowledge_id)
    return ids


def router_signals(question: str) -> dict[str, Any]:
    normalized = normalize_query(question)
    module_codes = set(MODULE_CODE_PATTERN.findall(normalized))
    form_codes = set(FORM_CODE_PATTERN.findall(normalized))
    error_codes = set(ERROR_CODE_PATTERN.findall(normalized))

    return {
        "knowledge_ids": extract_knowledge_ids(question),
        "module_count": len(module_codes),
        "form_count": len(form_codes),
        "error_code_count": len(error_codes),
        "aspect_count": count_terms(normalized, ASPECT_TERMS),
        "technical_count": count_terms(normalized, TECHNICAL_TERMS),
        "semantic_count": count_terms(normalized, SEMANTIC_TERMS),
        "diversity_count": count_terms(normalized, DIVERSITY_TERMS),
        "conceptual_count": count_terms(normalized, CONCEPTUAL_TERMS),
        "has_quote": bool(QUOTED_TEXT_PATTERN.search(question or "")),
        "has_path": bool(PATH_PATTERN.search(normalized)),
        "question_marks": (question or "").count("?"),
    }


def route_by_rules(question: str) -> tuple[str, str] | None:
    signals = router_signals(question)
    normalized = normalize_query(question)

    if signals["knowledge_ids"]:
        return "BM25", "rule: knowledge id detected"

    multi_aspect = signals["aspect_count"] >= 3 and ("," in (question or "") or " va " in normalized)
    comparison = any(term in normalized for term in {"so sanh", "compare", "dong thoi"})
    if comparison or multi_aspect or signals["question_marks"] > 1:
        return "Decomposition", "rule: multi-part question"

    if signals["diversity_count"]:
        return "MMR", "rule: diverse examples or groups requested"

    strong_keyword = (
        signals["has_quote"]
        or signals["has_path"]
        or signals["form_count"] > 0
        or signals["error_code_count"] > 0
        or "invalid object name" in normalized
    )
    technical_count = (
        signals["technical_count"]
        + signals["module_count"]
        + signals["form_count"]
        + signals["error_code_count"]
    )
    semantic_count = signals["semantic_count"]

    if strong_keyword:
        return "BM25", "rule: exact technical keyword"
    if signals["conceptual_count"] >= 2 and technical_count <= 1:
        return "HyDE", "rule: conceptual low-overlap query"
    if technical_count and semantic_count:
        return "Hybrid RRF", "rule: keyword plus semantic intent"
    if technical_count >= 2:
        return "BM25", "rule: keyword-heavy query"
    if semantic_count >= 2 and technical_count == 0:
        return "Semantic", "rule: paraphrased similarity query"

    return None


def parse_focused_answer(text: str) -> str:
    text = (text or "").strip()
    if "[TRẢ LỜI]:" in text:
        text = text.split("[TRẢ LỜI]:")[-1].strip()
    text = re.sub(r"^\s*[\u2022\-\*]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_context(documents: list[RetrievedDocument]) -> str:
    formatted: list[str] = []
    seen: set[str] = set()
    for index, doc in enumerate(documents, start=1):
        content = (doc.text or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        metadata = doc.metadata
        source = metadata.get("source_doc_id") or metadata.get("doc_id") or doc.id
        header_parts = [f"[{index}] Source={source}"]
        for label, key in [
            ("KnowledgeID", "knowledge_id"),
            ("Keywords", "keywords"),
            ("Title", "title"),
        ]:
            value = metadata.get(key)
            if value not in (None, ""):
                header_parts.append(f"{label}={value}")
        formatted.append(f"{'; '.join(header_parts)}\n{content}")
    return "\n\n".join(formatted)


def merge_exact_matches(
    exact_docs: list[RetrievedDocument],
    retrieved_docs: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    merged: list[RetrievedDocument] = []
    seen: set[str] = set()
    for doc in [*exact_docs, *retrieved_docs]:
        if doc.id in seen:
            continue
        seen.add(doc.id)
        merged.append(doc)
    return merged


def choose_retrieval_mode(config: AppConfig, question: str) -> tuple[str, str]:
    rule_route = route_by_rules(question)
    if rule_route:
        return rule_route

    try:
        router_config = config.with_overrides(
            ollama_host=config.local_ollama_host,
            ollama_model=config.router_model,
        )
        signals = json.dumps(router_signals(question), ensure_ascii=True, separators=(",", ":"))
        raw = chat(
            router_config,
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": ROUTER_TEMPLATE.format(question=question, signals=signals)},
            ],
            response_format="json",
            options_override={
                "temperature": 0,
                "top_p": 0.1,
                "num_ctx": 2048,
                "num_predict": 512,
            },
        )
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        mode = canonicalize_router_mode(str(payload.get("mode", "")))
        reason = str(payload.get("reason", "")).strip()
    except Exception as exc:
        return "Hybrid RRF", f"router fallback: {exc.__class__.__name__}"

    if not mode:
        return "Hybrid RRF", f"router returned unsupported mode: {payload.get('mode') or 'empty'}"
    return mode, reason or "model routed"


def retrieve_documents(
    config: AppConfig,
    store: ChromaKnowledgeStore,
    bm25: BM25Index,
    question: str,
    mode: str,
    k: int,
    fetch_k: int,
    lambda_mult: float,
) -> tuple[list[RetrievedDocument], dict[str, Any]]:
    mode_key = mode.lower()
    diagnostics: dict[str, Any] = {}
    exact_docs: list[RetrievedDocument] = []
    exact_knowledge_ids = extract_knowledge_ids(question)
    if exact_knowledge_ids:
        for knowledge_id in exact_knowledge_ids:
            exact_docs.extend(store.get_by_knowledge_id(knowledge_id))
        diagnostics["exact_knowledge_ids"] = exact_knowledge_ids
        diagnostics["exact_knowledge_matches"] = [doc.id for doc in exact_docs]

    if mode_key == "semantic":
        return merge_exact_matches(exact_docs, semantic_search(store, question, k=k))[:k], diagnostics

    if mode_key == "bm25":
        diagnostics["retriever"] = "BM25Okapi"
        return merge_exact_matches(exact_docs, bm25_search(bm25, question, k=k))[:k], diagnostics

    if mode_key == "mmr":
        return (
            merge_exact_matches(
                exact_docs,
                mmr_search(store, question, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult),
            )[:k],
            diagnostics,
        )

    if mode_key == "hyde":
        hypothetical_doc = chat(
            config,
            [{"role": "user", "content": HYDE_TEMPLATE.format(question=question)}],
        )
        diagnostics["hypothetical_document"] = hypothetical_doc
        return merge_exact_matches(exact_docs, semantic_search(store, hypothetical_doc, k=k))[:k], diagnostics

    if mode_key == "decomposition":
        raw = chat(
            config,
            [{"role": "user", "content": DECOMPOSE_TEMPLATE.format(question=question)}],
        )
        sub_questions = [
            re.sub(r"^\s*[\d\.\-\*\)]+\s*", "", line).strip()
            for line in raw.splitlines()
            if line.strip()
        ]
        sub_questions = [q for q in sub_questions if q][:4]
        diagnostics["sub_questions"] = sub_questions
        ranked_lists = [semantic_search(store, question, k=fetch_k)]
        ranked_lists.extend(semantic_search(store, sub_q, k=fetch_k) for sub_q in sub_questions)
        return (
            merge_exact_matches(
                exact_docs,
                reciprocal_rank_fusion(ranked_lists, limit=k, rrf_k=config.rrf_k),
            )[:k],
            diagnostics,
        )

    return (
        merge_exact_matches(
            exact_docs,
            hybrid_rrf_search(
                store,
                bm25,
                question,
                k=k,
                fetch_k=fetch_k,
                rrf_k=config.rrf_k,
            ),
        )[:k],
        diagnostics,
    )


def prepare_answer_request(
    config: AppConfig,
    store: ChromaKnowledgeStore,
    bm25: BM25Index,
    question: str,
    mode: str = "BM25",
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> PreparedRagRequest:
    router_diagnostics: dict[str, Any] = {}
    if mode.lower() in {"auto", "auto router", "router"}:
        selected_mode, reason = choose_retrieval_mode(config, question)
        router_diagnostics = {
            "requested_mode": mode,
            "selected_mode": selected_mode,
            "reason": reason,
        }
        mode = selected_mode

    documents, diagnostics = retrieve_documents(
        config=config,
        store=store,
        bm25=bm25,
        question=question,
        mode=mode,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    if router_diagnostics:
        diagnostics = {"router": router_diagnostics, **diagnostics}

    context = format_context(documents)
    if not context:
        return PreparedRagRequest(
            messages=[],
            sources=[],
            mode=mode,
            diagnostics=diagnostics,
            fallback_answer=NO_CONTEXT_ANSWER,
        )

    prompt = ANSWER_TEMPLATE.format(context=context, question=question)
    return PreparedRagRequest(
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        sources=documents,
        mode=mode,
        diagnostics=diagnostics,
    )


def answer_question(
    config: AppConfig,
    store: ChromaKnowledgeStore,
    bm25: BM25Index,
    question: str,
    mode: str = "BM25",
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> RagResponse:
    prepared = prepare_answer_request(
        config=config,
        store=store,
        bm25=bm25,
        question=question,
        mode=mode,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    if prepared.fallback_answer is not None:
        return RagResponse(
            answer=prepared.fallback_answer,
            sources=prepared.sources,
            mode=prepared.mode,
            diagnostics=prepared.diagnostics,
        )

    answer = chat(config, prepared.messages)
    return RagResponse(
        answer=parse_focused_answer(answer),
        sources=prepared.sources,
        mode=prepared.mode,
        diagnostics=prepared.diagnostics,
    )


def stream_answer_question(
    config: AppConfig,
    store: ChromaKnowledgeStore,
    bm25: BM25Index,
    question: str,
    mode: str = "BM25",
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> RagStreamResponse:
    prepared = prepare_answer_request(
        config=config,
        store=store,
        bm25=bm25,
        question=question,
        mode=mode,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    chunks = (
        [prepared.fallback_answer]
        if prepared.fallback_answer is not None
        else chat_stream(config, prepared.messages)
    )
    return RagStreamResponse(
        chunks=chunks,
        sources=prepared.sources,
        mode=prepared.mode,
        diagnostics=prepared.diagnostics,
    )
