from __future__ import annotations

from rag_app.rag import extract_knowledge_ids, route_by_rules, router_signals


def test_router_detects_knowledge_ids():
    assert extract_knowledge_ids("knowledge 51493") == [51493]
    assert extract_knowledge_ids("bai id 29013") == [29013]


def test_router_signals_detect_oms_form_codes():
    signals = router_signals("D05F1602 va W89F1000 cau hinh report nhu the nao?")

    assert signals["form_count"] == 2
    assert signals["module_count"] == 0


def test_router_routes_exact_technical_queries_to_bm25():
    mode, reason = route_by_rules('Man hinh D03F1000 tab "Thiet lap ma chung tu" bi loi gi?')

    assert mode == "BM25"
    assert "exact technical" in reason


def test_router_routes_exact_knowledge_id_to_bm25():
    mode, reason = route_by_rules("knowledge 51493")

    assert mode == "BM25"
    assert "knowledge id" in reason


def test_router_routes_exact_w_form_to_bm25():
    mode, reason = route_by_rules("W89F1000 bao cao tai LW6 HR portal")

    assert mode == "BM25"
    assert "exact technical" in reason


def test_router_routes_keyword_plus_semantic_to_hybrid():
    mode, reason = route_by_rules(
        "Tim cac bai lien quan den in hoa don nhung thieu dong hoac sai mau in"
    )

    assert mode == "Hybrid RRF"
    assert "keyword plus semantic" in reason


def test_router_routes_knowledge_keyword_plus_semantic_to_hybrid():
    mode, reason = route_by_rules(
        "Tim cac bai lien quan den phan quyen user tren Web6 nhung khong thay menu"
    )

    assert mode == "Hybrid RRF"
    assert "keyword plus semantic" in reason


def test_router_routes_paraphrase_to_semantic():
    mode, reason = route_by_rules(
        "Nguoi dung sua du lieu nhung he thong khong ghi nhan thay doi, co bai nao giong vay khong?"
    )

    assert mode == "Semantic"
    assert "paraphrased" in reason


def test_router_routes_diverse_requests_to_mmr():
    mode, reason = route_by_rules("Liet ke cac nhom huong dan khac nhau thuong gap trong module D54")

    assert mode == "MMR"
    assert "diverse" in reason


def test_router_routes_multi_part_questions_to_decomposition():
    mode, reason = route_by_rules(
        "So sanh loi khong luu duoc du lieu va loi khong in duoc report: mo ta, cause, solution"
    )

    assert mode == "Decomposition"
    assert "multi-part" in reason


def test_router_routes_conceptual_queries_to_hyde():
    mode, reason = route_by_rules(
        "Khi chuc nang nghiep vu hoat dong khac ky vong sau cau hinh, nen tim dang bai nao?"
    )

    assert mode == "HyDE"
    assert "conceptual" in reason
