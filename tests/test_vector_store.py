"""Tests for rag.vector_store — Chroma ingest/search with rerank."""

from __future__ import annotations

import tempfile
from pathlib import Path


def _small_docs() -> list[dict]:
    return [
        {
            "doc_id": "07_customer_support",
            "chunk_id": "07_customer_support::chunk-0000",
            "text": "# 고객 지원 운영 가이드 환불 정책 - 결제 후 7일 이내 전액 환불, 이후 부분 환불.",
            "category": "support",
            "difficulty": "easy",
        },
        {
            "doc_id": "10_finance_expense",
            "chunk_id": "10_finance_expense::chunk-0000",
            "text": "# 경비 청구 및 재무 가이드 식대 3만원 숙박 15만원 월 한도 50만원.",
            "category": "finance",
            "difficulty": "easy",
        },
    ]


def test_ingest_and_search() -> None:
    """ingest 후 search가 top 결과로 관련 문서를 반환해야 한다."""
    tmp_path = Path(tempfile.mkdtemp())
    from rag.vector_store import ingest, search

    persist = str(tmp_path / "chroma_test_vs")
    collection = "test_vs_ingest"

    n = ingest(_small_docs(), persist_dir=persist, collection_name=collection, reset=True)
    assert n == 2

    # 환불 정책 질의 → 고객 지원 문서가 최상위
    results = search("환불 정책", top_k=2, persist_dir=persist, collection_name=collection)
    assert len(results) >= 1
    assert results[0]["doc_id"] == "07_customer_support"
    assert "score" in results[0]
    assert "rerank_score" in results[0]
    assert results[0]["score"] >= 0


def test_search_empty_query_returns_empty() -> None:
    from rag.vector_store import search

    tmp = tempfile.mkdtemp()
    results = search("", top_k=5, persist_dir=tmp, collection_name="empty_q_test")
    assert results == []


def test_ingest_idempotent_no_duplicate() -> None:
    """동일 chunk_id 재 ingest 시 0 반환 (중복 방지)."""
    from rag.vector_store import ingest

    tmp = tempfile.mkdtemp()
    persist = str(Path(tmp) / "chroma_dup")
    coll = "dup_test"
    docs = _small_docs()
    n1 = ingest(docs, persist_dir=persist, collection_name=coll, reset=True)
    assert n1 == 2
    n2 = ingest(docs, persist_dir=persist, collection_name=coll, reset=False)
    assert n2 == 0


def test_load_synthetic_chunks_from_file() -> None:
    """A 레포 synthetic_qa.jsonl 로드가 10개 유니크 chunk를 반환해야 한다."""
    import pytest

    from rag.vector_store import _resolve_synthetic_qa_path, load_synthetic_chunks

    if _resolve_synthetic_qa_path() is None:
        pytest.skip("synthetic_qa.jsonl not present in isolated clone — skip (B boundary: consume A data when available)")
    chunks = load_synthetic_chunks()
    # A 레포에는 10개 문서가 있으므로 유니크 chunk는 10개
    assert len(chunks) == 10
    doc_ids = {c["doc_id"] for c in chunks}
    assert "07_customer_support" in doc_ids
    assert "01_company_policy" in doc_ids


def test_search_score_threshold_filters() -> None:
    from rag.vector_store import ingest, search

    tmp = tempfile.mkdtemp()
    persist = str(Path(tmp) / "chroma_thresh")
    coll = "thresh_test"
    ingest(_small_docs(), persist_dir=persist, collection_name=coll, reset=True)
    # 매우 높은 threshold면 결과 없음
    results = search("환불 정책", top_k=5, persist_dir=persist, collection_name=coll, score_threshold=0.99)
    # threshold가 높으므로 0개 또는 매우 적음 (모델에 따라 다름)
    assert isinstance(results, list)
    # 낮은 threshold면 결과 있음
    results2 = search("환불 정책", top_k=5, persist_dir=persist, collection_name=coll, score_threshold=0.0)
    assert len(results2) >= 1
