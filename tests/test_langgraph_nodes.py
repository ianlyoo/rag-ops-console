"""Tests for graph.nodes + StateGraph routing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch


def test_retrieve_grade_generate_flow() -> None:
    """retrieve -> grade(relevant) -> generate -> human_review approved 경로."""
    from graph.nodes import generate, grade, human_review, retrieve

    mock_results = [
        {
            "doc_id": "07_customer_support",
            "chunk_id": "07_customer_support::chunk-0000",
            "text": "환불 정책 - 결제 후 7일 이내 전액 환불, 이후 부분 환불.",
            "score": 0.6,
            "rerank_score": 0.65,
            "metadata": {},
        }
    ]

    state: dict = {"query": "환불 정책", "iterations": 0}

    with patch("rag.vector_store.search", return_value=mock_results):
        r = retrieve(state)  # type: ignore[arg-type]
        assert "retrieved" in r
        assert len(r["retrieved"]) == 1

    state2 = {**state, "retrieved": mock_results}
    g = grade(state2)  # type: ignore[arg-type]
    assert g["graded"] == "relevant"

    state3 = {**state2, **g}
    gen = generate(state3)  # type: ignore[arg-type]
    assert "generation" in gen
    assert len(gen["citations"]) >= 1
    assert "환불" in gen["generation"] or "7일" in gen["generation"]

    state4 = {**state3, **gen}
    hr = human_review(state4)  # type: ignore[arg-type]
    assert hr["review_status"] == "approved"


def test_grade_irrelevant_and_rewrite() -> None:
    """검색 결과 없으면 grade irrelevant, rewrite가 쿼리를 확장."""
    from graph.nodes import grade, rewrite, route_after_grade

    state = {"query": "완전히 관련 없는 질의 xyz123", "retrieved": [], "iterations": 0}
    g = grade(state)  # type: ignore[arg-type]
    assert g["graded"] == "irrelevant"

    route = route_after_grade({**state, **g})  # type: ignore[arg-type]
    assert route == "rewrite"

    rw = rewrite(state)  # type: ignore[arg-type]
    # rewrite는 iterations를 증가시켜야 함
    assert rw["iterations"] == 1
    # 환불이 아닌 질의는 그대로이거나 약간 확장
    assert "rewritten_query" in rw


def test_rewrite_limits_one_retry() -> None:
    """iterations >=1 이면 grade irrelevant이어도 generate로 라우팅."""
    from graph.nodes import route_after_grade

    state = {"query": "환불 정책", "retrieved": [], "graded": "irrelevant", "iterations": 1}
    assert route_after_grade(state) == "generate"  # type: ignore[arg-type]
    state0 = {"query": "환불 정책", "retrieved": [], "graded": "irrelevant", "iterations": 0}
    assert route_after_grade(state0) == "rewrite"  # type: ignore[arg-type]


def test_human_review_citation_fail() -> None:
    """citation 없으면 needs_review."""
    from graph.nodes import human_review

    state = {"generation": "어떤 답변", "citations": [], "retrieved": []}
    hr = human_review(state)  # type: ignore[arg-type]
    assert hr["review_status"] == "needs_review"


def test_human_review_forbidden_phrase() -> None:
    """금지 구문 포함 시 needs_review."""
    from graph.nodes import human_review

    state = {
        "generation": "100% 환불 보장해드립니다",
        "citations": [{"doc_id": "x", "chunk_id": "x::0"}],
        "retrieved": [{"text": "100% 환불 보장해드립니다"}],
    }
    hr = human_review(state)  # type: ignore[arg-type]
    assert hr["review_status"] == "needs_review"


def test_state_graph_integration_with_mock_search() -> None:
    """StateGraph 전체 실행 — mock search로 통합 테스트."""
    from graph.run import build_graph

    mock_results = [
        {
            "doc_id": "07_customer_support",
            "chunk_id": "07_customer_support::chunk-0000",
            "text": "환불 정책 - 결제 후 7일 이내 전액 환불.",
            "score": 0.7,
            "rerank_score": 0.8,
            "metadata": {"doc_id": "07_customer_support"},
        }
    ]

    with patch("rag.vector_store.search", return_value=mock_results), patch(
        "rag.vector_store.ingest", return_value=0
    ):
        app = build_graph()
        result = app.invoke(
            {
                "query": "환불 정책",
                "retrieved": [],
                "graded": None,
                "rewritten_query": None,
                "generation": None,
                "citations": [],
                "review_status": "pending",
                "iterations": 0,
            }
        )
        assert result["graded"] == "relevant"
        assert result["generation"] is not None
        assert len(result["citations"]) >= 1
        assert result["review_status"] in ("approved", "needs_review")


def test_rewrite_retrieve_loop() -> None:
    """grade irrelevant -> rewrite -> retrieve 재시도 경로가 동작."""
    from graph.run import build_graph

    call_count = {"n": 0}

    def fake_search(query: str, top_k: int = 5, **kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] == 1:
            return []  # 첫 시도는 빈 결과
        return [
            {
                "doc_id": "07_customer_support",
                "chunk_id": "07_customer_support::chunk-0000",
                "text": "환불 정책 - 결제 후 7일 이내 전액 환불.",
                "score": 0.7,
                "rerank_score": 0.8,
                "metadata": {},
            }
        ]

    with patch("rag.vector_store.search", side_effect=fake_search), patch(
        "rag.vector_store.ingest", return_value=0
    ):
        app = build_graph()
        result = app.invoke(
            {
                "query": "환불 정책",
                "retrieved": [],
                "graded": None,
                "rewritten_query": None,
                "generation": None,
                "citations": [],
                "review_status": "pending",
                "iterations": 0,
            }
        )
        # rewrite 후 재시도하므로 search가 2회 호출되어야 함
        assert call_count["n"] == 2
        assert result["iterations"] >= 1
        assert result["generation"] is not None


def test_sheets_mock_append() -> None:
    """Sheets mock CSV 적재 동작."""
    from monitor.feedback_to_sheet import append_to_sheet

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "sheets_mock.csv"
        rows = [
            {
                "query": "테스트 질문",
                "response": "테스트 답변",
                "failure_category": "pending",
                "root_cause": "relevant",
                "review_status": "approved",
                "citations": "07_customer_support/07_customer_support::chunk-0000",
            }
        ]
        n = append_to_sheet(rows, csv_path=csv_path)
        assert n == 1
        assert csv_path.is_file()
        content = csv_path.read_text(encoding="utf-8")
        assert "테스트 질문" in content
        # gspread 경로는 코드에 존재해야 함
        import monitor.feedback_to_sheet as m

        src = Path(m.__file__).read_text(encoding="utf-8")
        assert "gspread" in src
        assert "CSV mock" in src or "csv" in src.lower()
