"""Behavior tests for customer-feedback classification and curation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from monitor.feedback_classifier import classify_feedback, classify_file
from monitor.feedback_to_sheet import curate_top_questions, run_pipeline


@pytest.mark.parametrize(
    ("feedback", "expected_sentiment", "expected_category"),
    [
        ("정확하고 정말 도움됐어요. 감사합니다!", "positive", "긍정"),
        ("답변이 틀렸고 부정확합니다.", "negative", "정확도"),
        ("500 에러가 발생하고 응답이 없습니다.", "negative", "버그"),
        ("내용이 불충분해요. 더 자세히 설명해 주세요.", "negative", "개선"),
    ],
)
def test_classify_feedback_assigns_sentiment_and_category(
    feedback: str,
    expected_sentiment: str,
    expected_category: str,
) -> None:
    """Given feedback, when classified, then its user-visible triage labels are correct."""
    result = classify_feedback("환불 정책", "결제 후 7일 이내 환불됩니다.", feedback)

    assert result["sentiment"] == expected_sentiment
    assert result["failure_category"] == expected_category


@pytest.mark.parametrize(
    ("feedback", "expected_root_cause"),
    [
        ("관련 문서를 검색하지 못하고 모른다고만 합니다.", "retrieval_miss"),
        ("근거 없는 답변이고 출처가 없습니다.", "citation_missing"),
        ("옛날 정보라서 최신 정책과 다릅니다.", "outdated_doc"),
    ],
)
def test_classify_feedback_maps_root_cause(feedback: str, expected_root_cause: str) -> None:
    """Given a failure signal, when classified, then it records the actionable root cause."""
    result = classify_feedback("환불 정책", "답변", feedback)

    assert result["root_cause"] == expected_root_cause
    assert result["review_status"] == "needs_human_review"


def test_classify_file_returns_all_sample_rows_with_review_state() -> None:
    """Given the sample CSV, when parsed, then all 30 feedback rows are classified."""
    sample = Path(__file__).parents[1] / "monitor" / "data" / "feedback_sample.csv"

    rows, stats = classify_file(sample)

    assert len(rows) == 30
    assert stats["total"] == 30
    assert {"failure_category", "root_cause", "review_status"} <= rows[0].keys()
    assert stats["by_review_status"]["auto_resolved"] == 10


def test_curate_top_questions_returns_five_failure_topic_recommendations() -> None:
    """Given repeated failure topics, when curated, then five ranked questions are returned."""
    rows = [
        {"query": "환불 정책 자세히 알려줘", "failure_category": "개선"},
        {"query": "환불 API 오류 해결 방법", "failure_category": "버그"},
        {"query": "연차 신청 오류", "failure_category": "버그"},
        {"query": "API 인증 방법", "failure_category": "개선"},
        {"query": "VPN 연결 오류", "failure_category": "버그"},
        {"query": "장애 대응 SLA", "failure_category": "개선"},
    ]

    recommendations = curate_top_questions(rows)

    assert len(recommendations) == 5
    assert all(item["category"] != "긍정" for item in recommendations)
    assert {item["keyword"] for item in recommendations} >= {"환불", "연차", "API"}


def test_pipeline_writes_classified_sheet_rows_and_five_recommendations(tmp_path: Path) -> None:
    """Given sample feedback, when the pipeline runs, then its CSV contains classified 30 rows."""
    sample = Path(__file__).parents[1] / "monitor" / "data" / "feedback_sample.csv"
    sheet_csv = tmp_path / "sheets_mock.csv"
    log_path = tmp_path / "feedback.log"

    result = run_pipeline(sample, sheet_csv, log_path)

    with sheet_csv.open(encoding="utf-8", newline="") as source:
        written_rows = list(csv.DictReader(source))
    assert len(written_rows) == 30
    assert all(row["failure_category"] for row in written_rows)
    assert len(result["recommended"]) == 5
    assert "classified 30 rows" in log_path.read_text(encoding="utf-8")
