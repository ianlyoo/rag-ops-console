"""monitor.feedback_classifier — 고객 피드백 분류 (스텁).

고객 피드백을 `긍정 / 부정 / 버그 / 개선` 4개 카테고리로 분류하고,
root-cause 태그를 부여한다.

스키마 (예정):
    - query: str — 원본 질문
    - response: str — 모델 답변
    - user_feedback: str — 고객이 남긴 피드백 텍스트
    - failure_category: str — 긍정 | 부정 | 버그 | 개선
    - root_cause: str | None — 예: retrieval_miss, hallucination, policy_violation 등
    - review_status: str — pending | reviewed | escalated

Task 16에서 실제 분류 로직 구현 예정 — 현재는 docstring + 시그니처만 둔다.
"""

from __future__ import annotations


def classify_feedback(
    query: str,
    response: str,
    user_feedback: str,
) -> dict:
    """고객 피드백을 긍정/부정/버그/개선 + root-cause로 분류 (스텁).

    Args:
        query: 원본 사용자 질문.
        response: 모델이 생성한 답변.
        user_feedback: 고객이 남긴 피드백 텍스트.

    Returns:
        분류 결과 dict — keys: failure_category, root_cause, review_status.
        현재 스텁은 빈 분류를 반환한다. Task 16에서 구현 예정.
    """
    _ = (query, response, user_feedback)
    return {
        "failure_category": "",
        "root_cause": None,
        "review_status": "pending",
    }
