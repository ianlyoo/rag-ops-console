"""monitor.feedback_to_sheet — Google Sheets 적재 + 추천 질문 큐레이션 (스텁).

분류된 피드백을 Google Sheets(gspread)에 적재하고,
빈도/실패 유형 기반 상위 5개 추천 질문을 큐레이션한다.

- 적재: `gspread`로 Sheets에 행 단위 append (query, response, failure_category, root_cause 등)
- 큐레이션: 부정/버그/개선 피드백 빈도 집계 → Top-5 추천 질문 추출
- 운영: 카카오뱅크 스타일 — 고객 피드백 루프를 Sheets에서 모니터링하고
  추천 질문을 선제안 콘텐츠로 운영

Task 16에서 실제 gspread 연동 + 큐레이션 로직 구현 예정.
현재는 docstring + 시그니처만 두며, 로컬에서는 CSV 모킹으로 대체 가능하다.
"""

from __future__ import annotations


def append_to_sheet(rows: list[dict], sheet_url: str | None = None) -> int:
    """분류된 피드백 행을 Google Sheets에 적재 (스텁).

    Args:
        rows: 적재할 피드백 dict 목록.
        sheet_url: 대상 Google Sheets URL (None이면 모킹 CSV로 대체).

    Returns:
        적재된 행 수. 현재 스텁은 0을 반환한다. Task 16에서 구현 예정.
    """
    _ = (rows, sheet_url)
    return 0


def curate_top_questions(rows: list[dict], top_k: int = 5) -> list[dict]:
    """피드백 기반 추천 질문 Top-K 큐레이션 (스텁).

    Args:
        rows: 분류된 피드백 목록.
        top_k: 추출할 추천 질문 수 (기본 5).

    Returns:
        추천 질문 목록 — 각 항목은 question/frequency/category를 포함.
        현재 스텁은 빈 목록을 반환한다. Task 16에서 구현 예정.
    """
    _ = (rows, top_k)
    return []
