"""graph.state — LangGraph state 정의 (스텁).

Task 15 구현 예정. 현재는 docstring + 최소 플레이스홀더만 둔다.

State 스키마 (예정):
    - question: str — 사용자 질문
    - documents: list[dict] — VectorDB 검색 결과 (chunk + metadata)
    - grade: str | None — 문서 적합도 판정 (relevant / irrelevant)
    - rewritten_question: str | None — grade 실패 시 재작성된 질문
    - generation: str | None — 최종 답변 (citation 포함)
    - human_review_required: bool — 답변 검증 실패 시 human review 큐 라우팅 여부
"""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """LangGraph 상태.

    Attributes:
        question: 원본 사용자 질문.
        documents: 검색된 문서 청크 목록.
        grade: 문서 적합도 (relevant/irrelevant).
        rewritten_question: 재작성된 질문 (rewrite 노드 산출).
        generation: 생성된 답변.
        human_review_required: human review 큐 라우팅 여부.
    """

    question: str
    documents: list[dict]
    grade: str | None
    rewritten_question: str | None
    generation: str | None
    human_review_required: bool
