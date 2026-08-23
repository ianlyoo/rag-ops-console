"""graph.state — LangGraph state 정의.

LangGraph StateGraph에서 사용되는 상태 스키마.
GraphState 8개 필드:
    query, retrieved, graded, rewritten_query, generation, citations,
    review_status, iterations
"""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """LangGraph RAG 상태.

    Attributes:
        query: 원본 사용자 질문.
        retrieved: VectorDB 검색 결과 목록 (각 항목 doc_id/chunk_id/text/score 포함).
        graded: 문서 적합도 판정 — "relevant" | "irrelevant" | None.
        rewritten_query: grade 실패 시 재작성된 질문 (재시도용).
        generation: 생성된 답변 (citation 포함).
        citations: 답변에 포함된 인용 목록 [{doc_id, chunk_id}].
        review_status: human review 상태 — "approved" | "needs_review" | "pending".
        iterations: 재시도 횟수 (rewrite→retrieve 루프 카운터).
    """

    query: str
    retrieved: list[dict]
    graded: str | None
    rewritten_query: str | None
    generation: str | None
    citations: list[dict]
    review_status: str
    iterations: int
