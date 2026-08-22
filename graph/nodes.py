"""graph.nodes — LangGraph 노드 정의 (스텁).

파이프라인: retrieve → grade → rewrite → generate → human review

- retrieve: VectorDB(Chroma)에서 질문 임베딩 기반 문서 검색
- grade: 검색 문서의 질문 적합도 판정 (relevant/irrelevant)
- rewrite: irrelevant 시 질문 재작성 후 retrieve 재시도 (조건부 라우팅)
- generate: 적합 문서 기반 답변 생성 (citation 포함)
- human review: 답변 검증 실패 시 운영자 검토 큐로 라우팅

Task 15에서 실제 LangGraph StateGraph + 조건부 엣지 구현 예정.
현재는 시그니처와 docstring만 제공 — 로직 없음.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph.state import GraphState


def retrieve(state: GraphState) -> GraphState:
    """VectorDB(Chroma) 검색 노드 — 질문 임베딩으로 문서 검색 (스텁)."""
    _ = state
    return {}


def grade(state: GraphState) -> GraphState:
    """검색 문서 적합도 판정 노드 — relevant/irrelevant 분류 (스텁)."""
    _ = state
    return {}


def rewrite(state: GraphState) -> GraphState:
    """질문 재작성 노드 — grade 실패 시 질문을 재작성해 retrieve 재시도 (스텁)."""
    _ = state
    return {}


def generate(state: GraphState) -> GraphState:
    """답변 생성 노드 — 적합 문서 기반 citation 포함 답변 생성 (스텁)."""
    _ = state
    return {}


def human_review(state: GraphState) -> GraphState:
    """Human review 큐 라우팅 노드 — 검증 실패 답변을 운영자 검토 큐로 전달 (스텁)."""
    _ = state
    return {}
