"""graph.nodes — LangGraph 5노드 정의.

파이프라인: retrieve → grade → rewrite → generate → human_review
조건부 라우팅:
    grade == "relevant" → generate
    grade == "irrelevant" + iterations < 1 → rewrite → retrieve (최대 1회 재시도)
    grade == "irrelevant" + iterations >= 1 → generate (가용한 문서로 최선 답변)

모든 노드는 GraphState를 입력받아 부분 업데이트 dict를 반환한다.
StateGraph 빌드는 graph/run.py에서 수행한다.
"""

from __future__ import annotations

import re
from typing import Any

from graph.state import GraphState

# ---------------------------------------------------------------------------
# Policy / 검증 상수
# ---------------------------------------------------------------------------
FORBIDDEN_PHRASES = [
    "100% 환불 보장",
    "무조건 환불",
    "법적 책임 없음",
]

GRADE_SCORE_THRESHOLD = 0.25  # cosine similarity 임계값 (해시 폴백 고려 낮게 설정)
GRADE_OVERLAP_THRESHOLD = 1  # 질의 토큰과 문서 토큰 최소 겹침 수


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict[str, Any]:
    """VectorDB(Chroma) 검색 노드."""
    from rag.vector_store import search

    query = state.get("rewritten_query") or state.get("query", "")
    if not query:
        print("[retrieve] empty query — skip")
        return {"retrieved": []}

    display_q = query[:80]
    print(f"[retrieve] query='{display_q}' (iterations={state.get('iterations', 0)})")

    try:
        results = search(query, top_k=5)
    except Exception as e:
        print(f"[retrieve] search failed: {e}")
        results = []

    print(f"[retrieve] found {len(results)} chunks")
    for r in results[:3]:
        print(f"  - {r.get('chunk_id','')} score={r.get('score',0):.3f} rerank={r.get('rerank_score',0):.3f}")

    return {"retrieved": results}


def grade(state: GraphState) -> dict[str, Any]:
    """검색 문서 적합도 판정 — relevant / irrelevant.

    판정 로직:
    - retrieved가 비어있으면 irrelevant
    - 최상위 문서의 score가 임계값 미만이면 irrelevant
    - 질의 토큰과 문서 토큰 겹침이 0이면 irrelevant
    - 그 외 relevant
    """
    retrieved: list[dict] = state.get("retrieved", [])  # type: ignore[assignment]
    query: str = state.get("rewritten_query") or state.get("query", "")  # type: ignore[assignment]

    if not retrieved:
        print("[grade] no documents -> irrelevant")
        return {"graded": "irrelevant"}

    top_score = retrieved[0].get("rerank_score", retrieved[0].get("score", 0))
    query_tokens = set(re.findall(r"\w+", query.lower()))

    # 최상위 문서와 질의 토큰 겹침 체크
    top_text = retrieved[0].get("text", "")
    top_tokens = set(re.findall(r"\w+", top_text.lower()))
    overlap = len(query_tokens & top_tokens)

    print(f"[grade] top_score={top_score:.3f} overlap={overlap} threshold_score={GRADE_SCORE_THRESHOLD}")

    if top_score < GRADE_SCORE_THRESHOLD and overlap < GRADE_OVERLAP_THRESHOLD:
        print("[grade] -> irrelevant (low score + no overlap)")
        return {"graded": "irrelevant"}
    if overlap == 0 and top_score < 0.5:
        print("[grade] -> irrelevant (no token overlap)")
        return {"graded": "irrelevant"}

    print("[grade] -> relevant")
    return {"graded": "relevant"}


def rewrite(state: GraphState) -> dict[str, Any]:
    """질문 재작성 — 규칙 기반, 최대 1회 재시도.

    재작성 규칙:
    - 원본 질의에 핵심 키워드가 부족하면 동의어/관련어 추가
    - 예: '환불 정책' → '환불 정책 결제 7일 전액 환불'
    """
    query: str = state.get("query", "")  # type: ignore[assignment]
    iterations: int = int(state.get("iterations", 0))  # type: ignore[arg-type]

    # 이미 재작성된 쿼리가 있으면 그대로 유지 (중복 재작성 방지)
    if state.get("rewritten_query"):
        print(f"[rewrite] already rewritten -> skip (iterations={iterations})")
        return {"iterations": iterations + 1}

    # 규칙 기반 재작성
    rewritten = query

    # 한국어 환불 관련
    if "환불" in query:
        if "7일" not in query:
            rewritten = query + " 7일 이내 전액 환불 결제"
        if "정책" in query and "고객 지원" not in rewritten:
            rewritten = rewritten + " 고객 지원"

    # 영어 fallback
    if "refund" in query.lower() and "7 days" not in query.lower():
        rewritten = query + " 7 days full refund payment"

    # 일반: 질의가 짧으면 문서 키워드 추가
    if len(query.strip()) < 10 and "정책" not in query:
        rewritten = query + " 정책 규정"

    rewritten = rewritten.strip()
    new_iterations = iterations + 1

    print(f"[rewrite] '{query}' -> '{rewritten}' (iterations {iterations} -> {new_iterations})")

    return {"rewritten_query": rewritten, "iterations": new_iterations}


def generate(state: GraphState) -> dict[str, Any]:
    """답변 생성 — extractive/template 기반 (LLM API 키 불필요).

    로직:
    - retrieved 중 최상위 문서를 기반으로 답변 템플릿 생성
    - citations는 사용된 문서의 doc_id/chunk_id로 구성
    - 문서가 없으면 fallback 답변 + 빈 citations
    """
    retrieved: list[dict] = state.get("retrieved", [])  # type: ignore[assignment]
    query: str = state.get("query", "")  # type: ignore[assignment]

    if not retrieved:
        answer = (
            f"질문 '{query}'에 대해 관련 문서를 찾지 못했습니다. "
            "고객 지원(support@acme.example)으로 문의해 주세요."
        )
        print("[generate] no documents -> fallback answer")
        return {"generation": answer, "citations": []}

    # 최상위 1~2개 문서를 기반으로 답변 구성
    top = retrieved[0]
    text = top.get("text", "")

    # 환불 정책 질문에 대한 특화 템플릿 (실제 문서 내용 기반)
    if "환불" in query or "refund" in query.lower():
        # 문서에서 환불 정책 부분 추출 시도
        # source 문서(07_customer_support)에는 "결제 후 7일 이내 전액 환불" 포함
        answer = (
            f"환불 정책에 대해 안내드립니다. {text[:400].strip()} "
            f"추가 문의는 support@acme.example 또는 1588-0000으로 연락해 주세요."
        )
    else:
        # 일반 extractive 답변
        snippet = text[:500].strip()
        answer = f"질문 '{query}'에 대한 답변입니다. {snippet}"

    # 정책: 답변 길이가 너무 짧으면 문서 전체 추가
    if len(answer) < 50:
        answer = text[:600]

    citations = [
        {"doc_id": r.get("doc_id", ""), "chunk_id": r.get("chunk_id", "")}
        for r in retrieved[:2]
    ]

    # citation 검증: 최소 1개 citation 필요
    if not citations:
        print("[generate] WARNING: no citations generated")

    print(f"[generate] answer_len={len(answer)} citations={len(citations)}")
    print(f"  citations: {[c['chunk_id'] for c in citations]}")

    return {"generation": answer, "citations": citations}


def human_review(state: GraphState) -> dict[str, Any]:
    """Human review — citation / policy 검증.

    검증:
    - citation이 없으면 needs_review
    - generation에 금지 구문(FORBIDDEN_PHRASES)이 포함되면 needs_review
    - generation이 retrieved 문서 텍스트와 무관하면(표절도 낮으면) needs_review
    - 그 외 approved
    """
    generation: str | None = state.get("generation")
    citations: list[dict] = state.get("citations", [])  # type: ignore[assignment]
    retrieved: list[dict] = state.get("retrieved", [])  # type: ignore[assignment]

    if not generation:
        print("[human_review] no generation -> needs_review")
        return {"review_status": "needs_review"}

    if not citations:
        print("[human_review] no citations -> needs_review (citation 검증 실패)")
        return {"review_status": "needs_review"}

    # 금지 구문 체크
    for phrase in FORBIDDEN_PHRASES:
        if phrase in generation:
            print(f"[human_review] forbidden phrase '{phrase}' -> needs_review")
            return {"review_status": "needs_review"}

    # 문서 기반 검증: generation의 주요 문장이 retrieved에 포함되는지
    # (간단: generation 토큰 중 일부가 retrieved 텍스트에 존재하는지)
    gen_tokens = set(re.findall(r"\w+", generation.lower()))
    if retrieved and gen_tokens:
        doc_text_combined = " ".join(r.get("text", "") for r in retrieved).lower()
        doc_tokens = set(re.findall(r"\w+", doc_text_combined))
        overlap_ratio = len(gen_tokens & doc_tokens) / max(len(gen_tokens), 1)
        print(f"[human_review] token overlap ratio={overlap_ratio:.2f}")
        if overlap_ratio < 0.15:
            print("[human_review] low overlap -> needs_review (hallucination 의심)")
            return {"review_status": "needs_review"}

    print("[human_review] -> approved")
    return {"review_status": "approved"}


# ---------------------------------------------------------------------------
# Routing helpers (graph/run.py에서 StateGraph 조건부 엣지로 사용)
# ---------------------------------------------------------------------------

def route_after_grade(state: GraphState) -> str:
    """grade 결과에 따른 라우팅 결정."""
    graded = state.get("graded", "irrelevant")
    iterations = int(state.get("iterations", 0))  # type: ignore[arg-type]
    if graded == "relevant":
        return "generate"
    # irrelevant이지만 재시도 한도(1회) 내이면 rewrite
    if iterations < 1:
        return "rewrite"
    # 재시도 소진 후에도 irrelevant이면 generate로 진행 (최선 답변)
    return "generate"
