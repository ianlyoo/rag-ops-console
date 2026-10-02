"""Pure conversion and keyword reranking of Chroma query results."""

from __future__ import annotations

import re
from typing import Any


def rank_results(
    result: dict[str, Any],
    query: str,
    top_k: int,
    score_threshold: float | None,
    *,
    disable_rerank: bool = False,
) -> list[dict[str, Any]]:
    """Preserve cosine thresholds and stable ordering before limiting the result."""
    docs = result.get("documents", [[]])[0] or []
    metas = result.get("metadatas", [[]])[0] or []
    dists = result.get("distances", [[]])[0] or []
    ids = result.get("ids", [[]])[0] or []

    items: list[dict[str, Any]] = []
    for doc_text, meta, dist, cid in zip(docs, metas, dists, ids, strict=False):
        # Chroma cosine distance: 0=동일, 2=반대. score = 1 - distance (유사도)
        score = 1.0 - float(dist) if dist is not None else 0.0
        # hash fallback은 distance 범위가 다를 수 있어 클리핑
        score = max(0.0, min(1.0, score))
        if score_threshold is not None and score < score_threshold:
            continue
        m = meta or {}
        items.append(
            {
                "doc_id": m.get("doc_id", ""),
                "chunk_id": m.get("chunk_id", cid),
                "text": doc_text or "",
                "score": score,
                "distance": float(dist) if dist is not None else None,
                "metadata": m,
            }
        )

    # rerank: 1) 키워드 겹침 보너스 2) 점수 내림차순
    # RAG_DISABLE_RERANK=1 이면 보너스 없이 score 순으로 정렬 (baseline 재현용)
    if disable_rerank:
        for item in items:
            item["rerank_score"] = item["score"]
    else:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        for item in items:
            doc_tokens = set(re.findall(r"\w+", item["text"].lower()))
            overlap = len(query_tokens & doc_tokens)
            # 보너스: overlap 비율 * 0.1 (최대 0.1 가산, rerank 목적)
            bonus = (overlap / max(len(query_tokens), 1)) * 0.1
            item["rerank_score"] = item["score"] + bonus

    items.sort(key=lambda x: x["rerank_score"], reverse=True)

    return items[:top_k]
