"""Embedding backends and deterministic offline fallback for the vector store."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Embedding — sentence-transformers with hash fallback
# ---------------------------------------------------------------------------
_model_cache: dict[str, Any] = {}


def _resolve_model_name() -> str:
    """환경변수 RAG_EMBEDDING_MODEL > multilingual 기본 > mini 기본."""
    env = os.environ.get("RAG_EMBEDDING_MODEL", "").strip()
    if env:
        return env
    return "paraphrase-multilingual-MiniLM-L12-v2"


def _load_sentence_transformer(model_name: str):  # type: ignore[no-untyped-def]
    """SentenceTransformer 로드 — 캐시. 실패 시 None 반환 (caller가 fallback)."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        return model
    except Exception as e:
        print(f"[vector_store] sentence-transformers load failed ({model_name}): {e}")
        _model_cache[model_name] = None
        return None


def _hash_embed(texts: list[str], dim: int = 384) -> list[list[float]]:
    """결정론적 hash 기반 임베딩 — 모델 다운로드 실패/오프라인 폴백.

    각 텍스트를 토큰 단위로 해시하여 dim 차원 sparse 벡터 생성 후 L2 정규화.
    Chroma cosine 유사도와 호환되도록 정규화한다.
    """
    import math

    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            tokens = [text.lower()]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % dim
            # TF 가중치 유사 효과: 동일 토큰 반복 시 누적
            vec[idx] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        vectors.append(vec)
    return vectors


def _embed(texts: list[str]) -> list[list[float]]:
    """임베딩 — sentence-transformers 우선, 실패 시 hash fallback.

    Env RAG_EMBEDDING_MODEL=hash 이면 즉시 hash fallback (baseline 재현용).
    """
    if not texts:
        return []
    model_name = _resolve_model_name()
    # baseline repro: hash 강제 시 ST 시도 없이 즉시 hash
    if model_name == "hash":
        return _hash_embed(texts)
    model = _load_sentence_transformer(model_name)
    if model is not None:
        try:
            embeddings = model.encode(texts, normalize_embeddings=True)
            # encode returns numpy array
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()  # type: ignore[no-any-return]
            return [list(e) for e in embeddings]
        except Exception as e:
            print(f"[vector_store] embedding encode failed, hash fallback: {e}")
    # 폴백: all-MiniLM-L6-v2 시도
    if model_name != "all-MiniLM-L6-v2":
        model2 = _load_sentence_transformer("all-MiniLM-L6-v2")
        if model2 is not None:
            try:
                embeddings = model2.encode(texts, normalize_embeddings=True)
                if hasattr(embeddings, "tolist"):
                    return embeddings.tolist()  # type: ignore[no-any-return]
                return [list(e) for e in embeddings]
            except Exception:
                pass
    return _hash_embed(texts)
