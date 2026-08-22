"""rag.vector_store — Chroma PersistentClient 기반 VectorDB.

실명 VectorDB Chroma를 사용한 임베딩·검색·rerank 모듈.

- Chroma PersistentClient: ./data/chroma (로컬 persistent directory)
- Embedding: sentence-transformers (로컬, API 키 불필요)
  기본 `paraphrase-multilingual-MiniLM-L12-v2` (한국어 고려, Task 명세 권장)
  폴백 `all-MiniLM-L6-v2` / hash-embedding (오프라인 / 다운로드 실패 시)

API:
    ingest(docs, persist_dir, collection_name, reset) -> int
    search(query, top_k, persist_dir, collection_name) -> list[dict]

데이터 소스:
    - A 레포 `data/synthetic_qa.jsonl`의 source_chunks를 절대경로로 로드
      (복사 금지 — 경로 참조/로드만). 환경변수 EVAL_PIPELINE_HOME 또는
      하드코딩 폴백 경로 `C:/Users/torch/Documents/code/llm-eval-pipeline/...`
    - B 자체 `data/` 샘플 + synthetic_chunks 모두 Chroma에 적재
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parents[1] / "data" / "chroma")
DEFAULT_COLLECTION = "rag_chunks"
SYNTHETIC_QA_FALLBACK = Path(
    r"C:\Users\torch\Documents\code\llm-eval-pipeline\data\synthetic_qa.jsonl"
)

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
    """임베딩 — sentence-transformers 우선, 실패 시 hash fallback."""
    if not texts:
        return []
    model_name = _resolve_model_name()
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


# ---------------------------------------------------------------------------
# Chroma client helpers
# ---------------------------------------------------------------------------

def get_chroma_client(persist_dir: str | Path | None = None) -> chromadb.PersistentClient:
    """Chroma PersistentClient 반환 (실명 VectorDB)."""
    p = str(persist_dir or DEFAULT_PERSIST_DIR)
    Path(p).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=p)


def _get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str = DEFAULT_COLLECTION,
) -> Any:
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


# ---------------------------------------------------------------------------
# Synthetic QA 로드 (A 레포 — 복사 금지, 경로 참조만)
# ---------------------------------------------------------------------------

def _resolve_synthetic_qa_path() -> Path | None:
    """EVAL_PIPELINE_HOME 환경변수 > 폴백 절대경로."""
    env_home = os.environ.get("EVAL_PIPELINE_HOME", "").strip()
    if env_home:
        p = Path(env_home) / "data" / "synthetic_qa.jsonl"
        if p.is_file():
            return p
    if SYNTHETIC_QA_FALLBACK.is_file():
        return SYNTHETIC_QA_FALLBACK
    # 상대경로 폴백: ../../llm-eval-pipeline/data/synthetic_qa.jsonl
    rel = Path(__file__).resolve().parents[1].parent / "llm-eval-pipeline" / "data" / "synthetic_qa.jsonl"
    if rel.is_file():
        return rel
    return None


def load_synthetic_chunks() -> list[dict[str, Any]]:
    """A 레포 synthetic_qa.jsonl에서 source_chunks 추출.

    Returns:
        list of {doc_id, chunk_id, text, metadata: {source, category, difficulty}}
    """
    path = _resolve_synthetic_qa_path()
    if path is None or not path.is_file():
        print(f"[vector_store] synthetic_qa.jsonl not found — skip (path={path})")
        return []
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sc in obj.get("source_chunks", []):
                cid = sc.get("chunk_id", "")
                if cid in seen:
                    continue
                seen.add(cid)
                chunks.append(
                    {
                        "doc_id": sc.get("doc_id", "unknown"),
                        "chunk_id": cid,
                        "text": sc.get("text", ""),
                        "category": obj.get("category", ""),
                        "difficulty": obj.get("difficulty", ""),
                    }
                )
    return chunks


def load_local_sample_docs() -> list[dict[str, Any]]:
    """B 자체 data/ 하위 샘플 문서가 있으면 로드 (선택). 현재는 synthetic 기반."""
    # data/*.md 또는 data/*.jsonl 이 있으면 추가 — 없으면 빈 목록
    data_dir = Path(__file__).resolve().parents[1] / "data"
    docs: list[dict[str, Any]] = []
    for pattern in ["*.md", "*.jsonl"]:
        for p in data_dir.glob(pattern):
            if p.name == "synthetic_qa.jsonl":
                continue
            try:
                if p.suffix == ".md":
                    text = p.read_text(encoding="utf-8")
                    docs.append(
                        {
                            "doc_id": p.stem,
                            "chunk_id": f"{p.stem}::chunk-0000",
                            "text": text,
                            "category": "local",
                            "difficulty": "medium",
                        }
                    )
            except Exception:
                continue
    return docs


# ---------------------------------------------------------------------------
# Public API: ingest / search
# ---------------------------------------------------------------------------

def ingest(
    docs: list[dict[str, Any]] | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = False,
) -> int:
    """문서 청크를 Chroma에 임베딩 후 적재.

    Args:
        docs: [{doc_id, chunk_id, text, ...}] — None이면 synthetic_qa + 로컬 샘플 자동 로드.
        persist_dir: Chroma persistent directory (기본 ./data/chroma).
        collection_name: Chroma 컬렉션 이름.
        reset: True이면 기존 컬렉션 삭제 후 재생성.

    Returns:
        적재된 문서 수.
    """
    if docs is None:
        docs = load_synthetic_chunks()
        docs.extend(load_local_sample_docs())
        if not docs:
            print("[vector_store] no docs to ingest")
            return 0

    # text 비어있는 항목 필터
    docs = [d for d in docs if d.get("text", "").strip()]
    if not docs:
        return 0

    client = get_chroma_client(persist_dir)

    if reset:
        import contextlib

        with contextlib.suppress(Exception):
            client.delete_collection(name=collection_name)

    collection = _get_or_create_collection(client, collection_name)

    # 기존 ids 조회 — 중복 방지
    try:
        existing = set(collection.get(include=[])["ids"])
    except Exception:
        existing = set()

    new_docs = [d for d in docs if d.get("chunk_id") not in existing]
    if not new_docs:
        print(f"[vector_store] all {len(docs)} docs already ingested — skip")
        return 0

    ids = [d["chunk_id"] for d in new_docs]
    texts = [d["text"] for d in new_docs]
    metadatas: list[dict[str, Any]] = []
    for d in new_docs:
        metadatas.append(
            {
                "doc_id": str(d.get("doc_id", "")),
                "chunk_id": str(d.get("chunk_id", "")),
                "category": str(d.get("category", "")),
                "difficulty": str(d.get("difficulty", "")),
            }
        )

    embeddings = _embed(texts)

    # Chroma add — batch
    batch_size = 50
    total = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,  # type: ignore[arg-type]
            embeddings=batch_embs,
        )
        total += len(batch_ids)

    print(f"[vector_store] Chroma ingest: {total} chunks -> {persist_dir or DEFAULT_PERSIST_DIR} / {collection_name}")
    return total


def search(
    query: str,
    top_k: int = 5,
    persist_dir: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Chroma 유사도 검색 + 점수 기반 rerank.

    Args:
        query: 검색 질의.
        top_k: 반환할 상위 문서 수.
        persist_dir: Chroma persistent directory.
        collection_name: 컬렉션 이름.
        score_threshold: 점수 임계값 — 이 점수 미만 문서는 제외 (None이면 비활성화).

    Returns:
        list of {doc_id, chunk_id, text, score, metadata} — score는 cosine 유사도 (높을수록 유사).
        rerank: 점수 내림차순 정렬 + 임계값 필터링 후 반환.
    """
    if not query.strip():
        return []

    client = get_chroma_client(persist_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        # 컬렉션이 없으면 빈 결과
        print(f"[vector_store] collection '{collection_name}' not found — return []")
        return []

    # 임베딩
    q_emb = _embed([query])[0]

    # Chroma query — n_results를 top_k보다 크게 잡아 rerank 여유 확보
    n_fetch = min(top_k * 3, 20)
    try:
        result = collection.query(
            query_embeddings=[q_emb],
            n_results=n_fetch,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[vector_store] query failed: {e}")
        return []

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
    # 키워드 보너스: 질의 토큰이 문서에 포함되면 score에 가중치
    query_tokens = set(re.findall(r"\w+", query.lower()))
    for item in items:
        doc_tokens = set(re.findall(r"\w+", item["text"].lower()))
        overlap = len(query_tokens & doc_tokens)
        # 보너스: overlap 비율 * 0.1 (최대 0.1 가산, rerank 목적)
        bonus = (overlap / max(len(query_tokens), 1)) * 0.1
        item["rerank_score"] = item["score"] + bonus

    items.sort(key=lambda x: x["rerank_score"], reverse=True)

    return items[:top_k]


def ingest_from_synthetic_qa(
    persist_dir: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = False,
) -> int:
    """편의 함수 — A synthetic_qa.jsonl을 로드해 Chroma에 적재."""
    chunks = load_synthetic_chunks()
    if not chunks:
        return 0
    return ingest(chunks, persist_dir=persist_dir, collection_name=collection_name, reset=reset)
