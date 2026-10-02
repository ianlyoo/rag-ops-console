"""rag.vector_store — Chroma PersistentClient 기반 VectorDB.

실명 VectorDB Chroma를 사용한 임베딩·검색·rerank 모듈.

- Chroma PersistentClient: ./data/chroma (로컬 persistent directory)
- Embedding: sentence-transformers (로컬, API 키 불필요)
  기본 `paraphrase-multilingual-MiniLM-L12-v2` (Recommended default per design — multilingual, Korean coverage)
  폴백 `all-MiniLM-L6-v2` / hash-embedding (오프라인 / 다운로드 실패 시)

API:
    ingest(docs, persist_dir, collection_name, reset) -> int
    search(query, top_k, persist_dir, collection_name) -> list[dict]

데이터 소스:
    - A 레포 `data/synthetic_qa.jsonl`의 source_chunks 로드
      (복사 금지 — 경로 참조/로드만). 폴백: 환경변수 EVAL_PIPELINE_HOME >
      CLI --input > bundled fixture > 상대경로 ../llm-eval-pipeline/
    - B 자체 `data/` 샘플 + synthetic_chunks 모두 Chroma에 적재
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb

from rag.embeddings import (
    _embed,
)
from rag.embeddings import (
    _hash_embed as _hash_embed,
)
from rag.embeddings import (
    _load_sentence_transformer as _load_sentence_transformer,
)
from rag.embeddings import (
    _model_cache as _model_cache,
)
from rag.embeddings import (
    _resolve_model_name as _resolve_model_name,
)
from rag.retrieval import rank_results

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parents[1] / "data" / "chroma")
DEFAULT_COLLECTION = "rag_chunks"


def _resolve_synthetic_qa_path() -> Path | None:
    # 1. Env var
    env = os.environ.get("EVAL_PIPELINE_HOME", "").strip()
    if env:
        p = Path(env) / "data" / "synthetic_qa.jsonl"
        if p.exists():
            return p
    # 2. CLI --input handled in caller (no fallback here)
    # 3. Bundled fixture (if any)
    bundled = Path(__file__).resolve().parents[1] / "data" / "synthetic_qa.jsonl"
    if bundled.exists():
        return bundled
    # 4. Relative sibling repo (dev convenience, not committed as absolute)
    sibling = Path(__file__).resolve().parents[2] / "llm-eval-pipeline" / "data" / "synthetic_qa.jsonl"
    if sibling.exists():
        return sibling
    return None


SYNTHETIC_QA_FALLBACK = None  # resolved lazily via _resolve_synthetic_qa_path()

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
        적재된 문서 수. Repeated chunk IDs keep the first non-empty document;
        IDs already stored in the collection are skipped.
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

    # Chroma requires unique IDs in each add call. Keep the first occurrence,
    # matching synthetic chunk loading and idempotence across ingest calls.
    new_docs = []
    for doc in docs:
        chunk_id = doc["chunk_id"]
        if chunk_id not in existing:
            new_docs.append(doc)
            existing.add(chunk_id)
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

    return rank_results(
        result, query, top_k, score_threshold,
        disable_rerank=os.environ.get("RAG_DISABLE_RERANK", "").strip().lower() in ("1", "true", "yes"),
    )


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
