# rag-ops-console — 한계와 트레이드오프 (Online / VectorDB / LangGraph)

## 1. Offline / Online 경계

- **B는 online 운영**: VectorDB 검색, LangGraph 라우팅, RAGAS 평가, Sheets 모니터링, Streamlit 대시보드. 무거운 임베딩/Chroma/LangGraph는 CI에서도 동작하되, B에서만 유지.
- **A는 offline 품질**: synthetic 50, Rule, 목 Judge, proxy Hit@5. B는 A 데이터를 입력으로만 소비, 평가 로직은 복사하지 않음 — `diff -r llm-eval-pipeline/eval rag-ops-console/eval` 시 파일명도 내용도 diverge (metrics.py vs ragas_eval.py).
- **이유**: A가 RAGAS를 돌리면 API 비용·불안정, B가 Rule proxy를 돌리면 운영 검색 품질 왜곡. 경계로 책임 분리.

## 2. VectorDB 선택 — Chroma vs Lexical (A)

- **선택**: Chroma PersistentClient + sentence-transformers 임베딩 (경량, 로컬 파일 친화적).
- **vs A lexical**: A는 `rag/pipeline.py` LexicalIndex(토큰 set Jaccard + chunk header). Chroma cosine은 동의어/임베딩 공간에서 Hit 향상 — RAGAS recall 0.627 vs A proxy 0.66은 lexical의 keyword 매칭 편향 대비 의미 분포 반영.
- **트레이드오프**:
  - Chroma: 임베딩 모델 다운로드(사전 적재), persist_dir `data/chroma` 관리, distance->score=1-distance 변환, rerank +0.1 keyword boost로 lexical 가산. but 모델 버전 교체 시 점수 드리프트.
  - Lexical: 무의존·deterministic·빠름, but `환불 정책`->`환불` 형태소 미처리 시 miss.
  - 대안 검토: FAISS(HNSW 빠름, but persist 복잡), Qdrant(서버 필요), Pinecone(클라우드 비용). 로컬 demo·CI 친화성으로 Chroma 선택.

## 3. RAGAS deterministic fallback 한계

- **현재**: 토큰 겹침 기반 precision/recall/faithfulness 근사. ragas 라이브러리 경로는 코드에 명시(`ragas.metrics.*`, `ragas.evaluate`)하나 LLM 키 없으면 fallback.
- **한계**: 진짜 RAGAS는 LLM이 문장 단위 entailment/정확도를 평가. 토큰 overlap 15% threshold는 관대 — precision 0.42가 실 LLM precision보다 높게 나올 수 있음. faithfulness 0.9953도 토큰 set이면 과대평가.
- **트레이드오프**: LLM 평가시 비용·지연·재현성 불안정 vs 토큰 근사 0비용·deterministic. 운영에서는 샘플 50/50 혼합(토큰 자동 + 월 1회 LLM 샘플링)으로 보정 권장.

## 4. LangGraph 라우팅 제한

- **설계**: 5노드 retrieve->grade->rewrite(최대1)->generate->human_review. `route_after_grade`는 iterations>=1이면 rewrite 금지, grade irrelevant라도 generate로 강제 진행.
- **한계**: 1회 rewrite만으로 retrieval_miss 13건 중 복구율은 제한. grade는 top_score 0.25 threshold + 토큰 overlap 2개 — threshold 민감. human_review는 `FORBIDDEN_PHRASES`("100% 환불 보장") + citation 유무 + token overlap <0.15로만 판정, 뉘앙스 판정 불가.
- **트레이드오프**: 노드 추가(예: query_decompose, self-refine)하면 정확도↑ but 지연·비용↑, 스트리밍 중단. 현재는 30초 데모/운영 99.9% SLO(p95 <300ms) 목표에 맞춰 경량 유지.

## 5. 모델 분산·운영 규모 한계

- **임베딩 분산**: sentence-transformers 모델 교체 시 벡터 드리프트 → 재ingest 필요. A 목 Judge처럼 프로파일 strict/lenient로 분산 시뮬은 B에 없음.
- **규모**: Chroma 10 chunks(10 docs), Sheets 30 rows, RAGAS 50 — 운영 실 트래픽(일 수천 쿼리)에서 `get_collection().count()` 10은 편향. feedback 30은 bootstrap, 월 수백으로 확장 시 `curate_top_questions` 키워드 클러스터링(15개 도메인 키워드)으로는 topic drift 대응 한계 — 추세는 별도 임베딩 클러스터링 필요.
- **Sheets**: gspread 실 Sheets + CSV mock 이중화는 CI 친화적이나, 동시 쓰기 충돌·스키마 변경(컬럼 추가) 시 마이그레이션 수동.

## 6. 개선 로드맵 (B 전용)

1. 임베딩 교체 실험: hash-embedding(before) vs sentence-transformers(after) delta +0.11을 월 1회 LLM RAGAS로 교차 검증
2. LangGraph rewrite 1->2 확장 + self-RAG critic 노드 실험, p95 지연 측정
3. feedback 30->300 확장 시 카테고리 임베딩 클러스터링으로 keyword 15->50 확장
4. Chroma persist 백업/버전 태깅(`data/chroma` snapshot) — drift 시 rollback
