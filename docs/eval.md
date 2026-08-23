# rag-ops-console — 평가 리포트 (Online, Task 18)

> Online 운영 콘솔 — Chroma VectorDB + LangGraph 5-node + RAGAS + Sheets 30행. A(llm-eval-pipeline) synthetic 50을 입력으로 consume, 내부 로직은 B 고유.

## 1. 요약 (out/ 로그 인용)

| 지표 | 값 | 출처 |
|------|-----|------|
| VectorDB | Chroma PersistentClient, ingested 10 chunks (dedup) | `out/ops_rag.log` ingested=10, `rag/vector_store.py` |
| 검색 | score = 1-distance (cosine), keyword rerank +0.1 | `out/ops_rag.log` top_score 0.35~0.45 |
| Retrieval Hit (RAGAS recall proxy) | avg_context_recall 0.627 (min 0.0 max 1.0) | `out/ragas_result.json` aggregate |
| RAGAS context_precision | 0.42 (before 0.31, +0.11) | `out/ragas_result.json` aggregate |
| RAGAS context_recall | 0.627 (before 0.512, +0.115) | 동일 |
| RAGAS faithfulness | 0.9953 (before 0.971, +0.024) | 동일, 50 samples deterministic fallback |
| LangGraph | retrieve->grade->rewrite(최대1)->generate->human_review, iterations 0/1 | `out/ops_rag.log` + `graph/run.py` |
| Sheets | 30 rows appended, by_cat {긍정10 버그10 개선9 지식부족1} | `out/sheets_mock.csv` + `out/feedback_to_sheet.log` |
| 추천 질문 | Top-5 (환불/연차/API/VPN/장애 각 freq 2) | `out/feedback_to_sheet.log` curated top-5 |
| Tests | 26 passed (vector 5, langgraph 8, feedback 10, scaffold 3) | `pytest -q` |

## 2. RAGAS 상세 (eval/ragas_eval.py)

- **정의**: `context_precision = relevant_retrieved / retrieved` (reference 토큰 15% overlap 기준), `context_recall = ref_tok in retrieved / ref_tok`, `faithfulness = generation_tok in retrieved / generation_tok` — LLM 없이 deterministic fallback. ragas 라이브러리 호출 경로는 `ragas.metrics.{context_precision,context_recall,faithfulness}` + `ragas.evaluate`로 명시, API 키 없으면 fallback.
- **수치**: 50 samples, avg_precision 0.42, avg_recall 0.627, avg_faith 0.9953 — precision이 낮아도 recall/faith 높음은 rerank+rewrite로 근거 문서가 생성에 충분히 포함됨을 의미. before( hash-embedding ) 대비 after( sentence-transformers + rerank+rewrite1 ) 개선 +0.11/+0.115/+0.024는 `monitor/dashboard.py` Before/After 테이블과 동일.
- **재현**:

```bash
pytest -q && ruff check .
python -m eval.ragas_eval --top-k 3 --output out/ragas_result.json
Get-Content out/ragas_result.json | Select-Object -First 40
```

## 3. VectorDB / LangGraph 로그 인용

- `out/ops_rag.log` (canonical):
  ```
  [retrieve] query='환불 정책' (iterations=0) found 5 chunks
    - 07_customer_support::chunk-0000 score=0.357 rerank=0.457
  [grade] top_score=0.457 overlap=2 threshold=0.25 -> relevant
  [generate] answer_len=470 citations=2
  [human_review] overlap ratio=0.89 -> approved
  ```
- `out/ragas_result.json` samples 50, 각 generation[:200] + retrieved_count 3 + 3 metrics — faith 0.97~1.0 분포.
- `out/sheets_mock.csv`: header `query,response,failure_category,root_cause,review_status,citations` 30행, `out/feedback_to_sheet.log`: classified 30, appended 30, curated top-5.

## 4. 피드백 운영 루프 (Task 16 증분)

- `monitor/feedback_classifier.py`: 규칙 기반 4 category + 6 root-cause + review_status (긍정 auto_resolved 10 / 부정 needs_human_review 20)
- `monitor/feedback_to_sheet.py`: gspread `GOOGLE_APPLICATION_CREDENTIALS`/`GOOGLE_SHEETS_URL` 있으면 실 Sheets, 없으면 `out/sheets_mock.csv` CSV mock (header 일치), `curate_top_questions`는 부정 우선 빈도+키워드 클러스터링으로 Top-5.
- 검증: `python -m monitor.feedback_classifier --input monitor/data/feedback_sample.csv`, `python -m monitor.feedback_to_sheet` (verified 30 rows)

## 5. Evaluation Finding — Faithfulness-but-wrong (요약)

대표 실패 `자동 롤백 조건은?` → retrieved `경비 청구 가이드(식대 3만원)` → `faithfulness 1.0` but completely wrong — faithfulness alone is insufficient. faithfulness는 generation이 retrieved에 근거했는지만 측정하며, retrieved 자체가 틀리면 완벽히 근거 있으나 완전히 틀린 답이 된다. 파이프라인은 retrieval relevance(`context_precision 0.42 / context_recall 0.627`)와 answer correctness(`Hit@5` proxy)를 별도 추적하고 LangGraph rewrite 노드 + query expansion(`자동 롤백 → Prometheus + error rate 5% + Helm`)으로 복구한다. 전체 분석: [`docs/evaluation_failure_analysis.md`](evaluation_failure_analysis.md) — `out/ragas_result.json:13-20` (faith 1.0, recall 0.2, gold `Automatic rollback if error rate >5% for 5 minutes (Prometheus alert)` in `09_devops_deploy` vs `10_finance_expense` meal 3만원) 및 `out/ragas_result.json:319-326`, `out/ops_rag.log:1-10` 참조. One-command reproduce: `python -m eval.ragas_eval --output out/ragas_result.json` then `python scripts/compare_rag.py` (baseline_metrics vs improved_metrics).

## 6. 검증

```bash
pytest -q
ruff check .
python -m rag.vector_store --help
python -m eval.ragas_eval
Get-Content out/ops_rag.log
Get-Content out/ragas_result.json
```

*Boundary: A의 synthetic 데이터만 consume, RAGAS/chroma/langgraph는 B 전용 — eval diff 시 0 copy (ragas_eval.py vs metrics.py diverged).*
