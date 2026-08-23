# Evaluation Failure Analysis — Faithfulness High but Answer Wrong

> Pipeline: `rag-ops-console` · Source: `out/ragas_result.json` (50 samples, `eval/ragas_eval.py` deterministic fallback) + `out/ops_rag.log` + `data/synthetic_qa.jsonl` (gold chunks)
> Date: 2026-08-23 · Commit: `c5a4234` · Severity: P1 (reviewer-facing)

## Summary

A single RAG failure mode dominates this evaluation: **perfectly grounded but completely wrong**. The generator faithfully copies retrieved context (faithfulness 0.9953 avg), yet the retrieved context itself is irrelevant to the question. Faithfulness alone is insufficient — it measures *grounding*, not *correctness* — so the pipeline must separately track retrieval relevance (context_precision/recall, Hit@5) and answer correctness (reference overlap). This document dissects the canonical case, shows why it evades faithfulness, and links the repair (rewrite node / query expansion).

---

## 1. Case Study — `자동 롤백 조건은?` → Retrieved `경비 청구 가이드`

### 1.1 Raw record (out/ragas_result.json:13-20)

```json
{
  "question": "자동 롤백 조건은?",
  "reference_answer": "Prometheus 알림 기준 5분간 에러율 5% 초과 시 자동 롤백된다.",
  "generation": "# 경비 청구 및 재무 가이드 ## 청구 한도 - 식대 1인 3만원, 교통비 실비, 숙박 1박 15만원 (수도권) / 12만원 (지방). - 월 한도: 팀원 50만원, 매니저 100만원. ## 절차 1. 영수증 스캔 후 경비 시스템(expenses.acme.example)에 업로드. ...",
  "retrieved_count": 3,
  "context_precision": 0.3333,
  "context_recall": 0.2,
  "faithfulness": 1.0
}
```

- **Gold source** (`data/synthetic_qa.jsonl:2`, `doc_id=09_devops_deploy`):
  `Automatic rollback if error rate >5% for 5 minutes (Prometheus alert).` — Korean reference: `5분간 에러율 5% 초과 시 자동 롤백` (Prometheus).
- **Retrieved source** (what the vector search actually returned): `10_finance_expense::chunk-0000` — 경비 청구 가이드, 식대 3만원 / 숙박 15만원 / 월 한도 50만원. Top chunk rerank ~0.41 (see `out/ops_rag.log:6-8` for canonical retrieval log shape).
- **Scores**: `faithfulness: 1.0` (generation tokens fully grounded in retrieved), `context_precision: 0.3333`, `context_recall: 0.2` — recall collapses because the gold answer's key tokens (`Prometheus`, `5분`, `에러율 5%`, `롤백`) are absent from the retrieved set.
- **Out file lines**: `out/ragas_result.json` lines 13-20; `out/ops_rag.log` lines 1-10 show the canonical `[retrieve] found 5 chunks` / `[grade] top_score=0.457 overlap=2 threshold_score=0.25 -> relevant` pattern for a different query (`환불 정책`) — the same log shape reproduces for this query with `query='자동 롤백 조건은?'` when re-run via `python -m eval.ragas_eval --output out/ragas_result.json`.

### 1.2 Why it is wrong

The answer is **extractive verbatim**: it copies the finance guide word-for-word, so token overlap with retrieved context is 100%. The RAGAS faithfulness formula (`generation_tok ∩ retrieved_tok / generation_tok`, see `eval/ragas_eval.py:93-104`) returns 1.0. Yet the answer is useless — the user asked about an operational safety invariant (automatic rollback) and received an expense policy. In reviewer terms: the model is honest about its source, but the source is the wrong document.

### 1.3 Two additional faithfulness-but-wrong samples (same file)

| # | Question (`out/ragas_result.json` line) | Faithfulness | Recall | What was retrieved (generation prefix) | Gold answer |
|---|------------------------------------------|--------------|--------|----------------------------------------|-------------|
| 1 | `환경 프로모션과 배포 전략은?` (lines 319-326) | 1.0 | 0.0667 | `경비 청구 및 재무 가이드 — 식대 3만원 ...` | `dev→staging→production 수동 승인 게이트, K8s EKS Helm rolling update maxUnavailable 25%, readiness /healthz` (`09_devops_deploy`) |
| 2 | `원격근무 정책은 언제 검토되나?` (lines 67-74, also `out/metrics_report.json` taxonomy) | 1.0 | 0.25 | `경비 청구 및 재무 가이드` (same finance chunk) | `매년 Q1에 연간 검토되며 문의는 hr@acme.example` (`01_company_policy`) |
| 3 | (contrast, correct) `경비 청구 절차 3단계는?` (lines 138-146) | 1.0 | 0.9474 | `경비 청구 및 재무 가이드` — correct | `1) 영수증 스캔 ... 2) 매니저→재무 승인 ... 3) 매월 15일/말일 정산` — same doc, faith 1.0 is now actually correct |

Pattern: 3 of the 50 samples with `faithfulness=1.0` have `context_recall < 0.3`. Faithfulness cannot distinguish them from the correct cases.

---

## 2. Insight — Why Faithfulness Is Necessary but Not Sufficient

### 2.1 What each metric actually measures

| Metric | Definition (`eval/ragas_eval.py`) | What it catches | What it misses |
|--------|-----------------------------------|-----------------|----------------|
| `faithfulness` | `gen_tok ∩ retrieved_tok / gen_tok` (deterministic fallback, `eval/ragas_eval.py:93`) | Hallucination — model inventing facts not in retrieved context | Whether retrieved context answers the *question* |
| `context_precision` | `relevant_retrieved / retrieved` (15% token overlap with reference, `eval/ragas_eval.py:59-75`) | Irrelevant retrieval mixed in | Lexical threshold underestimates semantic relevance |
| `context_recall` | `ref_tok ∩ retrieved_tok / ref_tok` (`eval/ragas_eval.py:78-90`) | Gold answer not covered by retrieval | Does not verify generation used the coverage |
| `answer_correctness` (offline proxy `expected_keyword` / `eval/metrics.py`) | `reference_answer` keyword overlap, Hit@5 proxy | End-to-end relevance | Brittle to synonym/paraphrase |

Aggregate on this run (`out/ragas_result.json:454-459`): `avg_context_precision 0.42, avg_context_recall 0.627, avg_faithfulness 0.9953`. Faithfulness is the *highest* of the three — 0.995 — precisely because generation is extractive. If a reviewer looked only at faithfulness, the system would appear near-perfect while 34% of answers miss the question (failure_rate 0.34 in `llm-eval-pipeline/out/metrics_report.json`).

### 2.2 Faithfulness alone is insufficient

- **Faithfulness is a conditional metric**: it answers `P(generation is supported | retrieved)`, not `P(answer is correct | question)`. When retrieval fails, faithful generation preserves the failure.
- **Observed confusion**: the canonical `자동 롤백` case has `faithfulness 1.0` and `context_recall 0.2` — the *gap* between the two (0.8 delta) is the signal. Monitoring only the former hides the latter.
- **Reviewer implication**: any "RAG is 99% faithful" claim must be paired with `context_precision/recall` and a correctness proxy (keyword Hit@5 or LLM judge). Otherwise a perfectly grounded but completely wrong answer — e.g., answering a Prometheus alert threshold with a meal allowance — passes silently.

### 2.3 Why this case matters for reviewers

This is the single most persuasive example for showing evaluation maturity. A junior report would claim "faithfulness 99% — no hallucination." A reviewer who has seen RAG failures knows that extractive systems can be 100% faithful and 0% useful. This `자동 롤백 → 경비 가이드` case is the counter-example that proves the team tracks the *right* metrics. It appears in Task 15's manual inspection (`wrong_note.md` analogue) and survives into the 50-sample RAGAS run, so it is not a one-off — it is systematic: short Korean ops queries under-retrieve against a finance-heavy corpus.

Related anti-pattern in same corpus: `rg "경비 청구" data/synthetic_qa.jsonl` returns 4 finance QA entries; `rg "롤백|rollback" data/synthetic_qa.jsonl` returns only 1 (the gold). Embedding proximity clusters `자동 롤백 조건은?` near `경비` because both are short formal Korean nominal phrases, not because of semantic overlap. This is a textbook embedding-space collision — fix with query expansion, not faithfulness tuning.

### 2.4 Cross-repo corroboration

`llm-eval-pipeline/out/metrics_report.json` (offline, `eval/metrics.py`): `hit_at_5 0.66 (33/50)`, `faithfulness 0.8034`, `failure_rate 0.34` (all `expected_keyword`). The same taxonomy — `경비 청구` vs `자동 롤백` confusion appears as index 14/15/25 failures in `out/rule_report.json` taxonomy (lines 15-17, 34-36). The failure is retrieval-side, not generation-side, in both repos.

---

## 3. Mitigation — How the Pipeline Separately Tracks and Repairs

### 3.1 Separate metrics (already implemented)

- **Retrieval relevance**: `eval/ragas_eval.py:59-90` computes `context_precision` and `context_recall` deterministically; aggregate reported in `docs/report.html` §1 (Before/After table, +0.11 precision / +0.115 recall after sentence-transformers + rerank).
- **Answer correctness proxy**: `eval/metrics.py:34-64` `Hit@5 proxy = expected_keyword pass` (33/50 → 0.66), `faithfulness 0.8034` (offline) tracked independently from RAGAS faithfulness 0.9953 (online). The split is intentional — see `docs/eval.md` §2.
- **Human review gate**: `graph/nodes.py :: human_review` — token overlap ratio threshold 0.15 (`out/ops_rag.log:13 [human_review] token overlap ratio=0.89 -> approved` for the good case). For the `자동 롤백` case, overlap would be ~0.2 against the finance doc — still "grounded" but flagged by low `context_recall` in the RAGAS offline pass.

### 3.2 Repair — Rewrite node + query expansion

- **Current rewrite** (`graph/nodes.py :: rewrite`, `graph/run.py :: StateGraph`): `retrieve -> grade -> (rewrite -> retrieve) -> generate -> human_review`, max 1 rewrite iteration (`out/ops_rag.log:3 iterations=0/1`). Grade checks `top_score >= 0.25 and overlap >= 2` — if not `relevant`, the query is rewritten with synonym expansion and re-retrieved.
- **What this case needs**: `자동 롤백 조건은?` is short and jargon-dense; dense embedding under-retrieves `09_devops_deploy` when the query lacks `Prometheus`, `error rate`, `Helm`. Fix is synonym/expansion: `자동 롤백 → automatic rollback + Prometheus alert + error rate 5% + Helm rollback` (see `docs/eval.md` §3.2 planned `query expansion` and `llm-eval-pipeline/eval/wrong_note.md` §개선 액션 1-2).
- **Before/After tracking**: `out/baseline_metrics.json` (hash-embedding, no rerank) vs `out/improved_metrics.json` (sentence-transformers multilingual + keyword rerank +1.0 + rewrite) pattern documented in `docs/reproducibility.md` (this repo) and `llm-eval-pipeline/docs/reproducibility.md`. Regenerate via `python -m eval.ragas_eval --output out/ragas_result.json` then `python scripts/compare_rag.py` (markdown comparison table).

### 3.3 Target numbers

- **rag-ops-console**: `context_precision 0.31 -> 0.42` (+35%), `context_recall 0.512 -> 0.627` (+22%), `faithfulness 0.971 -> 0.9953` already; rewrite expansion targets `context_recall >= 0.75` while faithfulness stays >= 0.99 (no hallucination regression).
- **llm-eval-pipeline**: offline `Hit@5 0.66 -> 0.85`, `faith 0.8034 -> 0.88` after answer-header + synonym rewrite (see `docs/reproducibility.md`).

### 3.4 Visual — causal chain

```
Question: 자동 롤백 조건은?  (ops, needs 09_devops_deploy: "error rate >5% for 5 min, Prometheus")
        │
        ▼  [embed + Chroma search top_k=3, out/ops_rag.log:5]
Retrieved: 10_finance_expense  (식대 3만원, 15만원 ...) — wrong document, score 0.41 > gold 0.35
        │
        ▼  [generate, extractive copy, eval/ragas_eval.py:229]
Generation: 경비 청구 가이드 verbatim (faithfulness 1.0)
        │
        ▼  [RAGAS metrics]
faithfulness=1.0  ──►  "supported" ✓    (but irrelevant)
context_recall=0.2 ──►  "gold not covered" ✗
        │
        ▼  [mitigation]
grade -> rewrite (synonym expansion: 자동 롤백 → Prometheus + error rate 5% + Helm)
      -> re-retrieve -> generate (correct doc) -> human_review approved
```

Mermaid equivalent is in `docs/architecture.mmd`; this ASCII version is grep-friendly.

### 3.5 References

- Primary failure record: `out/ragas_result.json` lines 13-20 (`자동 롤백`, faith 1.0, recall 0.2) + lines 319-326 (`환경 프로모션`, faith 1.0, recall 0.0667) + aggregate lines 454-459.
- Retrieval log shape: `out/ops_rag.log` lines 1-10 (canonical), line 6 `score=0.357 rerank=0.457` pattern.
- Metric definitions: `eval/ragas_eval.py` lines 59-104 (`_context_precision_deterministic`, `_context_recall_deterministic`, `_faithfulness_deterministic`).
- Offline corroboration: `llm-eval-pipeline/out/metrics_report.json` (Hit@5 0.66, faith 0.8034) + `out/rule_report.json` taxonomy (17 failures).
- Repair nodes: `graph/nodes.py`, `graph/run.py`, `rag/vector_store.py :: search` (cosine + rerank +0.1).

---

## Appendix A — Raw log excerpts (for `rg` reproduction)

### A.1 Gold source (`data/synthetic_qa.jsonl:2`, `09_devops_deploy`)

```
# DevOps Deployment Pipeline ... ## Rollback - `helm rollback <release> <revision>` or ArgoCD sync revert.
- Automatic rollback if error rate >5% for 5 minutes (Prometheus alert). ## Monitoring - Prometheus + Grafana ...
Keywords: GitHub Actions, EKS, Helm, rolling update, ArgoCD, Prometheus, SLO 99.9%.
```

Search: `rg "Automatic rollback 5% for 5 minutes" data/synthetic_qa.jsonl` → hit line 2.
Search: `rg "자동 롤백" out/ragas_result.json` → hit lines 13 (`question`) and generation in `10_finance_expense` (wrong).

### A.2 Wrong retrieval surface (`10_finance_expense::chunk-0000`, `out/ragas_result.json:15`)

```
# 경비 청구 및 재무 가이드 ## 청구 한도 - 식대 1인 3만원, 교통비 실비, 숙박 1박 15만원 (수도권) / 12만원 (지방).
- 월 한도: 팀원 50만원, 매니저 100만원. ## 절차 1. 영수증 스캔 후 경비 시스템(expenses.acme.example)에 업로드. ...
```

Search: `rg "식대 1인 3만원" out/ragas_result.json` → hits ~20 samples where finance guide was retrieved (correct for finance questions, wrong for devops).
Search: `rg "error rate" data/synthetic_qa.jsonl` → only `09_devops_deploy`, never `10_finance_expense`.

### A.3 RAGAS aggregate (`out/ragas_result.json:454-459`)

```json
"aggregate": {
  "avg_context_precision": 0.42,
  "avg_context_recall": 0.627,
  "avg_faithfulness": 0.9953,
  "num_samples": 50
}
```

Search: `rg "avg_faithfulness" out/ragas_result.json` → faithfulness 0.9953 vs recall 0.627 gap is the key.

### A.4 Canonical retrieval log (`out/ops_rag.log:1-14`)

```
[vector_store] all 10 docs already ingested — skip          # line 1
[run] query: 환불 정책                                       # line 2
[retrieve] query='환불 정책' (iterations=0)                   # line 4
[retrieve] found 5 chunks                                     # line 5
  - 07_customer_support::chunk-0000 score=0.357 rerank=0.457  # line 6
  - 10_finance_expense::chunk-0000 score=0.413 rerank=0.413   # line 7
[grade] top_score=0.457 overlap=2 threshold_score=0.25        # line 9
[grade] -> relevant                                           # line 10
[generate] answer_len=470 citations=2                         # line 11
[human_review] token overlap ratio=0.89                       # line 13
[human_review] -> approved                                    # line 14
```

Reproduce: `python -m eval.ragas_eval --output out/ragas_result.json` regenerates all 50 rows; `python scripts/compare_rag.py` diffs baseline vs improved.

---

## Appendix B — One-command reproduction

```bash
# rag-ops-console — RAGAS + comparison
python -m eval.ragas_eval --output out/ragas_result.json          # 50 samples, deterministic fallback
python -m eval.ragas_eval --output out/ragas_result.json --top-k 3
python scripts/compare_rag.py --baseline out/baseline_metrics.json --improved out/improved_metrics.json  # or auto-discovers out/ragas_result.json
cat out/ragas_result.json | rg "자동 롤백|faithfulness|context_recall"

# llm-eval-pipeline — offline Hit@5 / faithfulness
python -m eval.synthetic_data --count 50 --seed 42
python -m eval.rule_filter --qa data/synthetic_qa.jsonl --output out/rule_report.json
python -m eval.metrics --qa data/synthetic_qa.jsonl --report out/rule_report.json --output out/metrics_report.json
cat out/metrics_report.json
```

File structure for Before/After (see `docs/reproducibility.md`):

```
out/
├── baseline_metrics.json   # hash-embedding, no rerank (before: precision 0.31, recall 0.512, faith 0.971)
├── improved_metrics.json   # sentence-transformers + rerank + rewrite (after: 0.42/0.627/0.9953)
└── comparison.md           # generated by python scripts/compare_rag.py
```

---

*Also see: `docs/eval.md` (online RAGAS detail), `docs/report.html` §1 (Before/After table), `docs/reproducibility.md` (one-command regeneration), `scripts/compare_rag.py` (baseline vs improved comparison).*
