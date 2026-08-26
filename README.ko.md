# rag-ops-console

[English](README.md)

[![CI](https://github.com/ianlyoo/rag-ops-console/actions/workflows/ci.yml/badge.svg)](https://github.com/ianlyoo/rag-ops-console/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release: v0.1.0](https://img.shields.io/badge/Release-v0.1.0-blue.svg)](https://github.com/ianlyoo/rag-ops-console/releases/tag/v0.1.0)
[![Pages](https://img.shields.io/badge/Pages-live-brightgreen.svg)](https://ianlyoo.github.io/rag-ops-console/)

Online RAG Ops console with Chroma VectorDB and LangGraph 5 nodes — monitor retrieval quality with RAGAS and Sheets feedback loop in production.

> 온라인 RAG 운영 콘솔입니다. Chroma VectorDB 검색과 LangGraph 오케스트레이션, Deterministic 지표와 선택적 RAGAS, Sheets 피드백 루프를 운영합니다. 오프라인 평가는 [llm-eval-pipeline](https://github.com/ianlyoo/llm-eval-pipeline)에서 담당합니다.

## 빠른 시작 — rag와 vectordb

```bash
gh release download v0.1.0 --pattern "*.tgz" --repo ianlyoo/rag-ops-console
npm install ./rag-ops-console-*.tgz
```

```bash
git clone https://github.com/ianlyoo/rag-ops-console.git
cd rag-ops-console
bun install --frozen-lockfile
bun run build
```

```bash
python -m venv .venv
# Windows alternative: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

```bash
python -c "from rag.vector_store import ingest; print(ingest(reset=False))"
python -m eval.ragas_eval --output out/ragas_result.json
```

## 활용 사례 — ragas와 llm-judge

- Chroma vectordb 검색과 rerank
- LangGraph 5노드 라우팅과 human review gate
- Deterministic 지표와 선택적 `ragas.evaluate()` 비교
- 고객 피드백 분류와 Sheets 적재, Top-5 큐레이션

## 아키텍처 — langgraph 파이프라인

`vectordb → langgraph → ragas → feedback-loop → google-sheets`

## 벤치마크 — 측정된 워크로드에서의 ragas 평가

Deterministic 지표 (50 샘플): `precision 0.42 / recall 0.627 / faithfulness 0.9953`. LLM 기반 비교는 키 보유 시에만 동작합니다.

Limitations adjacent to benchmark: token-overlap proxy, one run per condition, 50 samples, answer relevancy 미구현, policy may change, no billing.

## 라이선스

MIT — [LICENSE](LICENSE)를 참고하세요.
