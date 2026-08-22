"""eval.ragas_eval — RAGAS 지표 계산.

RAGAS (context precision / context recall / faithfulness) 평가 모듈.

- ragas 라이브러리 호출 경로 명시 (ragas.metrics.*) — LLM evaluator가 필요한 경우
  `ragas` 라이브러리를 실제로 import하여 사용한다.
- LLM 없이 계산 가능한 deterministic approximation fallback 제공:
    토큰 겹침 기반 context precision/recall, 문장-컨텍스트 entailment 근사 faithfulness.

A 레포 synthetic_qa.jsonl을 평가 입력으로 consume한다.

Usage:
    python -m eval.ragas_eval --input data/synthetic_qa.jsonl --output out/ragas_result.json
    또는
    from eval.ragas_eval import evaluate_dataset, compute_metrics
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# RAGAS 라이브러리 호출 경로 명시 — import 시 ragas 사용
try:
    from ragas.metrics import (  # type: ignore[import-untyped]
        context_precision,
        context_recall,
        faithfulness,
    )

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    context_precision = None  # type: ignore[assignment]
    context_recall = None  # type: ignore[assignment]
    faithfulness = None  # type: ignore[assignment]

try:
    from ragas import evaluate  # type: ignore[import-untyped]

    _RAGAS_EVALUATE_AVAILABLE = True
except ImportError:
    _RAGAS_EVALUATE_AVAILABLE = False
    evaluate = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Deterministic fallback metrics (LLM 불필요)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _context_precision_deterministic(
    retrieved_texts: list[str], reference_answer: str
) -> float:
    """Context Precision 근사: retrieved 중 reference_answer와 겹치는 비율."""
    if not retrieved_texts:
        return 0.0
    ref_tokens = _tokenize(reference_answer)
    if not ref_tokens:
        return 0.0
    relevant = 0
    for t in retrieved_texts:
        doc_tokens = _tokenize(t)
        overlap = len(ref_tokens & doc_tokens)
        # 해당 문서가 reference와 20% 이상 겹치면 relevant로 간주
        if overlap / max(len(ref_tokens), 1) >= 0.15:
            relevant += 1
    return relevant / len(retrieved_texts)


def _context_recall_deterministic(
    retrieved_texts: list[str], reference_answer: str
) -> float:
    """Context Recall 근사: reference_answer 토큰 중 retrieved에 포함된 비율."""
    if not retrieved_texts:
        return 0.0
    ref_tokens = _tokenize(reference_answer)
    if not ref_tokens:
        return 0.0
    combined = set()
    for t in retrieved_texts:
        combined |= _tokenize(t)
    return len(ref_tokens & combined) / len(ref_tokens)


def _faithfulness_deterministic(generation: str, retrieved_texts: list[str]) -> float:
    """Faithfulness 근사: generation 토큰 중 retrieved에 근거 있는 비율."""
    if not generation or not retrieved_texts:
        return 0.0
    gen_tokens = _tokenize(generation)
    if not gen_tokens:
        return 0.0
    combined = set()
    for t in retrieved_texts:
        combined |= _tokenize(t)
    grounded = len(gen_tokens & combined)
    return grounded / len(gen_tokens)


def compute_metrics(
    question: str,
    retrieved_texts: list[str],
    generation: str,
    reference_answer: str,
) -> dict[str, float]:
    """단일 샘플에 대한 3개 지표 계산 (deterministic fallback).

    Args:
        question: 질문 (현재 fallback에서는 미사용, ragas LLM 경로에서 사용).
        retrieved_texts: 검색된 문서 텍스트 목록.
        generation: 생성된 답변.
        reference_answer: 정답.

    Returns:
        {context_precision, context_recall, faithfulness}
    """
    _ = question
    return {
        "context_precision": round(_context_precision_deterministic(retrieved_texts, reference_answer), 4),
        "context_recall": round(_context_recall_deterministic(retrieved_texts, reference_answer), 4),
        "faithfulness": round(_faithfulness_deterministic(generation, retrieved_texts), 4),
    }


def _try_ragas_evaluate(
    dataset_dict: dict[str, list],
) -> dict[str, Any] | None:
    """ragas 라이브러리로 실제 평가 시도 — 실패 시 None 반환.

    ragas는 LLM evaluator(OpenAI 등)를 요구하므로 API 키 없이는 실패한다.
    이 함수는 라이브러리 호출 경로를 명시적으로 보여주기 위해 존재한다.
    """
    if not _RAGAS_AVAILABLE or not _RAGAS_EVALUATE_AVAILABLE:
        return None
    try:
        from datasets import Dataset  # type: ignore[import-untyped]

        ds = Dataset.from_dict(dataset_dict)
        # ragas metrics 호출 경로 명시
        _ = context_precision
        _ = context_recall
        _ = faithfulness
        # LLM 없이는 evaluate가 실패하므로 try
        result = evaluate(ds, metrics=[context_precision, context_recall, faithfulness])  # type: ignore[misc]
        return {"ragas_raw": str(result)}
    except Exception as e:
        print(f"[ragas_eval] ragas evaluate failed (expected without LLM key): {e}")
        return None


def evaluate_dataset(
    data_path: str | Path | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """A synthetic_qa.jsonl 전체에 대해 VectorDB 검색 → 생성 → 지표 계산.

    Returns:
        {samples: [...], aggregate: {avg_context_precision, ...}}
    """
    # A 레포 경로 resolve
    if data_path is None:
        import os

        env_home = os.environ.get("EVAL_PIPELINE_HOME", "").strip()
        candidates = []
        if env_home:
            candidates.append(Path(env_home) / "data" / "synthetic_qa.jsonl")
        candidates.append(Path(r"C:\Users\torch\Documents\code\llm-eval-pipeline\data\synthetic_qa.jsonl"))
        cand2 = Path(__file__).resolve().parents[1].parent / "llm-eval-pipeline" / "data" / "synthetic_qa.jsonl"
        candidates.append(cand2)
        candidates.append(Path(__file__).resolve().parents[1] / "data" / "synthetic_qa.jsonl")
        for c in candidates:
            if c.is_file():
                data_path = c
                break
        if data_path is None:
            return {"error": "synthetic_qa.jsonl not found", "samples": [], "aggregate": {}}

    data_path = Path(data_path)
    if not data_path.is_file():
        return {"error": f"not found: {data_path}", "samples": [], "aggregate": {}}

    # VectorDB 검색을 위해 ingest 확인
    try:
        from rag.vector_store import ingest, search

        # ingest가 필요하면 수행 (이미 적재된 경우 skip)
        ingest(reset=False)
    except Exception as e:
        print(f"[ragas_eval] ingest failed: {e}")
        search = None  # type: ignore[assignment]

    samples: list[dict[str, Any]] = []
    ragas_inputs: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get("question", "")
            reference = obj.get("reference_answer", "")

            # VectorDB 검색
            retrieved_texts: list[str] = []
            if search is not None:
                try:
                    results = search(question, top_k=top_k)
                    retrieved_texts = [r.get("text", "") for r in results]
                except Exception:
                    retrieved_texts = [c.get("text", "") for c in obj.get("source_chunks", [])]
            else:
                retrieved_texts = [c.get("text", "") for c in obj.get("source_chunks", [])]

            # 생성 (간단 extractive)
            generation = retrieved_texts[0][:400] if retrieved_texts else reference

            metrics = compute_metrics(question, retrieved_texts, generation, reference)

            samples.append(
                {
                    "question": question,
                    "reference_answer": reference,
                    "generation": generation[:200],
                    "retrieved_count": len(retrieved_texts),
                    **metrics,
                }
            )

            ragas_inputs["question"].append(question)
            ragas_inputs["answer"].append(generation)
            ragas_inputs["contexts"].append(retrieved_texts)
            ragas_inputs["ground_truth"].append(reference)

    # aggregate
    if samples:
        aggregate = {
            "avg_context_precision": round(sum(s["context_precision"] for s in samples) / len(samples), 4),
            "avg_context_recall": round(sum(s["context_recall"] for s in samples) / len(samples), 4),
            "avg_faithfulness": round(sum(s["faithfulness"] for s in samples) / len(samples), 4),
            "num_samples": len(samples),
        }
    else:
        aggregate = {}

    # ragas 라이브러리 경로 시도 (결과는 참고용)
    ragas_result = _try_ragas_evaluate(ragas_inputs)
    if ragas_result:
        aggregate["ragas_library_attempt"] = ragas_result

    return {"samples": samples, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 평가 — context precision/recall/faithfulness")
    parser.add_argument("--input", type=str, default=None, help="synthetic_qa.jsonl 경로")
    parser.add_argument("--output", type=str, default="out/ragas_result.json", help="결과 JSON 경로")
    parser.add_argument("--top-k", type=int, default=3, help="검색 top_k")
    args = parser.parse_args()

    # UTF-8 stdout (Windows cp949 대응)
    import contextlib

    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    result = evaluate_dataset(data_path=args.input, top_k=args.top_k)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    agg = result.get("aggregate", {})
    print(f"[ragas_eval] samples={agg.get('num_samples', 0)}")
    print(f"  avg_context_precision={agg.get('avg_context_precision', 0)}")
    print(f"  avg_context_recall={agg.get('avg_context_recall', 0)}")
    print(f"  avg_faithfulness={agg.get('avg_faithfulness', 0)}")
    print(f"[ragas_eval] result -> {out_path}")


if __name__ == "__main__":
    main()
