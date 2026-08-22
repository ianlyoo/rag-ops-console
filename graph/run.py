"""graph.run — LangGraph CLI 엔트리.

Usage:
    python -m graph.run --query "환불 정책"
    python -m graph.run --query "환불 정책" --log out/ops_rag.log

실행 흐름:
    1) Chroma ingest (필요 시 synthetic_qa 로드)
    2) LangGraph StateGraph 빌드 (실명 LangGraph)
    3) 쿼리 실행 + 노드별 로그 출력 ([retrieve] [grade] 등)
    4) 최종 JSON 답변 + citations 출력
    5) Sheets 적재 (out/sheets_mock.csv) — gspread + CSV fallback
    6) out/ops_rag.log 저장
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

# Windows cp949 대응
if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def _fix_query_encoding(query: str) -> str:
    """PowerShell cp949 mojibake 복구.

    Windows 터미널에서 UTF-8 쿼리가 cp949로 잘못 디코딩되면
    '환불 정책' -> 'ȯ�� ��å' 처럼 깨진다. 이를 감지해 복구한다.
    """
    # 한글이 하나라도 포함되어 있으면 정상으로 간주
    if any("\uAC00" <= ch <= "\uD7A3" for ch in query):
        return query
    # mojibake 패턴: ȯ, �� 등 비정상 문자 포함 시 재디코딩 시도
    try:
        # 원래 UTF-8 바이트가 cp949로 디코딩된 경우 → cp949로 인코딩 후 utf-8로 디코딩
        fixed = query.encode("cp949").decode("utf-8")
        if any("\uAC00" <= ch <= "\uD7A3" for ch in fixed):
            return fixed
    except Exception:
        pass
    try:
        fixed = query.encode("utf-8").decode("cp949")
        if any("\uAC00" <= ch <= "\uD7A3" for ch in fixed):
            return fixed
    except Exception:
        pass
    return query

from langgraph.graph import END, StateGraph  # noqa: E402

from graph.nodes import generate, grade, human_review, retrieve, rewrite, route_after_grade  # noqa: E402
from graph.state import GraphState  # noqa: E402


def build_graph() -> Any:
    """LangGraph StateGraph 빌드 — 5노드 + 조건부 라우팅."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("human_review", human_review)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")

    # 조건부 라우팅: grade -> generate or rewrite
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "human_review")
    workflow.add_edge("human_review", END)

    return workflow.compile()


def run_query(query: str, verbose: bool = True) -> dict[str, Any]:
    """단일 쿼리 실행 — LangGraph StateGraph invoke."""
    # 1) Chroma ingest (최초 1회)
    try:
        from rag.vector_store import ingest

        ingest(reset=False)
    except Exception as e:
        print(f"[run] ingest warning: {e}")

    # 2) Graph 빌드 및 실행
    app = build_graph()

    initial_state: GraphState = {
        "query": query,
        "retrieved": [],
        "graded": None,
        "rewritten_query": None,
        "generation": None,
        "citations": [],
        "review_status": "pending",
        "iterations": 0,
    }

    if verbose:
        print(f"[run] query: {query}")
        print("[run] StateGraph nodes: retrieve -> grade -> (rewrite -> retrieve) -> generate -> human_review")

    result: dict[str, Any] = app.invoke(initial_state)  # type: ignore[arg-type]

    return result


def _save_sheets_and_log(
    query: str,
    result: dict[str, Any],
    log_path: Path,
    captured_output: str,
) -> None:
    """Sheets 적재 + 로그 파일 저장."""
    # Sheets 적재
    try:
        from monitor.feedback_to_sheet import append_to_sheet

        citations_str = ";".join(
            f"{c.get('doc_id','')}/{c.get('chunk_id','')}" for c in result.get("citations", [])
        )
        rows = [
            {
                "query": query,
                "response": (result.get("generation") or "")[:500],
                "failure_category": "pending",
                "root_cause": result.get("graded", ""),
                "review_status": result.get("review_status", ""),
                "citations": citations_str,
            }
        ]
        n = append_to_sheet(rows)
        sheets_log = f"[sheets] appended {n} rows -> out/sheets_mock.csv"
        print(sheets_log)
        captured_output += "\n" + sheets_log + "\n"
    except Exception as e:
        err = f"[sheets] failed: {e}"
        print(err)
        captured_output += "\n" + err + "\n"

    # 로그 파일 저장 — repo out/ + portfolio/out 복사
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_content = captured_output + "\n[final_result]\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    log_path.write_text(log_content, encoding="utf-8")
    print(f"[run] log -> {log_path}")

    # portfolio/out 복사
    # rag-ops-console은 portfolio/code/rag-ops-console 이므로 parents[2]가 code, parents[3]이 portfolio
    # 정확히 계산
    try:
        repo_root = Path(__file__).resolve().parents[1]
        portfolio_out = repo_root.parent.parent / "portfolio" / "out" / "ops_rag.log"
        if portfolio_out.parent.is_dir() or portfolio_out.parent.exists():
            portfolio_out.parent.mkdir(parents=True, exist_ok=True)
            portfolio_out.write_text(log_content, encoding="utf-8")
            print(f"[run] log copy -> {portfolio_out}")
        else:
            # fallback: C:/Users/torch/Documents/code/portfolio/out
            alt = Path(r"C:\Users\torch\Documents\code\portfolio\out\ops_rag.log")
            alt.parent.mkdir(parents=True, exist_ok=True)
            alt.write_text(log_content, encoding="utf-8")
            print(f"[run] log copy -> {alt}")
    except Exception as e:
        print(f"[run] portfolio log copy failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Ops Console — LangGraph RAG 실행")
    parser.add_argument("--query", type=str, required=True, help="검색 질의")
    parser.add_argument("--log", type=str, default="out/ops_rag.log", help="로그 파일 경로")
    parser.add_argument("--top-k", type=int, default=5, help="검색 top_k (reserved)")
    args = parser.parse_args()
    args.query = _fix_query_encoding(args.query)

    # stdout 캡처 (로그 파일용)
    buffer = io.StringIO()
    old_stdout = sys.stdout

    class TeeStdout:
        def __init__(self, *streams: Any) -> None:
            self.streams = streams

        def write(self, data: str) -> int:
            for s in self.streams:
                s.write(data)
            return len(data)

        def flush(self) -> None:
            for s in self.streams:
                s.flush()

    sys.stdout = TeeStdout(old_stdout, buffer)  # type: ignore[assignment]

    try:
        result = run_query(args.query, verbose=True)

        # 최종 JSON 출력
        print("\n[result] final JSON:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # citation 검증 로그
        citations = result.get("citations", [])
        if citations:
            print(f"\n[citation] PASS — {len(citations)} citations present")
        else:
            print("\n[citation] FAIL — no citations")

        review_status = result.get("review_status", "unknown")
        print(f"[review_status] {review_status}")

        captured = buffer.getvalue()

    finally:
        sys.stdout = old_stdout  # type: ignore[assignment]
        captured = buffer.getvalue()

        # 파일에는 buffer 내용 + result JSON 포함하여 저장
        # (이미 Tee로 출력했으므로 buffer에 노드 로그가 있음)
        # result가 정의되지 않았을 수 있으므로 방어
        try:
            _result = locals().get("result", {})
            # Sheets + 로그 저장 (captured 이용)
            # captured에 result JSON이 이미 포함되어 있으므로 중복 방지
            log_path = Path(args.log)
            if not log_path.is_absolute():
                log_path = Path(__file__).resolve().parents[1] / log_path

            # Sheets 적재 — captured에 추가 로그 포함
            sheets_extra = ""
            try:
                from monitor.feedback_to_sheet import append_to_sheet

                citations_str = ";".join(
                    f"{c.get('doc_id','')}/{c.get('chunk_id','')}" for c in _result.get("citations", [])
                )
                rows = [
                    {
                        "query": args.query,
                        "response": (_result.get("generation") or "")[:500],
                        "failure_category": "pending",
                        "root_cause": _result.get("graded", ""),
                        "review_status": _result.get("review_status", ""),
                        "citations": citations_str,
                    }
                ]
                n = append_to_sheet(rows)
                sheets_extra = f"[sheets] appended {n} rows -> out/sheets_mock.csv\n"
                print(sheets_extra, end="")
                captured += sheets_extra
            except Exception as e:
                sheets_extra = f"[sheets] failed: {e}\n"
                print(sheets_extra, end="")
                captured += sheets_extra

            # 로그 파일 저장
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # captured에는 이미 노드 로그 + result JSON이 있음 (Tee). result JSON이 없으면 추가
            if "[final_result]" not in captured:
                captured += "\n[final_result]\n" + json.dumps(_result, ensure_ascii=False, indent=2) + "\n"
            log_path.write_text(captured, encoding="utf-8")
            print(f"[run] log -> {log_path}")

            # portfolio/out 복사
            for cand in [
                Path(r"C:\Users\torch\Documents\code\portfolio\out\ops_rag.log"),
                Path(__file__).resolve().parents[1].parent.parent / "portfolio" / "out" / "ops_rag.log",
            ]:
                try:
                    cand.parent.mkdir(parents=True, exist_ok=True)
                    cand.write_text(captured, encoding="utf-8")
                    print(f"[run] log copy -> {cand}")
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"[run] log save failed: {e}", file=old_stdout)


if __name__ == "__main__":
    main()
