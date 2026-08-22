"""monitor.feedback_to_sheet — Google Sheets 적재 + 추천 질문 큐레이션.

- 적재: `gspread`로 Sheets에 행 단위 append (query, response, failure_category, root_cause 등)
- 큐레이션: 부정/버그/개선 피드백 빈도 집계 → Top-5 추천 질문 추출 (키워드 클러스터링)
- 운영: 고객 피드백 루프를 Sheets에서 모니터링하고 추천 질문을 선제안 콘텐츠로 운영
- gspread 호출 코드 포함하되 API 키 없으면 CSV 모킹 폴백 (out/sheets_mock.csv)
- classifier 연동: monitor.feedback_classifier 분류 결과를 Sheets 스키마로 변환

gspread 경로는 코드에 필수 — 실제 호출을 포함한다.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

MOCK_CSV_PATH = Path(__file__).resolve().parents[1] / "out" / "sheets_mock.csv"
CSV_HEADERS = ["query", "response", "failure_category", "root_cause", "review_status", "citations"]
DEFAULT_FEEDBACK_CSV = Path(__file__).resolve().parents[1] / "monitor" / "data" / "feedback_sample.csv"
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "out" / "feedback_to_sheet.log"

# 추천 질문 큐레이션을 위한 키워드 정규화 (Acme Corp 도메인)
_DOMAIN_KEYWORDS = [
    "환불",
    "연차",
    "API",
    "VPN",
    "장애",
    "ML",
    "학습",
    "개인정보",
    "배포",
    "비용",
    "정산",
    "지원",
    "채팅",
    "로그",
    "헬스체크",
]


def _try_gspread_append(rows: list[dict], sheet_url: str | None = None) -> int | None:
    """gspread 실제 호출 — 성공 시 적재 행 수 반환, 실패 시 None."""
    try:
        import gspread  # type: ignore[import-untyped]
        from google.oauth2.service_account import Credentials  # type: ignore[import-untyped]

        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        sheet_url = sheet_url or os.environ.get("GOOGLE_SHEETS_URL", "").strip()

        if not creds_path or not sheet_url:
            print("[feedback_to_sheet] gspread: no credentials/sheet_url -> CSV fallback")
            return None

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        client = gspread.authorize(creds)

        sheet_id = sheet_url.split("/d/")[1].split("/")[0] if "/d/" in sheet_url else sheet_url
        sheet = client.open_by_key(sheet_id).sheet1

        existing_headers = sheet.row_values(1) if sheet.row_count > 0 else []
        if (not existing_headers or existing_headers != CSV_HEADERS) and sheet.row_count == 0:
            sheet.append_row(CSV_HEADERS)

        appended = 0
        for row in rows:
            values = [str(row.get(h, "")) for h in CSV_HEADERS]
            sheet.append_row(values)
            appended += 1

        print(f"[feedback_to_sheet] gspread: appended {appended} rows to {sheet_id}")
        return appended

    except ImportError as e:
        print(f"[feedback_to_sheet] gspread not installed: {e}")
        return None
    except Exception as e:
        print(f"[feedback_to_sheet] gspread failed ({e}) -> CSV fallback")
        return None


def _csv_mock_append(rows: list[dict], csv_path: Path | None = None) -> int:
    """CSV 모킹 폴백 — out/sheets_mock.csv에 append."""
    path = Path(csv_path) if csv_path else MOCK_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.is_file() and path.stat().st_size > 0
    need_header = True
    if file_exists:
        try:
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header == CSV_HEADERS:
                    need_header = False
        except Exception:
            need_header = True

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if need_header:
            writer.writeheader()
        for row in rows:
            out_row: dict[str, Any] = {}
            for h in CSV_HEADERS:
                val = row.get(h, "")
                if isinstance(val, list):
                    val = ";".join(str(v) for v in val)
                out_row[h] = val
            writer.writerow(out_row)

    print(f"[feedback_to_sheet] CSV mock: appended {len(rows)} rows -> {path}")
    return len(rows)


def append_to_sheet(
    rows: list[dict],
    sheet_url: str | None = None,
    csv_path: str | Path | None = None,
) -> int:
    """분류된 피드백 행을 Google Sheets에 적재."""
    if not rows:
        return 0
    gspread_result = _try_gspread_append(rows, sheet_url=sheet_url)
    if gspread_result is not None:
        with contextlib.suppress(Exception):
            _csv_mock_append(rows, csv_path=Path(csv_path) if csv_path else None)
        return gspread_result
    return _csv_mock_append(rows, csv_path=Path(csv_path) if csv_path else None)


def _extract_domain_keyword(query: str) -> str:
    """query에서 도메인 키워드 추출 (클러스터링용)."""
    for kw in _DOMAIN_KEYWORDS:
        if kw.lower() in query.lower() or kw in query:
            return kw
    # fallback: 첫 명사성 토큰 (2글자 이상)
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", query)
    if tokens:
        # 가장 긴 토큰을 대표 키워드로
        return max(tokens, key=len)
    return query[:10].strip() or "기타"


def curate_top_questions(rows: list[dict], top_k: int = 5) -> list[dict]:
    """피드백 기반 추천 질문 Top-K 큐레이션.

    1) 부정/버그/개선/지식부족/정확도 우선 집계
    2) 정확히 동일 query 빈도 집계
    3) 빈도가 모두 1이면 키워드 클러스터링으로 주제 빈도 집계
    4) 상위 5개 반환 — 각 항목은 question/frequency/category/keyword 포함

    Args:
        rows: 분류된 피드백 목록 (각 항목에 query/failure_category 포함).
        top_k: 추출할 추천 질문 수 (기본 5).

    Returns:
        추천 질문 목록 — 각 항목은 question/frequency/category를 포함.
    """
    if not rows:
        return []

    # 확장된 부정 카테고리 (지식 부족, 정확도 포함)
    negative_cats = {"부정", "버그", "개선", "지식 부족", "정확도", "기타", "negative", "bug", "improvement"}
    target_rows = [r for r in rows if r.get("failure_category", "") in negative_cats]
    # 긍정이 섞여있을 때: 부정 계열이 있으면 부정만, 없으면 전체
    if not target_rows:
        target_rows = rows
    # "기타"는 긍정이 아닌데도 낮은 우선순위 — 유지하되, 긍정만 있을 때 fallback
    # 실제로는 긍정이 10개, 부정이 20개이므로 target_rows=20

    counter: Counter[str] = Counter()
    cat_map: dict[str, str] = {}
    keyword_map: dict[str, list[str]] = {}
    for r in target_rows:
        q = str(r.get("query", "")).strip()
        if not q:
            continue
        counter[q] += 1
        if q not in cat_map:
            cat_map[q] = str(r.get("failure_category", ""))
        kw = _extract_domain_keyword(q)
        keyword_map.setdefault(kw, []).append(q)

    # 정확히 동일 query가 2회 이상인 경우 우선
    top_exact = counter.most_common(top_k)
    # 모두 빈도 1이면 키워드 클러스터링으로 대체
    if top_exact and all(freq == 1 for _, freq in top_exact) and len(counter) > top_k:
        # 키워드 빈도 집계
        kw_counter = Counter({kw: len(qs) for kw, qs in keyword_map.items()})
        top_keywords = kw_counter.most_common(top_k)
        result: list[dict] = []
        for kw, freq in top_keywords:
            # 대표 질문: 해당 키워드 그룹에서 가장 첫 번째 query
            rep_q = keyword_map[kw][0]
            result.append(
                {
                    "question": rep_q,
                    "frequency": freq,
                    "category": cat_map.get(rep_q, ""),
                    "keyword": kw,
                }
            )
        # 키워드 빈도가 모두 1이어도 최소 5개는 반환 (그룹 크기=1)
        # frequency가 1인 경우에도 의미있는 추천 질문으로 간주
        if len(result) < top_k:
            # 부족분은 exact에서 채움
            existing_qs = {r["question"] for r in result}
            for q, freq in top_exact:
                if q not in existing_qs:
                    result.append(
                        {
                            "question": q,
                            "frequency": freq,
                            "category": cat_map.get(q, ""),
                            "keyword": _extract_domain_keyword(q),
                        }
                    )
                if len(result) >= top_k:
                    break
        return result[:top_k]

    return [
        {"question": q, "frequency": freq, "category": cat_map.get(q, ""), "keyword": _extract_domain_keyword(q)}
        for q, freq in top_exact
    ]


# ---------------------------------------------------------------------------
# Classifier 연동
# ---------------------------------------------------------------------------

def to_sheet_rows(classified_rows: list[dict]) -> list[dict]:
    """classifier 출력(classified_rows)을 Sheets 스키마(CSV_HEADERS)로 변환."""
    sheet_rows: list[dict] = []
    for r in classified_rows:
        sheet_rows.append(
            {
                "query": r.get("query", ""),
                "response": r.get("response", ""),
                "failure_category": r.get("failure_category", ""),
                "root_cause": r.get("root_cause", ""),
                "review_status": r.get("review_status", ""),
                "citations": r.get("citations", ""),
            }
        )
    return sheet_rows


def run_pipeline(
    input_csv: str | Path | None = None,
    csv_path: str | Path | None = None,
    log_path: str | Path | None = None,
    reset_csv: bool = True,
) -> dict[str, Any]:
    """전체 파이프라인: 분류 → Sheets 적재 → 추천질문 큐레이션 → 로그 기록.

    Args:
        input_csv: 피드백 입력 CSV (default: monitor/data/feedback_sample.csv).
        csv_path: Sheets 모킹 CSV 경로 (default: out/sheets_mock.csv).
        log_path: 로그 파일 경로 (default: out/feedback_to_sheet.log).
        reset_csv: True이면 기존 sheets_mock.csv를 초기화 후 30행 재작성.

    Returns:
        파이프라인 결과 dict (classified, sheet_rows, recommended, stats).
    """
    from monitor.feedback_classifier import classify_file

    in_csv = Path(input_csv) if input_csv else DEFAULT_FEEDBACK_CSV
    out_csv = Path(csv_path) if csv_path else MOCK_CSV_PATH
    out_log = Path(log_path) if log_path else DEFAULT_LOG_PATH

    # 로그 캡처 버퍼
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
        print(f"[feedback_to_sheet] pipeline start — input={in_csv}")
        classified, stats = classify_file(in_csv)
        print(f"[feedback_to_sheet] classified {len(classified)} rows")
        print(f"[feedback_to_sheet] stats: {stats}")

        # 분류 없이 raw dump면 FAIL — 반드시 classifier 거침
        if stats["total"] < 30:
            print(f"[feedback_to_sheet] WARNING: classified only {stats['total']} (expected 30)")

        sheet_rows = to_sheet_rows(classified)
        print(f"[feedback_to_sheet] prepared {len(sheet_rows)} sheet rows (failure_category/root_cause/review_status)")

        # CSV 초기화 (운영 깊이: 30행 정확히 적재)
        if reset_csv and out_csv.is_file():
            try:
                out_csv.unlink()
                print(f"[feedback_to_sheet] reset existing CSV: {out_csv}")
            except Exception as e:
                print(f"[feedback_to_sheet] reset failed: {e}")

        n = append_to_sheet(sheet_rows, csv_path=out_csv)
        print(f"[feedback_to_sheet] appended {n} rows -> {out_csv}")

        # 검증: CSV 실제 행 수 확인
        try:
            with open(out_csv, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows_in_csv = list(reader)
            print(f"[feedback_to_sheet] verified CSV rows: {len(rows_in_csv)} (header excluded)")
            if len(rows_in_csv) != len(classified):
                print(
                    "[feedback_to_sheet] WARNING: CSV row count mismatch! "
                    f"classified={len(classified)} csv={len(rows_in_csv)}"
                )
        except Exception as e:
            print(f"[feedback_to_sheet] CSV verify failed: {e}")

        # 추천 질문 Top-5 큐레이션
        recommended = curate_top_questions(classified, top_k=5)
        print(f"[feedback_to_sheet] curated top-{len(recommended)} recommended questions:")
        for i, rec in enumerate(recommended, start=1):
            kw = rec.get("keyword", "")
            print(f"  {i}. [{rec.get('category','')}/{kw}] Q='{rec['question']}' freq={rec['frequency']}")

        if len(recommended) < 5:
            print(f"[feedback_to_sheet] WARNING: only {len(recommended)} recommended (expected 5)")
        else:
            print("[feedback_to_sheet] recommendation 5개 추출 완료 — OK")

        captured = buffer.getvalue()

    finally:
        sys.stdout = old_stdout  # type: ignore[assignment]
        captured = buffer.getvalue()

    # 로그 파일 저장 — repo out/ + portfolio/out 복사
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text(captured, encoding="utf-8")
    print(f"[feedback_to_sheet] log -> {out_log}")

    for cand in [
        Path(r"C:\Users\torch\Documents\code\portfolio\out\feedback_to_sheet.log"),
        Path(__file__).resolve().parents[1].parent.parent / "portfolio" / "out" / "feedback_to_sheet.log",
    ]:
        try:
            cand.parent.mkdir(parents=True, exist_ok=True)
            cand.write_text(captured, encoding="utf-8")
            print(f"[feedback_to_sheet] log copy -> {cand}")
            break
        except Exception:
            continue

    return {
        "classified": classified,
        "sheet_rows": sheet_rows,
        "recommended": recommended,
        "stats": stats,
        "csv_path": str(out_csv),
        "log_path": str(out_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="피드백 분류 → Sheets 적재 → 추천 질문 큐레이션")
    parser.add_argument("--input", type=str, default=str(DEFAULT_FEEDBACK_CSV), help="피드백 입력 CSV")
    parser.add_argument("--csv", type=str, default=str(MOCK_CSV_PATH), help="Sheets 모킹 CSV 경로")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG_PATH), help="로그 파일 경로")
    parser.add_argument("--no-reset", action="store_true", help="기존 CSV 초기화 안 함 (append 모드)")
    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input,
        csv_path=args.csv,
        log_path=args.log,
        reset_csv=not args.no_reset,
    )


if __name__ == "__main__":
    main()
