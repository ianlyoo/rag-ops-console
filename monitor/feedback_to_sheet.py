"""monitor.feedback_to_sheet — Google Sheets 적재 + 추천 질문 큐레이션.

- 적재: `gspread`로 Sheets에 행 단위 append (query, response, failure_category, root_cause 등)
- 큐레이션: 부정/버그/개선 피드백 빈도 집계 → Top-5 추천 질문 추출
- 운영: 고객 피드백 루프를 Sheets에서 모니터링하고 추천 질문을 선제안 콘텐츠로 운영
- gspread 호출 코드 포함하되 API 키 없으면 CSV 모킹 폴백 (out/sheets_mock.csv)

gspread 경로는 코드에 필수 — 실제 호출을 포함한다.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any

MOCK_CSV_PATH = Path(__file__).resolve().parents[1] / "out" / "sheets_mock.csv"
CSV_HEADERS = ["query", "response", "failure_category", "root_cause", "review_status", "citations"]


def _try_gspread_append(rows: list[dict], sheet_url: str | None = None) -> int | None:
    """gspread 실제 호출 — 성공 시 적재 행 수 반환, 실패 시 None."""
    try:
        import gspread  # type: ignore[import-untyped]
        from google.oauth2.service_account import Credentials  # type: ignore[import-untyped]

        # sheet_url이 없거나 credentials가 없으면 실패로 처리 (CSV fallback)
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        sheet_url = sheet_url or os.environ.get("GOOGLE_SHEETS_URL", "").strip()

        if not creds_path or not sheet_url:
            # gspread import는 성공했으나 인증 정보 없음 — fallback 유도
            print("[feedback_to_sheet] gspread: no credentials/sheet_url -> CSV fallback")
            return None

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        client = gspread.authorize(creds)

        # sheet_url에서 spreadsheet ID 추출
        # URL 형태: https://docs.google.com/spreadsheets/d/<ID>/edit
        sheet_id = sheet_url.split("/d/")[1].split("/")[0] if "/d/" in sheet_url else sheet_url
        sheet = client.open_by_key(sheet_id).sheet1

        # 헤더 확인 및 추가
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
    # 헤더가 이미 있는지 확인
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
            # citations가 list면 문자열로 변환
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
    """분류된 피드백 행을 Google Sheets에 적재.

    gspread 호출을 우선 시도하고, 인증/네트워크 실패 시 CSV 모킹으로 폴백한다.
    코드에 gspread 경로가 필수이므로 항상 _try_gspread_append를 먼저 호출한다.

    Args:
        rows: 적재할 피드백 dict 목록.
        sheet_url: 대상 Google Sheets URL (None이면 환경변수 GOOGLE_SHEETS_URL 참조).
        csv_path: CSV 모킹 경로 (테스트용 오버라이드).

    Returns:
        적재된 행 수.
    """
    if not rows:
        return 0

    # 1) gspread 시도 (코드 경로 필수)
    gspread_result = _try_gspread_append(rows, sheet_url=sheet_url)
    if gspread_result is not None:
        # gspread 성공 — 추가로 CSV에도 기록 (이중 적재, 운영 로그용)
        import contextlib

        with contextlib.suppress(Exception):
            _csv_mock_append(rows, csv_path=Path(csv_path) if csv_path else None)
        return gspread_result

    # 2) CSV 폴백
    return _csv_mock_append(rows, csv_path=Path(csv_path) if csv_path else None)


def curate_top_questions(rows: list[dict], top_k: int = 5) -> list[dict]:
    """피드백 기반 추천 질문 Top-K 큐레이션.

    Args:
        rows: 분류된 피드백 목록 (각 항목에 query/failure_category 포함).
        top_k: 추출할 추천 질문 수 (기본 5).

    Returns:
        추천 질문 목록 — 각 항목은 question/frequency/category를 포함.
    """
    if not rows:
        return []

    # 부정/버그/개선 피드백만 집계 (긍정은 제외 — 개선 필요 질문이 우선)
    negative_cats = {"부정", "버그", "개선", "negative", "bug", "improvement"}
    target_rows = [r for r in rows if r.get("failure_category", "") in negative_cats]
    if not target_rows:
        target_rows = rows

    counter: Counter[str] = Counter()
    cat_map: dict[str, str] = {}
    for r in target_rows:
        q = str(r.get("query", "")).strip()
        if not q:
            continue
        counter[q] += 1
        if q not in cat_map:
            cat_map[q] = str(r.get("failure_category", ""))

    top = counter.most_common(top_k)
    return [
        {"question": q, "frequency": freq, "category": cat_map.get(q, "")}
        for q, freq in top
    ]
