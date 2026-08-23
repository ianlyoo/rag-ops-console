"""compare_rag.py — Baseline vs Improved markdown comparison.

Loads two JSONs (baseline, improved) and prints a markdown table.
Supports both shapes:
- RAGAS (rag-ops-console): out/ragas_result.json with {aggregate: {avg_context_precision, avg_context_recall, avg_faithfulness}} or flat baseline_metrics.json
- Metrics (llm-eval-pipeline): out/metrics_report.json with {hit_at_5: {hit_at_k}, faithfulness: {faithfulness}, failure_rate: {...}}

Usage:
    python scripts/compare_rag.py
    python scripts/compare_rag.py --baseline out/baseline_metrics.json --improved out/improved_metrics.json
    python scripts/compare_rag.py --baseline out/baseline_ragas_result.json --improved out/ragas_result.json --output out/comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def _extract_metrics(data: dict) -> dict[str, float]:
    """Normalize to flat metric dict."""
    out: dict[str, float] = {}
    # RAGAS aggregate shape
    if "aggregate" in data:
        agg = data["aggregate"] or {}
        for k in ("avg_context_precision", "avg_context_recall", "avg_faithfulness"):
            if k in agg:
                out[k] = float(agg[k])
        if "num_samples" in agg:
            out["num_samples"] = float(agg["num_samples"])
    # Flat RAGAS (baseline_metrics.json saved as aggregate)
    for k in ("avg_context_precision", "avg_context_recall", "avg_faithfulness", "context_precision", "context_recall", "faithfulness"):
        if k in data and isinstance(data[k], (int, float)):
            out[k] = float(data[k])
    # llm-eval-pipeline shape
    if "hit_at_5" in data and isinstance(data["hit_at_5"], dict):
        out["hit_at_5"] = float(data["hit_at_5"].get("hit_at_k", 0))
    if "faithfulness" in data and isinstance(data["faithfulness"], dict):
        out["faithfulness_offline"] = float(data["faithfulness"].get("faithfulness", 0))
    if "failure_rate" in data and isinstance(data["failure_rate"], dict):
        out["failure_rate"] = float(data["failure_rate"].get("failure_rate", 0))
    # generic flat fallbacks
    for k in ("hit_at_k", "hit_at_5", "failure_rate"):
        if k in data and isinstance(data[k], (int, float)):
            out[k] = float(data[k])
    return out


def _resolve_default(baseline: str | None, improved: str | None) -> tuple[Path | None, Path | None]:
    candidates_baseline = [
        Path("out/baseline_metrics.json"),
        Path("out/baseline_ragas_result.json"),
        Path("out/ragas_result.json"),
    ]
    candidates_improved = [
        Path("out/improved_metrics.json"),
        Path("out/ragas_result.json"),
        Path("out/metrics_report.json"),
    ]
    b_path = Path(baseline) if baseline else next((p for p in candidates_baseline if p.exists()), None)
    i_path = Path(improved) if improved else next((p for p in candidates_improved if p.exists()), None)
    return b_path, i_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Compare baseline vs improved metrics (markdown)")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline JSON (e.g., out/baseline_metrics.json)")
    parser.add_argument("--improved", type=str, default=None, help="Improved JSON (e.g., out/improved_metrics.json)")
    parser.add_argument("--output", type=str, default="out/comparison.md", help="Output markdown path")
    args = parser.parse_args()

    b_path, i_path = _resolve_default(args.baseline, args.improved)

    if b_path is None or i_path is None:
        # Use synthetic before/after from docs if files missing — still produce a helpful table
        print("[compare_rag] baseline or improved not found; using documented Before/After defaults.", file=sys.stderr)
        print("[compare_rag] Expected: out/baseline_metrics.json and out/improved_metrics.json")
        print("[compare_rag] Example: python scripts/compare_rag.py --baseline out/baseline_metrics.json --improved out/improved_metrics.json")
        # Fallback demo table (rag-ops-console documented numbers)
        md = (
            "# RAG Comparison — Baseline vs Improved (fallback, no JSON found)\n\n"
            "| Metric | Baseline (before) | Improved (after) | Δ | Δ% |\n"
            "|--------|-------------------|------------------|---|----|\n"
            "| context_precision | 0.3102 | 0.4200 | +0.1098 | +35.4% |\n"
            "| context_recall | 0.5120 | 0.6270 | +0.1150 | +22.5% |\n"
            "| faithfulness | 0.9710 | 0.9953 | +0.0243 | +2.5% |\n"
            "\n"
            "_Source: docs/report.html §1, out/ragas_result.json aggregate. Run `python -m eval.ragas_eval --output out/ragas_result.json` then re-run this script._\n"
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(md)
        print(f"[compare_rag] wrote fallback -> {out_path} (exit 0, create real JSONs to diff)")
        return

    if not b_path.exists():
        print(f"[compare_rag] baseline not found: {b_path}", file=sys.stderr)
        sys.exit(2)
    if not i_path.exists():
        print(f"[compare_rag] improved not found: {i_path}", file=sys.stderr)
        sys.exit(2)

    b_data = _load(b_path)
    i_data = _load(i_path)
    b_m = _extract_metrics(b_data)
    i_m = _extract_metrics(i_data)

    keys = sorted(set(b_m) | set(i_m))
    # Prefer known order
    preferred = ["avg_context_precision", "context_precision", "avg_context_recall", "context_recall", "avg_faithfulness", "faithfulness", "hit_at_5", "hit_at_k", "faithfulness_offline", "failure_rate", "num_samples"]
    ordered = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]

    lines = ["# Comparison — Baseline vs Improved", "", f"Baseline: `{b_path}` | Improved: `{i_path}`", "", "| Metric | Baseline | Improved | Δ | Δ% |", "|--------|----------|----------|---|----|"]
    for k in ordered:
        bv = b_m.get(k)
        iv = i_m.get(k)
        if bv is None or iv is None:
            bv_s = f"{bv:.4f}" if isinstance(bv, float) else str(bv) if bv is not None else "—"
            iv_s = f"{iv:.4f}" if isinstance(iv, float) else str(iv) if iv is not None else "—"
            lines.append(f"| {k} | {bv_s} | {iv_s} | — | — |")
            continue
        delta = iv - bv
        pct = (delta / bv * 100) if bv != 0 else 0.0
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {k} | {bv:.4f} | {iv:.4f} | {sign}{delta:.4f} | {sign}{pct:.1f}% |")
    lines += ["", f"_Generated by `python scripts/compare_rag.py --baseline {b_path} --improved {i_path}`_", ""]
    md = "\n".join(lines)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[compare_rag] wrote -> {out_path}")


if __name__ == "__main__":
    main()
