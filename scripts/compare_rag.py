"""Compare measured baseline and improved evaluation artifacts.

This script intentionally fails closed: it never invents or falls back to
hard-coded metric values. A comparison is produced only when both artifacts
exist and are distinct files.

Supported shapes:
- RAGAS-style: {"aggregate": {"avg_context_precision": ..., ...}}
- Flat RAG metrics: {"context_precision": ..., "context_recall": ..., ...}
- llm-eval metrics: nested hit_at_5 / faithfulness / failure_rate objects

Usage:
    python scripts/compare_rag.py \
      --baseline out/baseline_metrics.json \
      --improved out/improved_metrics.json

    python scripts/compare_rag.py \
      --baseline out/baseline_ragas_result.json \
      --improved out/ragas_result.json \
      --output out/comparison.md
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def _extract_metrics(data: dict) -> dict[str, float]:
    """Normalize supported result shapes to a flat metric mapping."""
    out: dict[str, float] = {}

    if "aggregate" in data and isinstance(data["aggregate"], dict):
        agg = data["aggregate"] or {}
        for key in (
            "avg_context_precision",
            "avg_context_recall",
            "avg_faithfulness",
            "num_samples",
        ):
            if isinstance(agg.get(key), (int, float)):
                out[key] = float(agg[key])

    for key in (
        "avg_context_precision",
        "avg_context_recall",
        "avg_faithfulness",
        "context_precision",
        "context_recall",
        "faithfulness",
        "num_samples",
    ):
        if isinstance(data.get(key), (int, float)):
            out[key] = float(data[key])

    if isinstance(data.get("hit_at_5"), dict):
        value = data["hit_at_5"].get("hit_at_k")
        if isinstance(value, (int, float)):
            out["hit_at_5"] = float(value)

    if isinstance(data.get("faithfulness"), dict):
        value = data["faithfulness"].get("faithfulness")
        if isinstance(value, (int, float)):
            out["faithfulness_offline"] = float(value)

    if isinstance(data.get("failure_rate"), dict):
        value = data["failure_rate"].get("failure_rate")
        if isinstance(value, (int, float)):
            out["failure_rate"] = float(value)

    for key in ("hit_at_k", "failure_rate"):
        if isinstance(data.get(key), (int, float)):
            out[key] = float(data[key])

    return out


def _first_existing(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.is_file()), None)


def _resolve_paths(
    baseline: str | None,
    improved: str | None,
) -> tuple[Path | None, Path | None]:
    baseline_path = Path(baseline) if baseline else _first_existing(
        [Path("out/baseline_metrics.json"), Path("out/baseline_ragas_result.json")]
    )
    improved_path = Path(improved) if improved else _first_existing(
        [Path("out/improved_metrics.json"), Path("out/improved_ragas_result.json")]
    )
    return baseline_path, improved_path


def _require_artifact(path: Path | None, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"{label} artifact not found. Provide --{label} or create a measured "
            f"out/{label}_metrics.json artifact first."
        )
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact not found: {path}")
    return path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Compare measured baseline vs improved evaluation artifacts"
    )
    parser.add_argument("--baseline", default=None, help="Measured baseline JSON")
    parser.add_argument("--improved", default=None, help="Measured improved JSON")
    parser.add_argument("--output", default="out/comparison.md", help="Output markdown")
    args = parser.parse_args()

    baseline_path, improved_path = _resolve_paths(args.baseline, args.improved)

    try:
        baseline_path = _require_artifact(baseline_path, "baseline")
        improved_path = _require_artifact(improved_path, "improved")
    except FileNotFoundError as exc:
        print(f"[compare_rag] ERROR: {exc}", file=sys.stderr)
        print(
            "[compare_rag] No synthetic fallback is used. Generate or restore both "
            "measured artifacts before comparing.",
            file=sys.stderr,
        )
        sys.exit(2)

    if baseline_path.resolve() == improved_path.resolve():
        print(
            "[compare_rag] ERROR: baseline and improved resolve to the same file. "
            "Use two independently measured artifacts.",
            file=sys.stderr,
        )
        sys.exit(2)

    baseline_metrics = _extract_metrics(_load(baseline_path))
    improved_metrics = _extract_metrics(_load(improved_path))

    if not baseline_metrics or not improved_metrics:
        print(
            "[compare_rag] ERROR: no supported metrics found in one or both artifacts.",
            file=sys.stderr,
        )
        sys.exit(2)

    keys = sorted(set(baseline_metrics) | set(improved_metrics))
    preferred = [
        "avg_context_precision",
        "context_precision",
        "avg_context_recall",
        "context_recall",
        "avg_faithfulness",
        "faithfulness",
        "hit_at_5",
        "hit_at_k",
        "faithfulness_offline",
        "failure_rate",
        "num_samples",
    ]
    ordered = [key for key in preferred if key in keys] + [
        key for key in keys if key not in preferred
    ]

    lines = [
        "# Comparison — Baseline vs Improved",
        "",
        f"Baseline: `{baseline_path}` | Improved: `{improved_path}`",
        "",
        "| Metric | Baseline | Improved | Δ | Δ% |",
        "|--------|----------|----------|---|----|",
    ]

    for key in ordered:
        before = baseline_metrics.get(key)
        after = improved_metrics.get(key)
        if before is None or after is None:
            before_s = "—" if before is None else f"{before:.4f}"
            after_s = "—" if after is None else f"{after:.4f}"
            lines.append(f"| {key} | {before_s} | {after_s} | — | — |")
            continue

        delta = after - before
        pct = (delta / before * 100.0) if before != 0 else 0.0
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {key} | {before:.4f} | {after:.4f} | "
            f"{sign}{delta:.4f} | {sign}{pct:.1f}% |"
        )

    lines += [
        "",
        "_Generated only from the two artifacts named above; no fallback values are embedded._",
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = "\n".join(lines)
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\n[compare_rag] wrote -> {output_path}")


if __name__ == "__main__":
    main()
