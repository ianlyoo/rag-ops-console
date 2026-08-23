"""Fail-closed regression tests for scripts/compare_rag.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_rag.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
HARDCODED_METRICS = ["0.3102", "0.5120", "0.9710"]


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_rag", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_missing_baseline_raises(tmp_path: Path) -> None:
    mod = _load_module()
    with pytest.raises(FileNotFoundError, match="baseline"):
        mod._require_artifact(None, "baseline")
    with pytest.raises(FileNotFoundError, match="baseline"):
        mod._require_artifact(tmp_path / "no_such.json", "baseline")
    missing = tmp_path / "missing_baseline.json"
    improved = tmp_path / "improved.json"
    improved.write_text(json.dumps({"aggregate": {"avg_context_precision": 0.5}}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline", str(missing), "--improved", str(improved)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "baseline" in result.stderr.lower()


def test_missing_improved_raises(tmp_path: Path) -> None:
    mod = _load_module()
    with pytest.raises(FileNotFoundError, match="improved"):
        mod._require_artifact(None, "improved")
    with pytest.raises(FileNotFoundError, match="improved"):
        mod._require_artifact(tmp_path / "no_such.json", "improved")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"aggregate": {"avg_context_precision": 0.5}}), encoding="utf-8")
    missing = tmp_path / "missing_improved.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline", str(baseline), "--improved", str(missing)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "improved" in result.stderr.lower()


def test_same_path_exits_nonzero(tmp_path: Path) -> None:
    artifact = tmp_path / "same.json"
    artifact.write_text(json.dumps({"aggregate": {"avg_context_precision": 0.42}}), encoding="utf-8")
    output = tmp_path / "comparison.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(artifact),
            "--improved",
            str(artifact),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "same file" in result.stderr.lower()


def test_valid_two_artifacts_creates_comparison_md(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    improved = tmp_path / "improved.json"
    baseline.write_text(
        json.dumps({"aggregate": {"avg_context_precision": 0.1, "avg_context_recall": 0.2}}),
        encoding="utf-8",
    )
    improved.write_text(
        json.dumps({"aggregate": {"avg_context_precision": 0.9, "avg_context_recall": 0.8}}),
        encoding="utf-8",
    )
    output = tmp_path / "comparison.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--improved",
            str(improved),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    content = output.read_text(encoding="utf-8")
    assert "0.1000" in content
    assert "0.9000" in content
    assert "0.2000" in content
    assert "0.8000" in content
    assert f"`{baseline}`" in content or str(baseline) in content
    assert f"`{improved}`" in content or str(improved) in content


def test_generated_comparison_uses_only_input_values(tmp_path: Path) -> None:
    baseline_val = 0.2468
    improved_val = 0.1357
    baseline = tmp_path / "baseline.json"
    improved = tmp_path / "improved.json"
    baseline.write_text(
        json.dumps({"aggregate": {"avg_context_precision": baseline_val}}),
        encoding="utf-8",
    )
    improved.write_text(
        json.dumps({"aggregate": {"avg_context_precision": improved_val}}),
        encoding="utf-8",
    )
    output = tmp_path / "comparison.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--improved",
            str(improved),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    content = output.read_text(encoding="utf-8")
    assert f"{baseline_val:.4f}" in content
    assert f"{improved_val:.4f}" in content
    for hardcoded in HARDCODED_METRICS:
        assert hardcoded not in content, f"hardcoded {hardcoded} leaked into output"


def test_source_has_no_hardcoded_fallback_metric() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for hardcoded in HARDCODED_METRICS:
        assert hardcoded not in source, f"hardcoded fallback {hardcoded} found in {SCRIPT}"
    lowered = source.lower()
    assert "fail" in lowered or "fallback" in lowered
