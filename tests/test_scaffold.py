"""Scaffold sanity tests — Task 14 verification."""

from pathlib import Path


def test_repo_structure_exists() -> None:
    """Required directories and scaffold files must exist."""
    root = Path(__file__).resolve().parents[1]
    for d in ["rag", "graph", "eval", "monitor", "docs", "data", "tests"]:
        assert (root / d).is_dir(), f"missing directory: {d}"
    for f in [
        "README.md",
        "requirements.txt",
        "LICENSE",
        ".github/workflows/ci.yml",
        "monitor/feedback_classifier.py",
        "monitor/feedback_to_sheet.py",
        "graph/state.py",
        "graph/nodes.py",
    ]:
        assert (root / f).is_file(), f"missing file: {f}"


def test_readme_contains_jd_keywords() -> None:
    """README must contain JD keywords and offline/online boundary."""
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for kw in ["VectorDB", "LangGraph", "RAGAS", "Sheets", "추천질문"]:
        assert kw in text, f"README missing JD keyword: {kw}"
    assert "llm-eval-pipeline" in text
    assert "https://github.com/ianlyoo/llm-eval-pipeline" in text


def test_no_eval_code_copy() -> None:
    """B must not copy A eval code — guard for Task 18 diff check (stub phase)."""
    # Scaffold phase: eval dir should be empty except __init__.py
    root = Path(__file__).resolve().parents[1]
    eval_files = [p.name for p in (root / "eval").iterdir() if p.is_file()]
    assert "__init__.py" in eval_files
