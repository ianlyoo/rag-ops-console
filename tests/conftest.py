"""Keep fixture tests independent of model downloads and Sheets credentials."""

import pytest


@pytest.fixture(autouse=True)
def offline_backends(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setattr("monitor.feedback_to_sheet._try_gspread_append", lambda *args, **kwargs: None)
