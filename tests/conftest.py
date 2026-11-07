"""Общие фикстуры: изолированный каталог данных, фейковые роутер, гейты, эмбеддер."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="reflexdesk-tests-"))
os.environ.update(
    REFLEXDESK_DATA_DIR=str(_TMP),
    DEEPSEEK_API_KEY="test-key",
    SEC_USER_AGENT="ReflexDesk tests tests@example.com",
    TRACING_ENABLED="false",
    LANGFUSE_PUBLIC_KEY="",
    LANGFUSE_SECRET_KEY="",
    LANGFUSE_HOST="",
)

import pytest
from fakes import FakeEmbedder
from qdrant_client import QdrantClient

from reflexdesk.config import get_settings

get_settings.cache_clear()


@pytest.fixture
def tmp_data_dir() -> Path:
    return _TMP


@pytest.fixture(autouse=True)
def fake_pdf(monkeypatch):
    """PDF-рендер подменяется: pandoc есть только в образе app."""

    def render_pdf(md_path: Path, title: str) -> Path:
        pdf = md_path.with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-1.4 test")
        return pdf

    monkeypatch.setattr("reflexdesk.reporting.render.render_pdf", render_pdf)


@pytest.fixture
def semantic_store():
    from reflexdesk.memory.semantic import SemanticStore

    return SemanticStore(QdrantClient(location=":memory:"), FakeEmbedder())


@pytest.fixture
def episodic(tmp_path):
    from sqlalchemy import create_engine

    from reflexdesk.memory.episodic import EpisodicStore

    return EpisodicStore(create_engine(f"sqlite:///{(tmp_path / 'l2.db').as_posix()}"))
