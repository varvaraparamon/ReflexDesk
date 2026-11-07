"""Сохранение отчёта: Markdown и PDF (pandoc + pdf-движок)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PDF_ENGINES = ("tectonic", "xelatex", "wkhtmltopdf", "weasyprint")
LATEX_ENGINES = ("tectonic", "xelatex")
RENDER_TIMEOUT_S = 180
DISCLAIMER = "Материал носит исследовательский характер и не является индивидуальной инвестиционной рекомендацией."


class RenderError(RuntimeError):
    """PDF не может быть собран."""


@dataclass(frozen=True)
class RenderedReport:
    md_path: Path
    pdf_path: Path


def ensure_disclaimer(markdown: str) -> str:
    """Добавляет обязательный дисклеймер, если его нет."""
    return markdown if DISCLAIMER in markdown else f"{markdown.rstrip()}\n\n---\n{DISCLAIMER}\n"


def pdf_engine() -> str:
    """Первый доступный pdf-движок pandoc."""
    if not shutil.which("pandoc"):
        raise RenderError("pandoc is not installed")
    engine = next((e for e in PDF_ENGINES if shutil.which(e)), None)
    if engine is None:
        raise RenderError(f"no pdf engine found: {PDF_ENGINES}")
    return engine


def render_pdf(md_path: Path, title: str) -> Path:
    """md → pdf рядом с исходником; pandoc запускается в каталоге отчёта (временные файлы пишутся туда)."""
    engine = pdf_engine()
    pdf_path = md_path.with_suffix(".pdf")
    command = ["pandoc", md_path.name, "-o", pdf_path.name, f"--pdf-engine={engine}", "--metadata", f"title={title}"]
    if engine in LATEX_ENGINES:
        command += ["-V", "mainfont=DejaVu Sans"]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=RENDER_TIMEOUT_S, cwd=md_path.parent)
    except subprocess.CalledProcessError as exc:
        raise RenderError(exc.stderr.decode(errors="ignore")[-1000:]) from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"pandoc timed out after {RENDER_TIMEOUT_S}s") from exc
    return pdf_path


def render(markdown: str, directory: Path, name: str) -> RenderedReport:
    """Пишет <name>.md и собирает <name>.pdf."""
    directory.mkdir(parents=True, exist_ok=True)
    md_path = directory / f"{name}.md"
    md_path.write_text(markdown, encoding="utf-8")
    return RenderedReport(md_path, render_pdf(md_path, name))
