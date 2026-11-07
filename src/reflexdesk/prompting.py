"""Скиллы и сборка системных промптов: DESK + RISK + роль + активные скиллы, с хэшем версии."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from reflexdesk.config import get_settings

BASE_PROMPTS = ("DESK.md", "RISK.md")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Skill:
    """Скилл: метаданные из frontmatter и тело методики."""

    name: str
    description: str
    body: str
    tools_required: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: str = ""
    quality_checklist: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        return _sha(self.render())

    def render(self) -> str:
        checklist = "\n".join(f"- {item}" for item in self.quality_checklist)
        return (
            f"### Скилл: {self.name}\nКогда применять: {self.description}\n\n{self.body.strip()}\n\n"
            f"Чек-лист качества:\n{checklist}"
        )


@dataclass(frozen=True)
class SystemPrompt:
    """Собранный системный промпт и версии его частей."""

    text: str
    parts: dict[str, str] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return _sha(self.text)

    def trace_metadata(self) -> dict[str, str]:
        return {"prompt_hash": self.version, **{f"v:{k}": v for k, v in self.parts.items()}}


def parse_skill(text: str) -> Skill:
    """Разбирает markdown-файл скилла с YAML-frontmatter."""
    if not text.startswith("---"):
        raise ValueError("skill file must start with YAML frontmatter")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    return Skill(
        name=meta["name"],
        description=meta.get("description", ""),
        body=body,
        tools_required=tuple(meta.get("tools_required") or ()),
        inputs=tuple(meta.get("inputs") or ()),
        outputs=str(meta.get("outputs", "")),
        quality_checklist=tuple(meta.get("quality_checklist") or ()),
    )


class PromptLibrary:
    """Доступ к семантическим .md и скиллам; собирает промпты агентов."""

    def __init__(self, prompts_dir: Path, skills_dir: Path) -> None:
        self._prompts_dir = prompts_dir
        self._skills = {
            skill.name: skill
            for skill in (parse_skill(p.read_text(encoding="utf-8")) for p in sorted(skills_dir.glob("*.md")))
        }

    def skill(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"unknown skill {name!r}")
        return self._skills[name]

    def document(self, name: str) -> str:
        return (self._prompts_dir / name).read_text(encoding="utf-8")

    def build(
        self,
        agent: str,
        skills: tuple[str, ...] = (),
        tools: tuple[str, ...] = (),
        documents: tuple[str, ...] = (),
    ) -> SystemPrompt:
        """Системный промпт агента; скилл монтируется, если его tools_required доступны агенту."""
        parts: dict[str, str] = {}
        sections: list[str] = []
        for name in (*BASE_PROMPTS, *documents):
            text = self.document(name)
            parts[name] = _sha(text)
            sections.append(text)
        role = self.document(f"agents/{agent}.md")
        parts[f"agents/{agent}.md"] = _sha(role)
        sections.append(role)
        available = set(tools)
        for name in skills:
            skill = self.skill(name)
            if set(skill.tools_required) <= available:
                parts[f"skill:{name}"] = skill.version
                sections.append(skill.render())
        return SystemPrompt(text="\n\n".join(s.strip() for s in sections), parts=parts)


@lru_cache(maxsize=1)
def get_library() -> PromptLibrary:
    """Синглтон библиотеки промптов."""
    settings = get_settings()
    return PromptLibrary(settings.prompts_dir, settings.skills_dir)
