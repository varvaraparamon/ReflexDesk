"""Контейнер зависимостей узлов графа."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

from reflexdesk.agents.base import Agent, AgentSpec
from reflexdesk.config import Settings, get_settings
from reflexdesk.gateway import Gates, get_gateway
from reflexdesk.memory import EpisodicStore, Knowledge, get_episodic, get_knowledge
from reflexdesk.prompting import PromptLibrary, get_library
from reflexdesk.router.router import LLMRouter, get_router


@dataclass
class Services:
    """Роутер, гейты, промпты и память; L3 подгружается лениво при первом обращении."""

    settings: Settings
    router: LLMRouter
    gates: Gates
    library: PromptLibrary
    episodic: EpisodicStore
    knowledge_factory: Callable[[], Knowledge]

    def agent(self, spec: AgentSpec) -> Agent:
        return Agent(spec, self.router, self.library)

    @cached_property
    def knowledge(self) -> Knowledge:
        return self.knowledge_factory()


def default_services() -> Services:
    """Сервисы процесса из синглтонов."""
    return Services(
        settings=get_settings(),
        router=get_router(),
        gates=Gates(get_gateway()),
        library=get_library(),
        episodic=get_episodic(),
        knowledge_factory=get_knowledge,
    )
