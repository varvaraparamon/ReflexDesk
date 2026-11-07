"""Bull и Bear: независимые позиции по одному AnalysisDraft."""

from __future__ import annotations

from typing import Any, Literal

from reflexdesk.agents.base import Usage
from reflexdesk.agents.services import Services
from reflexdesk.agents.specs import BEAR, BULL
from reflexdesk.schemas import DebateCase
from reflexdesk.state import DeskState

SPECS = {"bull": BULL, "bear": BEAR}


class DebateNode:
    """Одна сторона дебатов; не видит позицию оппонента."""

    def __init__(self, services: Services, side: Literal["bull", "bear"]) -> None:
        self._side = side
        self._agent = services.agent(SPECS[side])

    def __call__(self, state: DeskState) -> dict[str, Any]:
        usage = Usage()
        draft = state["analysis_draft"]
        prompt = (
            f"Сторона: {self._side}. Аналитический драфт:\n{draft.model_dump_json(exclude={'computations'})}"
        )
        case = self._agent.ask(prompt, DebateCase, usage)
        if case is not None:
            case.side = self._side
        return {f"{self._side}_case": case, **usage.as_update()}
