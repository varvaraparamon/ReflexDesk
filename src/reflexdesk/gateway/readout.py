"""Readout в формате AnyJev/Tacit: промпт State/Question/Options, метки-токены, softmax по меткам."""

from __future__ import annotations

import math
import string
from collections.abc import Mapping

from reflexdesk.gateway.spec import DecisionSpec

DECISION_SYSTEM = (
    "You are a decision function. You will be given a state and one question. "
    "Reply with the answer label only: no words, no punctuation, no explanation."
)
REASONING_SYSTEM = "You are a careful assistant that answers questions about a given context."
LETTERS = string.ascii_uppercase
MAX_DIGIT_LEVELS = 9


def labels(spec: DecisionSpec) -> list[str]:
    """Метки ответа: Yes/No для noul, цифры 1..k для score (до 9 уровней), иначе буквы."""
    if spec.kind == "noul":
        return ["Yes", "No"]
    if spec.kind == "score" and len(spec.options) <= MAX_DIGIT_LEVELS:
        return [str(i + 1) for i in range(len(spec.options))]
    return list(LETTERS[: len(spec.options)])


def _option_lines(spec: DecisionSpec, marks: list[str], order: list[str]) -> list[str]:
    """Строки вариантов в заданном порядке: «метка. имя — описание»."""
    lines = []
    for mark, option in zip(marks, order, strict=True):
        desc = spec.criteria.get(option)
        lines.append(f"{mark}. {option}" + (f" — {desc}" if desc else ""))
    return lines


def _answer_hint(marks: list[str]) -> str:
    return "Answer with the number only." if marks[0].isdigit() else "Answer with the letter only."


def _noul_criteria(spec: DecisionSpec) -> list[str]:
    names = {"yes": "Yes", "no": "No"}
    return [f"{names[o]}: {spec.criteria[o]}" for o in spec.options if o in spec.criteria]


def decision_prompt(spec: DecisionSpec, order: list[str] | None = None, state: str | None = None) -> str:
    """Пользовательская часть промпта в формате Tacit; order — порядок показа, state — подмена состояния."""
    marks = labels(spec)
    order = order or spec.options
    parts = ["State:", (spec.context if state is None else state) or "(empty)", "", f"Question: {spec.question}"]
    if spec.kind == "noul":
        parts += [*_noul_criteria(spec), "Answer Yes or No."]
    elif spec.kind == "score":
        parts += ["Pick the level that applies (the levels are ordered):", *_option_lines(spec, marks, order), _answer_hint(marks)]
    else:
        parts += ["Options:", *_option_lines(spec, marks, order), _answer_hint(marks)]
    return "\n".join(parts)


def reasoning_prompt(spec: DecisionSpec) -> str:
    """Промпт эскалации: рассуждение и ответ последней строкой «Answer: <метка>»."""
    marks = labels(spec)
    parts = [spec.context.strip(), "", f"Question: {spec.question}", ""]
    if spec.kind == "noul":
        parts += [*_noul_criteria(spec), "Answer Yes or No."]
        form = "Answer: Yes  or  Answer: No"
    elif spec.kind == "score":
        parts += ["Levels:", *_option_lines(spec, marks, spec.options)]
        form = "Answer: <level number>" if marks[0].isdigit() else "Answer: <level letter>"
    else:
        parts += ["Options:", *_option_lines(spec, marks, spec.options)]
        form = "Answer: <option letter>"
    parts += ["", "Think through the question, then give your final answer on the last line in exactly this form:", form]
    return "\n".join(parts)


def chatml(system: str, user: str, *, thinking: bool) -> str:
    """Шаблон чата Qwen3 с generation prompt; без размышлений — с пустым блоком think."""
    head = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return head if thinking else head + "<think>\n\n</think>\n\n"


def close_thought(text: str) -> str:
    """Рассуждение до </think> включительно; закрывается вручную, если бюджет токенов исчерпан."""
    end = text.find("</think>")
    return text[: end + len("</think>")] if end >= 0 else text + "\n</think>"


def label_distribution(top_logprobs: Mapping[str, float], spec: DecisionSpec) -> dict[str, float]:
    """Softmax только по меткам; метка вне top-k получает вероятность 0."""
    mass = dict.fromkeys(labels(spec), 0.0)
    for token, logprob in top_logprobs.items():
        key = token.strip()
        if key in mass:
            mass[key] += math.exp(logprob)
    total = sum(mass.values())
    if total == 0.0:
        raise ValueError("no option label among top logprobs")
    return {mark: value / total for mark, value in mass.items()}


def rotations(spec: DecisionSpec, limit: int) -> list[list[str]]:
    """Циклические сдвиги порядка вариантов (L0); noul и score не переставляются."""
    if spec.kind != "choice":
        return [list(spec.options)]
    k = len(spec.options)
    return [spec.options[i:] + spec.options[:i] for i in range(min(k, limit))]


def to_options(label_dist: Mapping[str, float], spec: DecisionSpec, order: list[str]) -> dict[str, float]:
    """Распределение по меткам позиции → распределение по вариантам для данного порядка (noul — по смыслу)."""
    if spec.kind == "noul":
        return {"yes": label_dist["Yes"], "no": label_dist["No"]}
    return {option: label_dist[mark] for mark, option in zip(labels(spec), order, strict=True)}


def divide_prior(label_dist: Mapping[str, float], prior: Mapping[str, float]) -> dict[str, float]:
    """Контекстная калибровка: деление на распределение меток при пустом состоянии, с нижней границей prior."""
    floor = 0.1 / len(label_dist)
    raw = {k: v / max(prior[k], floor) for k, v in label_dist.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def average(dists: list[dict[str, float]]) -> dict[str, float]:
    keys = dists[0].keys()
    return {k: sum(d[k] for d in dists) / len(dists) for k in keys}
