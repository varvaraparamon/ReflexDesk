"""Тексты и HTML-форматирование сообщений бота."""

from __future__ import annotations

import re
from datetime import datetime
from html import escape
from typing import Any

from reflexdesk.reporting.parse import section

SUMMARY_LIMIT = 1500
STATUS_ICONS = {"running": "⏳", "done": "✅", "failed": "❌", "rejected": "🚫", "interrupted": "⏸"}
DIRECTION_ICONS = {"up": "📈", "down": "📉", "flat": "➖"}
PREDICTION_ICONS = {"pending": "⏳", "hit": "✅", "miss": "❌", "expired": "⌛"}

HELP = (
    "<b>ReflexDesk</b> — исследовательский деск: собирает данные (рынок, SEC, новости, FRED), "
    "анализирует с дебатами Bull/Bear, проверяет числа валидатором и присылает отчёт PDF.\n\n"
    "<b>Как пользоваться</b>\n"
    "Просто напишите запрос, например:\n"
    "<code>Проанализируй NVDA на 90 дней</code>\n"
    "<code>Проанализируй MSFT 2024-03-01</code> — анализ на историческую дату\n\n"
    "<b>Команды</b>\n"
    "/analyze &lt;запрос&gt; [ГГГГ-ММ-ДД] — запустить анализ\n"
    "/runs — мои последние анализы\n"
    "/status &lt;id&gt; — статус анализа\n"
    "/report &lt;id&gt; — отчёт и PDF\n"
    "/predictions [ТИКЕР] — мои прогнозы и их сверка\n"
    "/metrics — качество прогнозов деска\n\n"
    "Отчёт приходит сам, когда готов (обычно 3–5 минут). Когда наступает горизонт прогноза, "
    "деск сверяет его с рынком и сообщает результат. Материалы носят исследовательский характер "
    "и не являются индивидуальной инвестиционной рекомендацией."
)


def denied(chat_id: int) -> str:
    return f"Доступ закрыт. Передайте администратору ваш chat_id: <code>{chat_id}</code>"


def accepted(run_id: str, as_of: str | None) -> str:
    when = f" на дату {escape(as_of)}" if as_of else ""
    return f"⏳ Анализ{when} запущен. ID: <code>{short(run_id)}</code>\nПришлю отчёт, когда будет готов."


def short(run_id: str) -> str:
    return run_id[:8]


def runs_list(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "Анализов пока нет. Напишите запрос, например: <code>Проанализируй AAPL</code>"
    lines = ["<b>Последние анализы</b>"]
    for r in runs:
        icon = STATUS_ICONS.get(r["status"], "•")
        as_of = f" · {r['as_of_date']}" if r.get("as_of_date") else ""
        lines.append(f"{icon} <code>{short(r['run_id'])}</code> {escape(r['query'][:60])}{as_of}")
    return "\n".join(lines)


def run_status(summary: dict[str, Any]) -> str:
    icon = STATUS_ICONS.get(summary["status"], "•")
    parts = [f"{icon} <b>{escape(summary.get('ticker') or '—')}</b> · <code>{short(summary['run_id'])}</code> · {summary['status']}"]
    if summary.get("as_of_date"):
        parts.append(f"Дата среза: {summary['as_of_date']}")
    parts.append(f"Фактов собрано: {summary.get('evidence_count', 0)}")
    validation = summary.get("validation")
    if summary.get("validation_round"):
        verdict = "пройдена" if validation and validation.get("verdict") == "ok" else "не пройдена"
        parts.append(f"Проверка валидатором: {verdict} (раундов: {summary['validation_round']})")
    if "UNVALIDATED" in summary.get("flags", []):
        parts.append("⚠️ Отчёт помечен UNVALIDATED — числа требуют ручной проверки")
    parts.append(f"Стоимость: ${summary.get('cost_usd', 0):.3f}")
    return "\n".join(parts)


def report_summary(summary: dict[str, Any], markdown: str, predictions: list[dict[str, Any]]) -> str:
    """Карточка готового отчёта: статус, резюме, прогнозы."""
    text = [run_status(summary)]
    resume = section(markdown, "Резюме")
    if resume:
        clean = re.sub(r"\[\d+(?:[,\s–-]+\d+)*\]", "", resume)
        text.append("\n<b>Резюме</b>\n" + escape(clean[:SUMMARY_LIMIT]))
    if predictions:
        text.append("\n<b>Прогнозы</b>\n" + predictions_block(predictions))
    return "\n".join(text)


def predictions_block(predictions: list[dict[str, Any]]) -> str:
    lines = []
    for p in predictions:
        lines.append(
            f"{PREDICTION_ICONS.get(p['status'], '•')} {DIRECTION_ICONS.get(p['direction'], '')} "
            f"<b>{escape(p['ticker'])}</b> {p['direction']} · {p['horizon_days']} дн · уверенность {p['confidence']:.2f} · "
            f"сверка {p['check_at'][:10]}" + _actual(p)
        )
    return "\n".join(lines)


def predictions_list(predictions: list[dict[str, Any]]) -> str:
    if not predictions:
        return "Прогнозов пока нет."
    return "<b>Мои прогнозы</b>\n" + predictions_block(predictions[-20:])


def verified(prediction: dict[str, Any]) -> str:
    outcome = {"hit": "сбылся ✅", "miss": "не сбылся ❌", "expired": "истёк без данных ⌛"}[prediction["status"]]
    return (
        f"🔔 Прогноз {outcome}\n"
        f"<b>{escape(prediction['ticker'])}</b> {prediction['direction']} на {prediction['horizon_days']} дн "
        f"(уверенность {prediction['confidence']:.2f}){_actual(prediction)}\n"
        f"Тезис: {escape(prediction['thesis'][:300])}\n"
        "Деск извлёк из этого урок и учтёт его в следующих анализах."
    )


def metrics(summary: dict[str, Any]) -> str:
    p, r = summary["predictions"], summary["reports"]
    hit = f"{p['hit_rate']:.2f}" if p["hit_rate"] is not None else "н/д"
    brier = f"{p['brier']:.3f}" if p["brier"] is not None else "н/д"
    cost = f"${r['cost_usd_mean']:.3f}" if r["cost_usd_mean"] is not None else "н/д"
    p50 = f"{r['latency_s_p50'] / 60:.1f} мин" if r["latency_s_p50"] is not None else "н/д"
    return (
        "<b>Качество деска</b>\n"
        f"Прогнозов: {p['total']} (ожидают сверки: {p['by_status']['pending']})\n"
        f"Hit rate: {hit} · Brier: {brier}\n"
        f"Отчётов: {r['count']} · средняя стоимость {cost} · медиана времени {p50}"
    )


def run_finished(summary: dict[str, Any]) -> str:
    if summary["status"] == "rejected":
        return f"🚫 Запрос <code>{short(summary['run_id'])}</code> отклонён: он вне компетенции деска (не про анализ рынка или просьба совершить сделку)."
    if summary["status"] == "failed":
        errors = "\n".join(f"• {escape(e[:200])}" for e in summary.get("errors", [])[:3])
        return f"❌ Анализ <code>{short(summary['run_id'])}</code> не удался.\n{errors}"
    return f"⏸ Анализ <code>{short(summary['run_id'])}</code> прерван сбоем; он продолжится автоматически после восстановления сервиса."


def _actual(p: dict[str, Any]) -> str:
    return f" · факт {p['actual_return']:+.1%}" if p.get("actual_return") is not None else ""


def iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
