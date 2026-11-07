"""Продуктовые метрики прогнозов: исход, hit rate, Brier, калибровка по бинам."""

from __future__ import annotations

from dataclasses import dataclass

from reflexdesk.schemas import Direction, Prediction

FLAT_THRESHOLD = 0.03
CALIBRATION_BINS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001)


def realized_direction(actual_return: float, threshold: float = FLAT_THRESHOLD) -> Direction:
    """Фактическое направление: |r| ≤ порога — flat."""
    if abs(actual_return) <= threshold:
        return "flat"
    return "up" if actual_return > 0 else "down"


def is_hit(direction: Direction, actual_return: float, threshold: float = FLAT_THRESHOLD) -> bool:
    return realized_direction(actual_return, threshold) == direction


def verified(predictions: list[Prediction]) -> list[Prediction]:
    return [p for p in predictions if p.status in ("hit", "miss")]


def hit_rate(predictions: list[Prediction]) -> float | None:
    """Доля hit среди сверенных."""
    done = verified(predictions)
    return sum(p.status == "hit" for p in done) / len(done) if done else None


def brier(predictions: list[Prediction]) -> float | None:
    """Brier score: среднее (confidence − 1[hit])²."""
    done = verified(predictions)
    if not done:
        return None
    return sum((p.confidence - (p.status == "hit")) ** 2 for p in done) / len(done)


@dataclass(frozen=True)
class CalibrationBin:
    low: float
    high: float
    count: int
    mean_confidence: float
    hit_rate: float
    brier: float


def calibration(predictions: list[Prediction]) -> list[CalibrationBin]:
    """Калибровка по бинам уверенности."""
    done = verified(predictions)
    bins = []
    for low, high in zip(CALIBRATION_BINS, CALIBRATION_BINS[1:], strict=False):
        members = [p for p in done if low <= p.confidence < high]
        if not members:
            continue
        bins.append(CalibrationBin(
            low=low, high=min(high, 1.0), count=len(members),
            mean_confidence=sum(p.confidence for p in members) / len(members),
            hit_rate=sum(p.status == "hit" for p in members) / len(members),
            brier=brier(members) or 0.0,
        ))
    return bins
