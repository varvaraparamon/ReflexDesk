"""Ядро бэктеста для исполнения в sandbox: только numpy/pandas/math, без импорта reflexdesk."""

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SIGN = {"up": 1.0, "down": -1.0, "flat": 0.0}


def backtest(signals, prices, benchmark):
    """signals: [{ticker, direction, start, end}], prices: {ticker: {date: close}}, benchmark: {date: close}."""
    frame = pd.DataFrame({t: pd.Series(p) for t, p in prices.items()})
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().ffill()
    returns = frame.pct_change().fillna(0.0)
    exposure = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    hits = []
    for s in signals:
        window = (returns.index > pd.Timestamp(s["start"])) & (returns.index <= pd.Timestamp(s["end"]))
        exposure.loc[window, s["ticker"]] += SIGN[s["direction"]]
        series = frame[s["ticker"]].loc[pd.Timestamp(s["start"]):pd.Timestamp(s["end"])].dropna()
        if len(series) > 1:
            realized = series.iloc[-1] / series.iloc[0] - 1
            actual = "flat" if abs(realized) <= 0.03 else ("up" if realized > 0 else "down")
            hits.append(actual == s["direction"])
    gross = exposure.abs().sum(axis=1).replace(0.0, np.nan)
    strategy = (exposure * returns).sum(axis=1).div(gross).fillna(0.0)
    bench = pd.Series(benchmark)
    bench.index = pd.to_datetime(bench.index)
    bench_ret = bench.sort_index().pct_change().reindex(strategy.index).fillna(0.0)
    return {
        "strategy": stats(strategy),
        "benchmark": stats(bench_ret),
        "hit_rate": float(np.mean(hits)) if hits else None,
        "signals": len(signals),
        "equity_curve": {
            d.date().isoformat(): [round(float(a), 6), round(float(b), 6)]
            for d, a, b in zip(strategy.index, (1 + strategy).cumprod(), (1 + bench_ret).cumprod(), strict=True)
        },
    }


def stats(daily):
    """Sharpe (годовой), max drawdown, суммарная доходность."""
    equity = (1 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1
    std = daily.std()
    sharpe = float(daily.mean() / std * math.sqrt(TRADING_DAYS)) if std and std > 0 else 0.0
    return {"sharpe": round(sharpe, 4), "max_dd": round(float(drawdown.min()), 6), "total_return": round(float(equity.iloc[-1] - 1), 6)}
