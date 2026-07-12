"""Phase 4.4: performance metrics on a daily (simple) return series.

Pure functions -- no loader, no I/O. Everything annualises by ``sqrt(periods/yr)``
consistently with the factor harness. A dollar-neutral long-short return is already
self-financing (an excess return), so no risk-free rate is subtracted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns) -> pd.Series:
    """Cumulative wealth ``prod(1 + r)`` (starts just above/below 1)."""
    return (1.0 + pd.Series(returns, dtype=float).fillna(0.0)).cumprod()


def total_return(returns) -> float:
    """Compounded total return ``prod(1 + r) - 1``."""
    eq = equity_curve(returns)
    return float(eq.iloc[-1] - 1.0) if len(eq) else float("nan")


def cagr(returns, periods_per_year: int = 252) -> float:
    """Compound annual growth rate implied by the total return over ``n`` periods."""
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)
    if n == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / n) - 1.0


def ann_vol(returns, periods_per_year: int = 252) -> float:
    """Annualised volatility ``std * sqrt(periods/yr)``."""
    sd = float(pd.Series(returns, dtype=float).std(ddof=1))
    return sd * np.sqrt(periods_per_year)


def sharpe(returns, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ``mean/std * sqrt(periods/yr)`` (no risk-free; LS is self-financing)."""
    r = pd.Series(returns, dtype=float).dropna()
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def sortino(returns, target: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sortino using the *target semideviation* (downside squared deviations
    normalised by the full ``N``, hence never above total volatility)."""
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) == 0:
        return float("nan")
    excess = r - target
    downside = np.sqrt(float((np.minimum(excess, 0.0) ** 2).mean()))
    return float(excess.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else float("nan")


def drawdown_series(returns) -> pd.Series:
    """Drawdown path ``equity / running_peak - 1`` (<= 0)."""
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def max_drawdown(returns) -> float:
    """Worst peak-to-trough drawdown, as a positive magnitude."""
    dd = drawdown_series(returns)
    return float(abs(dd.min())) if len(dd) else float("nan")


def calmar(returns, periods_per_year: int = 252) -> float:
    """CAGR divided by the max-drawdown magnitude."""
    mdd = max_drawdown(returns)
    return float(cagr(returns, periods_per_year) / mdd) if mdd > 0 else float("nan")


def conditional_drawdown(returns, alpha: float = 0.05) -> float:
    """Conditional Drawdown at Risk: mean of the worst ``alpha`` drawdowns (magnitude)."""
    dd = drawdown_series(returns).dropna()
    if len(dd) == 0:
        return float("nan")
    threshold = float(dd.quantile(alpha))
    worst = dd[dd <= threshold]
    return float(-worst.mean()) if len(worst) else 0.0


def hit_rate(returns) -> float:
    """Fraction of periods with a positive return."""
    r = pd.Series(returns, dtype=float).dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def profit_factor(returns) -> float:
    """Gross profit over gross loss."""
    r = pd.Series(returns, dtype=float).dropna()
    gain = float(r[r > 0].sum())
    loss = float(-r[r < 0].sum())
    return gain / loss if loss > 0 else float("inf") if gain > 0 else float("nan")


def avg_turnover(turnover) -> float:
    """Average per-rebalance two-sided turnover."""
    t = pd.Series(turnover, dtype=float).dropna()
    return float(t.mean()) if len(t) else float("nan")


def monthly_return_heatmap(returns) -> pd.DataFrame:
    """Year x month table of monthly compounded returns."""
    m = (1.0 + pd.Series(returns, dtype=float).fillna(0.0)).resample("ME").prod() - 1.0
    frame = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.to_numpy()})
    return frame.pivot(index="year", columns="month", values="ret")


def performance_summary(returns, turnover=None, periods_per_year: int = 252) -> dict:
    """The full scalar performance metric set as a dict (used for the report and benchmarks)."""
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "ann_vol": ann_vol(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, 0.0, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
    }
    if turnover is not None:
        out["avg_turnover"] = avg_turnover(turnover)
    return out
