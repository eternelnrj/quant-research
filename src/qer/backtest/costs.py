"""Phase 4.2: transaction costs, short borrow, and capacity.

Turns the gross return series from :mod:`qer.backtest.engine` into a *net* one under
a documented, explicit cost model, and reports capacity as a fraction of ADV. The
three pitfalls this closes:

* **Linear costs in everything** -- costs are a linear spread charge on turnover
  *plus* a convex market-impact term (per-trade cost ``coef * sqrt(participation)``,
  Almgren-Chriss square-root law), so the total impact drag scales as ``size^{3/2}``
  and dominates at scale.
* **Free-money shorts** -- daily borrow carry on the short book, and a hard-to-borrow
  proxy (:func:`htb_mask`) whose names are excluded from the short leg
  (:func:`exclude_htb_shorts`).
* **Capacity ignored** -- :func:`capacity_report` reports each name's position as a
  fraction of its 21-day ADV.

Everything is numpy/pandas and reuses the 4.1 weight/trade path and
``loader.{close, volume, dollar_volume, market_cap}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Explicit, documented cost assumptions -- every reported number traces back here."""

    spread_bps: float = 8.0        # one-way linear cost (half-spread + commission), bps of notional
    impact_coef: float = 0.1       # Almgren-Chriss square-root coefficient (calibratable)
    borrow_bps: float = 75.0       # annualised borrow cost on the short book, bps
    aum: float = 5e8               # capital deployed ($): sets impact participation and capacity


# ---------------------------------------------------------------------------
# Cost primitives
# ---------------------------------------------------------------------------

def turnover(w_prev, w_target) -> float:
    """Two-sided turnover ``sum_i |w_target_i - w_prev_i|`` (charges both legs)."""
    a = pd.Series(w_prev, dtype=float)
    b = pd.Series(w_target, dtype=float)
    names = a.index.union(b.index)
    return float((b.reindex(names).fillna(0.0) - a.reindex(names).fillna(0.0)).abs().sum())


def linear_cost(turnover_value: float, spread_bps: float) -> float:
    """Linear spread/commission cost = ``c * turnover`` as a fraction of AUM."""
    return float(turnover_value) * spread_bps * 1e-4


def impact_cost(trade_notional, adv, coef: float):
    """Per-trade market impact, as a *fraction of the traded notional*.

    ``coef * sqrt(trade_notional / ADV)`` (participation rate; Almgren-Chriss). Because
    this per-unit cost grows with size, the total drag (notional times this) is convex,
    ``~ size^{3/2}``. Vectorises over Series/arrays; a non-positive ADV yields 0 (an
    untradeable name contributes no modelled impact rather than an infinity).
    """
    tn = np.asarray(trade_notional, dtype=float)
    a = np.asarray(adv, dtype=float)
    participation = np.divide(tn, a, out=np.zeros_like(tn), where=a > 0)
    return coef * np.sqrt(np.clip(participation, 0.0, None))


def borrow_cost(weights, borrow_bps: float) -> float:
    """Daily borrow carry on the short book = ``short_gross * borrow_bps / 1e4 / 252``."""
    w = pd.Series(weights, dtype=float)
    short_gross = float(w[w < 0].abs().sum())
    return short_gross * (borrow_bps * 1e-4) / 252.0


# ---------------------------------------------------------------------------
# Liquidity: ADV, hard-to-borrow, capacity
# ---------------------------------------------------------------------------

def adv(loader, window: int = 21) -> pd.DataFrame:
    """Trailing ``window``-day average dollar volume (date x ticker)."""
    return loader.dollar_volume.rolling(window, min_periods=1).mean()


def htb_mask(loader, quantile: float = 0.2, window: int = 21) -> pd.DataFrame:
    """Hard-to-borrow proxy: names that are *both* small-cap and illiquid.

    True where market cap and ADV are both in the bottom ``quantile`` of the
    cross-section (short interest is not in the dataset, so small+illiquid is the
    proxy). Such names are excluded from the short leg by :func:`exclude_htb_shorts`.
    """
    mcap = loader.market_cap
    advol = adv(loader, window).reindex_like(mcap)
    small = mcap.le(mcap.quantile(quantile, axis=1), axis=0)
    illiquid = advol.le(advol.quantile(quantile, axis=1), axis=0)
    return small & illiquid


def exclude_htb_shorts(weights: pd.DataFrame, htb: pd.DataFrame) -> pd.DataFrame:
    """Zero the *short* weights of hard-to-borrow names; never touch longs.

    Leaves a small net imbalance (fewer shorts) that Phase 4.3's constraints
    re-neutralise; here it just reflects that some names cannot be shorted. Names or
    dates absent from ``htb`` are treated as borrowable.
    """
    mask = htb.reindex(index=weights.index, columns=weights.columns).fillna(False).astype(bool)
    blocked = mask & (weights < 0)          # aligned bool frame: hard-to-borrow AND short
    return weights.mask(blocked, 0.0)


def capacity_report(weight_path: pd.DataFrame, loader, aum: float,
                    window: int = 21) -> pd.DataFrame:
    """Per-name position as a fraction of 21-day ADV (the capacity guardrail).

    Returns a frame indexed by ticker with ``max_pos_pct_adv`` and ``mean_pos_pct_adv``
    (position notional ``|w| * aum`` over ADV), sorted worst-first so the capacity-
    binding names are on top.
    """
    advol = adv(loader, window).reindex(index=weight_path.index, columns=weight_path.columns)
    pos_notional = weight_path.abs() * aum
    pos_pct_adv = pos_notional.divide(advol.where(advol > 0))
    report = pd.DataFrame({
        "max_pos_pct_adv": pos_pct_adv.max(axis=0),
        "mean_pos_pct_adv": pos_pct_adv.where(pos_pct_adv > 0).mean(axis=0),
    })
    return report[report["max_pos_pct_adv"] > 0].sort_values("max_pos_pct_adv", ascending=False)


# ---------------------------------------------------------------------------
# Apply the model to a backtest
# ---------------------------------------------------------------------------

@dataclass
class CostedResult:
    """Net returns plus the per-day cost breakdown (for the 4.4 gross-vs-net chart)."""

    net_returns: pd.Series
    gross_returns: pd.Series
    linear: pd.Series
    impact: pd.Series
    borrow: pd.Series

    def summary(self, periods_per_year: int = 252) -> dict:
        g, n = self.gross_returns.dropna(), self.net_returns.dropna()

        def sharpe(r):
            sd = float(r.std(ddof=1))
            return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")

        return {
            "gross_sharpe": sharpe(g),
            "net_sharpe": sharpe(n),
            "ann_cost": float((self.linear + self.impact + self.borrow).sum())
            / max(len(n) / periods_per_year, 1e-9),
        }


def apply_costs(result, loader, cost_model: CostModel) -> CostedResult:
    """Net the gross returns of a :class:`~qer.backtest.engine.BacktestResult`.

    Charges, per day: linear spread on that day's turnover, convex per-name impact on
    the trades, and daily borrow carry on the short book. Trade costs land on the
    rebalance (effective) dates; borrow accrues every day a short is held.

    Impact and capacity use a *fixed nominal* ``cost_model.aum`` for the trade notional
    (the standard "at \\$X AUM" capacity view), independent of the compounding equity
    curve. Linear and borrow are turnover/short-gross fractions and so are already
    NAV-consistent with the gross return.
    """
    gross = result.returns
    trades = result.trades
    advol = adv(loader).reindex(index=trades.index, columns=trades.columns)

    # linear: c * turnover on each day trades occur
    linear = trades.abs().sum(axis=1) * (cost_model.spread_bps * 1e-4)

    # impact: sum_i |dw_i| * coef * sqrt(|dw_i| * aum / ADV_i)   (convex, per name)
    trade_notional = trades.abs() * cost_model.aum
    impact_frac = cost_model.impact_coef * np.sqrt(
        trade_notional.divide(advol.where(advol > 0)).clip(lower=0).fillna(0.0)
    )
    impact = (trades.abs() * impact_frac).sum(axis=1)

    # borrow: daily carry on the short book, every day
    short_gross = result.weights.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_gross * (cost_model.borrow_bps * 1e-4) / 252.0

    linear = linear.reindex(gross.index).fillna(0.0)
    impact = impact.reindex(gross.index).fillna(0.0)
    borrow = borrow.reindex(gross.index).fillna(0.0)
    net = gross - linear - impact - borrow
    return CostedResult(net_returns=net, gross_returns=gross,
                        linear=linear, impact=impact, borrow=borrow)
