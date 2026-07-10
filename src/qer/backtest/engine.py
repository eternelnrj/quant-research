"""Phase 4.1: the causal, walk-forward, T+1 cross-sectional long-short executor.

``Backtest`` is a *pure executor*: at each rebalance it forms dollar-neutral,
unit-gross target weights from that date's oriented signal (restricted to the
point-in-time universe), executes them with an ``exec_lag``-bar delay (T+1 by
default), lets the positions drift with returns between rebalances, and accrues
the daily *simple* portfolio return. It fits nothing -- IS/OOS separation is the
caller's concern (see :mod:`qer.backtest.schedule`).

Timing (matches the harness ``factor_long_short(lag=1)`` convention): a signal
computed from ``close(t)`` is executed at ``close(t + exec_lag)``, so the target
is first *held* on day ``t + 1 + exec_lag`` and earns that day's return onward.
``exec_lag=0`` is the optimistic same-close cheat, kept only as an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qer.backtest.schedule import rebalance_schedule
from qer.backtest.weights import signal_to_weights


@dataclass
class BacktestResult:
    """Outputs of one backtest run."""

    returns: pd.Series          # daily gross portfolio simple return
    weights: pd.DataFrame       # date x ticker: weights HELD during each day (start-of-day)
    equity: pd.Series           # cumulative (1 + returns).cumprod()
    turnover: pd.Series         # per-rebalance two-sided turnover (target vs drifted weights)
    rebalance_dates: pd.DatetimeIndex   # effective (first-held) dates

    def summary(self, periods_per_year: int = 252) -> dict:
        """A few headline stats for a quick sanity check (4.4 expands on these)."""
        r = self.returns.dropna()
        sd = float(r.std(ddof=1))
        mean = float(r.mean())
        return {
            "n_days": int(len(r)),
            "ann_return": mean * periods_per_year,
            "ann_vol": sd * np.sqrt(periods_per_year),
            "sharpe": (mean / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
            "total_return": float(self.equity.iloc[-1] - 1.0) if len(self.equity) else float("nan"),
            "avg_turnover": float(self.turnover.mean()) if len(self.turnover) else float("nan"),
        }


class Backtest:
    """Vectorised, walk-forward, T+1 long-short executor (Phase 4.1)."""

    def __init__(self, *, freq: str = "M", scheme: str = "equal", n_buckets: int = 10,
                 exec_lag: int = 1):
        self.freq = freq
        self.scheme = scheme
        self.n_buckets = n_buckets
        self.exec_lag = int(exec_lag)

    def run(self, loader, factor, *, start=None, end=None) -> BacktestResult:
        cal = pd.DatetimeIndex(loader.close.index).drop_duplicates().sort_values()
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal <= pd.Timestamp(end)]
        K = len(cal)
        pos = {d: k for k, d in enumerate(cal)}

        returns = loader.get_returns("simple").reindex(cal)
        direction = int(getattr(factor, "direction", 1) or 1)
        signal = factor.compute_panel(loader) * direction   # orient: high => long

        # Target weights keyed by the first calendar index at which they are HELD.
        # signal known at close(t) -> traded at close(t + exec_lag) -> first held
        # on day index pos[t] + 1 + exec_lag.
        targets: dict[int, pd.Series] = {}
        for t in rebalance_schedule(cal, freq=self.freq):
            if t not in pos or t not in signal.index:
                continue
            eff = pos[t] + 1 + self.exec_lag
            if eff >= K:
                continue
            universe = set(loader.get_universe(t))
            row = signal.loc[t]
            row = row[[c for c in row.index if c in universe]]
            w = signal_to_weights(row, scheme=self.scheme, n_buckets=self.n_buckets)
            if len(w) and float(w.abs().sum()) > 0:
                targets[eff] = w

        port_ret = pd.Series(0.0, index=cal, dtype=float)
        turnover: dict[pd.Timestamp, float] = {}
        weight_rows: dict[pd.Timestamp, pd.Series] = {}
        current: pd.Series | None = None    # drifted weights held at the start of the day

        for k, d in enumerate(cal):
            if k in targets:                # rebalance takes effect at the start of day k
                new = targets[k]
                if current is not None:
                    names = current.index.union(new.index)
                    prev = current.reindex(names).fillna(0.0)
                    tgt = new.reindex(names).fillna(0.0)
                    turnover[d] = float((tgt - prev).abs().sum())
                else:
                    turnover[d] = float(new.abs().sum())
                current = new.copy()
            if current is None:
                continue                    # flat before the first position
            r_d = returns.loc[d].reindex(current.index).fillna(0.0)
            weight_rows[d] = current                       # start-of-day held weights (fractions of NAV)
            rp = float((current * r_d).sum())              # portfolio return on NAV
            port_ret.loc[d] = rp
            # Buy-and-hold drift to fractions of the *new* NAV: positions grow by
            # (1 + r_i), the book by (1 + rp), so weights scale by their ratio. Without
            # the (1 + rp) divisor the compounded equity double-counts within-period growth.
            growth = 1.0 + rp
            current = current * (1.0 + r_d) / growth if growth != 0.0 else current * (1.0 + r_d)

        weights = (pd.DataFrame(weight_rows).T.reindex(cal).fillna(0.0)
                   if weight_rows else pd.DataFrame(index=cal))
        equity = (1.0 + port_ret.fillna(0.0)).cumprod()
        return BacktestResult(
            returns=port_ret,
            weights=weights,
            equity=equity,
            turnover=pd.Series(turnover, dtype=float).sort_index(),
            rebalance_dates=pd.DatetimeIndex(sorted(cal[k] for k in targets)),
        )
