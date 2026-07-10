"""Integration test for the Phase 4.1 causal engine on the synthetic loader.

The load-bearing test is causality: perturbing future signal values must not move
any past portfolio return, and the T+1 execution lag must actually cost the
look-ahead edge a clairvoyant signal would otherwise capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qer.backtest import Backtest
from qer.backtest.schedule import train_test_split
from qer.factors.graph.base import PanelFactor


def _rand_signal(loader, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx, cols = loader.close.index, loader.close.columns
    return pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)


def test_engine_produces_causal_return_series(synthetic_loader):
    f = PanelFactor(_rand_signal(synthetic_loader), name="rand", direction=1)
    res = Backtest(freq="M", scheme="signal", exec_lag=1).run(synthetic_loader, f)
    assert res.returns.index.equals(synthetic_loader.close.index)
    assert np.isfinite(res.returns.to_numpy()).all()
    assert len(res.rebalance_dates) > 0
    # the equity curve compounds the daily portfolio returns
    assert np.isclose(res.equity.iloc[-1], float((1.0 + res.returns).prod()))
    assert set(res.summary()) >= {"sharpe", "ann_vol", "avg_turnover"}


def test_no_look_ahead(synthetic_loader):
    sig = _rand_signal(synthetic_loader, 1)
    base = Backtest(exec_lag=1).run(synthetic_loader, PanelFactor(sig, name="s", direction=1))
    t0 = synthetic_loader.close.index[250]

    sig2 = sig.copy()
    future = sig2.index > t0
    rng = np.random.default_rng(2)
    sig2.loc[future] = rng.normal(size=(int(future.sum()), sig2.shape[1]))
    perturbed = Backtest(exec_lag=1).run(synthetic_loader, PanelFactor(sig2, name="s2", direction=1))

    # perturbing only future signals leaves every past return bit-identical
    assert base.returns.loc[:t0].equals(perturbed.returns.loc[:t0])


def test_exec_lag_zero_is_the_look_ahead_cheat(synthetic_loader):
    # a clairvoyant signal (tomorrow's return) only pays off if you can trade at today's close
    clair = synthetic_loader.get_returns("simple").shift(-1)
    f = PanelFactor(clair, name="clair", direction=1)
    r_cheat = Backtest(scheme="equal", exec_lag=0).run(synthetic_loader, f).returns.mean()
    r_t1 = Backtest(scheme="equal", exec_lag=1).run(synthetic_loader, f).returns.mean()
    assert r_cheat > r_t1                                  # T+1 removes the look-ahead edge


def test_weights_dollar_neutral_and_unit_gross_each_rebalance(synthetic_loader):
    f = PanelFactor(_rand_signal(synthetic_loader, 3), name="s", direction=1)
    res = Backtest(scheme="signal", exec_lag=1).run(synthetic_loader, f)
    w = res.weights.loc[res.rebalance_dates]
    assert w.sum(axis=1).abs().max() < 1e-9                # dollar-neutral
    assert ((w.abs().sum(axis=1) - 1.0).abs() < 1e-9).all()  # unit gross


def test_constant_signal_gives_zero_return(synthetic_loader):
    idx, cols = synthetic_loader.close.index, synthetic_loader.close.columns
    flat = pd.DataFrame(1.0, index=idx, columns=cols)
    res = Backtest(scheme="signal", exec_lag=1).run(
        synthetic_loader, PanelFactor(flat, name="flat", direction=1))
    assert res.returns.abs().max() < 1e-12                 # zero weights -> zero return


def test_is_oos_split_partitions_the_calendar(synthetic_loader):
    cal = synthetic_loader.close.index
    is_, oos = train_test_split(cal, cal[250])
    assert is_.intersection(oos).empty
    assert len(is_) + len(oos) == len(cal)


def test_equity_matches_buy_and_hold_within_a_holding_period(synthetic_loader):
    # Within one holding period the compounded equity must equal the analytic
    # buy-and-hold NAV of the (drifting) target weights -- the drift/compounding
    # accounting regression guard.
    cal = synthetic_loader.close.index
    f = PanelFactor(_rand_signal(synthetic_loader, 5), name="s", direction=1)
    res = Backtest(freq="M", scheme="signal", exec_lag=1).run(synthetic_loader, f)

    t0, t1 = res.rebalance_dates[0], res.rebalance_dates[1]
    base_day = cal[cal.get_loc(t0) - 1]                 # positions established at this close
    assert np.isclose(res.equity.loc[base_day], 1.0)    # flat before the first position

    w0 = res.weights.loc[t0]
    w0 = w0[w0 != 0]
    prices = synthetic_loader.close[w0.index]
    period = cal[(cal >= t0) & (cal < t1)]
    ratio = prices.loc[period] / prices.loc[base_day]
    nav_true = 1.0 + (w0 * (ratio - 1.0)).sum(axis=1)   # analytic buy-and-hold NAV

    assert np.allclose(res.equity.loc[period].to_numpy(), nav_true.to_numpy(), atol=1e-9)
