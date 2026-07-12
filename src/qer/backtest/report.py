"""Phase 4.5: assemble a standardised, deterministic backtest report.

``compute_report_data`` runs the full 4.1--4.4 pipeline (backtest -> costs -> metrics ->
risk -> cost curve -> capacity -> deflated Sharpe) and returns a :class:`ReportData` whose
scalar tables are fully deterministic (two runs of the same config give a byte-identical
metrics table). ``build_report`` renders it to a self-contained HTML file (embedded charts)
or a matplotlib PDF. SPY/FF5/deflated-Sharpe sections are optional and skipped cleanly when
their data is absent, so the report runs on the synthetic loader with no network or extras.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qer.backtest.costs import CostModel, apply_costs, capacity_report  # noqa: E402
from qer.backtest.engine import Backtest  # noqa: E402
from qer.backtest.metrics import (  # noqa: E402
    drawdown_series,
    equity_curve,
    monthly_return_heatmap,
    performance_summary,
    sharpe,
)
from qer.backtest.risk import (  # noqa: E402
    benchmark_stats,
    ff5_exposures,
    realised_beta,
    rolling_sharpe,
)
from qer.backtest.schedule import train_test_split  # noqa: E402
from qer.config import FF5_FILE  # noqa: E402
from qer.diagnostics.deflated_sharpe import deflated_sharpe  # noqa: E402


@dataclass
class ReportData:
    config: dict
    headline: str
    performance_net: dict
    performance_gross: dict
    benchmark: pd.DataFrame | None
    cost_totals: dict
    cost_curve: pd.DataFrame
    capacity: pd.DataFrame
    ff5: dict | None
    realised_beta: float | None
    deflated_sharpe: float | None
    equity_gross: pd.Series
    equity_net: pd.Series
    drawdown: pd.Series
    monthly: pd.DataFrame
    rolling_sharpe: pd.Series

    def metrics_table(self) -> pd.DataFrame:
        """The deterministic performance table (used for the reproducibility test)."""
        return pd.DataFrame({"net": self.performance_net, "gross": self.performance_gross}).T


def compute_report_data(loader, factor, *, freq="M", scheme="equal", weigher=None,
                        cost_model=None, oos_split=None, n_trials=None,
                        periods_per_year: int = 252) -> ReportData:
    cost_model = cost_model or CostModel()
    result = Backtest(freq=freq, scheme=scheme, weigher=weigher).run(loader, factor)
    costed = apply_costs(result, loader, cost_model)
    net, gross = costed.net_returns, costed.gross_returns

    if oos_split is not None:
        _, oos = train_test_split(net.index, oos_split)
        net_eval, gross_eval = net.loc[oos], gross.loc[oos]
        eval_label = f"OOS (>= {pd.Timestamp(oos_split).date()})"
    else:
        net_eval, gross_eval, eval_label = net, gross, "full sample"

    perf_net = performance_summary(net_eval, turnover=result.turnover,
                                   periods_per_year=periods_per_year)
    perf_gross = performance_summary(gross_eval, periods_per_year=periods_per_year)

    benchmark = beta = ff5 = None
    try:
        mkt = loader.market_return
        benchmark = benchmark_stats(net_eval, market_return=mkt, periods_per_year=periods_per_year)
        beta = realised_beta(net_eval, mkt)
    except Exception:                          # noqa: BLE001 -- SPY optional; skip section
        pass
    if FF5_FILE.exists():
        try:
            ff5 = ff5_exposures(net_eval, pd.read_parquet(FF5_FILE))
        except Exception:                      # noqa: BLE001 -- FF5 optional; skip section
            pass

    # Sharpe-vs-cost over the SAME evaluation period as the headline, varying the assumed
    # round-trip spread on top of the model's impact/borrow -- so the curve at the assumed
    # spread is the headline net Sharpe. Reuses the gross result; no second backtest.
    eval_idx = net_eval.index
    grid = sorted({0.0, 5.0, 10.0, 15.0, 20.0, 25.0, float(cost_model.spread_bps)})
    curve_rows = []
    for bps in grid:
        ne = apply_costs(result, loader, replace(cost_model, spread_bps=bps)).net_returns.reindex(eval_idx)
        curve_rows.append({"bps": bps, "net_sharpe": sharpe(ne, periods_per_year),
                           "mean_net_return": float(ne.mean())})
    curve = pd.DataFrame(curve_rows).set_index("bps")
    capacity = capacity_report(result.weights, loader, cost_model.aum)

    dsr = None
    r = net_eval.dropna()
    sd = float(r.std(ddof=1)) if len(r) > 2 else 0.0
    if n_trials and sd > 0:
        dsr = deflated_sharpe(float(r.mean() / sd), n_trials=int(n_trials), n_obs=len(r),
                              skew=float(r.skew()), kurtosis=float(r.kurtosis() + 3.0))

    cal = net.index
    config = {
        "factor": getattr(factor, "name", "factor"), "scheme": scheme, "freq": str(freq),
        "spread_bps": cost_model.spread_bps, "impact_coef": cost_model.impact_coef,
        "borrow_bps": cost_model.borrow_bps, "aum": cost_model.aum,
        "data_start": str(cal.min().date()), "data_end": str(cal.max().date()),
        "universe_size": int((result.weights != 0).any(axis=0).sum()),
        "n_rebalances": len(result.rebalance_dates), "evaluated_on": eval_label,
    }
    headline = (
        f"Turnover {perf_net.get('avg_turnover', float('nan')):.1%} per rebalance; "
        f"assumed round-trip cost {cost_model.spread_bps:.0f} bps; "
        f"net Sharpe {perf_net['sharpe']:.2f} ({eval_label})."
    )
    return ReportData(
        config=config, headline=headline, performance_net=perf_net, performance_gross=perf_gross,
        benchmark=benchmark, cost_totals={"linear": float(costed.linear.sum()),
                                          "impact": float(costed.impact.sum()),
                                          "borrow": float(costed.borrow.sum())},
        cost_curve=curve, capacity=capacity, ff5=ff5, realised_beta=beta, deflated_sharpe=dsr,
        equity_gross=equity_curve(gross), equity_net=equity_curve(net),
        drawdown=drawdown_series(net), monthly=monthly_return_heatmap(net),
        rolling_sharpe=rolling_sharpe(net),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _charts(data: ReportData) -> dict:
    charts = {}

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(data.equity_gross.index, data.equity_gross.to_numpy(), label="gross", lw=1)
    ax.plot(data.equity_net.index, data.equity_net.to_numpy(), label="net", lw=1)
    ax.set_title("Equity curve (gross vs net)")
    ax.legend()
    ax.grid(alpha=0.3)
    charts["equity"] = _b64(fig)

    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.fill_between(data.drawdown.index, data.drawdown.to_numpy(), 0, color="#8A3B2E", alpha=0.5)
    ax.set_title("Drawdown")
    ax.grid(alpha=0.3)
    charts["drawdown"] = _b64(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(data.cost_curve.index, data.cost_curve["net_sharpe"].to_numpy(), marker="o")
    ax.set_title("Sharpe vs assumed round-trip cost")
    ax.set_xlabel("bps")
    ax.set_ylabel("net Sharpe")
    ax.grid(alpha=0.3)
    charts["cost_curve"] = _b64(fig)

    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.plot(data.rolling_sharpe.index, data.rolling_sharpe.to_numpy())
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Rolling 252-day Sharpe")
    ax.grid(alpha=0.3)
    charts["rolling_sharpe"] = _b64(fig)
    return charts


def render_html(data: ReportData) -> str:
    charts = _charts(data)
    cfg = data.config

    def table(obj) -> str:
        df = obj.to_frame("value") if isinstance(obj, pd.Series) else pd.DataFrame(obj)
        return df.to_html(float_format=lambda v: f"{v:.4f}", border=0)

    sections = [
        f"<h1>Backtest report: {cfg['factor']}</h1>",
        f"<p class='headline'>{data.headline}</p>",
        "<h2>Configuration</h2>" + table(pd.Series(cfg)),
        "<h2>Performance (net vs gross)</h2>" + data.metrics_table().to_html(
            float_format=lambda v: f"{v:.4f}", border=0),
        f"<h2>Equity &amp; drawdown</h2><img src='data:image/png;base64,{charts['equity']}'>"
        f"<img src='data:image/png;base64,{charts['drawdown']}'>",
    ]
    if not data.monthly.empty:
        heat = data.monthly.copy()
        heat.columns = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in heat.columns]
        sections.append("<h2>Monthly returns</h2>" + heat.to_html(
            float_format=lambda v: f"{v:.2%}", border=0, na_rep=""))
    sections += [
        f"<h2>Costs: Sharpe vs assumed cost (centrepiece)</h2>"
        f"<img src='data:image/png;base64,{charts['cost_curve']}'>"
        + "<p>Cost totals (fraction of AUM): " + ", ".join(
            f"{k} {v:.4f}" for k, v in data.cost_totals.items()) + "</p>",
        f"<h2>Rolling Sharpe</h2><img src='data:image/png;base64,{charts['rolling_sharpe']}'>",
    ]
    if data.benchmark is not None:
        sections.append("<h2>Benchmark comparison</h2>" + data.benchmark.to_html(
            float_format=lambda v: f"{v:.4f}", border=0))
    if data.realised_beta is not None:
        sections.append(f"<p>Realised beta to SPY: {data.realised_beta:.3f}</p>")
    if data.ff5 is not None:
        sections.append(f"<h2>FF5 exposures</h2><pre>{data.ff5}</pre>")
    sections.append("<h2>Capacity (position as % of 21-day ADV)</h2>"
                    + data.capacity.head(15).to_html(float_format=lambda v: f"{v:.3f}", border=0))
    if data.deflated_sharpe is not None:
        sections.append(f"<p>Deflated Sharpe (vs {cfg['n_rebalances']} rebalances of trials): "
                        f"{data.deflated_sharpe:.4f}</p>")

    style = ("<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#1F3A5F;max-width:900px}"
             "h1{font-size:1.5rem}h2{font-size:1.1rem;border-bottom:1px solid #C9D3DD;margin-top:1.5rem}"
             ".headline{background:#F2F4F7;padding:0.6rem;border-radius:4px;font-weight:600}"
             "table{border-collapse:collapse;font-size:0.85rem}td,th{padding:2px 10px;text-align:right}"
             "img{max-width:100%;margin:4px 0}</style>")
    return f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body>" \
           + "\n".join(sections) + "</body></html>"


def build_report(loader, factor, *, out_dir, fmt: str = "html", **kwargs):
    """Run the full backtest and write a standardised report; returns ``(path, ReportData)``."""
    data = compute_report_data(loader, factor, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = data.config["factor"]
    data.metrics_table().to_csv(out / f"backtest_{name}_metrics.csv")   # deterministic artefact

    if fmt == "pdf":
        path = out / f"backtest_{name}.pdf"
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].plot(data.equity_gross.index, data.equity_gross.to_numpy(), label="gross")
        axes[0, 0].plot(data.equity_net.index, data.equity_net.to_numpy(), label="net")
        axes[0, 0].set_title("Equity")
        axes[0, 0].legend()
        axes[0, 1].fill_between(data.drawdown.index, data.drawdown.to_numpy(), 0, alpha=0.5)
        axes[0, 1].set_title("Drawdown")
        axes[1, 0].plot(data.cost_curve.index, data.cost_curve["net_sharpe"].to_numpy(), marker="o")
        axes[1, 0].set_title("Sharpe vs cost (bps)")
        axes[1, 1].plot(data.rolling_sharpe.index, data.rolling_sharpe.to_numpy())
        axes[1, 1].set_title("Rolling Sharpe")
        fig.suptitle(f"{name}: {data.headline}", fontsize=9)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    else:
        path = out / f"backtest_{name}.html"
        path.write_text(render_html(data))
    return path, data
