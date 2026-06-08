# Glossary

Working definitions for the terms and conventions used across this project.

Where a term has a project-specific convention (e.g. how membership intervals

are bounded), that convention is stated explicitly rather than the textbook

generality, because the convention is what the code actually relies on.

### Cross-sectional factor

A signal computed across the universe of stocks *on a single date*, producing

one value per ticker that ranks names relative to each other.

### Long-short equity

A strategy that buys (goes long) names ranked high by the signal and sells

(goes short) names ranked low, ideally cancelling broad market exposure so the

return reflects the signal rather than the market's direction.

### Alpha

Return not explained by known systematic risk factors. 

### 12-1 month momentum

The headline factor: cumulative return from roughly 12 months ago to roughly 1

month ago. In trading days that is a 252-day lookback with a 21-day skip.

### Skip month

The most recent ~21 trading days, deliberately excluded from the momentum

window. Very recent moves tend to mean-revert (short-term reversal), so

including them would contaminate a medium-term momentum signal with noise.

### Short-term reversal

The empirical tendency of stocks that moved sharply over the last few weeks to

move back. It is the contamination the skip month guards against, and a planned

factor in its own right.

### Information coefficient (IC)

The cross-sectional rank correlation (Spearman) between a factor's values on

date *T* and the realised forward return measured from *T*. Computed per date,

then summarised over time. A daily IC of 0.01–0.04 is typical for a real

classical factor.

### IC IR (information ratio)

`mean(IC) / std(IC) * sqrt(252)`, an annualised measure of how *stable* the

signal is rather than how strong it is on any one day. The project treats IR

above ~0.5 as meaningful and below ~0.2 as noise. Undefined when the IC has

zero variance.

### t-stat of IC

`mean(IC) / (std(IC) / sqrt(n))`, testing whether the average IC is

distinguishable from zero. Distinct from the IR: it answers "is this real?"

rather than "is this economically interesting?" Production work uses

Newey-West standard errors here to account for autocorrelation in the IC.

### Hit rate

Fraction of dates on which the daily IC shares the sign of the mean IC — a

crude consistency check on a signal.

### Spearman rank correlation

Correlation of ranks rather than raw values. Used for IC because factor signals

are about *ordering* names, and ranks are robust to the heavy tails and

nonlinearities of return data.

### Forward return

The return realised *after* the signal date, used as the prediction target.

Here it is the log return from close of *T* to close of *T+21* trading days.

Aligning factor(*T*) with forward-return(*T*) is the whole content of the IC.

### Look-ahead bias

Letting information from the future leak into a decision made in the past — the

cardinal sin of backtesting. The momentum function is constructed so the only

data it touches is the slice up to and including the as-of date.

### Survivorship bias

Evaluating a strategy only on names that survived to today, which flatters

results because failures are excluded. The membership table preserves

delisted/acquired names precisely to avoid this; the 5% minimum-churn audit is

a canary for it.

### Point-in-time (PIT)

Data reconstructed *as it would have been known on the historical date*, with no

later corrections folded back in. The universe is PIT: `get_universe(date)`

returns the index constituents as of that date.

### S&P 500 membership table

The project's universe backbone: one row per `(ticker, contiguous membership

interval)` with `start_date` and `end_date`. Built by scraping the current

constituents and the changes log from Wikipedia, then walking events into

intervals.

### Half-open interval convention

Membership is treated as `[start_date, end_date)`: a name is a member *on* its

start date but *not* on its end date. This avoids double-counting on same-day

index swaps (e.g. a ticker removed and re-added the same day yields two clean

intervals meeting at that date).

### Sentinel end date

`2099-12-31`, the placeholder `end_date` meaning "still in the index". Using a

far-future timestamp lets range queries avoid special-casing NaT/None.

### Entity resolution

Collapsing a single economic entity's history under one canonical ticker across

renames (e.g. `FB -> META`, `WLP -> ANTM -> ELV`). Renames are folded together;

acquisitions (the entity ceases to exist) are *not* renames and instead get an

explicit exit date.

### Ticker rename vs. exit

A rename keeps the same entity in the index under a new symbol (use the rename

map, walk the chain). An exit is an acquisition or delisting (use the exits map

to supply the date the changes log failed to record).

### Churn

The fraction of all ever-members no longer in the index today. Used as a

two-sided sanity check: too low signals survivorship bias (today's names leaked

backwards), too high signals over-counted exits or unresolved renames.

### Auto-adjust

A price-ingestion flag controlling whether OHLC are corrected for splits and

dividends. The loader computes returns from an adjusted close so corporate

actions don't appear as spurious jumps. (Note: the loader's price accessor

reads an `adj close` field, a detail worth keeping straight when re-ingesting.)

### Wide matrix

A `date x ticker` DataFrame for one field (close, volume, etc.), built lazily

from the per-ticker parquet files and cached to disk. This is the shape every

downstream computation consumes.

### Log return

`log(P_t / P_{t-1})`. Preferred here because log returns are additive across

time, which makes the momentum window a simple difference of log prices.

### Universe

The set of tradeable names on a given date — concretely, the S&P 500

constituents returned by `get_universe(date)`. Restricting cross-sectional

computations to the universe is what makes the IC meaningful.

### Fama-French 5-factor (FF5)

A standard risk model (market, size, value, profitability, investment) used to

strip known exposures from a signal so the residual alpha can be attributed to

the factor under test.

### Newey-West standard errors

Heteroskedasticity- and autocorrelation-consistent standard errors, needed

because daily IC and factor returns are serially correlated and naive standard

errors would overstate significance.

### Deflated Sharpe ratio

A Sharpe ratio adjusted downward for the number of strategy variants tried,

guarding against the multiple-testing illusion of finding "the" good backtest

among many.

### Graph features

The project's differentiator: relational signals between stocks rather than

asset-level ones — correlation-network centralities, lead-lag directed graphs,

and text-similarity networks from 10-K business descriptions.

### Centrality

A node's importance in a network (degree, eigenvector, betweenness). Computed on

correlation graphs (e.g. minimum spanning trees, k-NN graphs) to capture how

central a stock is in the cross-asset correlation structure.
