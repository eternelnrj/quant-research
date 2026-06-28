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

### Correlation network

A graph whose nodes are stocks and whose edges encode the trailing-window return
correlation between them. The raw correlation matrix is shrunk and then sparsified
(see *sparsifier*) before any node feature is computed, because the dense matrix is
mostly estimation noise at 500 names.

### Minimum spanning tree (MST)

The cheapest connected subgraph touching every node, built here on the Mantegna
distance so that strongly correlated stocks sit close together. A parameter-free,
connected backbone of the correlation network — unlike a hard-threshold graph, it
cannot fragment.

### Mantegna distance

The metric `d = sqrt(2(1 - rho))` that turns a correlation `rho` into a distance
(perfectly correlated -> 0, uncorrelated -> sqrt(2), perfectly anti-correlated ->
2). Converts the correlation matrix into something a spanning-tree / distance
algorithm can consume.

### Ledoit-Wolf shrinkage

A well-conditioned covariance estimator that pulls the noisy sample covariance
toward a structured target. Applied before forming correlations so high-dimensional
instability does not propagate into the centralities — the cheapest single defence
against the correlation-instability pitfall.

### Sparsifier

A rule that keeps only the informative edges of a dense correlation graph. The
project uses at least two (a hard threshold on `|rho|` and an MST; PMFG optional)
so a result is not an artefact of one filtering method.

### Eigenvector centrality

A centrality scoring a node highly when it connects to other high-scoring nodes
(co-moving with well-connected stocks). It assumes non-negative edge weights
(Perron-Frobenius), so on a signed correlation graph it is computed on the MST
topology or on `|rho|`, never on distance weights — which would invert the ordering.

### Betweenness centrality

How often a node lies on shortest paths between other nodes — a measure of being a
bridge between otherwise separate clusters. Exact computation is O(n^3); the plan
operates on the MST / largest component and uses `igraph` or an approximation.

### Community detection

Partitioning the network into densely connected groups (Louvain / Leiden,
maximising modularity). Used both as a membership factor and as the input to the
cluster-vs-sector confusion matrix that checks whether the communities are just
GICS sectors in disguise.

### Label alignment

Matching the arbitrary integer labels Louvain/Leiden assign to communities across
consecutive snapshots (greedy Jaccard / Hungarian on membership overlap). Required
before any "did this stock change community" feature, or relabelling noise
masquerades as real migration.

### Lead-lag network

A *directed* graph with an edge i -> j when stock i's market-residualised returns
lead j's at lags of 1-5 days. Edges are tested with a Bartlett standard error,
FDR-controlled, and compared to a shuffled-returns null; in-degree and out-degree
(followers and leaders) become separate factors.

### Bartlett edge test

The approximation that, under a no-lead-lag null, the lag-k cross-correlation of
two near-white residual series has standard error `~ 1/sqrt(T)`, giving each
candidate edge a z-score and p-value. The primary lead-lag engine here, since
`statsmodels` (and thus a reliable Granger test) is not importable in this
environment.

### Shuffled-returns null

A baseline built by circularly shuffling the return series to destroy genuine
lead-lag structure while preserving marginal properties. Surviving edge density is
reported against this null so that "no real edge structure" is a defensible finding
rather than a dead end.

### Text-similarity network (TNIC)

A graph linking firms whose 10-K *Business* (Item 1) descriptions are similar — a
data-driven, Hoberg-Phillips-style alternative to fixed GICS sectors. Built as a
cosine k-nearest-neighbour graph over sentence-transformer embeddings.

### Neighbour-return signal

The average recent return of a firm's text-graph neighbours — an
industry-momentum-style predictor that uses the text-similarity network in place of
a sector label.

### Cross-sectional neutralisation

Removing a feature's mechanical tilt toward known characteristics before testing
it: rank-transform the feature, regress out log market cap, Amihud illiquidity, and
sector, and form the long-short on the residual. If neutralisation destroys the
signal, the "relational" feature was a known exposure in disguise — a valuable
early negative result.

### Spanning test

A regression of a candidate factor's long-short returns on an existing set of
factor returns; a non-zero HAC intercept ("unspanned alpha") means the candidate
adds something the existing set does not. Distinct from neutralisation: this
removes covariance with factor *returns*, neutralisation removes a cross-sectional
*characteristic* tilt.

### Gibbons-Ross-Shanken (GRS) statistic

The standard test of whether adding one or more assets improves the mean-variance
frontier. For a *single* test asset it reduces to the squared t-stat of the
spanning-regression intercept, which is why that intercept t *is* the spanning
statistic; genuine extra power comes only from a joint test across many assets.

### Trials ledger

A pass-or-fail log of every configuration evaluated in the pre-registered grid
(`data/graphs/trials.parquet`). The deflated Sharpe and Bonferroni/BH steps are fed
the total number of trials run, not the number of reported winners — the single
most important guard against fooling yourself in Phase 3.
