# Theory Index (Phases 0-3)

An index of the mathematical, statistical, econometric, and finance-theory
concepts and named results that underpin the code through Phase 2, plus the
planned Phase 3 graph layer (marked as such). It is a *pointer*, not an
explanation: each entry is something to look up to find the relevant theory.
Canonical papers are named where one exists, since that is usually the fastest
route to the underlying result.

See `design_notes.md` for how these are used (and where the code deliberately
approximates them) and `glossary.md` for project conventions.

## Probability and statistics foundations

- Moments: mean, variance, third and fourth central moments
- Standardized moments; skewness (Fisher-Pearson coefficient; sample / bias-corrected G1); excess kurtosis
- Covariance and correlation; Pearson correlation; Spearman rank correlation
- Variance of a sum / difference of random variables
- Sample estimators and the degrees-of-freedom (Bessel's) correction
- Central Limit Theorem; asymptotic normality of estimators
- Law of Large Numbers
- Order statistics; expected value of the maximum of N draws; extreme-value theory; Gumbel distribution; Euler-Mascheroni constant
- Normal distribution: CDF (Phi), quantile / inverse-CDF (probit), survival function; relation to the error function (erf / erfc)
- Student's t-distribution; degrees of freedom

## Time series and robust inference

- Autocorrelation / serial correlation
- Overlapping observations and the effective (independent) sample size
- Heteroskedasticity- and autocorrelation-consistent (HAC) covariance
- Newey-West estimator (1987); Bartlett kernel / weights; bandwidth (lag) selection
- White heteroskedasticity-robust standard errors (1980)
- Fixed-b asymptotics for HAC inference (Kiefer-Vogelsang)
- Rolling / moving-window estimators

## Hypothesis testing and multiple comparisons

- t-statistic; two-sided p-value from a test statistic
- Single-sample mean test vs regression-coefficient test (degrees of freedom: n-1 vs n-k)
- Family-wise error rate; Bonferroni correction
- False discovery rate; Benjamini-Hochberg procedure (1995)
- Multiple-testing / "factor zoo" problem in asset pricing (Harvey-Liu-Zhu 2016; Harvey-Liu)
- Deflated Sharpe Ratio and Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2014); variance of the Sharpe-ratio estimator; selection bias under multiple trials

## Linear regression and econometrics

- Ordinary least squares; normal equations; (X'X)^-1 X'y; solution via QR / SVD (lstsq)
- Residuals, projection, and the role of the intercept
- Gauss-Markov theorem (OLS as BLUE)
- Frisch-Waugh-Lovell theorem (partialling-out; the intercept after removing factors)
- Sandwich covariance estimator (bread / meat form): (X'X)^-1 S (X'X)^-1
- Single-index (market-model) regression; rolling beta as cov / var

## Asset-pricing models and factor theory

- CAPM / single-index market model; systematic vs idiosyncratic risk
- Jensen's alpha
- Fama-French three-factor (1993) and five-factor (2015) models; factor loadings (betas); FF5-adjusted alpha
- Risk premium; total return vs price return (dividend reinvestment)

## The factors (originating literature)

- Cross-sectional momentum, 12-1 with skip-month (Jegadeesh-Titman 1993)
- Short-term reversal (Jegadeesh 1990; Lehmann 1990)
- Idiosyncratic volatility / low-volatility anomaly (Ang-Hodrick-Xing-Zhang 2006)
- Idiosyncratic skewness and its pricing (Kumar 2009; Boyer-Mitton-Vorkink 2010; Bali-Cakici-Whitelaw 2011, MAX)
- Amihud illiquidity measure (Amihud 2002)
- Size effect (Banz 1981)
- Value / book-to-market (Fama-French 1992)
- Gross profitability / quality (Novy-Marx 2013)
- Cross-sectional ranking; percentile / quantile sorts; decile long-short spreads
- Information Coefficient (rank IC) and Information Ratio; annualization by sqrt(periods)
- Portfolio turnover; Sharpe ratio (per-observation and annualized)

## Universe, data, and point-in-time correctness

- Survivorship bias; point-in-time data; look-ahead bias
- Index reconstruction as interval / set algebra over membership events
- Entity resolution / ticker-rename chains as transitive closure (graph reachability; connected components)
- Log (continuously compounded) vs simple returns; additivity of log returns
- Split- and dividend-adjustment factors; total-return series
- As-of / forward-fill joins for irregularly sampled point-in-time fundamentals; reporting lag

## Numerical methods

- Solving least squares via QR / SVD rather than an explicit matrix inverse
- Inverse-normal-CDF evaluation: rational approximations (e.g. Acklam) vs library quantile functions
- Numerical stability of variance / skew computations over rolling windows

## Graph-based features (Phase 3, planned)

The entries below underpin the planned graph layer; the code does not yet exist,
so they index the *plan*, not a shipped implementation.

Correlation networks and filtering
- Sample correlation/covariance conditioning in high dimension; eigenvalue spread
- Ledoit-Wolf linear shrinkage of the covariance matrix (2004); structured target
- Random-matrix-theory view of noisy correlation eigenvalues (Marchenko-Pastur)
- Network filtering: minimum spanning tree; Mantegna distance metric `sqrt(2(1-rho))` (Mantegna 1999)
- Planar maximally filtered graph (PMFG; Tumminello-Aste-Di Matteo-Mantegna 2005)
- Correlation / MST dynamics over time (Onnela et al. 2003)

Graph centrality and community structure
- Degree, weighted degree, clustering coefficient
- Eigenvector centrality; Perron-Frobenius non-negativity requirement; signed-network (PN) centrality
- Betweenness centrality; Brandes algorithm; O(n^3) cost and approximation on the largest component
- Modularity maximisation; Louvain and Leiden community detection
- Label / permutation alignment across snapshots (Hungarian assignment; Jaccard overlap)

Lead-lag and connectedness
- Lagged cross-correlation; Bartlett standard error `~ 1/sqrt(T)` for cross-correlations of white series
- Granger causality (gated on `statsmodels`; off the critical path here)
- Econometric connectedness networks (Billio-Getmansky-Lo-Pelizzon 2012)
- Directed connectedness from variance decompositions (Diebold-Yilmaz 2014)
- Circular-shuffle / block permutation nulls for edge-density significance

Text-similarity networks
- Sentence / document embeddings from transformer encoders; cosine similarity
- k-nearest-neighbour similarity graphs
- Text-based network industries, TNIC (Hoberg-Phillips 2016)
- Supply-chain / economic-link return predictability (Cohen-Frazzini 2008) - documented fallback

Incremental value and spanning
- Spanning regression; unspanned alpha as the HAC intercept
- Gibbons-Ross-Shanken test (1989); GRS = t^2 for a single test asset; joint multi-asset spanning
- Appraisal ratio (Treynor-Black); ex-post max-Sharpe gain = squared appraisal ratio `alpha^2 / sigma^2(eps)`
- Block bootstrap for HAC-robust inference on one fat-tailed daily series (Kunsch 1989)
- Cross-sectional neutralisation as partialling-out (Frisch-Waugh-Lovell, above) of characteristics

## Caveats

A few entries name the *correct* theory behind a design choice or a fix even
where the final code takes a deliberate shortcut, so reading the reference will
describe something slightly different from what the code computes:

- The idiosyncratic-skewness factor uses a rolling per-date beta rather than the
  single-window market regression the Kumar / Bali / Boyer-Mitton-Vorkink
  references define (an approximation; see `design_notes.md`).
- The deflated Sharpe uses a heuristic effective sample size (`n / horizon`)
  rather than formal fixed-b / autocorrelation-based HAC sample-size theory.
- HAC standard errors are the asymptotic (large-sample) object; the fixed-b
  literature describes the finite-sample reference distribution they approximate.
- The planned Phase 3 lead-lag network indexes Granger causality as the canonical
  result, but the implementation substitutes a numpy lagged cross-correlation with
  a Bartlett edge test (Granger is gated on `statsmodels` becoming importable), so
  the reference again describes something stronger than the code will compute.

The canonical references are listed regardless, since they are what you would
read to understand what the code is approximating.
