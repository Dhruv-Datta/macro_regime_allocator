# Macro Regime Allocator

A tactical equity/T-bills allocation system that uses macroeconomic regime classification to dynamically shift portfolio weights between equities (SPY) and the risk-free rate (fed funds). The core thesis: macro conditions are persistent and classifiable, and a disciplined model that recognizes when the environment favors cash over stocks can deliver equity-like returns with substantially less drawdown.

## Table of Contents

- [How It Works](#how-it-works)
  - [Data Pipeline](#data-pipeline)
  - [Feature Engineering](#feature-engineering)
  - [The Model](#the-model)
  - [Allocation Pipeline](#allocation-pipeline)
  - [Crash Overlay](#crash-overlay)
- [Why Logistic Regression](#why-logistic-regression)
- [Backtest Methodology](#backtest-methodology)
  - [Walk-Forward Design](#walk-forward-design)
  - [Lookahead Prevention](#lookahead-prevention)
  - [Sample Weighting](#sample-weighting)
  - [What Makes This Backtest Honest](#what-makes-this-backtest-honest)
- [Results](#results)
- [Why This Could Work in Real Life](#why-this-could-work-in-real-life)
  - [The Macro Persistence Argument](#the-macro-persistence-argument)
  - [Structural Advantages](#structural-advantages)
  - [What Could Go Wrong](#what-could-go-wrong)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

---

## How It Works

The system answers one question each month: **will equities outperform T-bills over the next month?** It frames this as a binary classification problem, trains a logistic regression on lagged macroeconomic features, and maps the output probability to a portfolio weight between equities and cash.

### Data Pipeline

All data is sourced from two public APIs:

- **Yahoo Finance**: SPY (equity proxy), VIX (implied volatility), VIX3M (3-month VIX)
- **FRED (Federal Reserve Economic Data)**: CPI, Core CPI, unemployment rate, 10-year and 2-year Treasury yields, fed funds rate, high-yield credit spreads (ICE BofA), industrial production

Revisable macro series (CPI, unemployment, industrial production) are downloaded using FRED's **first-release** endpoint via ALFRED. This means the model trains on the data that was actually available at the time, not the revised figures published months later. This is a critical distinction — most backtests inadvertently use revised data, which introduces a subtle but real lookahead bias.

All series are resampled to monthly frequency and forward-filled to handle publication lag differences.

### Feature Engineering

The model uses 12 engineered features spanning four macro domains:

**Inflation (2 features)**
- `inflation_yoy` — Year-over-year CPI change. Captures the level of inflationary pressure. High inflation historically erodes equity valuations and favors cash.
- `inflation_impulse` — The difference between annualized 3-month CPI momentum and YoY CPI. Captures inflation acceleration. A positive impulse means inflation is worsening faster than the trailing average suggests — historically the worst environment for equities.

**Labor & Credit (3 features)**
- `unemployment_rate` — Current unemployment level. Low unemployment often coincides with late-cycle dynamics and tighter monetary policy.
- `credit_spread_level` — ICE BofA High Yield Option-Adjusted Spread. The market's real-time pricing of default risk. Wide spreads signal stress; tight spreads signal complacency or genuine strength.
- `credit_spread_3m_change` — 3-month change in HY OAS. Captures the direction of credit conditions. Widening spreads are often the first signal that equity risk is mispriced.

**Rates (2 features)**
- `real_fed_funds` — Fed funds rate minus trailing 12-month inflation. Measures the actual restrictiveness of monetary policy. Negative real rates are historically bullish for equities.
- `yield_curve_slope` — 10-year minus 2-year Treasury yield. The classic recession indicator. Inversions (negative slope) have preceded every US recession since the 1960s.

**Volatility (2 features)**
- `vix_1m_change` — One-month change in VIX. Captures volatility shocks. Sudden VIX spikes accompany rapid equity declines.
- `vix_term_structure` — VIX / VIX3M ratio. When >1 (backwardation), the market prices near-term risk higher than medium-term — a sign of active panic, not just elevated worry.

**Equity Market (3 features)**
- `equity_momentum_3m` — 3-month SPY return. Short-term momentum. Interestingly, the model learns this with a *positive* coefficient (favoring T-bills when recent equity returns are strong), suggesting it captures mean-reversion or late-rally exhaustion signals.
- `equity_vol_3m` — 3-month rolling realized volatility, annualized. The strongest equity-favoring feature — low realized volatility is the single best predictor of continued equity outperformance.
- `equity_drawdown_from_high` — Current distance from the 12-month rolling high. Captures whether the market is in a drawdown.

**All features are lagged by 1 month** to prevent lookahead bias. The model at time *t* only sees data available through *t-1*.

### The Model

**Algorithm**: Logistic Regression (scikit-learn, L2 regularization, LBFGS solver)

The classifier predicts `P(equity outperforms T-bills over the next 1 month)`. This probability is the input to the allocation pipeline.

Key model parameters:
- **Regularization (C=0.5)**: Moderate inverse regularization. Lower C means stronger L2 penalty, which shrinks coefficients toward zero and prevents any single feature from dominating. This is deliberate — we want the model to rely on the *ensemble* of macro signals, not overfit to any one.
- **Class weighting**: Balanced. Automatically upweights the minority class (T-bills-favored months) during training so the model doesn't just learn to always predict "equity wins" (which would be correct ~65% of the time but useless for risk management).
- **Feature scaling**: All features are standardized (zero mean, unit variance) before training via `StandardScaler`. This is essential for logistic regression because the regularization penalty treats all coefficients equally — without scaling, features with larger numeric ranges would be implicitly penalized less.

### Allocation Pipeline

The raw probability from the classifier goes through a four-stage pipeline to produce final portfolio weights:

**Stage 1: Sigmoid Weight Map**

```
weight = 1 / (1 + exp(-steepness * (p - 0.5) + bias))
```

The bias term is derived from the baseline allocation (90% equity / 10% T-bills), so at `P = 0.50` the model outputs 90% equity, not 50%. This encodes the prior that equities generally outperform cash — the model must see clear danger to go defensive. The steepness parameter (10.0) creates a sharp S-curve: small probability changes near 0.5 produce large weight swings, while extreme probabilities saturate. This means the model is decisive in the ambiguous middle range but doesn't overreact to high-confidence predictions.

**Stage 2: Crash Overlay** (see below)

**Stage 3: Asymmetric Smoothing**

```
smoothed = alpha * target + (1 - alpha) * previous_weight
```

- Ramping up equity: `alpha = 0.7` (slow — takes multiple months to fully re-enter)
- Cutting equity: `alpha = 1.0` (instant — no delay on defensive moves)

This asymmetry is deliberate. The cost of being slow to buy back in after a crisis is a few missed percentage points of recovery. The cost of being slow to cut during a crash is catastrophic drawdown. The system is designed to prioritize capital preservation.

**Stage 4: Weight Caps**

Final equity weight is clipped to [3%, 97%] to prevent full concentration in either asset.

### Crash Overlay

The crash overlay is a **separate, rule-based defense layer** that operates on current (unlagged) market data. While the classifier uses lagged features to prevent lookahead in training, the overlay deliberately uses real-time signals because its purpose is crisis response, not prediction.

Three independent triggers:

1. **VIX Spike** — Fires when VIX jumps >10 points in one month. Severity scales linearly up to a 50% weight reduction. This catches sudden volatility events like March 2020 or the 2008 Lehman collapse.

2. **VIX Panic (Backwardation + Rising)** — Fires when VIX term structure is in backwardation (VIX/VIX3M > 1.08) AND VIX is rising (>3 points in a month). Backwardation alone can persist during elevated-but-stable fear; requiring the rising condition filters for active, worsening panic. Up to 35% weight reduction.

3. **Drawdown Crash** — Fires when the equity market is >15% below its 12-month high AND the drawdown is accelerating (>4% worse than last month) AND VIX term structure confirms stress (>0.98). The triple confirmation prevents firing during orderly corrections. Up to 30% weight reduction.

The penalties are additive but capped at 60% total reduction. The overlay fired in 18 of 254 backtest months (7.1%), averaging 55.6% equity weight when active vs. 82.7% when off.

---

## Why Logistic Regression

This is a deliberate choice over more complex alternatives (gradient boosting, neural networks, random forests). The reasons are structural, not just practical:

**1. Interpretability maps to tradability.** Every coefficient directly tells you "if credit spreads widen by 1 standard deviation, the log-odds of T-bills outperforming increase by X." You can look at the model's current feature contributions and understand *why* it's recommending a given allocation. This matters because in practice, you need to trust a model enough to follow it during drawdowns — the exact moments when it matters most.

**2. Low capacity prevents overfitting to macro noise.** Macroeconomic regimes change slowly. There are roughly 3-5 distinct regime shifts per decade. A linear model with 12 features cannot memorize the 250-month training set — it is forced to learn genuine statistical relationships. A gradient-boosted tree with 100 estimators has enough capacity to memorize idiosyncratic month-to-month patterns that won't repeat.

**3. Stable out-of-sample behavior.** Logistic regression's prediction surface is monotonic in each feature. If credit spreads widen, the model always moves toward defensiveness — it can't learn bizarre non-monotonic interactions like "wide spreads are bad unless unemployment is between 4.2% and 4.5%." This stability means the model degrades gracefully as market regimes evolve.

**4. Calibrated probabilities.** Unlike tree-based methods, logistic regression produces naturally calibrated probabilities without post-hoc calibration. The sigmoid output directly represents the model's uncertainty, which feeds cleanly into the allocation pipeline's weight mapping.

**5. Minimal hyperparameter surface.** The model has one meaningful hyperparameter (regularization strength C). Compare this to XGBoost (max_depth, n_estimators, learning_rate, min_child_weight, subsample, colsample_bytree, gamma, lambda, alpha) — each of which introduces an optimization axis that can be overfit to the validation set.

The 55.3% classification accuracy might look modest, but in a context where the base rate is ~65% equity (simply always predicting equity would be 65% "accurate" on one class and 0% on the other), balanced accuracy of 54.6% means the model has genuine predictive signal on *both* classes. And because the allocation pipeline amplifies edge — a small probability tilt produces a large weight shift through the steep sigmoid — even modest classification skill translates into meaningful risk-adjusted outperformance.

---

## Backtest Methodology

### Walk-Forward Design

The backtest uses a strictly **walk-forward expanding window** design:

```
Month 37:  Train on months 1-35,  predict month 37,  earn return in month 38
Month 38:  Train on months 1-36,  predict month 38,  earn return in month 39
Month 39:  Train on months 1-37,  predict month 39,  earn return in month 40
...
```

At each step:
1. The model is retrained **from scratch** on all available history up to `t - horizon` (where horizon = 1 month). This means the training set only includes labels whose forward return windows have fully realized — no partial lookahead on the prediction target.
2. A fresh `StandardScaler` is fit on the training data only.
3. The model predicts on the current month's features.
4. Weights are computed, smoothed, and the portfolio earns the next month's realized return.

There is no single train/test split and no fixed holdout set. Every month in the backtest is genuinely out-of-sample relative to the model that produces its prediction.

### Lookahead Prevention

Five distinct mechanisms prevent the model from seeing the future:

1. **Feature lag**: All 12 features are shifted by 1 month. The model at time *t* sees macro data from *t-1*.
2. **Label gap**: Training labels stop at `t - horizon` (t-1 for 1-month horizon). The label for month *t* requires knowing the return from *t* to *t+1*, which hasn't happened yet.
3. **First-release FRED data**: Revisable series (CPI, unemployment, industrial production) use ALFRED first-release values — the number that was published at the time, not the revised figure available months later.
4. **Per-step scaler fitting**: The StandardScaler is fit only on training data at each step. Test-time features are transformed using training statistics.
5. **Crash overlay uses only concurrent data**: The overlay reads VIX and drawdown values for the current month — data that would be available to a live trader at rebalance time.

### Sample Weighting

Training uses the product of two weighting schemes:

**Recency weighting** (half-life: 18 months): Exponential decay gives recent observations higher weight. The macro environment that mattered 15 years ago is less informative than the environment from 2 years ago. This allows the model to adapt to structural changes (e.g., the post-2008 low-rate regime, post-COVID inflation dynamics) without discarding long-history statistical power.

**Class balancing**: Minority-class samples (months where T-bills outperformed) are upweighted proportionally. Without this, the model would learn the trivially high "always predict equity" strategy. Combined with logistic regression's `class_weight='balanced'` parameter, this produces a model that allocates its capacity to identifying *both* regimes.

### What Makes This Backtest Honest

Most tactical allocation backtests are some combination of overfit, leaked, or unrealistic. Here's what this one gets right:

**No parameter optimization on test data.** The config.yaml parameters (regularization C, sigmoid steepness, crash overlay thresholds, smoothing rates) were set based on financial reasoning, not grid-searched against backtest returns. C=0.5 is moderate regularization. The 90% equity baseline reflects the historical equity risk premium. The 10-point VIX spike threshold corresponds to roughly a 2-standard-deviation daily VIX move. These are not magic numbers discovered by trying 10,000 combinations.

**Monthly rebalancing with realistic execution.** The model rebalances once per month at month-end. There are no intraday signals, no look-ahead to daily data within the month, and no assumption of mid-month execution. The assets (SPY + fed funds) are among the most liquid in the world — execution slippage on monthly rebalancing of a two-asset portfolio is negligible.

**Benchmark selection is fair.** The model is compared against:
- 90/10 static baseline (its own "neutral" allocation — this is the hardest benchmark to beat because it captures most of the equity premium)
- 60/40 traditional balanced portfolio
- Pure equity (SPY)
- Pure T-bills

The model doesn't cherry-pick a weak benchmark. It's compared against the allocation it would hold if it had *no* signal at all.

**Survivorship bias is not applicable.** The system trades SPY (the S&P 500 ETF) and the fed funds rate. There's no stock selection, no sector rotation, no universe construction that could introduce survivorship bias.

**Transaction costs are trivially small.** Average monthly turnover is 0.176 (17.6% of the portfolio changes hands per month). For SPY, round-trip transaction costs are <1 basis point. Even at 5bps per trade, annual drag would be ~10bps — a rounding error on 10.76% CAGR.

---

## Results

Over the 21.2-year backtest period (Dec 2004 to Jan 2026, 254 months):

| Metric | Model | Equity | 60/40 | 90/10 Baseline |
|:---|---:|---:|---:|---:|
| CAGR | 10.76% | 10.60% | 7.29% | 9.80% |
| Total Return | 770% | 744% | 343% | 624% |
| Volatility | 10.77% | 14.78% | 8.87% | 13.30% |
| Sharpe | 1.00 | 0.72 | 0.82 | 0.74 |
| Sortino | 1.56 | 0.97 | 1.11 | 0.99 |
| Max Drawdown | -22.08% | -50.78% | -33.15% | -46.79% |
| Max DD Duration | 20 mo | 52 mo | 41 mo | 52 mo |

The key result is not the CAGR — it's the **drawdown reduction**. The model captures nearly all of the equity premium (10.76% vs 10.60%) while cutting the maximum drawdown by more than half (-22% vs -51%). The Sharpe ratio improves from 0.72 to 1.00, and the Sortino ratio (which specifically penalizes downside volatility) improves from 0.97 to 1.56.

The model's strongest performance came in 2008 (-11.5% vs equity's -36.8%), 2009 (+36.1% vs equity's +26.4%), and 2022 (-14.1% vs equity's -18.2%). It tends to underperform in strong bull markets where going defensive has no payoff (2010, 2016, 2021).

---

## Why This Could Work in Real Life

### The Macro Persistence Argument

The fundamental bet is that **macroeconomic regimes persist for months, not days**. Recessions don't begin and end in a week. Credit stress builds over quarters. Inflation regimes last years. This persistence means that a monthly-frequency model has a legitimate information advantage: by the time the macro environment has deteriorated enough for the classifier to detect it, there are usually still months of pain ahead.

This is different from trying to time the stock market on daily or weekly data, where noise dominates signal. At the monthly frequency, the signal-to-noise ratio of macro variables is much higher.

### Structural Advantages

**1. The features are economically grounded.** Every feature has a clear causal story for why it should predict equity/cash relative performance. Credit spreads widen because the market prices in higher default risk — this directly impairs equity valuations. Real rates go negative because the Fed is stimulating — this directly inflates asset prices. These aren't data-mined correlations; they're well-understood economic transmission mechanisms that have been documented in academic literature for decades.

**2. The model has low complexity relative to the signal.** Twelve features, one linear boundary, monthly frequency. The model cannot memorize patterns. It can only learn broad statistical tendencies that are robust enough to survive L2 regularization and class balancing. This is the opposite of a deep learning model that finds fragile patterns in tick data.

**3. The allocation is asymmetric by design.** The 90% equity baseline means the model defaults to being fully invested. It only deviates when macro signals are clearly negative. This asymmetry aligns with the base rate: equities outperform cash roughly 65% of the time. The model doesn't need to be right often — it needs to be right during the 35% of months that matter.

**4. The crash overlay provides non-model defense.** The VIX spike and backwardation triggers operate independently of the classifier. They react to market microstructure signals (implied volatility term structure) that are available in real time and have near-zero false positive rates for genuine crises. This layered defense means the system isn't relying on a single point of failure.

**5. The execution is trivial.** Two assets (SPY + cash), monthly rebalancing, no leverage, no shorting, no options. This can be implemented in any brokerage account with minimal operational complexity. There are no execution risks from illiquidity, wide bid-ask spreads, or complex instrument pricing.

### What Could Go Wrong

No model is permanent. Here are the real risks:

**Regime structural breaks.** The model learns from 20+ years of history that includes specific Fed behavior, specific inflation dynamics, and specific market microstructure. If the fundamental structure of how macro variables transmit to equity returns changes (e.g., the Fed permanently abandons inflation targeting, or credit markets are restructured), the learned coefficients could become miscalibrated.

**Slow-burn drawdowns.** The crash overlay is designed for acute crises (sharp VIX spikes, rapid drawdowns). A slow, grinding bear market that unfolds over 18 months without acute volatility events could slip past the overlay while the classifier's lagged features take time to catch up.

**Inflation regime novelty.** The training data includes one major inflation episode (2021-2023). The model's strongest T-bills-favoring feature is inflation impulse. If a future inflation regime behaves differently from 2021-2023 (e.g., stagflation with simultaneous credit tightening), the model has limited training data for that specific combination.

**Overfitting to 2008.** The model's most dramatic outperformance came during the GFC. If the 2008 crisis was sufficiently unique (a once-in-80-year financial system event), the model may have overweighted the features that happened to predict that specific crisis. The expanding window design mitigates this over time as 2008 becomes a smaller fraction of training data.

**Crash overlay might fire during V-shaped recoveries.** The VIX spike trigger could reduce equity exposure right before a sharp rebound (like March 2020). The asymmetric smoothing (slow re-entry) means the model would miss the initial recovery bounce, though the 0.7 smoothing rate means it re-enters within 2-3 months.

---

## Project Structure

```
macro_regime_indicator/
├── macro_regime_allocator/
│   ├── main.py              # Entry point — orchestrates the pipeline
│   ├── config.py             # Configuration (loads from config.yaml)
│   ├── config.yaml           # User-facing parameters
│   ├── data.py               # Data download, feature engineering, labels
│   ├── model.py              # RegimeClassifier (LogisticRegression wrapper)
│   ├── backtest.py           # Walk-forward backtest, allocation logic, crash overlay
│   ├── results.py            # Evaluation metrics, report generation, plots
│   ├── Makefile              # run, predict, signal, fast, install, clean
│   ├── data/                 # Downloaded and processed CSVs
│   └── outputs/              # Backtest results, metrics, plots, model.joblib
│
└── frontend/
    ├── src/                  # React + TypeScript dashboard
    │   ├── App.tsx           # Two-tab layout (Charts / Report)
    │   ├── pages/
    │   │   ├── Dashboard.tsx # 7 charts matching generated plots
    │   │   └── Report.tsx    # Model vs Equity vs 60/40 deep dive
    │   ├── data.ts           # CSV loading and computation helpers
    │   └── components/       # ChartCard wrapper
    ├── scripts/
    │   └── prepare_data.py   # Copies backend outputs to public/data/
    └── Makefile              # install, dev, build, clean
```

## Getting Started

**Requirements**: Python 3.10+, Node.js 18+, a FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)

```bash
# Backend — run the full pipeline
cd macro_regime_allocator
export FRED_API_KEY=your_key_here
make install    # install Python dependencies with uv
make run        # download data, train, backtest, generate plots

# Frontend — view the dashboard
cd ../frontend
make install    # npm install
make dev        # copies data + starts dev server on port 3000
```

Other backend commands:
- `make fast` — skip download, re-run backtest with cached data
- `make predict` — full run + show current allocation signal
- `make signal` — cached data + current allocation signal
