# Macro Regime Allocator

A dynamic portfolio allocation system that shifts between equities (SPY) and T-Bills based on a machine-learning regime classifier. Rather than maintaining a static allocation, the model predicts whether equities will outperform T-Bills over the next month and adjusts exposure accordingly.

**Core philosophy:** The equity risk premium is real and persistent. The model's job is not to constantly trade — it's to *defend* during the rare months that matter. Stay nearly fully invested, cut hard when danger appears.

## How It Works

### Data Pipeline

The system ingests monthly macroeconomic and market data from two sources:

- **Market data** (via yfinance): SPY prices, VIX (spot and 3-month), used to compute returns, drawdowns, and volatility signals.
- **Macro data** (via FRED API): CPI, unemployment, fed funds rate, Treasury yields (2Y/10Y), high-yield credit spreads, and industrial production.

All series are resampled to **month-end frequency**. Intra-month drawdowns (peak-to-trough within each calendar month) are computed from daily prices before resampling.

### Feature Engineering

Thirteen features are engineered from the raw data, grouped by economic theme:

| Theme | Features |
|---|---|
| Inflation | Year-over-year CPI, inflation impulse (3m acceleration) |
| Labor | Unemployment rate |
| Rates & Credit | Credit spread level, 3-month credit spread change, real fed funds rate |
| Yield Curve | 10Y-2Y slope |
| Volatility | VIX 1-month change, VIX term structure (spot vs. 3-month ratio) |
| Equity Momentum | 3-month equity return |
| Equity Risk | 3-month realized volatility, drawdown from high, intra-month drawdown |

All features are **lagged by 1 month** to prevent lookahead bias and account for publication delays in macro data.

### Labels

The target variable is binary: **does the T-Bill rate beat the equity return over the next month?** This frames the problem as regime classification — the model learns to identify environments where defensive positioning is warranted.

### Model

The classifier is a **logistic regression** with moderate L2 regularization (C=0.5). Logistic regression was chosen deliberately:

- Outputs calibrated probabilities, which map naturally to allocation weights
- Coefficients are directly interpretable — you can see *which* macro factors drive the allocation
- Regularization prevents extreme probabilities that would cause unnecessary whipsaws
- Robust to the relatively small sample size (~300 monthly observations)

Features are standardized before fitting. No feature selection is performed — regularization handles irrelevant inputs.

### Walk-Forward Backtest

The model is evaluated using a strict **walk-forward** protocol to simulate real-time decision-making:

1. At each month *t*, train on all data up to *t - 1* (expanding window)
2. Apply **recency weighting** (exponential decay, 12-month halflife) so the model adapts to evolving regimes
3. Apply **class balancing** via sample weights so the minority class (T-Bill wins, ~35% of months) gets adequate representation
4. Predict P(equity outperforms) for month *t*
5. Convert probability to an allocation weight
6. Realize returns over the next month

A minimum of 48 months of training data is required before predictions begin. The model is never trained on future data.

### Allocation Pipeline

Raw model probabilities are transformed into portfolio weights through a multi-stage pipeline:

1. **Sigmoid mapping:** Probabilities are passed through a steep sigmoid centered on the baseline allocation (95% equity). This creates a bias toward staying invested — the model needs strong conviction to deviate.

2. **Crash overlay:** A rule-based layer that reacts to *current-month* market stress, since the model only sees lagged data. Three triggers:
   - **VIX spike** (1-month change exceeds threshold): Reduces equity weight proportionally
   - **VIX backwardation** (spot > 3-month futures while rising): Indicates acute fear, triggers additional reduction
   - **Accelerating drawdown** (drawdown beyond threshold with elevated VIX): Catches cascading sell-offs

3. **Asymmetric smoothing:** Weights are exponentially smoothed with different speeds for increases vs. decreases. Re-entry is near-instant (alpha=0.98) to avoid missing recoveries. Defense is slightly delayed (alpha=0.97) to require signal confirmation and prevent whipsaws on single bad months.

4. **Weight caps:** Final weight is clamped between 10% and 97% equity. The floor ensures some equity exposure is always maintained; the ceiling prevents leverage.

### Live Prediction

For live use, the model is trained on the full historical dataset and saved. Given new monthly data, it engineers features, generates a probability, and outputs a recommended equity/T-Bill split for the current month.

```bash
uv run python -m macro_regime_allocator --predict
```

## Validation

The model undergoes a comprehensive robustness validation suite to guard against overfitting and fragility.

### Parameter Sensitivity

Each key parameter is varied one-at-a-time while holding others fixed:

- **Regularization (C):** Performance is stable across a range of values around the chosen setting
- **Sigmoid steepness:** A broad plateau exists around the selected value — results aren't knife-edged on this choice
- **Recency halflife:** Robust across a reasonable range (6-18 months)
- **Minimum training months:** More history generally helps, with diminishing returns beyond ~60 months
- **Smoothing parameters:** Sensitive in a tight range, but the up/down asymmetry is consistently beneficial

### Ablation Studies

Each component is removed individually to measure its marginal contribution:

- **Crash overlay:** Removing it degrades max drawdown protection meaningfully. The overlay's value is concentrated in tail-risk months.
- **Recency weighting:** Removing it causes the largest single drop in risk-adjusted returns, confirming that regime dynamics evolve over time.
- **Smoothing:** Minimal impact on aggregate metrics, but reduces turnover and transaction costs in practice.

### Subperiod Analysis

Performance is evaluated across distinct market regimes (e.g., 2000s bear markets, 2010s bull market, 2020s volatility) to ensure the model isn't relying on a single favorable period. The model's defensive value is most apparent during stressed periods, while it modestly underperforms during sustained bull runs — consistent with its design philosophy.

### Bootstrap Confidence Intervals

Returns are block-bootstrapped (preserving autocorrelation) to generate confidence intervals around key metrics (Sharpe ratio, excess returns). This quantifies the statistical uncertainty inherent in a ~25-year backtest.

### Coefficient Stability

Model coefficients are tracked across the walk-forward windows to verify that feature importance is stable over time. Most features (VIX changes, momentum, credit spreads) maintain consistent sign and magnitude. Some features show more variability, which is expected given evolving macro regimes.

### Holdout Testing

A configurable holdout period (default: 2020 onward) is excluded from all tuning and validation. The model is trained only on pre-holdout data and evaluated on the holdout period. This provides a final out-of-sample check for overfitting.

## Project Structure

```
macro_regime_allocator/
├── config.yaml          # All tunable parameters
├── config.py            # Config dataclass + FRED series definitions
├── main.py              # CLI entrypoint (backtest or predict)
├── data.py              # Download, merge, feature engineering
├── model.py             # RegimeClassifier wrapper
├── backtest.py          # Walk-forward engine + allocation logic
├── results.py           # Metrics, plots, report generation
├── validate.py          # Robustness validation suite
├── data/                # Cached raw + engineered data (CSV)
└── outputs/
    ├── backtest_results.csv
    ├── investment_metrics.csv
    ├── model.joblib
    ├── report.md
    ├── plots/           # 7 diagnostic plots
    └── validation/      # Sensitivity, ablation, bootstrap results
```

## Usage

### Run Full Backtest

```bash
uv run python -m macro_regime_allocator
```

This downloads data, engineers features, runs the walk-forward backtest, generates plots and metrics, and saves everything to `outputs/`.

### Run Validation Suite

```bash
uv run python -m macro_regime_allocator --validate
```

### Get Live Allocation

```bash
uv run python -m macro_regime_allocator --predict
```

### Configuration

All parameters are in `config.yaml`. Key settings:

- `start_date` / `end_date`: Data range
- `regularization_C`: Model regularization strength
- `baseline_equity`: Default equity allocation (0.95 = 95%)
- `allocation_steepness`: How aggressively the model deviates from baseline
- `crash_overlay`: Enable/disable the rule-based crash defense
- `holdout_start`: Date to begin the untouched holdout period

## Dependencies

Managed via `uv`. Key packages: scikit-learn, pandas, yfinance, fredapi, matplotlib, joblib.
