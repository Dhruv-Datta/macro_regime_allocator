# Robustness Validation Report

*Generated 2026-03-25 16:32*

## 1. Parameter Sensitivity

| Param | Value | Sharpe | Excess Sharpe | CAGR | Max DD | Calmar | Excess Calmar |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | - | 0.600 | +0.178 | 6.66% | -33.1% | 0.201 | +0.082 |
| regularization_C | 0.25 | 0.611 | +0.189 | 6.85% | -32.7% | 0.210 | +0.091 |
| regularization_C | 0.375 | 0.606 | +0.183 | 6.75% | -33.0% | 0.205 | +0.086 |
| regularization_C | 0.45 | 0.603 | +0.180 | 6.69% | -33.1% | 0.202 | +0.083 |
| regularization_C | 0.55 | 0.598 | +0.176 | 6.62% | -33.2% | 0.200 | +0.081 |
| regularization_C | 0.625 | 0.595 | +0.173 | 6.58% | -33.2% | 0.198 | +0.079 |
| regularization_C | 0.75 | 0.591 | +0.168 | 6.51% | -33.3% | 0.195 | +0.076 |
| allocation_steepness | 6.5 | 0.533 | +0.110 | 6.53% | -39.8% | 0.164 | +0.045 |
| allocation_steepness | 9.75 | 0.574 | +0.151 | 6.65% | -36.6% | 0.182 | +0.063 |
| allocation_steepness | 11.7 | 0.592 | +0.170 | 6.67% | -34.6% | 0.193 | +0.074 |
| allocation_steepness | 14.3 | 0.602 | +0.180 | 6.59% | -31.6% | 0.209 | +0.090 |
| allocation_steepness | 16.25 | 0.606 | +0.184 | 6.52% | -30.6% | 0.213 | +0.094 |
| allocation_steepness | 19.5 | 0.617 | +0.195 | 6.49% | -30.1% | 0.216 | +0.097 |
| recency_halflife_months | 6 | 0.685 | +0.263 | 7.32% | -31.5% | 0.232 | +0.113 |
| recency_halflife_months | 9 | 0.629 | +0.207 | 6.89% | -31.4% | 0.219 | +0.100 |
| recency_halflife_months | 11 | 0.608 | +0.185 | 6.72% | -32.6% | 0.206 | +0.087 |
| recency_halflife_months | 13 | 0.594 | +0.172 | 6.61% | -33.6% | 0.197 | +0.078 |
| recency_halflife_months | 15 | 0.584 | +0.162 | 6.53% | -34.3% | 0.190 | +0.071 |
| recency_halflife_months | 18 | 0.576 | +0.153 | 6.48% | -35.0% | 0.185 | +0.066 |
| min_train_months | 24 | 0.600 | +0.178 | 6.66% | -33.1% | 0.201 | +0.082 |
| min_train_months | 36 | 0.600 | +0.178 | 6.66% | -33.1% | 0.201 | +0.082 |
| min_train_months | 43 | 0.600 | +0.178 | 6.66% | -33.1% | 0.201 | +0.082 |
| min_train_months | 53 | 0.600 | +0.178 | 6.65% | -33.1% | 0.201 | +0.082 |
| min_train_months | 60 | 0.624 | +0.179 | 6.86% | -32.4% | 0.212 | +0.087 |
| min_train_months | 72 | 0.720 | +0.179 | 7.72% | -31.0% | 0.249 | +0.101 |
| weight_smoothing_down | 0.82 | 0.596 | +0.174 | 6.67% | -33.7% | 0.198 | +0.079 |
| weight_smoothing_down | 0.89 | 0.598 | +0.176 | 6.67% | -33.4% | 0.199 | +0.080 |
| weight_smoothing_down | 0.94 | 0.600 | +0.177 | 6.66% | -33.2% | 0.200 | +0.081 |
| weight_smoothing_down | 0.98 | 0.601 | +0.178 | 6.65% | -33.1% | 0.201 | +0.082 |
| weight_smoothing_down | 0.99 | 0.601 | +0.178 | 6.65% | -33.0% | 0.201 | +0.082 |
| weight_smoothing_down | 1.0 | 0.601 | +0.179 | 6.65% | -33.0% | 0.202 | +0.083 |
| weight_smoothing_up | 0.83 | 0.608 | +0.186 | 6.67% | -32.6% | 0.204 | +0.085 |
| weight_smoothing_up | 0.9 | 0.604 | +0.182 | 6.66% | -32.9% | 0.203 | +0.084 |
| weight_smoothing_up | 0.95 | 0.602 | +0.179 | 6.66% | -33.0% | 0.202 | +0.083 |
| weight_smoothing_up | 0.99 | 0.600 | +0.177 | 6.66% | -33.1% | 0.201 | +0.082 |

## 2. Ablation Studies

| Variant | Sharpe | Excess Sharpe | CAGR | Max DD | Calmar | Excess Calmar |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_system | 0.600 | +0.178 | 6.66% | -33.1% | 0.201 | +0.082 |
| no_crash_overlay | 0.570 | +0.147 | 6.49% | -35.5% | 0.183 | +0.064 |
| no_smoothing | 0.600 | +0.178 | 6.65% | -33.1% | 0.201 | +0.082 |
| no_recency_weighting | 0.582 | +0.160 | 6.94% | -36.0% | 0.193 | +0.074 |
| model_only | 0.568 | +0.146 | 6.48% | -35.7% | 0.182 | +0.063 |
| baseline_50_50 | 0.665 | +0.090 | 5.33% | -25.8% | 0.207 | +0.058 |
| baseline_60_40 | 0.661 | +0.135 | 5.61% | -26.8% | 0.210 | +0.072 |
| baseline_75_25 | 0.645 | +0.173 | 6.00% | -28.3% | 0.212 | +0.085 |
| baseline_95_5 | 0.600 | +0.178 | 6.66% | -33.1% | 0.201 | +0.082 |

## 3. Subperiod Analysis

| Period | Sharpe | Excess Sharpe | CAGR | Max DD | Excess CAGR | Calmar | Excess Calmar | Months |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_period | 0.706 | +0.165 | 8.30% | -33.1% | 0.53% | 0.251 | +0.091 | 313 |
| 2000s | 0.204 | +0.253 | 2.36% | -33.1% | 3.11% | 0.071 | +0.087 | 120 |
| 2010s | 1.058 | -0.025 | 11.13% | -13.2% | -1.67% | 0.845 | +0.017 | 120 |
| 2020s | 1.015 | +0.119 | 13.88% | -15.8% | -0.60% | 0.879 | +0.244 | 73 |
| ex_2008_crisis | 0.786 | +0.089 | 9.01% | -33.1% | -0.39% | 0.272 | +0.052 | 289 |
| ex_2020_covid | 0.739 | +0.163 | 8.56% | -33.1% | 0.50% | 0.259 | +0.094 | 307 |
| ex_both_crises | 0.826 | +0.079 | 9.31% | -33.1% | -0.43% | 0.281 | +0.053 | 283 |
| in_sample | 0.600 | +0.178 | 6.66% | -33.1% | 0.84% | 0.201 | +0.082 | 240 |
| holdout | 1.015 | +0.119 | 13.88% | -15.8% | -0.60% | 0.879 | +0.244 | 73 |

## 4. Bootstrap Confidence Intervals

| Metric | Mean | Std | 5th | 25th | 50th | 75th | 95th |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sharpe | 0.6164 | 0.2642 | 0.2008 | 0.4310 | 0.6083 | 0.7814 | 1.0752 |
| excess_sharpe | 0.1688 | 0.1052 | 0.0033 | 0.0944 | 0.1666 | 0.2409 | 0.3405 |
| excess_cagr | 0.0085 | 0.0155 | -0.0138 | -0.0029 | 0.0072 | 0.0177 | 0.0353 |

## 5. Coefficient Stability

| Feature | Mean Coef | Change Std | Max Jump | Spike Freq | Instability |
| :--- | ---: | ---: | ---: | ---: | ---: |
| inflation_yoy | +0.0612 | 0.0692 | 0.3046 | 24.61% | 0.600 |
| inflation_impulse | +0.1176 | 0.0750 | 0.2894 | 23.66% | 0.650 |
| unemployment_rate | -0.0657 | 0.0627 | 0.2964 | 25.24% | 0.340 |
| real_fed_funds | -0.0291 | 0.0570 | 0.2457 | 21.45% | 0.100 |
| yield_curve_slope | -0.0129 | 0.0682 | 0.3094 | 24.29% | 0.560 |
| vix_1m_change | -0.0643 | 0.0728 | 0.4019 | 26.18% | 0.860 |
| vix_term_structure | -0.0587 | 0.0649 | 0.5238 | 33.12% | 0.700 |
| equity_momentum_3m | +0.1835 | 0.0675 | 0.2517 | 28.08% | 0.470 |
| equity_vol_3m | -0.0214 | 0.0709 | 0.3143 | 22.71% | 0.650 |
| equity_drawdown_from_high | +0.0074 | 0.0643 | 0.3645 | 31.23% | 0.570 |

## 6. Defensive Accuracy

*Does the model defend when it matters and ride the wave when it should?*

| Metric | Value | Detail |
| :--- | ---: | :--- |
| Crisis hit rate | 58.1% | Went defensive in worst 31 equity months (bottom 10%) |
| Calm ride rate | 58.5% | Stayed invested when equity beat T-bills |
| Cost of false defense | 88.66% | Return given up on 80 false alarms |
| Defense payoff | 90.18% | Losses avoided on 48 correct defensive calls |
| Payoff ratio | 1.02x | Saved / cost (higher = better) |