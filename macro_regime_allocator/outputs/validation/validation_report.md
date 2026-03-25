# Robustness Validation Report

*Generated 2026-03-25 16:00*

## 1. Parameter Sensitivity

| Param | Value | Sharpe | CAGR | Max DD | Calmar | Excess Calmar |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | - | 0.764 | 8.33% | -27.2% | 0.307 | +0.163 |
| regularization_C | 0.25 | 0.758 | 8.38% | -28.1% | 0.298 | +0.154 |
| regularization_C | 0.375 | 0.764 | 8.38% | -27.4% | 0.306 | +0.163 |
| regularization_C | 0.45 | 0.764 | 8.35% | -27.2% | 0.307 | +0.163 |
| regularization_C | 0.55 | 0.763 | 8.31% | -27.1% | 0.307 | +0.163 |
| regularization_C | 0.625 | 0.761 | 8.28% | -27.0% | 0.306 | +0.163 |
| regularization_C | 0.75 | 0.758 | 8.23% | -27.0% | 0.305 | +0.161 |
| allocation_steepness | 6.5 | 0.671 | 7.98% | -35.4% | 0.225 | +0.082 |
| allocation_steepness | 9.75 | 0.731 | 8.27% | -30.2% | 0.274 | +0.131 |
| allocation_steepness | 11.7 | 0.757 | 8.36% | -27.8% | 0.300 | +0.157 |
| allocation_steepness | 14.3 | 0.766 | 8.27% | -27.1% | 0.305 | +0.162 |
| allocation_steepness | 16.25 | 0.763 | 8.14% | -27.2% | 0.300 | +0.156 |
| allocation_steepness | 19.5 | 0.757 | 7.93% | -27.0% | 0.294 | +0.150 |
| recency_halflife_months | 6 | 0.786 | 8.38% | -28.3% | 0.296 | +0.152 |
| recency_halflife_months | 9 | 0.775 | 8.37% | -27.8% | 0.301 | +0.158 |
| recency_halflife_months | 11 | 0.767 | 8.35% | -27.4% | 0.305 | +0.162 |
| recency_halflife_months | 13 | 0.761 | 8.32% | -27.0% | 0.309 | +0.165 |
| recency_halflife_months | 15 | 0.757 | 8.31% | -26.6% | 0.312 | +0.169 |
| recency_halflife_months | 18 | 0.753 | 8.30% | -27.1% | 0.307 | +0.163 |
| min_train_months | 24 | 0.653 | 7.32% | -34.1% | 0.215 | +0.096 |
| min_train_months | 36 | 0.687 | 7.63% | -33.9% | 0.225 | +0.101 |
| min_train_months | 43 | 0.722 | 8.05% | -31.6% | 0.255 | +0.119 |
| min_train_months | 53 | 0.806 | 8.86% | -27.2% | 0.326 | +0.160 |
| min_train_months | 60 | 0.857 | 9.44% | -27.2% | 0.347 | +0.178 |
| min_train_months | 72 | 0.995 | 10.44% | -27.2% | 0.384 | +0.194 |
| weight_smoothing_down | 0.82 | 0.749 | 8.25% | -28.3% | 0.291 | +0.148 |
| weight_smoothing_down | 0.89 | 0.756 | 8.29% | -27.8% | 0.299 | +0.155 |
| weight_smoothing_down | 0.94 | 0.761 | 8.32% | -27.4% | 0.304 | +0.160 |
| weight_smoothing_down | 0.98 | 0.764 | 8.34% | -27.1% | 0.308 | +0.164 |
| weight_smoothing_down | 0.99 | 0.765 | 8.34% | -27.0% | 0.309 | +0.165 |
| weight_smoothing_down | 1.0 | 0.766 | 8.35% | -26.9% | 0.310 | +0.166 |
| weight_smoothing_up | 0.83 | 0.782 | 8.41% | -25.6% | 0.328 | +0.185 |
| weight_smoothing_up | 0.9 | 0.773 | 8.38% | -26.2% | 0.320 | +0.176 |
| weight_smoothing_up | 0.95 | 0.767 | 8.35% | -26.8% | 0.312 | +0.168 |
| weight_smoothing_up | 0.99 | 0.762 | 8.33% | -27.3% | 0.305 | +0.162 |

## 2. Ablation Studies

| Variant | Sharpe | CAGR | Max DD | Calmar | Excess Calmar |
| :--- | ---: | ---: | ---: | ---: | ---: |
| full_system | 0.764 | 8.33% | -27.2% | 0.307 | +0.163 |
| no_crash_overlay | 0.727 | 8.19% | -31.5% | 0.260 | +0.116 |
| no_smoothing | 0.764 | 8.33% | -27.2% | 0.306 | +0.163 |
| no_recency_weighting | 0.702 | 8.02% | -29.3% | 0.274 | +0.131 |
| model_only | 0.726 | 8.18% | -31.6% | 0.258 | +0.115 |
| baseline_50_50 | 0.740 | 6.32% | -21.2% | 0.298 | +0.134 |
| baseline_60_40 | 0.745 | 6.67% | -22.4% | 0.298 | +0.142 |
| baseline_75_25 | 0.753 | 7.22% | -24.2% | 0.298 | +0.149 |
| baseline_95_5 | 0.764 | 8.33% | -27.2% | 0.307 | +0.163 |

## 3. Subperiod Analysis

| Period | Sharpe | CAGR | Max DD | Excess CAGR | Calmar | Excess Calmar | Months |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_period | 0.832 | 9.70% | -27.2% | 0.90% | 0.357 | +0.177 | 296 |
| 2000s | 0.454 | 5.09% | -27.2% | 4.47% | 0.188 | +0.175 | 103 |
| 2010s | 1.053 | 11.19% | -13.2% | -1.61% | 0.850 | +0.021 | 120 |
| 2020s | 1.022 | 14.00% | -15.7% | -0.48% | 0.889 | +0.253 | 73 |
| ex_2008_crisis | 0.881 | 9.97% | -25.9% | -0.66% | 0.385 | +0.037 | 272 |
| ex_2020_covid | 0.873 | 10.01% | -27.2% | 0.89% | 0.369 | +0.182 | 290 |
| ex_both_crises | 0.929 | 10.32% | -25.9% | -0.71% | 0.398 | +0.037 | 266 |
| in_sample | 0.764 | 8.33% | -27.2% | 1.33% | 0.307 | +0.163 | 223 |
| holdout | 1.022 | 14.00% | -15.7% | -0.48% | 0.889 | +0.253 | 73 |

## 4. Bootstrap Confidence Intervals

| Metric | Mean | Std | 5th | 25th | 50th | 75th | 95th |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sharpe | 0.7710 | 0.2812 | 0.3060 | 0.5823 | 0.7808 | 0.9588 | 1.2460 |
| excess_cagr | 0.0127 | 0.0194 | -0.0145 | -0.0020 | 0.0115 | 0.0242 | 0.0500 |

## 5. Coefficient Stability

| Feature | Mean Coef | Change Std | Max Jump | Spike Freq | Instability |
| :--- | ---: | ---: | ---: | ---: | ---: |
| inflation_yoy | +0.0559 | 0.0646 | 0.2955 | 25.42% | 0.675 |
| inflation_impulse | +0.1579 | 0.0731 | 0.2839 | 22.71% | 0.758 |
| unemployment_rate | -0.0405 | 0.0613 | 0.2646 | 26.44% | 0.542 |
| credit_spread_level | +0.0642 | 0.0475 | 0.1919 | 19.32% | 0.083 |
| credit_spread_3m_change | +0.0867 | 0.0587 | 0.2431 | 26.44% | 0.358 |
| real_fed_funds | -0.0219 | 0.0565 | 0.2519 | 23.73% | 0.317 |
| yield_curve_slope | -0.0065 | 0.0624 | 0.2614 | 23.73% | 0.508 |
| vix_1m_change | -0.0727 | 0.0713 | 0.3501 | 26.78% | 0.900 |
| vix_term_structure | -0.0732 | 0.0672 | 0.5210 | 31.86% | 0.875 |
| equity_momentum_3m | +0.2228 | 0.0602 | 0.2504 | 22.37% | 0.317 |
| equity_vol_3m | -0.0296 | 0.0697 | 0.3023 | 22.71% | 0.725 |
| equity_drawdown_from_high | +0.0140 | 0.0509 | 0.2808 | 27.80% | 0.442 |

## 6. Defensive Accuracy

*Does the model defend when it matters and ride the wave when it should?*

| Metric | Value | Detail |
| :--- | ---: | :--- |
| Crisis hit rate | 58.6% | Went defensive in worst 29 equity months (bottom 10%) |
| Calm ride rate | 62.2% | Stayed invested when equity beat T-bills |
| Cost of false defense | 70.46% | Return given up on 71 false alarms |
| Defense payoff | 80.74% | Losses avoided on 40 correct defensive calls |
| Payoff ratio | 1.15x | Saved / cost (higher = better) |