# Robustness Validation Report

*Generated 2026-03-25 14:44*

## 1. Parameter Sensitivity

| Param | Value | Sharpe | CAGR | Max DD | Calmar | Excess Calmar |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | - | 0.762 | 8.32% | -27.3% | 0.305 | +0.161 |
| regularization_C | 0.25 | 0.757 | 8.37% | -28.3% | 0.296 | +0.152 |
| regularization_C | 0.375 | 0.763 | 8.37% | -27.5% | 0.304 | +0.160 |
| regularization_C | 0.45 | 0.762 | 8.34% | -27.4% | 0.305 | +0.161 |
| regularization_C | 0.55 | 0.761 | 8.31% | -27.3% | 0.305 | +0.161 |
| regularization_C | 0.625 | 0.760 | 8.28% | -27.2% | 0.305 | +0.161 |
| regularization_C | 0.75 | 0.757 | 8.22% | -27.1% | 0.303 | +0.160 |
| allocation_steepness | 6.5 | 0.669 | 7.97% | -35.5% | 0.224 | +0.081 |
| allocation_steepness | 9.75 | 0.729 | 8.26% | -30.4% | 0.272 | +0.129 |
| allocation_steepness | 11.7 | 0.755 | 8.35% | -28.0% | 0.298 | +0.155 |
| allocation_steepness | 14.3 | 0.764 | 8.26% | -27.3% | 0.303 | +0.160 |
| allocation_steepness | 16.25 | 0.762 | 8.13% | -27.3% | 0.298 | +0.155 |
| allocation_steepness | 19.5 | 0.755 | 7.93% | -27.1% | 0.292 | +0.149 |
| recency_halflife_months | 6 | 0.784 | 8.37% | -28.5% | 0.294 | +0.151 |
| recency_halflife_months | 9 | 0.773 | 8.36% | -28.0% | 0.299 | +0.156 |
| recency_halflife_months | 11 | 0.766 | 8.34% | -27.5% | 0.303 | +0.160 |
| recency_halflife_months | 13 | 0.759 | 8.31% | -27.1% | 0.307 | +0.163 |
| recency_halflife_months | 15 | 0.755 | 8.30% | -26.7% | 0.310 | +0.167 |
| recency_halflife_months | 18 | 0.752 | 8.29% | -27.1% | 0.306 | +0.162 |
| min_train_months | 24 | 0.652 | 7.31% | -34.2% | 0.214 | +0.095 |
| min_train_months | 36 | 0.685 | 7.62% | -34.0% | 0.224 | +0.100 |
| min_train_months | 43 | 0.720 | 8.04% | -31.7% | 0.254 | +0.118 |
| min_train_months | 53 | 0.805 | 8.86% | -27.3% | 0.324 | +0.158 |
| min_train_months | 60 | 0.855 | 9.43% | -27.3% | 0.345 | +0.176 |
| min_train_months | 72 | 0.994 | 10.44% | -27.3% | 0.382 | +0.192 |
| weight_smoothing_down | 0.8 | 0.747 | 8.24% | -28.5% | 0.289 | +0.146 |
| weight_smoothing_down | 0.87 | 0.754 | 8.28% | -27.9% | 0.296 | +0.153 |
| weight_smoothing_down | 0.92 | 0.759 | 8.31% | -27.5% | 0.302 | +0.158 |
| weight_smoothing_down | 0.96 | 0.763 | 8.33% | -27.2% | 0.306 | +0.162 |
| weight_smoothing_down | 0.97 | 0.764 | 8.33% | -27.2% | 0.307 | +0.163 |
| weight_smoothing_down | 0.98 | 0.764 | 8.34% | -27.1% | 0.308 | +0.164 |
| weight_smoothing_up | 0.83 | 0.780 | 8.40% | -25.7% | 0.327 | +0.184 |
| weight_smoothing_up | 0.9 | 0.771 | 8.37% | -26.3% | 0.318 | +0.174 |
| weight_smoothing_up | 0.95 | 0.765 | 8.34% | -27.0% | 0.309 | +0.166 |
| weight_smoothing_up | 0.99 | 0.761 | 8.32% | -27.4% | 0.303 | +0.160 |

## 2. Ablation Studies

| Variant | Sharpe | CAGR | Max DD | Calmar | Excess Calmar |
| :--- | ---: | ---: | ---: | ---: | ---: |
| full_system | 0.762 | 8.32% | -27.3% | 0.305 | +0.161 |
| no_crash_overlay | 0.726 | 8.18% | -31.7% | 0.258 | +0.115 |
| no_smoothing | 0.764 | 8.33% | -27.2% | 0.306 | +0.163 |
| no_recency_weighting | 0.701 | 8.02% | -29.3% | 0.273 | +0.130 |
| model_only | 0.726 | 8.18% | -31.6% | 0.258 | +0.115 |
| baseline_50_50 | 0.740 | 6.33% | -21.4% | 0.296 | +0.132 |
| baseline_60_40 | 0.745 | 6.67% | -22.5% | 0.296 | +0.140 |
| baseline_75_25 | 0.752 | 7.22% | -24.4% | 0.296 | +0.148 |
| baseline_95_5 | 0.762 | 8.32% | -27.3% | 0.305 | +0.161 |

## 3. Subperiod Analysis

| Period | Sharpe | CAGR | Max DD | Excess CAGR | Calmar | Excess Calmar | Months |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_period | 0.831 | 9.70% | -27.3% | 0.90% | 0.355 | +0.175 | 296 |
| 2000s | 0.450 | 5.05% | -27.3% | 4.43% | 0.185 | +0.172 | 103 |
| 2010s | 1.054 | 11.21% | -13.2% | -1.59% | 0.851 | +0.023 | 120 |
| 2020s | 1.020 | 14.01% | -15.8% | -0.47% | 0.887 | +0.251 | 73 |
| ex_2008_crisis | 0.880 | 9.98% | -26.0% | -0.65% | 0.384 | +0.036 | 272 |
| ex_2020_covid | 0.872 | 10.01% | -27.3% | 0.89% | 0.367 | +0.180 | 290 |
| ex_both_crises | 0.929 | 10.33% | -26.0% | -0.70% | 0.398 | +0.037 | 266 |
| in_sample | 0.762 | 8.32% | -27.3% | 1.32% | 0.305 | +0.161 | 223 |
| holdout | 1.020 | 14.01% | -15.8% | -0.47% | 0.887 | +0.251 | 73 |

## 4. Bootstrap Confidence Intervals

| Metric | Mean | Std | 5th | 25th | 50th | 75th | 95th |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sharpe | 0.7694 | 0.2813 | 0.3057 | 0.5800 | 0.7793 | 0.9565 | 1.2464 |
| excess_cagr | 0.0126 | 0.0193 | -0.0145 | -0.0020 | 0.0114 | 0.0242 | 0.0497 |

## 5. Coefficient Stability

| Feature | Mean Coef | CV | Change Vol | Change Vol / Mean |
| :--- | ---: | ---: | ---: | ---: |
| inflation_yoy | +0.0559 | 4.10 | 0.0646 | 1.15 |
| inflation_impulse | +0.1579 | 0.88 | 0.0731 | 0.46 |
| unemployment_rate | -0.0405 | 3.37 | 0.0613 | 1.51 |
| credit_spread_level | +0.0642 | 1.56 | 0.0475 | 0.74 |
| credit_spread_3m_change | +0.0867 | 2.34 | 0.0587 | 0.68 |
| real_fed_funds | -0.0219 | 6.96 | 0.0565 | 2.58 |
| yield_curve_slope | -0.0065 | 26.26 | 0.0624 | 9.63 |
| vix_1m_change | -0.0727 | 3.08 | 0.0713 | 0.98 |
| vix_term_structure | -0.0732 | 1.84 | 0.0672 | 0.92 |
| equity_momentum_3m | +0.2228 | 0.72 | 0.0602 | 0.27 |
| equity_vol_3m | -0.0296 | 8.88 | 0.0697 | 2.36 |
| equity_drawdown_from_high | +0.0140 | 8.29 | 0.0509 | 3.64 |