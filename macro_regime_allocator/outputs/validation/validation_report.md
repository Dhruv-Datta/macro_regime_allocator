# Robustness Validation Report

*Generated 2026-03-25 13:48*

## 1. Parameter Sensitivity

| Param | Value | Sharpe | CAGR | Max DD | Excess CAGR |
| :--- | ---: | ---: | ---: | ---: | ---: |
| baseline | - | 0.762 | 8.32% | -27.3% | 1.32% |
| regularization_C | 0.1 | 0.726 | 8.22% | -31.8% | 1.22% |
| regularization_C | 0.25 | 0.757 | 8.37% | -28.3% | 1.37% |
| regularization_C | 0.5 | 0.762 | 8.32% | -27.3% | 1.32% |
| regularization_C | 1.0 | 0.751 | 8.13% | -27.1% | 1.13% |
| regularization_C | 2.0 | 0.737 | 7.87% | -26.9% | 0.86% |
| allocation_steepness | 5.0 | 0.643 | 7.82% | -37.6% | 0.82% |
| allocation_steepness | 8.0 | 0.698 | 8.12% | -33.1% | 1.12% |
| allocation_steepness | 13.0 | 0.762 | 8.32% | -27.3% | 1.32% |
| allocation_steepness | 18.0 | 0.759 | 8.02% | -27.2% | 1.02% |
| allocation_steepness | 25.0 | 0.748 | 7.68% | -26.2% | 0.67% |
| recency_halflife_months | 6 | 0.784 | 8.37% | -28.5% | 1.37% |
| recency_halflife_months | 12 | 0.762 | 8.32% | -27.3% | 1.32% |
| recency_halflife_months | 24 | 0.749 | 8.30% | -27.6% | 1.30% |
| recency_halflife_months | 48 | 0.739 | 8.26% | -28.2% | 1.26% |
| min_train_months | 36 | 0.685 | 7.62% | -34.0% | 1.55% |
| min_train_months | 48 | 0.762 | 8.32% | -27.3% | 1.32% |
| min_train_months | 60 | 0.855 | 9.43% | -27.3% | 1.16% |
| min_train_months | 72 | 0.994 | 10.44% | -27.3% | 1.15% |
| weight_smoothing_down | 0.7 | 0.735 | 8.17% | -29.4% | 1.17% |
| weight_smoothing_down | 0.85 | 0.752 | 8.27% | -28.1% | 1.27% |
| weight_smoothing_down | 0.95 | 0.762 | 8.32% | -27.3% | 1.32% |
| weight_smoothing_down | 1.0 | 0.766 | 8.35% | -26.9% | 1.34% |
| weight_smoothing_up | 0.7 | 0.796 | 8.44% | -25.2% | 1.44% |
| weight_smoothing_up | 0.85 | 0.777 | 8.39% | -25.7% | 1.39% |
| weight_smoothing_up | 0.95 | 0.766 | 8.34% | -27.0% | 1.34% |
| weight_smoothing_up | 1.0 | 0.760 | 8.31% | -27.6% | 1.31% |

## 2. Ablation Studies

| Variant | Sharpe | CAGR | Max DD | Excess CAGR |
| :--- | ---: | ---: | ---: | ---: |
| full_system | 0.762 | 8.32% | -27.3% | 1.32% |
| no_crash_overlay | 0.726 | 8.18% | -31.7% | 1.18% |
| no_smoothing | 0.764 | 8.33% | -27.2% | 1.33% |
| no_recency_weighting | 0.701 | 8.02% | -29.3% | 1.02% |
| model_only | 0.726 | 8.18% | -31.6% | 1.18% |
| baseline_50_50 | 0.740 | 6.33% | -21.4% | 1.73% |
| baseline_60_40 | 0.745 | 6.67% | -22.5% | 1.51% |
| baseline_75_25 | 0.752 | 7.22% | -24.4% | 1.24% |
| baseline_95_5 | 0.762 | 8.32% | -27.3% | 1.32% |

## 3. Subperiod Analysis

| Period | Sharpe | CAGR | Max DD | Excess CAGR | Months |
| :--- | ---: | ---: | ---: | ---: | ---: |
| full_period | 0.831 | 9.70% | -27.3% | 0.90% | 296 |
| 2000s | 0.450 | 5.05% | -27.3% | 4.43% | 103 |
| 2010s | 1.054 | 11.21% | -13.2% | -1.59% | 120 |
| 2020s | 1.020 | 14.01% | -15.8% | -0.47% | 73 |
| ex_2008_crisis | 0.880 | 9.98% | -26.0% | -0.65% | 272 |
| ex_2020_covid | 0.872 | 10.01% | -27.3% | 0.89% | 290 |
| ex_both_crises | 0.929 | 10.33% | -26.0% | -0.70% | 266 |
| in_sample | 0.762 | 8.32% | -27.3% | 1.32% | 223 |
| holdout | 1.020 | 14.01% | -15.8% | -0.47% | 73 |

## 4. Bootstrap Confidence Intervals

| Metric | Mean | Std | 5th | 25th | 50th | 75th | 95th |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sharpe | 0.7694 | 0.2813 | 0.3057 | 0.5800 | 0.7793 | 0.9565 | 1.2464 |
| excess_cagr | 0.0126 | 0.0193 | -0.0145 | -0.0020 | 0.0114 | 0.0242 | 0.0497 |

## 5. Coefficient Stability

| Feature | Mean Coef | Std | CV | Sign Flips |
| :--- | ---: | ---: | ---: | ---: |
| inflation_yoy | 0.0683 | 0.2206 | 3.23 | 7 |
| inflation_impulse | 0.1586 | 0.1447 | 0.91 | 5 |
| unemployment_rate | -0.0581 | 0.1351 | 2.32 | 8 |
| credit_spread_level | 0.0627 | 0.0906 | 1.44 | 9 |
| credit_spread_3m_change | 0.1120 | 0.2010 | 1.79 | 4 |
| real_fed_funds | -0.0215 | 0.1307 | 6.08 | 6 |
| yield_curve_slope | -0.0081 | 0.1764 | 21.89 | 9 |
| vix_1m_change | -0.0747 | 0.2642 | 3.54 | 9 |
| vix_term_structure | -0.0655 | 0.1347 | 2.06 | 9 |
| equity_momentum_3m | 0.2248 | 0.1846 | 0.82 | 4 |
| equity_vol_3m | -0.0178 | 0.2462 | 13.83 | 3 |
| equity_drawdown_from_high | 0.0091 | 0.1186 | 13.09 | 6 |