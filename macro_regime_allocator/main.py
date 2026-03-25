#!/usr/bin/env python3
"""
Macro Regime Allocator — Equities vs T-Bills

Predicts whether equities will outperform the risk-free rate (fed funds)
over the next N months. Converts prediction into equity/cash weights.

Usage:
    python main.py
    python main.py --predict-latest
    python main.py --model incremental --window rolling
"""

import argparse
import sys
import os
import warnings
import pandas as pd

from config import Config
from data import load_data, engineer_features, build_labels
from backtest import run_backtest, probabilities_to_weights
from results import evaluate, generate_all_plots

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Macro Regime Allocator")
    parser.add_argument("--window", choices=["expanding", "rolling"], default=None)
    parser.add_argument("--horizon", type=int, default=None,
                        help="Forecast horizon in months (default: 1)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--predict-latest", action="store_true")
    parser.add_argument("--validate", action="store_true",
                        help="Run robustness validation suite")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    if args.window:
        cfg.window_type = args.window
    if args.horizon:
        cfg.forecast_horizon_months = args.horizon

    print("=" * 60)
    print("  MACRO REGIME ALLOCATOR — Equities vs T-Bills")
    print("=" * 60)
    print(f"  Equity proxy:  {cfg.asset_tickers['equity']}")
    print(f"  T-Bills:       fed funds rate")
    print(f"  Horizon:       {cfg.forecast_horizon_months} months")
    print(f"  Window:        {cfg.window_type}")
    print(f"  Start date:    {cfg.start_date}  (data from {cfg.data_start_date})")
    print(f"  Min train:     {cfg.min_train_months} months (auto-backdated)")
    print("=" * 60)

    # Step 1: Load data
    if args.skip_download:
        print("\nLoading cached data...")
        merged_path = os.path.join(cfg.data_dir, "merged_monthly.csv")
        if not os.path.exists(merged_path):
            print(f"ERROR: {merged_path} not found. Run without --skip-download first.")
            sys.exit(1)
        merged = pd.read_csv(merged_path, index_col="date", parse_dates=True)
    else:
        print("\n── Step 1: Load Data ──────────────────────────────────────")
        merged = load_data(cfg)

    # Step 2: Features
    print("\n── Step 2: Feature Engineering ────────────────────────────")
    features = engineer_features(merged, cfg)

    # Step 3: Labels
    print("\n── Step 3: Build Labels ──────────────────────────────────")
    labels = build_labels(merged, cfg)

    # Step 4: Backtest
    print("\n── Step 4: Walk-Forward Backtest ──────────────────────────")
    results = run_backtest(features, labels, cfg)
    bt = results["backtest"]
    final_model = results["final_model"]

    # Step 5: Evaluate
    print("\n── Step 5: Evaluation ────────────────────────────────────")
    eval_results = evaluate(bt, final_model, cfg)

    # Step 6: Plots
    print("\n── Step 6: Generate Plots ────────────────────────────────")
    generate_all_plots(bt, eval_results, cfg)

    # Robustness validation suite
    if args.validate:
        from validate import run_validation
        print("\n── Step 7: Robustness Validation ─────────────────────────")
        run_validation(features, labels, cfg)

    # Latest prediction
    if args.predict_latest:
        from backtest import _gather_market_data
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║               CURRENT ALLOCATION SIGNAL                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        latest_date = features.index[-1]
        latest_features = features.reindex(columns=final_model.feature_names).iloc[[-1]]
        if latest_features.isna().any(axis=None):
            missing = latest_features.columns[latest_features.isna().iloc[0]].tolist()
            raise ValueError(f"Latest features missing: {missing}")

        proba = final_model.predict_proba(latest_features)[0]

        # Apply crash overlay with current market data
        market_data = _gather_market_data(merged, latest_date)
        raw_proba, weights, overlay_reason = probabilities_to_weights(proba, cfg, market_data)

        # Apply smoothing against last backtest weight if available
        if len(bt) > 0:
            prev_eq = bt["weight_equity"].iloc[-1]
            target_eq = weights[0]
            alpha = cfg.weight_smoothing_up if target_eq >= prev_eq else cfg.weight_smoothing_down
            import numpy as np
            smoothed_eq = np.clip(alpha * target_eq + (1 - alpha) * prev_eq,
                                  cfg.min_weight, cfg.max_weight)
            weights = np.array([smoothed_eq, 1.0 - smoothed_eq])

        print(f"  As of:                   {latest_date.strftime('%Y-%m')}")
        print(f"  P(equity beats T-bills): {proba[0]:.3f}")
        print(f"  P(T-bills win):          {proba[1]:.3f}")
        print(f"  Crash overlay:           {overlay_reason}")
        if market_data:
            print(f"  Market signals:")
            for k, v in sorted(market_data.items()):
                print(f"    {k:>28s}: {v:+.2f}")
        print(f"")
        print(f"  ┌─────────────────────────────────┐")
        print(f"  │  RECOMMENDED ALLOCATION          │")
        for i, name in enumerate(cfg.asset_classes):
            bar_len = int(weights[i] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  │  {name:>8s}: {weights[i]:6.1%}  {bar} │")
        print(f"  └─────────────────────────────────┘")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Results: {cfg.output_dir}/")
    print(f"  Plots:   {cfg.plot_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
