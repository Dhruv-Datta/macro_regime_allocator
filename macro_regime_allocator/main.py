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
    parser.add_argument("--model", choices=["logistic", "incremental"], default=None)
    parser.add_argument("--horizon", type=int, default=None,
                        help="Forecast horizon in months (default: 1)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--predict-latest", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    if args.window:
        cfg.window_type = args.window
    if args.model:
        cfg.model_type = args.model
    if args.horizon:
        cfg.forecast_horizon_months = args.horizon

    print("=" * 60)
    print("  MACRO REGIME ALLOCATOR — Equities vs T-Bills")
    print("=" * 60)
    print(f"  Equity proxy:  {cfg.asset_tickers['equity']}")
    print(f"  T-Bills:       fed funds rate")
    print(f"  Horizon:       {cfg.forecast_horizon_months} months")
    print(f"  Window:        {cfg.window_type}")
    print(f"  Model:         {cfg.model_type}")
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

    # Latest prediction
    if args.predict_latest:
        print("\n── Latest Predicted Allocation ────────────────────────────")
        latest_date = features.index[-1]
        latest_features = features.reindex(columns=final_model.feature_names).iloc[[-1]]
        if latest_features.isna().any(axis=None):
            missing = latest_features.columns[latest_features.isna().iloc[0]].tolist()
            raise ValueError(f"Latest features missing: {missing}")

        proba = final_model.predict_proba(latest_features)[0]
        _, weights, _ = probabilities_to_weights(proba, cfg)

        print(f"  As of: {latest_date.strftime('%Y-%m')}")
        print(f"  P(equity beats T-bills): {proba[0]:.3f}")
        print(f"  Final weights:")
        for i, name in enumerate(cfg.asset_classes):
            print(f"    {name:>10s}: {weights[i]:.1%}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Results: {cfg.output_dir}/")
    print(f"  Plots:   {cfg.plot_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
