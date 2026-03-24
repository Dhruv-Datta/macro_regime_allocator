"""
Build supervised classification labels.

The question: will equities outperform the risk-free rate over the next
N months? If yes → label 0 (equity). If no → label 1 (safe).

The safe rate is the fed funds rate, representing the return from parking
money in T-bills or money market — the actual opportunity cost of being
in equities.
"""

import os
import pandas as pd
import numpy as np
from config import Config


def compute_forward_returns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute forward N-month returns for equity (price-based) and safe
    (fed funds rate, compounded over the forward window).
    """
    horizon = cfg.forecast_horizon_months
    fwd = pd.DataFrame(index=df.index)

    # Equity: forward price return
    if "equity" not in df.columns:
        raise ValueError("'equity' not found in data columns.")
    fwd["fwd_ret_equity"] = (
        df["equity"].shift(-horizon) / df["equity"] - 1
    ) * 100

    # Safe rate: compound the monthly fed funds rate over the forward window
    rate_col = cfg.safe_rate_series
    if rate_col not in df.columns:
        raise ValueError(f"'{rate_col}' not found in data columns.")

    # Fed funds is annualized percent (e.g. 5.25 = 5.25%). Monthly rate:
    monthly_rate = df[rate_col] / 100 / 12

    # Compound over forward horizon months
    safe_cum = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - horizon):
        cum = 1.0
        for j in range(horizon):
            r = monthly_rate.iloc[i + j]
            if np.isnan(r):
                cum = np.nan
                break
            cum *= (1 + r)
        safe_cum.iloc[i] = (cum - 1) * 100

    fwd["fwd_ret_safe"] = safe_cum

    return fwd


def build_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Build labels: 0 = equity outperformed safe rate, 1 = safe rate won.
    """
    print("Building labels...")

    fwd = compute_forward_returns(df, cfg)

    # Label: did equity beat the safe rate?
    valid = fwd.dropna(how="any")
    labels = (valid["fwd_ret_safe"] >= valid["fwd_ret_equity"]).astype(int)
    labels.name = "label"

    # Equity excess return (for analysis)
    excess = valid["fwd_ret_equity"] - valid["fwd_ret_safe"]
    excess.name = "equity_excess_return"

    labeled = pd.concat([fwd, excess, labels], axis=1)
    labeled = labeled.dropna(subset=["label"])
    labeled["label"] = labeled["label"].astype(int)

    # Print distribution
    dist = labeled["label"].value_counts().sort_index()
    for idx, count in dist.items():
        pct = count / len(labeled) * 100
        print(f"  {cfg.class_labels[idx]:>10s}: {count:4d} ({pct:.1f}%)")

    avg_excess = labeled["equity_excess_return"].mean()
    print(f"  Avg equity excess return ({cfg.forecast_horizon_months}m): {avg_excess:.2f}%")
    print(f"  Labeled dataset shape: {labeled.shape}")

    os.makedirs(cfg.data_dir, exist_ok=True)
    path = os.path.join(cfg.data_dir, "labeled_dataset.csv")
    labeled.to_csv(path)
    print(f"  Saved to {path}")

    return labeled


if __name__ == "__main__":
    cfg = Config()
    df = pd.read_csv(os.path.join(cfg.data_dir, "merged_monthly.csv"),
                     index_col="date", parse_dates=True)
    labeled = build_labels(df, cfg)
    print(labeled.tail())
