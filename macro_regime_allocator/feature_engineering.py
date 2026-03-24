"""
Convert raw merged data into model features.
All features are lagged to prevent lookahead bias.

Features focused on: when should you be in equities vs safe rate?
Heavy emphasis on crash detection signals (VIX, drawdowns, credit).
"""

import os
import pandas as pd
import numpy as np
from config import Config


def compute_inflation_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    cpi_col = "cpi" if "cpi" in df.columns else "core_cpi"
    if cpi_col not in df.columns:
        return feats

    cpi = df[cpi_col]
    feats["inflation_yoy"] = cpi.pct_change(12) * 100
    inflation_3m = ((cpi / cpi.shift(3)) ** 4 - 1) * 100
    feats["inflation_impulse"] = inflation_3m - feats["inflation_yoy"]

    return feats


def compute_labor_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    if "unemployment" not in df.columns:
        return feats

    feats["unemployment_rate"] = df["unemployment"]

    return feats


def compute_rates_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    if "credit_spread" in df.columns:
        feats["credit_spread_level"] = df["credit_spread"]
        # Credit spread momentum — widening spreads = danger
        feats["credit_spread_3m_change"] = df["credit_spread"].diff(3)

    # Real fed funds rate
    if "fed_funds" in df.columns:
        cpi_col = "cpi" if "cpi" in df.columns else "core_cpi"
        if cpi_col in df.columns:
            inflation = df[cpi_col].pct_change(12) * 100
            feats["real_fed_funds"] = df["fed_funds"] - inflation

    # Yield curve slope: 10Y - 2Y (classic recession predictor)
    # Inverted curve (negative) = recession ahead = get defensive
    if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
        feats["yield_curve_slope"] = df["treasury_10y"] - df["treasury_2y"]

    return feats


def compute_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIX-based features for crash detection.

    VIX level: high VIX = fear, but often a contrarian buy signal
    VIX momentum: rapidly rising VIX = active crash
    VIX term structure: VIX > VIX3M (backwardation) = panic/crash mode
                        VIX < VIX3M (contango) = calm, normal
    """
    feats = pd.DataFrame(index=df.index)

    if "vix" not in df.columns:
        print("  WARNING: No VIX data, skipping VIX features")
        return feats

    # VIX 1-month change (spike = crash unfolding)
    feats["vix_1m_change"] = df["vix"].diff(1)

    # VIX term structure: VIX / VIX3M ratio
    # > 1 = backwardation (panic), < 1 = contango (calm)
    if "vix3m" in df.columns:
        feats["vix_term_structure"] = df["vix"] / df["vix3m"]
        # VIX3M only starts ~2006-07; fill earlier dates with 1.0 (neutral/contango)
        feats["vix_term_structure"] = feats["vix_term_structure"].fillna(1.0)
    else:
        print("  WARNING: No VIX3M data, defaulting term structure to 1.0 (neutral)")
        feats["vix_term_structure"] = 1.0

    return feats


def compute_equity_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Equity momentum, volatility, and drawdown features."""
    feats = pd.DataFrame(index=df.index)

    if "equity" not in df.columns:
        return feats

    monthly_ret = df["equity"].pct_change()

    # Momentum
    feats["equity_momentum_3m"] = df["equity"].pct_change(cfg.momentum_window) * 100

    # Volatility
    feats["equity_vol_3m"] = (
        monthly_ret.rolling(cfg.volatility_window).std() * np.sqrt(12) * 100
    )

    # Drawdown from 12-month high — how far are we from the peak?
    # Deep drawdown = potential crash or deep value
    rolling_high = df["equity"].rolling(12).max()
    feats["equity_drawdown_from_high"] = (
        (df["equity"] / rolling_high - 1) * 100
    )

    return feats


def apply_lag(features: pd.DataFrame, lag: int) -> pd.DataFrame:
    return features.shift(lag)


def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build the full feature matrix. All features lagged by macro_lag_months."""
    print("Engineering features...")

    inflation = compute_inflation_features(df)
    labor = compute_labor_features(df)
    rates = compute_rates_features(df)
    vix = compute_vix_features(df)
    equity = compute_equity_features(df, cfg)

    features = pd.concat([inflation, labor, rates, vix, equity], axis=1)
    features = apply_lag(features, cfg.macro_lag_months)
    features = features.dropna(how="all")

    print(f"  Feature columns ({len(features.columns)}): {list(features.columns)}")
    print(f"  Feature matrix shape: {features.shape}")

    os.makedirs(cfg.data_dir, exist_ok=True)
    path = os.path.join(cfg.data_dir, "features.csv")
    features.to_csv(path)
    print(f"  Saved to {path}")

    return features
