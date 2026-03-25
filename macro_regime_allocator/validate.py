"""
Robustness validation suite for the macro regime allocator.

Tests whether backtest results survive perturbation of parameters,
removal of components, restriction to subperiods, and statistical
resampling. Designed to detect overfitting and fragile tuning.

Usage:
    python main.py --validate
    python main.py --validate --skip-download
"""

import copy
import os
import itertools
import numpy as np
import pandas as pd
from dataclasses import fields as dc_fields

from config import Config
from backtest import run_backtest


# ── Helpers ────────────────────────────────────────────────────────────────

def _backtest_metrics(bt: pd.DataFrame) -> dict:
    """Extract key metrics from a backtest DataFrame."""
    # Filter out the synthetic start row (NaN port_return)
    bt_valid = bt.dropna(subset=["port_return"])
    if bt_valid.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan,
                "excess_cagr": np.nan, "hit_rate": np.nan, "n_months": 0}

    port = bt_valid["port_return"]
    ew = bt_valid["ew_return"]
    n_months = len(port)
    n_years = n_months / 12

    cum = (1 + port).prod()
    cagr = cum ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = port.std() * np.sqrt(12)
    sharpe = cagr / vol if vol > 0 else 0

    cum_series = (1 + port).cumprod()
    max_dd = (cum_series / cum_series.cummax() - 1).min()

    cum_ew = (1 + ew).prod()
    cagr_ew = cum_ew ** (1 / n_years) - 1 if n_years > 0 else 0
    excess_cagr = cagr - cagr_ew

    hit_rate = (port > 0).mean()

    return {
        "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
        "excess_cagr": excess_cagr, "hit_rate": hit_rate,
        "n_months": n_months,
    }


def _run_variant(features, labels, cfg, name="variant", score_before=None):
    """Run a backtest variant, returning metrics or NaNs on failure.

    If score_before is set (a pd.Timestamp), only score on data before that
    date. The backtest still runs walk-forward through the full period (so
    the model trains correctly), but metrics are computed on the pre-holdout
    slice only.
    """
    try:
        result = run_backtest(features, labels, cfg)
        bt = result["backtest"]
        if score_before is not None:
            bt = bt[bt.index < score_before]
        metrics = _backtest_metrics(bt)
        metrics["name"] = name
        return metrics
    except Exception as e:
        print(f"  WARNING: {name} failed: {e}")
        return {"name": name, "cagr": np.nan, "sharpe": np.nan,
                "max_dd": np.nan, "excess_cagr": np.nan,
                "hit_rate": np.nan, "n_months": 0}


# ── 1. Parameter Sensitivity ──────────────────────────────────────────────

def parameter_sensitivity(features: pd.DataFrame, labels: pd.DataFrame,
                          cfg: Config) -> pd.DataFrame:
    """
    Sweep key parameters one-at-a-time around their default values.
    Shows whether performance is stable to nearby settings.

    When holdout_start is configured, metrics are computed ONLY on the
    pre-holdout period so that parameter selection doesn't peek at holdout.
    """
    print("\n" + "=" * 60)
    print("  PARAMETER SENSITIVITY ANALYSIS")
    score_before = pd.Timestamp(cfg.holdout_start) if cfg.holdout_start else None
    if score_before:
        print(f"  (scoring on pre-holdout data only, before {cfg.holdout_start})")
    print("=" * 60)

    sweeps = {
        "regularization_C":       [0.1, 0.25, 0.5, 1.0, 2.0],
        "allocation_steepness":   [5.0, 8.0, 13.0, 18.0, 25.0],
        "recency_halflife_months": [6, 12, 24, 48],
        "min_train_months":       [36, 48, 60, 72],
        "weight_smoothing_down":  [0.7, 0.85, 0.95, 1.0],
        "weight_smoothing_up":    [0.7, 0.85, 0.95, 1.0],
    }

    rows = []
    baseline = _run_variant(features, labels, cfg, "baseline", score_before)
    baseline["param"] = "baseline"
    baseline["value"] = "-"
    rows.append(baseline)

    for param, values in sweeps.items():
        default_val = getattr(cfg, param)
        print(f"\n  Sweeping {param} (default={default_val}):")
        for val in values:
            variant_cfg = copy.deepcopy(cfg)
            setattr(variant_cfg, param, val)
            name = f"{param}={val}"
            print(f"    {name}...", end=" ", flush=True)
            m = _run_variant(features, labels, variant_cfg, name, score_before)
            m["param"] = param
            m["value"] = val
            is_default = " (DEFAULT)" if val == default_val else ""
            print(f"Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.3%}{is_default}")
            rows.append(m)

    df = pd.DataFrame(rows)
    return df


# ── 2. Ablation Studies ──────────────────────────────────────────────────

def ablation_studies(features: pd.DataFrame, labels: pd.DataFrame,
                     cfg: Config) -> pd.DataFrame:
    """
    Test each component in isolation and combination:
    - Full system (baseline)
    - No crash overlay
    - No smoothing
    - No recency weighting (flat halflife)
    - Different baseline weights (50/50, 60/40, 75/25, 95/5)
    - Model-only (no overlay, no smoothing)

    When holdout_start is configured, metrics are computed ONLY on the
    pre-holdout period so that component selection doesn't peek at holdout.
    """
    print("\n" + "=" * 60)
    print("  ABLATION STUDIES")
    score_before = pd.Timestamp(cfg.holdout_start) if cfg.holdout_start else None
    if score_before:
        print(f"  (scoring on pre-holdout data only, before {cfg.holdout_start})")
    print("=" * 60)

    variants = {}

    # Baseline
    variants["full_system"] = cfg

    # No crash overlay
    no_overlay = copy.deepcopy(cfg)
    no_overlay.crash_overlay = False
    variants["no_crash_overlay"] = no_overlay

    # No smoothing
    no_smooth = copy.deepcopy(cfg)
    no_smooth.weight_smoothing_up = 1.0
    no_smooth.weight_smoothing_down = 1.0
    variants["no_smoothing"] = no_smooth

    # Flat recency (very long halflife = effectively uniform)
    flat_recency = copy.deepcopy(cfg)
    flat_recency.recency_halflife_months = 999
    variants["no_recency_weighting"] = flat_recency

    # Model only: no overlay, no smoothing
    model_only = copy.deepcopy(cfg)
    model_only.crash_overlay = False
    model_only.weight_smoothing_up = 1.0
    model_only.weight_smoothing_down = 1.0
    variants["model_only"] = model_only

    # Different baseline weights
    for eq_pct in [50, 60, 75, 95]:
        v = copy.deepcopy(cfg)
        v.equal_weight = [eq_pct / 100, 1 - eq_pct / 100]
        variants[f"baseline_{eq_pct}_{100-eq_pct}"] = v

    rows = []
    for name, variant_cfg in variants.items():
        print(f"  {name}...", end=" ", flush=True)
        m = _run_variant(features, labels, variant_cfg, name, score_before)
        print(f"Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.3%}, MaxDD={m['max_dd']:.1%}")
        rows.append(m)

    return pd.DataFrame(rows)


# ── 3. Subperiod Analysis ────────────────────────────────────────────────

def subperiod_analysis(features: pd.DataFrame, labels: pd.DataFrame,
                       cfg: Config) -> pd.DataFrame:
    """
    Break results into subperiods to check consistency:
    - By decade
    - Excluding 2008 crisis
    - Excluding 2020 COVID crash
    - Excluding both
    """
    print("\n" + "=" * 60)
    print("  SUBPERIOD ANALYSIS")
    print("=" * 60)

    # Run the full backtest once
    result = run_backtest(features, labels, cfg)
    bt = result["backtest"].dropna(subset=["port_return"])

    rows = []

    # Full period
    m = _backtest_metrics(bt)
    m["name"] = "full_period"
    m["period"] = f"{bt.index[0].strftime('%Y-%m')} to {bt.index[-1].strftime('%Y-%m')}"
    rows.append(m)

    # By decade
    for decade_start in range(2000, 2030, 10):
        decade_end = decade_start + 10
        mask = (bt.index.year >= decade_start) & (bt.index.year < decade_end)
        sub = bt[mask]
        if len(sub) < 12:
            continue
        m = _backtest_metrics(sub)
        m["name"] = f"{decade_start}s"
        m["period"] = f"{decade_start}-{decade_end}"
        print(f"  {m['name']}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.3%}")
        rows.append(m)

    # Excluding crisis periods
    exclusions = {
        "ex_2008_crisis": lambda idx: ~((idx.year == 2008) | (idx.year == 2009)),
        "ex_2020_covid": lambda idx: ~((idx.year == 2020) & (idx.month <= 6)),
        "ex_both_crises": lambda idx: ~(
            (idx.year == 2008) | (idx.year == 2009) |
            ((idx.year == 2020) & (idx.month <= 6))
        ),
    }

    for name, mask_fn in exclusions.items():
        sub = bt[mask_fn(bt.index)]
        if len(sub) < 12:
            continue
        m = _backtest_metrics(sub)
        m["name"] = name
        m["period"] = f"full minus exclusion ({len(bt) - len(sub)} months removed)"
        print(f"  {name}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.3%}")
        rows.append(m)

    # Holdout split if configured
    if cfg.holdout_start:
        holdout_ts = pd.Timestamp(cfg.holdout_start)
        in_sample = bt[bt.index < holdout_ts]
        holdout = bt[bt.index >= holdout_ts]
        for label, sub in [("in_sample", in_sample), ("holdout", holdout)]:
            if len(sub) < 6:
                continue
            m = _backtest_metrics(sub)
            m["name"] = label
            m["period"] = f"{sub.index[0].strftime('%Y-%m')} to {sub.index[-1].strftime('%Y-%m')}"
            print(f"  {label}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.3%}")
            rows.append(m)

    return pd.DataFrame(rows)


# ── 4. Bootstrap Confidence Intervals ────────────────────────────────────

def bootstrap_confidence(features: pd.DataFrame, labels: pd.DataFrame,
                         cfg: Config, n_bootstrap: int = 1000,
                         block_size: int = 12) -> pd.DataFrame:
    """
    Block bootstrap on monthly returns to estimate confidence intervals
    for Sharpe ratio and excess CAGR. Uses circular block bootstrap
    to preserve autocorrelation structure.
    """
    print("\n" + "=" * 60)
    print("  BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"  ({n_bootstrap} iterations, block_size={block_size} months)")
    print("=" * 60)

    result = run_backtest(features, labels, cfg)
    bt = result["backtest"].dropna(subset=["port_return"])
    score_before = pd.Timestamp(cfg.holdout_start) if cfg.holdout_start else None
    if score_before:
        print(f"  (resampling pre-holdout returns only, before {cfg.holdout_start})")
        bt = bt[bt.index < score_before]
    port_returns = bt["port_return"].values
    ew_returns = bt["ew_return"].values
    n = len(port_returns)

    rng = np.random.default_rng(42)
    sharpes = []
    excess_cagrs = []

    for _ in range(n_bootstrap):
        # Circular block bootstrap
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_size) % n for s in starts
        ])[:n]

        boot_port = port_returns[indices]
        boot_ew = ew_returns[indices]

        n_years = n / 12
        cum_p = (1 + boot_port).prod()
        cagr_p = cum_p ** (1 / n_years) - 1 if n_years > 0 else 0
        vol_p = boot_port.std() * np.sqrt(12)
        sharpe_p = cagr_p / vol_p if vol_p > 0 else 0

        cum_ew = (1 + boot_ew).prod()
        cagr_ew = cum_ew ** (1 / n_years) - 1 if n_years > 0 else 0

        sharpes.append(sharpe_p)
        excess_cagrs.append(cagr_p - cagr_ew)

    sharpes = np.array(sharpes)
    excess_cagrs = np.array(excess_cagrs)

    ci_levels = [2.5, 5, 25, 50, 75, 95, 97.5]
    rows = []
    for metric_name, values in [("sharpe", sharpes), ("excess_cagr", excess_cagrs)]:
        row = {"metric": metric_name, "mean": values.mean(), "std": values.std()}
        for ci in ci_levels:
            row[f"p{ci:.0f}" if ci == int(ci) else f"p{ci}"] = np.percentile(values, ci)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Key results
    sharpe_ci = (np.percentile(sharpes, 5), np.percentile(sharpes, 95))
    excess_ci = (np.percentile(excess_cagrs, 5), np.percentile(excess_cagrs, 95))
    pct_positive_sharpe = (sharpes > 0).mean() * 100
    pct_positive_excess = (excess_cagrs > 0).mean() * 100

    print(f"\n  Sharpe ratio:")
    print(f"    Mean:     {sharpes.mean():.3f}")
    print(f"    90% CI:   [{sharpe_ci[0]:.3f}, {sharpe_ci[1]:.3f}]")
    print(f"    P(>0):    {pct_positive_sharpe:.1f}%")

    print(f"\n  Excess CAGR vs baseline:")
    print(f"    Mean:     {excess_cagrs.mean():.3%}")
    print(f"    90% CI:   [{excess_ci[0]:.3%}, {excess_ci[1]:.3%}]")
    print(f"    P(>0):    {pct_positive_excess:.1f}%")

    return df


# ── 5. Walk-Forward Stability ─────────────────────────────────────────────

def coefficient_stability(features: pd.DataFrame, labels: pd.DataFrame,
                          cfg: Config) -> pd.DataFrame:
    """
    Track how model coefficients evolve over the walk-forward window.
    Stable coefficients suggest the model is learning real signal, not noise.
    """
    from model import RegimeClassifier
    from backtest import _recency_weights, _class_balanced_weights

    print("\n" + "=" * 60)
    print("  COEFFICIENT STABILITY OVER TIME")
    print("=" * 60)

    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx].copy()
    labels = labels.loc[common_idx].copy()
    valid_mask = features.notna().all(axis=1)
    features, labels = features.loc[valid_mask], labels.loc[valid_mask]

    X, y = features, labels["label"]
    all_dates = X.index.tolist()
    horizon = cfg.forecast_horizon_months

    coef_records = []

    # Sample every 12 months to keep it manageable
    sample_points = range(cfg.min_train_months, len(all_dates) - horizon, 12)

    for i in sample_points:
        train_end = i - horizon
        if train_end < 1:
            continue

        train_start = 0 if cfg.window_type == "expanding" else max(0, train_end - cfg.rolling_window_months)
        train_idx = all_dates[train_start:train_end]
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]

        if len(y_train) < cfg.min_train_months or y_train.nunique() < 2:
            continue

        sw = _recency_weights(len(train_idx), cfg.recency_halflife_months)
        cw = _class_balanced_weights(y_train.values)
        model = RegimeClassifier(cfg)
        model.fit(X_train, y_train, sample_weight=sw * cw)

        coefs = model.classifier.coef_[0]
        record = {"date": all_dates[i], "train_size": len(train_idx)}
        for fname, cval in zip(X.columns, coefs):
            record[fname] = cval
        coef_records.append(record)

    df = pd.DataFrame(coef_records)
    if not df.empty:
        df = df.set_index("date")
        feature_cols = [c for c in df.columns if c != "train_size"]
        print(f"\n  Tracked {len(df)} refit points across {len(feature_cols)} features")
        print(f"\n  Coefficient std over time (lower = more stable):")
        for col in feature_cols:
            std = df[col].std()
            mean = df[col].mean()
            cv = abs(std / mean) if mean != 0 else float("inf")
            sign_changes = ((df[col].shift(1) * df[col]) < 0).sum()
            print(f"    {col:>30s}: std={std:.4f}, cv={cv:.2f}, sign_flips={sign_changes}")

    return df


# ── Master Runner ─────────────────────────────────────────────────────────

def run_validation(features: pd.DataFrame, labels: pd.DataFrame,
                   cfg: Config) -> dict:
    """Run the full validation suite and save results."""
    output_dir = os.path.join(cfg.output_dir, "validation")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 1. Parameter sensitivity
    print("\n\n" + "#" * 60)
    print("  1/5  PARAMETER SENSITIVITY")
    print("#" * 60)
    sens = parameter_sensitivity(features, labels, cfg)
    sens.to_csv(os.path.join(output_dir, "parameter_sensitivity.csv"), index=False)
    results["sensitivity"] = sens

    # 2. Ablation studies
    print("\n\n" + "#" * 60)
    print("  2/5  ABLATION STUDIES")
    print("#" * 60)
    abl = ablation_studies(features, labels, cfg)
    abl.to_csv(os.path.join(output_dir, "ablation_studies.csv"), index=False)
    results["ablation"] = abl

    # 3. Subperiod analysis
    print("\n\n" + "#" * 60)
    print("  3/5  SUBPERIOD ANALYSIS")
    print("#" * 60)
    sub = subperiod_analysis(features, labels, cfg)
    sub.to_csv(os.path.join(output_dir, "subperiod_analysis.csv"), index=False)
    results["subperiod"] = sub

    # 4. Bootstrap confidence intervals
    print("\n\n" + "#" * 60)
    print("  4/5  BOOTSTRAP CONFIDENCE")
    print("#" * 60)
    boot = bootstrap_confidence(features, labels, cfg)
    boot.to_csv(os.path.join(output_dir, "bootstrap_confidence.csv"), index=False)
    results["bootstrap"] = boot

    # 5. Coefficient stability
    print("\n\n" + "#" * 60)
    print("  5/5  COEFFICIENT STABILITY")
    print("#" * 60)
    coef = coefficient_stability(features, labels, cfg)
    coef.to_csv(os.path.join(output_dir, "coefficient_stability.csv"))
    results["coefficients"] = coef

    # Summary
    _print_summary(results)
    _save_validation_report(results, output_dir)

    return results


def _print_summary(results: dict):
    """Print a concise robustness verdict."""
    print("\n\n" + "=" * 60)
    print("  ROBUSTNESS SUMMARY")
    print("=" * 60)

    warnings = []
    passes = []

    # Check sensitivity: does Sharpe survive parameter perturbation?
    sens = results["sensitivity"]
    sens_valid = sens[sens["param"] != "baseline"].dropna(subset=["sharpe"])
    if len(sens_valid) > 0:
        pct_positive = (sens_valid["sharpe"] > 0).mean() * 100
        if pct_positive < 70:
            warnings.append(f"Sharpe positive in only {pct_positive:.0f}% of parameter variants")
        else:
            passes.append(f"Sharpe positive in {pct_positive:.0f}% of parameter variants")

    # Check ablation: does model add value vs simplest baseline?
    abl = results["ablation"]
    full = abl[abl["name"] == "full_system"]
    model_only = abl[abl["name"] == "model_only"]
    if not full.empty and not model_only.empty:
        if model_only.iloc[0]["sharpe"] > 0:
            passes.append("Model-only variant has positive Sharpe")
        else:
            warnings.append("Model-only variant has negative Sharpe — edge may come from overlay/smoothing")

    # Check subperiod consistency
    sub = results["subperiod"]
    decades = sub[sub["name"].str.contains("0s$", na=False)]
    if len(decades) > 1:
        all_positive = (decades["excess_cagr"] > 0).all()
        if all_positive:
            passes.append("Excess CAGR positive across all decades")
        else:
            neg_decades = decades[decades["excess_cagr"] <= 0]["name"].tolist()
            warnings.append(f"Negative excess CAGR in: {', '.join(neg_decades)}")

    # Check bootstrap
    boot = results["bootstrap"]
    sharpe_row = boot[boot["metric"] == "sharpe"]
    if not sharpe_row.empty:
        p5 = sharpe_row.iloc[0].get("p5", np.nan)
        if not np.isnan(p5) and p5 > 0:
            passes.append(f"Sharpe 90% CI lower bound > 0 ({p5:.3f})")
        elif not np.isnan(p5):
            warnings.append(f"Sharpe 90% CI includes zero (lower bound = {p5:.3f})")

    # Check coefficient stability
    coefs = results["coefficients"]
    if not coefs.empty:
        feature_cols = [c for c in coefs.columns if c != "train_size"]
        sign_flips = sum(
            ((coefs[c].shift(1) * coefs[c]) < 0).sum() for c in feature_cols
        )
        total_points = len(coefs) * len(feature_cols)
        flip_rate = sign_flips / total_points if total_points > 0 else 0
        if flip_rate > 0.15:
            warnings.append(f"High coefficient instability: {flip_rate:.0%} sign flip rate")
        else:
            passes.append(f"Coefficient sign flip rate: {flip_rate:.0%}")

    print("\n  PASSES:")
    for p in passes:
        print(f"    [+] {p}")

    print("\n  WARNINGS:")
    if warnings:
        for w in warnings:
            print(f"    [!] {w}")
    else:
        print("    None")

    verdict = "ROBUST" if len(warnings) == 0 else (
        "PARTIALLY ROBUST" if len(warnings) <= 2 else "FRAGILE"
    )
    print(f"\n  VERDICT: {verdict}")
    print(f"    {len(passes)} passes, {len(warnings)} warnings")


def _save_validation_report(results: dict, output_dir: str):
    """Save a markdown summary of all validation results."""
    from datetime import datetime
    lines = []
    w = lines.append

    w("# Robustness Validation Report")
    w(f"\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # Sensitivity
    w("## 1. Parameter Sensitivity")
    w("")
    sens = results["sensitivity"]
    w("| Param | Value | Sharpe | CAGR | Max DD | Excess CAGR |")
    w("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in sens.iterrows():
        w(f"| {row.get('param', row['name'])} | {row.get('value', '-')} | "
          f"{row['sharpe']:.3f} | {row['cagr']:.2%} | {row['max_dd']:.1%} | "
          f"{row['excess_cagr']:.2%} |")
    w("")

    # Ablation
    w("## 2. Ablation Studies")
    w("")
    abl = results["ablation"]
    w("| Variant | Sharpe | CAGR | Max DD | Excess CAGR |")
    w("| :--- | ---: | ---: | ---: | ---: |")
    for _, row in abl.iterrows():
        w(f"| {row['name']} | {row['sharpe']:.3f} | {row['cagr']:.2%} | "
          f"{row['max_dd']:.1%} | {row['excess_cagr']:.2%} |")
    w("")

    # Subperiod
    w("## 3. Subperiod Analysis")
    w("")
    sub = results["subperiod"]
    w("| Period | Sharpe | CAGR | Max DD | Excess CAGR | Months |")
    w("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in sub.iterrows():
        w(f"| {row['name']} | {row['sharpe']:.3f} | {row['cagr']:.2%} | "
          f"{row['max_dd']:.1%} | {row['excess_cagr']:.2%} | {row['n_months']} |")
    w("")

    # Bootstrap
    w("## 4. Bootstrap Confidence Intervals")
    w("")
    boot = results["bootstrap"]
    w("| Metric | Mean | Std | 5th | 25th | 50th | 75th | 95th |")
    w("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in boot.iterrows():
        w(f"| {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} | "
          f"{row.get('p5', np.nan):.4f} | {row.get('p25', np.nan):.4f} | "
          f"{row.get('p50', np.nan):.4f} | {row.get('p75', np.nan):.4f} | "
          f"{row.get('p95', np.nan):.4f} |")
    w("")

    # Coefficient stability
    w("## 5. Coefficient Stability")
    w("")
    coefs = results["coefficients"]
    if not coefs.empty:
        feature_cols = [c for c in coefs.columns if c != "train_size"]
        w("| Feature | Mean Coef | Std | CV | Sign Flips |")
        w("| :--- | ---: | ---: | ---: | ---: |")
        for col in feature_cols:
            mean = coefs[col].mean()
            std = coefs[col].std()
            cv = abs(std / mean) if mean != 0 else float("inf")
            flips = ((coefs[col].shift(1) * coefs[col]) < 0).sum()
            w(f"| {col} | {mean:.4f} | {std:.4f} | {cv:.2f} | {flips} |")

    report_path = os.path.join(output_dir, "validation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Validation report saved to {report_path}")
