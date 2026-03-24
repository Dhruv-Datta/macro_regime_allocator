"""
Classification and investment performance evaluation.
Equities vs safe rate framing.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from config import Config
from model import RegimeClassifier


def direction_accuracy(bt: pd.DataFrame, cfg: Config) -> dict:
    """
    Did the model tilt toward equities when equities actually beat the
    safe rate, and toward safe when equities underperformed?
    """
    valid = bt.dropna(subset=["ret_equity", "ret_safe"]).copy()

    equity_won = valid["ret_equity"] > valid["ret_safe"]
    model_favors_equity = valid["weight_equity"] > 0.5

    correct = equity_won == model_favors_equity
    accuracy = correct.mean()

    # Weighted by how much the call was worth
    spread = (valid["ret_equity"] - valid["ret_safe"]).abs()
    if spread.sum() > 0:
        weighted = (correct * spread).sum() / spread.sum()
    else:
        weighted = 0.5

    # How much excess return did we capture?
    excess = valid["ret_equity"] - valid["ret_safe"]
    port_excess = valid["port_return"] - valid["ret_safe"]
    equity_up_months = excess > 0
    if equity_up_months.any():
        upside_capture = (
            port_excess[equity_up_months].sum() / excess[equity_up_months].sum()
        )
    else:
        upside_capture = 0
    equity_down_months = excess < 0
    if equity_down_months.any():
        downside_capture = (
            port_excess[equity_down_months].sum() / excess[equity_down_months].sum()
        )
    else:
        downside_capture = 0

    print("\n── Direction Accuracy ──────────────────────────────────────")
    print(f"  Direction accuracy:          {accuracy:.3f}")
    print(f"    (tilted toward the 1-month winner)")
    print(f"  Magnitude-weighted accuracy: {weighted:.3f}")
    print(f"    (weighted by size of equity vs safe spread)")
    print(f"  Upside capture:             {upside_capture:.3f}")
    print(f"    (fraction of equity gains captured when equity won)")
    print(f"  Downside capture:           {downside_capture:.3f}")
    print(f"    (fraction of equity losses taken when equity lost)")
    print(f"    (lower is better — means you avoided losses)")

    return {
        "direction_accuracy": accuracy,
        "weighted_accuracy": weighted,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
    }


def classification_metrics(bt: pd.DataFrame, cfg: Config) -> dict:
    """Standard classification metrics against the 3-month label."""
    valid = bt.dropna(subset=["pred_class", "actual_label"])
    y_true = valid["actual_label"].astype(int)
    y_pred = valid["pred_class"].astype(int)

    class_names = [cfg.class_labels[i] for i in sorted(cfg.class_labels.keys())]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n── Classification Metrics (3-month label) ─────────────────")
    print(f"  Accuracy:          {acc:.3f}")
    print(f"  Balanced Accuracy: {bal_acc:.3f}")

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"\n  Confusion Matrix:")
    print(cm_df.to_string(col_space=10))

    print(f"\n  Per-class metrics:")
    print(classification_report(
        y_true, y_pred, target_names=class_names, labels=[0, 1], zero_division=0,
    ))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm_df,
    }


def investment_metrics(returns: pd.Series, label: str = "Portfolio") -> dict:
    n_months = len(returns)
    n_years = n_months / 12

    cum_ret = (1 + returns).prod()
    cagr = cum_ret ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(12)
    sharpe = cagr / vol if vol > 0 else 0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    hit_rate = (returns > 0).mean()

    return {
        "label": label,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "total_return": cum_ret - 1,
        "n_months": n_months,
    }


def feature_importance(model: RegimeClassifier, cfg: Config) -> pd.DataFrame:
    coefs = model.get_coefficients()

    print("\n── Feature Importance (Model Coefficients) ─────────────────")
    if len(coefs) == 1:
        row = coefs.iloc[0].sort_values()
        print("  Negative = favors equity, Positive = favors safe rate:")
        for feat, val in row.items():
            direction = "-> safe" if val > 0 else "-> equity"
            print(f"    {feat:>30s}: {val:+.3f}  {direction}")
    else:
        print(coefs.round(3).to_string())

    return coefs


def evaluate(bt: pd.DataFrame, model: RegimeClassifier, cfg: Config) -> dict:
    """Full evaluation."""
    dir_metrics = direction_accuracy(bt, cfg)
    clf_metrics = classification_metrics(bt, cfg)

    print("\n── Investment Metrics ──────────────────────────────────────")
    ew_label = f"{int(cfg.equal_weight[0]*100)}/{int(cfg.equal_weight[1]*100)}"
    strategies = {
        "Model Portfolio": bt["port_return"],
        ew_label: bt["ew_return"],
        "Equity Only": bt["ret_equity"],
        "Safe Rate Only": bt["ret_safe"],
    }

    inv_results = []
    for name, rets in strategies.items():
        m = investment_metrics(rets, name)
        inv_results.append(m)
        print(f"\n  {name}:")
        print(f"    CAGR:         {m['cagr']:.2%}")
        print(f"    Volatility:   {m['volatility']:.2%}")
        print(f"    Sharpe:       {m['sharpe']:.2f}")
        print(f"    Max Drawdown: {m['max_drawdown']:.2%}")
        print(f"    Hit Rate:     {m['hit_rate']:.2%}")

    inv_df = pd.DataFrame(inv_results).set_index("label")

    model_sharpe = inv_df.loc["Model Portfolio", "sharpe"]
    ew_sharpe = inv_df.loc[ew_label, "sharpe"]
    eq_sharpe = inv_df.loc["Equity Only", "sharpe"]
    print(f"\n  Sharpe improvement vs {ew_label}:  {model_sharpe - ew_sharpe:+.2f}")
    print(f"  Sharpe improvement vs equity: {model_sharpe - eq_sharpe:+.2f}")

    # Weight distribution
    print(f"\n── Weight Distribution ─────────────────────────────────────")
    eq_w = bt["weight_equity"]
    print(f"  Mean equity weight:  {eq_w.mean():.1%}")
    print(f"  Min equity weight:   {eq_w.min():.1%}")
    print(f"  Max equity weight:   {eq_w.max():.1%}")
    print(f"  Std equity weight:   {eq_w.std():.3f}")
    # Bucket distribution
    buckets = pd.cut(eq_w, bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
                     labels=["<30%", "30-50%", "50-70%", "70-85%", ">85%"])
    bucket_counts = buckets.value_counts().sort_index()
    for bucket, count in bucket_counts.items():
        print(f"    {bucket:>8s}: {count:3d} months ({count/len(eq_w):.0%})")

    coefs = feature_importance(model, cfg)

    avg_turnover = bt["turnover"].mean()
    print(f"\n  Average Monthly Turnover: {avg_turnover:.3f}")

    # Crash overlay stats
    if "overlay" in bt.columns:
        overlay_active = bt["overlay"] != "none"
        n_active = overlay_active.sum()
        print(f"\n── Crash Overlay Stats ─────────────────────────────────────")
        print(f"  Months overlay fired: {n_active}/{len(bt)} ({n_active/len(bt):.1%})")
        if n_active > 0:
            avg_eq_when_active = bt.loc[overlay_active, "weight_equity"].mean()
            avg_eq_when_off = bt.loc[~overlay_active, "weight_equity"].mean()
            print(f"  Avg equity weight (overlay active): {avg_eq_when_active:.1%}")
            print(f"  Avg equity weight (overlay off):    {avg_eq_when_off:.1%}")
            # Did the overlay help? Compare returns in overlay months
            overlay_port = bt.loc[overlay_active, "port_return"].mean()
            overlay_eq = bt.loc[overlay_active, "ret_equity"].mean()
            print(f"  Avg monthly return (overlay active): {overlay_port:.2%} (equity was {overlay_eq:.2%})")

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    inv_df.to_csv(os.path.join(cfg.output_dir, "investment_metrics.csv"))
    coefs.to_csv(os.path.join(cfg.output_dir, "coefficients.csv"))
    clf_metrics["confusion_matrix"].to_csv(
        os.path.join(cfg.output_dir, "confusion_matrix.csv")
    )

    return {
        "classification": clf_metrics,
        "direction": dir_metrics,
        "investment": inv_df,
        "coefficients": coefs,
    }
