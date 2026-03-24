"""
Evaluation metrics and plot generation for backtest results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, classification_report,
)
from config import Config
from model import RegimeClassifier


# ── Investment Metrics ──────────────────────────────────────────────────────

def _investment_metrics(returns: pd.Series, label: str) -> dict:
    n_months = len(returns)
    n_years = n_months / 12
    cum_ret = (1 + returns).prod()
    cagr = cum_ret ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(12)
    sharpe = cagr / vol if vol > 0 else 0
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    hit_rate = (returns > 0).mean()
    return {"label": label, "cagr": cagr, "volatility": vol, "sharpe": sharpe,
            "max_drawdown": max_dd, "hit_rate": hit_rate, "total_return": cum_ret - 1,
            "n_months": n_months}


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(bt: pd.DataFrame, model: RegimeClassifier, cfg: Config) -> dict:
    """Compute and print all evaluation metrics."""

    # Direction accuracy
    valid = bt.dropna(subset=["ret_equity", "ret_tbills"]).copy()
    equity_won = valid["ret_equity"] > valid["ret_tbills"]
    model_favors_equity = valid["weight_equity"] > 0.5
    correct = equity_won == model_favors_equity

    spread = (valid["ret_equity"] - valid["ret_tbills"]).abs()
    weighted_acc = (correct * spread).sum() / spread.sum() if spread.sum() > 0 else 0.5

    excess = valid["ret_equity"] - valid["ret_tbills"]
    port_excess = valid["port_return"] - valid["ret_tbills"]
    up = excess > 0
    upside_capture = port_excess[up].sum() / excess[up].sum() if up.any() else 0
    down = excess < 0
    downside_capture = port_excess[down].sum() / excess[down].sum() if down.any() else 0

    print("\n── Direction Accuracy ──────────────────────────────────────")
    print(f"  Direction accuracy:          {correct.mean():.3f}")
    print(f"  Magnitude-weighted accuracy: {weighted_acc:.3f}")
    print(f"  Upside capture:             {upside_capture:.3f}")
    print(f"  Downside capture:           {downside_capture:.3f} (lower = better)")

    # Classification metrics
    clf_valid = bt.dropna(subset=["pred_class", "actual_label"])
    y_true = clf_valid["actual_label"].astype(int)
    y_pred = clf_valid["pred_class"].astype(int)
    class_names = [cfg.class_labels[i] for i in sorted(cfg.class_labels.keys())]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    horizon = cfg.forecast_horizon_months
    print(f"\n── Classification Metrics ({horizon}-month label) ─────────────────")
    print(f"  Accuracy:          {acc:.3f}")
    print(f"  Balanced Accuracy: {bal_acc:.3f}")
    print(f"\n  Confusion Matrix:\n{cm_df.to_string(col_space=10)}")
    print(f"\n  Per-class metrics:")
    print(classification_report(y_true, y_pred, target_names=class_names,
                                labels=[0, 1], zero_division=0))

    # Investment metrics
    print("── Investment Metrics ──────────────────────────────────────")
    ew_label = f"{int(cfg.equal_weight[0]*100)}/{int(cfg.equal_weight[1]*100)}"
    strategies = {
        "Model Portfolio": bt["port_return"],
        ew_label: bt["ew_return"],
        "60/40 Reference": bt["ret_6040"],
        "Equity Only": bt["ret_equity"],
        "T-Bills Only": bt["ret_tbills"],
    }

    inv_results = []
    for name, rets in strategies.items():
        m = _investment_metrics(rets, name)
        inv_results.append(m)
        print(f"\n  {name}:")
        print(f"    CAGR:         {m['cagr']:.2%}")
        print(f"    Volatility:   {m['volatility']:.2%}")
        print(f"    Sharpe:       {m['sharpe']:.2f}")
        print(f"    Max Drawdown: {m['max_drawdown']:.2%}")
        print(f"    Hit Rate:     {m['hit_rate']:.2%}")

    inv_df = pd.DataFrame(inv_results).set_index("label")
    model_sharpe = inv_df.loc["Model Portfolio", "sharpe"]
    print(f"\n  Sharpe improvement vs {ew_label}:  {model_sharpe - inv_df.loc[ew_label, 'sharpe']:+.2f}")
    print(f"  Sharpe improvement vs equity: {model_sharpe - inv_df.loc['Equity Only', 'sharpe']:+.2f}")

    # Weight distribution
    eq_w = bt["weight_equity"]
    print(f"\n── Weight Distribution ─────────────────────────────────────")
    print(f"  Mean equity weight:  {eq_w.mean():.1%}")
    print(f"  Min / Max:           {eq_w.min():.1%} / {eq_w.max():.1%}")
    buckets = pd.cut(eq_w, bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
                     labels=["<30%", "30-50%", "50-70%", "70-85%", ">85%"])
    for bucket, count in buckets.value_counts().sort_index().items():
        print(f"    {bucket:>8s}: {count:3d} months ({count/len(eq_w):.0%})")

    # Feature importance
    coefs = model.get_coefficients()
    print("\n── Feature Importance (Model Coefficients) ─────────────────")
    if len(coefs) == 1:
        row = coefs.iloc[0].sort_values()
        print("  Negative = favors equity, Positive = favors T-bills:")
        for feat, val in row.items():
            print(f"    {feat:>30s}: {val:+.3f}  {'-> tbills' if val > 0 else '-> equity'}")
    else:
        print(coefs.round(3).to_string())

    print(f"\n  Average Monthly Turnover: {bt['turnover'].mean():.3f}")

    # Crash overlay stats
    if "overlay" in bt.columns:
        active = bt["overlay"] != "none"
        n_active = active.sum()
        print(f"\n── Crash Overlay Stats ─────────────────────────────────────")
        print(f"  Months overlay fired: {n_active}/{len(bt)} ({n_active/len(bt):.1%})")
        if n_active > 0:
            print(f"  Avg equity weight (overlay active): {bt.loc[active, 'weight_equity'].mean():.1%}")
            print(f"  Avg equity weight (overlay off):    {bt.loc[~active, 'weight_equity'].mean():.1%}")
            print(f"  Avg monthly return (overlay active): {bt.loc[active, 'port_return'].mean():.2%} "
                  f"(equity was {bt.loc[active, 'ret_equity'].mean():.2%})")

    # Save summary
    os.makedirs(cfg.output_dir, exist_ok=True)
    inv_df.to_csv(os.path.join(cfg.output_dir, "investment_metrics.csv"))

    return {"classification": {"confusion_matrix": cm_df},
            "investment": inv_df, "coefficients": coefs}


# ── Plots ───────────────────────────────────────────────────────────────────

def _ew_label(cfg):
    return f"{int(cfg.equal_weight[0]*100)}/{int(cfg.equal_weight[1]*100)}"


def _save(fig, cfg, name):
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_dir, name), dpi=150)
    plt.close(fig)
    print(f"  Saved {name}")


def generate_all_plots(bt: pd.DataFrame, eval_results: dict, cfg: Config):
    os.makedirs(cfg.plot_dir, exist_ok=True)
    print("\nGenerating plots...")
    ew_label = _ew_label(cfg)

    # Cumulative returns
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bt.index, bt["cum_port"], label="Model Portfolio", linewidth=2)
    ax.plot(bt.index, bt["cum_ew"], label=ew_label, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.plot(bt.index, bt["cum_6040"], label="60/40 Reference",
            linewidth=1.5, linestyle="-.", alpha=0.7, color="purple")
    ax.plot(bt.index, bt["cum_equity"], label="Equity Only (SPY)",
            linewidth=1, linestyle="--", alpha=0.6)
    ax.plot(bt.index, bt["cum_tbills"], label="T-Bills Only",
            linewidth=1, linestyle="--", alpha=0.6)
    ax.set_title("Cumulative Returns: Model vs Benchmarks")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, cfg, "cumulative_returns.png")

    # Drawdowns
    fig, ax = plt.subplots(figsize=(12, 4))
    for col, label in [("port_return", "Model"), ("ew_return", ew_label),
                        ("ret_6040", "60/40 Reference"), ("ret_equity", "Equity Only (SPY)")]:
        cum = (1 + bt[col]).cumprod()
        ax.fill_between(bt.index, cum / cum.cummax() - 1, 0, alpha=0.25, label=label)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, cfg, "drawdowns.png")

    # Equity weight over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(bt.index, bt["weight_equity"], 0, alpha=0.4, color="#2196F3", label="Equity weight")
    ax.fill_between(bt.index, bt["weight_equity"], 1, alpha=0.4, color="#4CAF50", label="T-Bills weight")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50/50")
    ax.axhline(0.6, color="orange", linestyle=":", alpha=0.5, label="60/40")
    ax.set_title("Portfolio Equity Weight Over Time")
    ax.set_ylabel("Equity Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, cfg, "equity_weight_over_time.png")

    # Predicted probabilities
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt.index, bt["prob_equity"], label="P(Equity beats T-bills)", color="#2196F3", alpha=0.8)
    ax.plot(bt.index, bt["prob_tbills"], label="P(T-Bills win)", color="#4CAF50", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Predicted Probabilities Over Time")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, cfg, "probabilities_over_time.png")

    # Rolling Sharpe
    fig, ax = plt.subplots(figsize=(12, 5))
    window = 24
    for col, label in [("port_return", "Model"), ("ew_return", ew_label),
                        ("ret_6040", "60/40 Reference"), ("ret_equity", "Equity Only (SPY)")]:
        rm = bt[col].rolling(window).mean() * 12
        rs = bt[col].rolling(window).std() * np.sqrt(12)
        ax.plot(bt.index, rm / rs, label=label, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, cfg, "rolling_sharpe.png")

    # Confusion matrix
    if "classification" in eval_results:
        cm = eval_results["classification"]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns)
        ax.set_yticklabels(cm.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(len(cm.index)):
            for j in range(len(cm.columns)):
                val = cm.values[i, j]
                color = "white" if val > cm.values.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=16)
        fig.colorbar(im)
        _save(fig, cfg, "confusion_matrix.png")

    # Feature coefficients
    if "coefficients" in eval_results:
        coefs = eval_results["coefficients"]
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(coefs) == 1:
            vals = coefs.iloc[0].sort_values()
            colors = ["#2196F3" if v < 0 else "#4CAF50" for v in vals]
            ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(vals.index, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Coefficient (negative = equity, positive = T-bills)")
            ax.set_title("What Pushes Toward Equities vs T-Bills")
        else:
            im = ax.imshow(coefs.values, cmap="RdBu_r", aspect="auto",
                           vmin=-np.abs(coefs.values).max(), vmax=np.abs(coefs.values).max())
            ax.set_xticks(range(len(coefs.columns)))
            ax.set_yticks(range(len(coefs.index)))
            ax.set_xticklabels(coefs.columns, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(coefs.index)
            fig.colorbar(im)
        _save(fig, cfg, "coefficients.png")

    print("  All plots saved.")
