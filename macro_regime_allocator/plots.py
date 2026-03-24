"""
Visualization for backtest results. Equities vs safe rate.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import Config


def _benchmark_label(cfg: Config):
    return f"{int(cfg.equal_weight[0]*100)}/{int(cfg.equal_weight[1]*100)}"


def plot_cumulative_returns(bt: pd.DataFrame, cfg: Config):
    ew_label = _benchmark_label(cfg)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bt.index, bt["cum_port"], label="Model Portfolio", linewidth=2)
    ax.plot(bt.index, bt["cum_ew"], label=ew_label, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.plot(bt.index, bt["cum_6040"], label="60/40 Reference",
            linewidth=1.5, linestyle="-.", alpha=0.7, color="purple")
    ax.plot(bt.index, bt["cum_equity"], label="Equity Only (SPY)",
            linewidth=1, linestyle="--", alpha=0.6)
    ax.plot(bt.index, bt["cum_safe"], label="Safe Rate Only",
            linewidth=1, linestyle="--", alpha=0.6)
    ax.set_title("Cumulative Returns: Model vs Benchmarks")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_dir, "cumulative_returns.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved cumulative_returns.png")


def plot_drawdowns(bt: pd.DataFrame, cfg: Config):
    fig, ax = plt.subplots(figsize=(12, 4))

    ew_label = _benchmark_label(cfg)
    for col, label in [("port_return", "Model"), ("ew_return", ew_label),
                        ("ret_6040", "60/40 Reference"),
                        ("ret_equity", "Equity Only (SPY)")]:
        cum = (1 + bt[col]).cumprod()
        dd = cum / cum.cummax() - 1
        ax.fill_between(bt.index, dd, 0, alpha=0.25, label=label)

    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_dir, "drawdowns.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved drawdowns.png")


def plot_equity_weight_over_time(bt: pd.DataFrame, cfg: Config):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(bt.index, bt["weight_equity"], 0,
                    alpha=0.4, color="#2196F3", label="Equity weight")
    ax.fill_between(bt.index, bt["weight_equity"], 1,
                    alpha=0.4, color="#4CAF50", label="Safe rate weight")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50/50")
    ax.axhline(0.6, color="orange", linestyle=":", alpha=0.5, label="60/40")

    ax.set_title("Portfolio Equity Weight Over Time")
    ax.set_ylabel("Equity Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_dir, "equity_weight_over_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved equity_weight_over_time.png")


def plot_probabilities_over_time(bt: pd.DataFrame, cfg: Config):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(bt.index, bt["prob_equity"], label="P(Equity beats safe rate)",
            color="#2196F3", alpha=0.8)
    ax.plot(bt.index, bt["prob_safe"], label="P(Safe rate wins)",
            color="#4CAF50", alpha=0.8)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Predicted Probabilities Over Time")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_dir, "probabilities_over_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved probabilities_over_time.png")


def plot_rolling_sharpe(bt: pd.DataFrame, cfg: Config, window: int = 24):
    fig, ax = plt.subplots(figsize=(12, 5))

    ew_label = _benchmark_label(cfg)
    for col, label in [("port_return", "Model"), ("ew_return", ew_label),
                        ("ret_6040", "60/40 Reference"),
                        ("ret_equity", "Equity Only (SPY)")]:
        rolling_mean = bt[col].rolling(window).mean() * 12
        rolling_std = bt[col].rolling(window).std() * np.sqrt(12)
        rolling_sharpe = rolling_mean / rolling_std
        ax.plot(bt.index, rolling_sharpe, label=label, alpha=0.8)

    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_dir, "rolling_sharpe.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved rolling_sharpe.png")


def plot_confusion_matrix(cm: pd.DataFrame, cfg: Config):
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
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=16)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved confusion_matrix.png")


def plot_coefficient_bar(coefs: pd.DataFrame, cfg: Config):
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(coefs) == 1:
        vals = coefs.iloc[0].sort_values()
        colors = ["#2196F3" if v < 0 else "#4CAF50" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Coefficient (negative = equity, positive = safe rate)")
        ax.set_title("What Pushes Toward Equities vs Safe Rate")
    else:
        im = ax.imshow(coefs.values, cmap="RdBu_r", aspect="auto",
                       vmin=-np.abs(coefs.values).max(),
                       vmax=np.abs(coefs.values).max())
        ax.set_xticks(range(len(coefs.columns)))
        ax.set_yticks(range(len(coefs.index)))
        ax.set_xticklabels(coefs.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(coefs.index)
        fig.colorbar(im)

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_dir, "coefficients.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved coefficients.png")


def generate_all_plots(bt: pd.DataFrame, eval_results: dict, cfg: Config):
    os.makedirs(cfg.plot_dir, exist_ok=True)
    print("\nGenerating plots...")

    plot_cumulative_returns(bt, cfg)
    plot_drawdowns(bt, cfg)
    plot_equity_weight_over_time(bt, cfg)
    plot_probabilities_over_time(bt, cfg)
    plot_rolling_sharpe(bt, cfg)

    if "classification" in eval_results:
        plot_confusion_matrix(eval_results["classification"]["confusion_matrix"], cfg)

    if "coefficients" in eval_results:
        plot_coefficient_bar(eval_results["coefficients"], cfg)

    print("  All plots saved.")
