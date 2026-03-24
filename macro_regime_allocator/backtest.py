"""
Allocation logic and walk-forward backtest engine.

Allocation pipeline: model probabilities → sigmoid amplification → crash overlay
→ asymmetric smoothing → weight caps.

Backtest: monthly rebalancing, trains on realized labels only, applies crash
overlay using current (unlagged) market data for fast defensive switching.
"""

import os
import numpy as np
import pandas as pd
from config import Config
from model import RegimeClassifier


# ── Allocation ──────────────────────────────────────────────────────────────

def sigmoid_weight_map(p_equity: float, cfg: Config) -> float:
    """Map P(equity outperforms) through a steep sigmoid biased toward the baseline."""
    bias = np.log(cfg.equal_weight[0] / cfg.equal_weight[1])
    x = (p_equity - 0.5) * cfg.allocation_steepness + bias
    return 1.0 / (1.0 + np.exp(-x))


def crash_overlay(equity_weight: float, market_data: dict, cfg: Config) -> tuple:
    """
    Defensive overlay using current (unlagged) market conditions.
    Reacts to ACTIVE deterioration, not static levels.
    Returns (adjusted_weight, reason_string).
    """
    if not cfg.crash_overlay or not market_data:
        return equity_weight, "none"

    penalties = []
    vix_change = market_data.get("vix_1m_change")
    vix_ts = market_data.get("vix_term_structure")

    # VIX spike: sharp jump = crash unfolding
    if vix_change is not None and vix_change > cfg.vix_spike_threshold:
        severity = min((vix_change - cfg.vix_spike_threshold) / 15.0, 1.0)
        penalties.append(("vix_spike", severity * 0.50))

    # VIX backwardation + rising = panic mode
    if (vix_ts is not None and vix_ts > 1.08
            and vix_change is not None and vix_change > 3.0):
        severity = min((vix_ts - 1.08) * 5.0, 1.0)
        penalties.append(("vix_panic", severity * 0.35))

    # Drawdown accelerating with VIX stress confirmation
    drawdown = market_data.get("equity_drawdown_from_high")
    dd_change = market_data.get("drawdown_1m_change")
    if (drawdown is not None and drawdown < cfg.drawdown_defense_threshold
            and dd_change is not None and dd_change < -4.0
            and vix_ts is not None and vix_ts > 0.98):
        severity = min(-dd_change / 10.0, 1.0)
        penalties.append(("drawdown_crash", severity * 0.30))

    if not penalties:
        return equity_weight, "none"

    total_penalty = min(sum(p for _, p in penalties), 0.60)
    reasons = "+".join(name for name, _ in penalties)
    return equity_weight * (1.0 - total_penalty), reasons


def probabilities_to_weights(probabilities: np.ndarray, cfg: Config,
                             market_data: dict = None) -> tuple:
    """Full pipeline: probabilities → sigmoid → crash overlay → caps."""
    raw = probabilities.copy()
    eq_w = sigmoid_weight_map(probabilities[0], cfg)
    eq_w, overlay_reason = crash_overlay(eq_w, market_data or {}, cfg)

    # Cap and normalize
    eq_w = np.clip(eq_w, cfg.min_weight, cfg.max_weight)
    weights = np.array([eq_w, 1.0 - eq_w])
    return raw, weights, overlay_reason


# ── Sample Weighting ────────────────────────────────────────────────────────

def _recency_weights(n: int, halflife: int) -> np.ndarray:
    decay = np.log(2) / halflife
    return np.exp(-decay * np.arange(n)[::-1])


def _class_balanced_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n, n_c = len(y), len(classes)
    cw = {c: n / (n_c * cnt) for c, cnt in zip(classes, counts)}
    return np.array([cw[yi] for yi in y])


# ── Market Data for Crash Overlay ───────────────────────────────────────────

def _gather_market_data(merged: pd.DataFrame, rebalance_date) -> dict:
    """Collect current (unlagged) market signals for the crash overlay."""
    if rebalance_date not in merged.index:
        return {}

    md = {}
    row = merged.loc[rebalance_date]
    rd_loc = merged.index.get_loc(rebalance_date)

    # VIX 1-month change
    if "vix" in merged.columns and rd_loc > 0:
        md["vix_1m_change"] = merged["vix"].iloc[rd_loc] - merged["vix"].iloc[rd_loc - 1]

    # VIX term structure
    if "vix" in merged.columns and "vix3m" in merged.columns:
        v, v3 = row.get("vix"), row.get("vix3m")
        if pd.notna(v) and pd.notna(v3) and v3 > 0:
            md["vix_term_structure"] = v / v3

    # Equity drawdown from 12-month high + 1m change
    if "equity" in merged.columns:
        lookback = max(0, rd_loc - 11)
        rolling_high = merged["equity"].iloc[lookback:rd_loc + 1].max()
        if rolling_high > 0:
            dd_now = (row["equity"] / rolling_high - 1) * 100
            md["equity_drawdown_from_high"] = dd_now
            if rd_loc > 0:
                prev_lookback = max(0, rd_loc - 12)
                prev_high = merged["equity"].iloc[prev_lookback:rd_loc].max()
                if prev_high > 0:
                    dd_prev = (merged["equity"].iloc[rd_loc - 1] / prev_high - 1) * 100
                    md["drawdown_1m_change"] = dd_now - dd_prev

    # Credit spread 3-month change
    if "credit_spread" in merged.columns and rd_loc >= 3:
        md["credit_spread_3m_change"] = (
            merged["credit_spread"].iloc[rd_loc] - merged["credit_spread"].iloc[rd_loc - 3]
        )

    return md


# ── Walk-Forward Backtest ───────────────────────────────────────────────────

def run_backtest(features: pd.DataFrame, labels: pd.DataFrame, cfg: Config) -> dict:
    """
    Walk-forward backtest with monthly rebalancing.

    At each month t:
      1. Train on labels up to t - horizon (only fully realized)
      2. Predict probability of equity outperformance
      3. Map to weights via sigmoid + crash overlay + smoothing
      4. Earn 1-month return
    """
    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx].copy()
    labels = labels.loc[common_idx].copy()

    valid_mask = features.notna().all(axis=1)
    features, labels = features.loc[valid_mask], labels.loc[valid_mask]

    print(f"Backtest universe: {len(features)} months from "
          f"{features.index[0].strftime('%Y-%m')} to "
          f"{features.index[-1].strftime('%Y-%m')}")

    # Load merged data for returns and overlay signals
    merged = pd.read_csv(os.path.join(cfg.data_dir, "merged_monthly.csv"),
                         index_col="date", parse_dates=True)

    monthly_returns = pd.DataFrame(index=merged.index)
    monthly_returns["equity"] = merged["equity"].pct_change()
    rate_col = cfg.tbills_rate_series
    if rate_col not in merged.columns:
        raise ValueError(f"T-bills rate series '{rate_col}' not found in data.")
    monthly_returns["tbills"] = merged[rate_col].shift(1) / 100 / 12

    X, y = features, labels["label"]
    all_dates = X.index.tolist()
    horizon = cfg.forecast_horizon_months
    ew = np.array(cfg.equal_weight)

    results = []
    prev_equity_weight = cfg.equal_weight[0]

    for i in range(cfg.min_train_months, len(all_dates) - 1):
        rebalance_date = all_dates[i]
        next_date = all_dates[i + 1]

        # Only train on labels whose forward window is fully realized
        train_end = i - horizon
        if train_end < 1:
            continue

        # ── Train ───────────────────────────────────────────────────────
        train_start = 0 if cfg.window_type == "expanding" else max(0, train_end - cfg.rolling_window_months)
        train_idx = all_dates[train_start:train_end]
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]

        if len(y_train) < cfg.min_train_months or y_train.nunique() < 2:
            continue

        sw = _recency_weights(len(train_idx), cfg.recency_halflife_months)
        cw = _class_balanced_weights(y_train.values)
        model = RegimeClassifier(cfg)
        model.fit(X_train, y_train, sample_weight=sw * cw)

        # ── Predict & allocate ──────────────────────────────────────────
        proba = model.predict_proba(X.loc[[rebalance_date]])[0]
        pred_class = np.argmax(proba)

        market_data = _gather_market_data(merged, rebalance_date)
        _, weights, overlay_reason = probabilities_to_weights(proba, cfg, market_data)

        # Asymmetric smoothing: slow ramp up, instant defense
        target_eq = weights[0]
        alpha = cfg.weight_smoothing_up if target_eq >= prev_equity_weight else cfg.weight_smoothing_down
        smoothed_eq = np.clip(alpha * target_eq + (1 - alpha) * prev_equity_weight,
                              cfg.min_weight, cfg.max_weight)
        weights = np.array([smoothed_eq, 1.0 - smoothed_eq])
        prev_equity_weight = smoothed_eq

        # ── Realized returns ────────────────────────────────────────────
        if next_date not in monthly_returns.index:
            continue
        ret_eq = monthly_returns.loc[next_date, "equity"]
        ret_tbills = monthly_returns.loc[next_date, "tbills"]
        if np.isnan(ret_eq) or np.isnan(ret_tbills):
            continue

        realized = np.array([ret_eq, ret_tbills])
        actual_label = y.loc[rebalance_date] if rebalance_date in y.index else np.nan

        results.append({
            "rebalance_date": rebalance_date,
            "return_date": next_date,
            "pred_class": pred_class,
            "actual_label": actual_label,
            "prob_equity": proba[0],
            "prob_tbills": proba[1],
            "weight_equity": weights[0],
            "weight_tbills": weights[1],
            "overlay": overlay_reason,
            "ret_equity": ret_eq,
            "ret_tbills": ret_tbills,
            "port_return": np.dot(weights, realized),
            "ew_return": np.dot(ew, realized),
            "ret_6040": 0.60 * ret_eq + 0.40 * ret_tbills,
            "train_size": train_end,
        })

    print(f"  Predictions made: {len(results)}")
    if not results:
        raise RuntimeError("No predictions were made. Check data availability.")

    bt = pd.DataFrame(results).set_index("return_date")
    bt.index.name = "date"

    # Cumulative series
    for col, src in [("cum_port", "port_return"), ("cum_ew", "ew_return"),
                     ("cum_equity", "ret_equity"), ("cum_tbills", "ret_tbills"),
                     ("cum_6040", "ret_6040")]:
        bt[col] = (1 + bt[src]).cumprod()

    bt["turnover"] = bt[["weight_equity", "weight_tbills"]].diff().abs().sum(axis=1)
    bt.at[bt.index[0], "turnover"] = 0

    # Save backtest results
    os.makedirs(cfg.output_dir, exist_ok=True)
    bt.to_csv(os.path.join(cfg.output_dir, "backtest_results.csv"))
    print(f"  Backtest results saved to {cfg.output_dir}/")

    # Train final model on all available data
    final_model = RegimeClassifier(cfg)
    final_idx = all_dates[:len(all_dates) - horizon]
    y_final = y.loc[y.index.isin(final_idx)].dropna()
    common = X.index.intersection(y_final.index)
    sw = _recency_weights(len(common), cfg.recency_halflife_months)
    cw = _class_balanced_weights(y.loc[common].values)
    final_model.fit(X.loc[common], y.loc[common], sample_weight=sw * cw)
    final_model.save_model()

    return {"backtest": bt, "final_model": final_model}
