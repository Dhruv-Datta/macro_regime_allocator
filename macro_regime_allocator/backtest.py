"""
Walk-forward backtest engine.

Monthly rebalancing. The model predicts whether equities will outperform
the risk-free rate over the next 3 months. Weights are applied to 1-month
returns: equity price return vs fed funds rate / 12.
"""

import os
import numpy as np
import pandas as pd
from config import Config
from model import RegimeClassifier
from allocation import (
    probabilities_to_weights,
    get_equal_weights,
)


def compute_recency_weights(n_samples: int, halflife: int) -> np.ndarray:
    decay = np.log(2) / halflife
    ages = np.arange(n_samples)[::-1]
    weights = np.exp(-decay * ages)
    return weights


def compute_class_balanced_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_classes = len(classes)
    class_weights = {c: n / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([class_weights[yi] for yi in y])


def run_backtest(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: Config,
) -> dict:
    """
    Walk-forward backtest with monthly rebalancing.

    At each month t:
      1. Train on labels up to t - horizon (only fully realized labels)
      2. Predict: will equities beat the safe rate?
      3. Set weights based on prediction confidence
      4. Earn 1-month return: weight * equity_return + (1-weight) * safe_rate/12
    """
    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx].copy()
    labels = labels.loc[common_idx].copy()

    valid_mask = features.notna().all(axis=1)
    features = features.loc[valid_mask]
    labels = labels.loc[valid_mask]

    print(f"Backtest universe: {len(features)} months from "
          f"{features.index[0].strftime('%Y-%m')} to "
          f"{features.index[-1].strftime('%Y-%m')}")

    # Load merged data for returns
    merged_path = os.path.join(cfg.data_dir, "merged_monthly.csv")
    merged = pd.read_csv(merged_path, index_col="date", parse_dates=True)

    # Monthly returns
    monthly_returns = pd.DataFrame(index=merged.index)
    monthly_returns["equity"] = merged["equity"].pct_change()

    rate_col = cfg.safe_rate_series
    if rate_col in merged.columns:
        # Fed funds rate is annualized percent → monthly return
        monthly_returns["safe"] = merged[rate_col].shift(1) / 100 / 12
    else:
        raise ValueError(f"Safe rate series '{rate_col}' not found in data.")

    X = features
    y = labels["label"]
    all_dates = X.index.tolist()
    horizon = cfg.forecast_horizon_months

    if cfg.model_type == "incremental":
        persistent_model = RegimeClassifier(cfg)
        last_trained_idx = 0

    results = []
    n_predicted = 0
    prev_equity_weight = cfg.equal_weight[0]  # start at baseline

    for i in range(cfg.min_train_months, len(all_dates) - 1):
        rebalance_date = all_dates[i]
        next_date = all_dates[i + 1]

        # Only train on labels whose forward window is fully realized
        train_end = i - horizon
        if train_end < 1:
            continue

        if cfg.model_type == "logistic":
            if cfg.window_type == "expanding":
                train_start = 0
            else:
                train_start = max(0, train_end - cfg.rolling_window_months)

            train_idx = all_dates[train_start:train_end]
            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]

            if len(y_train) < cfg.min_train_months or y_train.nunique() < 2:
                continue

            sw = compute_recency_weights(len(train_idx), cfg.recency_halflife_months)
            cw = compute_class_balanced_weights(y_train.values)
            model = RegimeClassifier(cfg)
            model.fit(X_train, y_train, sample_weight=sw * cw)

        elif cfg.model_type == "incremental":
            if not persistent_model.is_fitted:
                warmup_idx = all_dates[:train_end]
                if len(warmup_idx) < cfg.min_train_months:
                    continue
                X_warmup = X.loc[warmup_idx]
                y_warmup = y.loc[warmup_idx]
                if y_warmup.nunique() < 2:
                    continue
                sw = compute_recency_weights(len(warmup_idx), cfg.recency_halflife_months)
                cw = compute_class_balanced_weights(y_warmup.values)
                persistent_model.fit(X_warmup, y_warmup, sample_weight=sw * cw)
                last_trained_idx = train_end
            else:
                new_idx = all_dates[last_trained_idx:train_end]
                if len(new_idx) > 0:
                    X_new = X.loc[new_idx]
                    y_new = y.loc[new_idx]
                    if len(y_new) > 0:
                        sw = compute_recency_weights(len(new_idx), cfg.recency_halflife_months)
                        cw = compute_class_balanced_weights(y_new.values)
                        persistent_model.partial_fit(X_new, y_new, sample_weight=sw * cw)
                    last_trained_idx = train_end

            model = persistent_model

            step_num = i - cfg.min_train_months
            if cfg.checkpoint_every > 0 and step_num % cfg.checkpoint_every == 0:
                model.save_checkpoint(step_num, cfg)

        # Predict
        X_pred = X.loc[[rebalance_date]]
        proba = model.predict_proba(X_pred)[0]  # [p_equity, p_safe]
        pred_class = np.argmax(proba)

        # Current (unlagged) market data for crash overlay
        market_data = {}
        if rebalance_date in merged.index:
            row = merged.loc[rebalance_date]
            # VIX 1-month change
            if "vix" in merged.columns:
                rd_loc = merged.index.get_loc(rebalance_date)
                if rd_loc > 0:
                    market_data["vix_1m_change"] = (
                        merged["vix"].iloc[rd_loc] - merged["vix"].iloc[rd_loc - 1]
                    )
            # VIX term structure
            if "vix" in merged.columns and "vix3m" in merged.columns:
                v, v3 = row.get("vix"), row.get("vix3m")
                if pd.notna(v) and pd.notna(v3) and v3 > 0:
                    market_data["vix_term_structure"] = v / v3
            # Equity drawdown from 12-month high + 1m change in drawdown
            if "equity" in merged.columns:
                rd_loc = merged.index.get_loc(rebalance_date)
                lookback = max(0, rd_loc - 11)
                rolling_high = merged["equity"].iloc[lookback:rd_loc + 1].max()
                if rolling_high > 0:
                    dd_now = (row["equity"] / rolling_high - 1) * 100
                    market_data["equity_drawdown_from_high"] = dd_now
                    # Was the drawdown worse or better than last month?
                    if rd_loc > 0:
                        prev_lookback = max(0, rd_loc - 12)
                        prev_high = merged["equity"].iloc[prev_lookback:rd_loc].max()
                        if prev_high > 0:
                            dd_prev = (merged["equity"].iloc[rd_loc - 1] / prev_high - 1) * 100
                            market_data["drawdown_1m_change"] = dd_now - dd_prev
            # Credit spread 3-month change
            if "credit_spread" in merged.columns:
                rd_loc = merged.index.get_loc(rebalance_date)
                if rd_loc >= 3:
                    market_data["credit_spread_3m_change"] = (
                        merged["credit_spread"].iloc[rd_loc]
                        - merged["credit_spread"].iloc[rd_loc - 3]
                    )

        raw_proba, weights, overlay_reason = probabilities_to_weights(
            proba, cfg, market_data=market_data
        )

        # Asymmetric weight smoothing: slow ramp up, fast defense
        target_equity = weights[0]
        if target_equity >= prev_equity_weight:
            # Increasing equity: smooth (slow ramp avoids churn in bulls)
            alpha = cfg.weight_smoothing_up
        else:
            # Decreasing equity: respond fast (quick defensive switch)
            alpha = cfg.weight_smoothing_down
        smoothed_equity = alpha * target_equity + (1 - alpha) * prev_equity_weight
        smoothed_equity = np.clip(smoothed_equity, cfg.min_weight, cfg.max_weight)
        weights = np.array([smoothed_equity, 1.0 - smoothed_equity])
        prev_equity_weight = smoothed_equity

        # 1-month realized returns
        if next_date not in monthly_returns.index:
            continue
        ret_eq = monthly_returns.loc[next_date, "equity"]
        ret_safe = monthly_returns.loc[next_date, "safe"]
        if np.isnan(ret_eq) or np.isnan(ret_safe):
            continue

        realized = np.array([ret_eq, ret_safe])
        port_ret = np.dot(weights, realized)

        ew = get_equal_weights(cfg)
        ew_ret = np.dot(ew, realized)

        # 60/40 reference line (not used in training, just for charting)
        ret_6040 = 0.60 * ret_eq + 0.40 * ret_safe

        actual_label = y.loc[rebalance_date] if rebalance_date in y.index else np.nan

        results.append({
            "rebalance_date": rebalance_date,
            "return_date": next_date,
            "pred_class": pred_class,
            "actual_label": actual_label,
            "prob_equity": proba[0],
            "prob_safe": proba[1],
            "weight_equity": weights[0],
            "weight_safe": weights[1],
            "overlay": overlay_reason,
            "ret_equity": ret_eq,
            "ret_safe": ret_safe,
            "port_return": port_ret,
            "ew_return": ew_ret,
            "ret_6040": ret_6040,
            "holding_period_months": 1,
            "train_size": train_end,
        })
        n_predicted += 1

    print(f"  Predictions made: {n_predicted}")

    if not results:
        raise RuntimeError("No predictions were made. Check data availability.")

    bt = pd.DataFrame(results)
    bt.set_index("return_date", inplace=True)
    bt.index.name = "date"

    bt["cum_port"] = (1 + bt["port_return"]).cumprod()
    bt["cum_ew"] = (1 + bt["ew_return"]).cumprod()
    bt["cum_equity"] = (1 + bt["ret_equity"]).cumprod()
    bt["cum_safe"] = (1 + bt["ret_safe"]).cumprod()
    bt["cum_6040"] = (1 + bt["ret_6040"]).cumprod()

    weight_cols = ["weight_equity", "weight_safe"]
    turnover = bt[weight_cols].diff().abs().sum(axis=1)
    turnover.iloc[0] = 0
    bt["turnover"] = turnover

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    bt.to_csv(os.path.join(cfg.output_dir, "backtest_results.csv"))
    bt[weight_cols].to_csv(os.path.join(cfg.output_dir, "monthly_weights.csv"))
    bt[["prob_equity", "prob_safe"]].to_csv(
        os.path.join(cfg.output_dir, "monthly_probabilities.csv")
    )
    bt[["port_return", "ew_return"]].to_csv(
        os.path.join(cfg.output_dir, "monthly_returns.csv")
    )
    print(f"  Backtest results saved to {cfg.output_dir}/")

    # Final model on all data
    final_model = RegimeClassifier(cfg)
    final_train_end = len(all_dates) - horizon
    final_idx = all_dates[:final_train_end]
    X_final = X.loc[final_idx]
    y_final = y.loc[final_idx].dropna()
    common = X_final.index.intersection(y_final.index)
    sw = compute_recency_weights(len(common), cfg.recency_halflife_months)
    cw = compute_class_balanced_weights(y.loc[common].values)
    final_model.fit(X.loc[common], y.loc[common], sample_weight=sw * cw)
    final_model.save_model()

    return {
        "backtest": bt,
        "final_model": final_model,
    }
