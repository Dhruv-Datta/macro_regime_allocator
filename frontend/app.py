#!/usr/bin/env python3
"""
Macro Regime Allocator — Streamlit Dashboard

Interactive frontend for visualizing backtest results, tuning parameters,
viewing current allocation signals, and exploring data.

Reads output files from ../macro_regime_allocator/
"""

import os
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(FRONTEND_DIR), "macro_regime_allocator")
DATA_DIR = os.path.join(BACKEND_DIR, "data")
OUTPUT_DIR = os.path.join(BACKEND_DIR, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
CONFIG_PATH = os.path.join(BACKEND_DIR, "config.yaml")

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Macro Regime Allocator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Data Loaders ───────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_backtest():
    path = os.path.join(OUTPUT_DIR, "backtest_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="date", parse_dates=True)


@st.cache_data(ttl=60)
def load_metrics():
    path = os.path.join(OUTPUT_DIR, "investment_metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data(ttl=60)
def load_features():
    path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="date", parse_dates=True)


@st.cache_data(ttl=60)
def load_merged():
    path = os.path.join(DATA_DIR, "merged_monthly.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="date", parse_dates=True)


@st.cache_data(ttl=60)
def load_config_yaml():
    import yaml
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def fmt_pct(v, decimals=2):
    if pd.isna(v):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def fmt_f(v, decimals=2):
    if pd.isna(v):
        return "—"
    return f"{v:.{decimals}f}"


def no_data_warning():
    st.warning(
        "No backtest results found. Go to **Run Model** to generate results, "
        f"or run `make run` in `macro_regime_allocator/`."
    )
    st.stop()


# ── Sidebar Navigation ────────────────────────────────────────────────────────

st.sidebar.title("Macro Regime Allocator")
st.sidebar.caption("Tactical equity / T-bills allocation")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Backtest Charts", "Annual Returns", "Features & Data",
     "Configuration", "Run Model"],
    index=0,
)

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "Dashboard":
    st.title("Macro Regime Allocator")
    st.caption("Tactical equity/T-bills allocation using macro regime signals")

    bt = load_backtest()
    metrics = load_metrics()
    if bt is None or metrics is None:
        no_data_warning()

    # ── Current Signal ──────────────────────────────────────────────────────
    st.header("Current Allocation Signal")
    latest = bt.iloc[-1]
    prev = bt.iloc[-2] if len(bt) > 1 else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Date", latest.name.strftime("%Y-%m"))
    with col2:
        eq_w = latest["weight_equity"]
        delta = f"{eq_w - prev['weight_equity']:.1%}" if prev is not None else None
        st.metric("Equity Weight", f"{eq_w:.1%}", delta=delta)
    with col3:
        st.metric("P(Equity Wins)", f"{latest['prob_equity']:.1%}")
    with col4:
        overlay = latest.get("overlay", "none")
        st.metric("Crash Overlay",
                  "ACTIVE" if overlay != "none" else "Inactive",
                  delta=overlay if overlay != "none" else None,
                  delta_color="inverse" if overlay != "none" else "off")

    # Allocation gauges
    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eq_w * 100,
            title={"text": "Equity Allocation"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#2196F3"},
                "steps": [
                    {"range": [0, 30], "color": "#ffebee"},
                    {"range": [30, 60], "color": "#fff3e0"},
                    {"range": [60, 85], "color": "#e8f5e9"},
                    {"range": [85, 100], "color": "#e3f2fd"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75, "value": 50,
                },
            },
            number={"suffix": "%"},
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest["prob_equity"] * 100,
            title={"text": "Model Confidence (Equity)"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#4CAF50"},
                "steps": [
                    {"range": [0, 30], "color": "#ffebee"},
                    {"range": [30, 50], "color": "#fff3e0"},
                    {"range": [50, 70], "color": "#e8f5e9"},
                    {"range": [70, 100], "color": "#c8e6c9"},
                ],
            },
            number={"suffix": "%"},
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    # ── Key Metrics ─────────────────────────────────────────────────────────
    st.header("Performance Summary")
    strats = list(metrics.index)
    metric_cols = st.columns(len(strats))
    for i, strat in enumerate(strats):
        with metric_cols[i]:
            st.subheader(strat)
            m = metrics.loc[strat]
            st.metric("CAGR", fmt_pct(m.get("cagr", 0)))
            st.metric("Sharpe", fmt_f(m.get("sharpe", 0)))
            st.metric("Max DD", fmt_pct(m.get("max_drawdown", 0)))
            st.metric("Sortino", fmt_f(m.get("sortino", 0)))

    # ── Cumulative Returns ──────────────────────────────────────────────────
    st.header("Cumulative Returns")
    fig = go.Figure()
    for col, name, dash in [
        ("cum_port", "Model Portfolio", "solid"),
        ("cum_ew", "Baseline", "dash"),
        ("cum_6040", "60/40 Reference", "dashdot"),
        ("cum_equity", "Equity Only", "dot"),
        ("cum_tbills", "T-Bills Only", "dot"),
    ]:
        if col in bt.columns:
            fig.add_trace(go.Scatter(
                x=bt.index, y=bt[col], name=name,
                line=dict(dash=dash),
            ))
    fig.update_layout(
        yaxis_title="Growth of $100", hovermode="x unified",
        height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Crash Overlay Stats ─────────────────────────────────────────────────
    if "overlay" in bt.columns:
        active = bt["overlay"] != "none"
        n_active = active.sum()
        st.header("Crash Overlay Activity")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Times Fired", f"{n_active}/{len(bt)}")
        with c2:
            st.metric("Fire Rate", f"{n_active / len(bt):.1%}")
        with c3:
            if n_active > 0:
                st.metric("Avg Return (Active)", fmt_pct(
                    bt.loc[active, "port_return"].mean()))


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST CHARTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Backtest Charts":
    st.title("Backtest Charts")

    bt = load_backtest()
    if bt is None:
        no_data_warning()

    metrics = load_metrics()

    chart = st.selectbox(
        "Select Chart",
        ["Cumulative Returns", "Drawdowns", "Equity Weight", "Probabilities",
         "Rolling Sharpe", "Monthly Returns Heatmap", "Weight Distribution"],
    )

    if chart == "Cumulative Returns":
        fig = go.Figure()
        for col, name in [("cum_port", "Model"), ("cum_ew", "Baseline"),
                          ("cum_6040", "60/40"), ("cum_equity", "Equity"),
                          ("cum_tbills", "T-Bills")]:
            if col in bt.columns:
                fig.add_trace(go.Scatter(x=bt.index, y=bt[col], name=name))
        fig.update_layout(yaxis_title="Growth of $100", hovermode="x unified",
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Drawdowns":
        fig = go.Figure()
        for col, name in [("port_return", "Model"), ("ew_return", "Baseline"),
                          ("ret_6040", "60/40"), ("ret_equity", "Equity")]:
            cum = (1 + bt[col]).cumprod()
            dd = cum / cum.cummax() - 1
            fig.add_trace(go.Scatter(
                x=bt.index, y=dd, name=name, fill="tozeroy", opacity=0.4,
            ))
        fig.update_layout(yaxis_title="Drawdown", hovermode="x unified",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Equity Weight":
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
            subplot_titles=["Equity Weight Over Time", "Monthly Turnover"],
        )
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt["weight_equity"], fill="tozeroy",
            name="Equity Weight", line=dict(color="#2196F3"),
        ), row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="50/50", row=1, col=1)
        fig.add_trace(go.Bar(
            x=bt.index, y=bt["turnover"], name="Turnover",
            marker_color="#FF9800", opacity=0.6,
        ), row=2, col=1)
        fig.update_layout(height=600, hovermode="x unified")
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Probabilities":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt["prob_equity"],
            name="P(Equity Wins)", line=dict(color="#2196F3"),
        ))
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt["prob_tbills"],
            name="P(T-Bills Win)", line=dict(color="#4CAF50"),
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_title="Probability", yaxis_range=[0, 1],
                          height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Rolling Sharpe":
        window = st.slider("Rolling Window (months)", 6, 60, 24)
        fig = go.Figure()
        for col, name in [("port_return", "Model"), ("ew_return", "Baseline"),
                          ("ret_6040", "60/40"), ("ret_equity", "Equity")]:
            rm = bt[col].rolling(window).mean() * 12
            rs = bt[col].rolling(window).std() * np.sqrt(12)
            fig.add_trace(go.Scatter(x=bt.index, y=rm / rs, name=name))
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
        fig.update_layout(
            yaxis_title="Sharpe Ratio", height=500, hovermode="x unified",
            title=f"Rolling {window}-Month Sharpe Ratio",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Monthly Returns Heatmap":
        bt_copy = bt.copy()
        bt_copy["year"] = bt_copy.index.year
        bt_copy["month"] = bt_copy.index.month
        pivot = bt_copy.pivot_table(
            values="port_return", index="year", columns="month", aggfunc="sum",
        )
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]
        fig = px.imshow(
            pivot.values * 100,
            x=pivot.columns, y=pivot.index.astype(str),
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            labels=dict(color="Return (%)"), aspect="auto",
        )
        fig.update_layout(height=600, title="Monthly Returns (%)")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Weight Distribution":
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=bt["weight_equity"], nbinsx=30,
            marker_color="#2196F3", opacity=0.7, name="Equity Weight",
        ))
        fig.add_vline(
            x=bt["weight_equity"].mean(), line_dash="dash", line_color="red",
            annotation_text=f"Mean: {bt['weight_equity'].mean():.1%}",
        )
        fig.update_layout(
            xaxis_title="Equity Weight", yaxis_title="Count",
            height=400, title="Distribution of Equity Weights",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Full Metrics Table ──────────────────────────────────────────────────
    if metrics is not None:
        st.header("Performance Metrics")
        display = metrics.copy()
        pct_cols = ["cagr", "volatility", "max_drawdown", "var_95", "cvar_95",
                    "hit_rate", "total_return", "best_month", "worst_month",
                    "avg_up_month", "avg_down_month"]
        for c in pct_cols:
            if c in display.columns:
                display[c] = display[c].apply(lambda x: fmt_pct(x))
        for c in ["sharpe", "sortino", "calmar", "up_down_ratio"]:
            if c in display.columns:
                display[c] = display[c].apply(lambda x: fmt_f(x))
        for c in ["max_dd_duration", "win_streak", "lose_streak", "n_months"]:
            if c in display.columns:
                display[c] = display[c].apply(lambda x: f"{int(x)}")
        st.dataframe(display, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANNUAL RETURNS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Annual Returns":
    st.title("Annual Returns")

    bt = load_backtest()
    if bt is None:
        no_data_warning()

    bt_copy = bt.copy()
    bt_copy["year"] = bt_copy.index.year
    years = sorted(bt_copy["year"].unique())

    rows = []
    for year in years:
        mask = bt_copy["year"] == year
        row = {"Year": year}
        for col, label in [("port_return", "Model"), ("ew_return", "Baseline"),
                           ("ret_6040", "60/40"), ("ret_equity", "Equity"),
                           ("ret_tbills", "T-Bills")]:
            row[label] = (1 + bt_copy.loc[mask, col]).prod() - 1
        cum_yr = (1 + bt_copy.loc[mask, "port_return"]).cumprod()
        row["Model DD"] = (cum_yr / cum_yr.cummax() - 1).min()
        row["Model Beat Equity"] = row["Model"] > row["Equity"]
        rows.append(row)

    annual_df = pd.DataFrame(rows)

    # Summary metric
    wins = annual_df["Model Beat Equity"].sum()
    total = len(annual_df)
    st.metric("Model Beat Equity", f"{wins}/{total} years ({wins / total:.0%})")

    # Formatted table
    display = annual_df.copy()
    for c in ["Model", "Baseline", "60/40", "Equity", "T-Bills", "Model DD"]:
        display[c] = display[c].apply(lambda x: fmt_pct(x, 1))
    display["Year"] = display["Year"].astype(str)
    st.dataframe(
        display[["Year", "Model", "Baseline", "60/40", "Equity", "T-Bills",
                 "Model DD", "Model Beat Equity"]],
        use_container_width=True, hide_index=True,
    )

    # Bar chart comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=annual_df["Year"], y=annual_df["Model"] * 100,
        name="Model", marker_color="#2196F3",
    ))
    fig.add_trace(go.Bar(
        x=annual_df["Year"], y=annual_df["Equity"] * 100,
        name="Equity", marker_color="#FF9800", opacity=0.6,
    ))
    fig.update_layout(
        barmode="group", yaxis_title="Annual Return (%)",
        height=500, title="Model vs Equity Annual Returns",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES & DATA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Features & Data":
    st.title("Features & Data Explorer")

    tab1, tab2, tab3 = st.tabs(["Features", "Raw Data", "Feature Correlations"])

    with tab1:
        features = load_features()
        if features is None:
            st.warning("No features data found.")
        else:
            st.subheader("Engineered Features")
            st.dataframe(features.tail(24), use_container_width=True)

            feat_col = st.selectbox("Plot Feature", features.columns.tolist())
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=features.index, y=features[feat_col], name=feat_col,
                line=dict(color="#2196F3"),
            ))
            fig.update_layout(height=400, title=f"{feat_col} Over Time",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        merged = load_merged()
        if merged is None:
            st.warning("No raw data found.")
        else:
            st.subheader("Merged Monthly Data")
            st.dataframe(merged.tail(24), use_container_width=True)

            data_col = st.selectbox("Plot Series", merged.columns.tolist())
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged.index, y=merged[data_col], name=data_col,
            ))
            fig.update_layout(height=400, title=f"{data_col} Over Time",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        features = load_features()
        if features is not None:
            corr = features.corr()
            fig = px.imshow(
                corr.values, x=corr.columns, y=corr.index,
                color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                aspect="auto",
            )
            fig.update_layout(height=600, title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Configuration":
    st.title("Configuration")
    st.caption("View and edit config.yaml parameters")

    cfg = load_config_yaml()
    if not cfg:
        st.warning(f"Could not load config.yaml at {CONFIG_PATH}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Date Range & Assets")
        start = st.text_input("Start Date", cfg.get("start_date", "2000-01-01"))
        end = st.text_input("End Date", cfg.get("end_date", "2026-03-01"))
        ticker = st.text_input("Equity Ticker", cfg.get("equity_ticker", "SPY"))
        horizon = st.number_input(
            "Forecast Horizon (months)",
            value=cfg.get("forecast_horizon_months", 1),
            min_value=1, max_value=12,
        )

        st.subheader("Feature Engineering")
        macro_lag = st.number_input(
            "Macro Lag (months)", value=cfg.get("macro_lag_months", 1),
            min_value=1, max_value=6,
        )
        momentum_win = st.number_input(
            "Momentum Window", value=cfg.get("momentum_window", 3),
            min_value=1, max_value=12,
        )
        vol_win = st.number_input(
            "Volatility Window", value=cfg.get("volatility_window", 3),
            min_value=1, max_value=12,
        )

        st.subheader("Model")
        reg_c = st.number_input(
            "Regularization C", value=cfg.get("regularization_C", 0.5),
            min_value=0.01, max_value=10.0, step=0.1, format="%.2f",
        )
        halflife = st.number_input(
            "Recency Halflife (months)",
            value=cfg.get("recency_halflife_months", 18),
            min_value=3, max_value=60,
        )
        window_type = st.selectbox(
            "Window Type", ["expanding", "rolling"],
            index=0 if cfg.get("window_type") == "expanding" else 1,
        )
        min_train = st.number_input(
            "Min Train Months", value=cfg.get("min_train_months", 36),
            min_value=12, max_value=120,
        )

    with col2:
        st.subheader("Allocation")
        baseline_eq = st.slider(
            "Baseline Equity Weight", 0.0, 1.0,
            cfg.get("baseline_equity", 0.90), 0.05,
        )
        steepness = st.slider(
            "Sigmoid Steepness", 1.0, 20.0,
            cfg.get("allocation_steepness", 10.0), 0.5,
        )
        min_w = st.slider(
            "Min Weight", 0.0, 0.20, cfg.get("min_weight", 0.03), 0.01,
        )
        max_w = st.slider(
            "Max Weight", 0.80, 1.0, cfg.get("max_weight", 0.97), 0.01,
        )
        smooth_up = st.slider(
            "Weight Smoothing Up", 0.0, 1.0,
            cfg.get("weight_smoothing_up", 0.7), 0.05,
        )
        smooth_down = st.slider(
            "Weight Smoothing Down", 0.0, 1.0,
            cfg.get("weight_smoothing_down", 1.0), 0.05,
        )

        st.subheader("Crash Overlay")
        crash_on = st.checkbox(
            "Enable Crash Overlay", value=cfg.get("crash_overlay", True),
        )
        vix_thresh = st.number_input(
            "VIX Spike Threshold", value=cfg.get("vix_spike_threshold", 10.0),
            min_value=1.0, max_value=30.0, step=0.5,
        )
        dd_thresh = st.number_input(
            "Drawdown Defense Threshold",
            value=cfg.get("drawdown_defense_threshold", -15.0),
            min_value=-50.0, max_value=-5.0, step=1.0,
        )
        credit_thresh = st.number_input(
            "Credit Spike Threshold",
            value=cfg.get("credit_spike_threshold", 1.5),
            min_value=0.5, max_value=5.0, step=0.1,
        )

    # Save button
    if st.button("Save Configuration", type="primary"):
        import yaml

        new_cfg = {
            "start_date": start,
            "end_date": end,
            "equity_ticker": ticker,
            "forecast_horizon_months": int(horizon),
            "macro_lag_months": int(macro_lag),
            "momentum_window": int(momentum_win),
            "volatility_window": int(vol_win),
            "regularization_C": float(reg_c),
            "class_weight": "balanced",
            "max_iter": 1000,
            "recency_halflife_months": int(halflife),
            "window_type": window_type,
            "rolling_window_months": cfg.get("rolling_window_months", 120),
            "min_train_months": int(min_train),
            "baseline_equity": float(baseline_eq),
            "baseline_tbills": round(1.0 - float(baseline_eq), 2),
            "min_weight": float(min_w),
            "max_weight": float(max_w),
            "allocation_steepness": float(steepness),
            "weight_smoothing_up": float(smooth_up),
            "weight_smoothing_down": float(smooth_down),
            "crash_overlay": bool(crash_on),
            "vix_spike_threshold": float(vix_thresh),
            "drawdown_defense_threshold": float(dd_thresh),
            "credit_spike_threshold": float(credit_thresh),
        }

        header = (
            "# ======================================================================\n"
            "#              MACRO REGIME ALLOCATOR -- Configuration\n"
            "#  Change these values to control how the model trains & allocates.\n"
            "# ======================================================================\n\n"
        )
        with open(CONFIG_PATH, "w") as f:
            f.write(header)
            yaml.dump(new_cfg, f, default_flow_style=False, sort_keys=False)

        st.success("Configuration saved! Re-run the model to apply changes.")
        load_config_yaml.clear()

    # ── Sigmoid visualizer ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Sigmoid Weight Preview")
    st.caption("How model probability maps to equity weight with current settings")

    probs = np.linspace(0, 1, 200)
    bias = np.log(baseline_eq / max(1 - baseline_eq, 0.001))
    x_vals = (probs - 0.5) * steepness + bias
    weights = 1.0 / (1.0 + np.exp(-x_vals))
    weights = np.clip(weights, min_w, max_w)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=probs, y=weights, name="Equity Weight",
        line=dict(color="#2196F3", width=3),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="P(Equity Outperforms)", yaxis_title="Equity Weight",
        height=400, title="Probability to Weight Mapping (Sigmoid)",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN MODEL
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Run Model":
    st.title("Run Model")
    st.caption("Execute the macro regime allocator pipeline")

    col1, col2 = st.columns(2)

    with col1:
        mode = st.radio(
            "Run Mode",
            ["Full Run (download + backtest)",
             "Quick Run (cached data)",
             "Predict Latest (fresh data)",
             "Quick Signal (cached)"],
        )

    with col2:
        st.info(
            "**Full Run** — Downloads fresh data from FRED & Yahoo Finance, "
            "trains the model, runs backtest, generates plots.\n\n"
            "**Quick Run** — Uses cached data, re-runs backtest.\n\n"
            "**Predict Latest** — Full run + shows current allocation.\n\n"
            "**Quick Signal** — Uses cached data, shows current allocation."
        )

    if st.button("Run", type="primary"):
        cmd_map = {
            "Full Run (download + backtest)": "run",
            "Quick Run (cached data)": "fast",
            "Predict Latest (fresh data)": "predict",
            "Quick Signal (cached)": "signal",
        }
        target = cmd_map[mode]

        with st.spinner(f"Running `make {target}` in backend..."):
            try:
                result = subprocess.run(
                    ["make", target],
                    cwd=BACKEND_DIR,
                    capture_output=True, text=True, timeout=600,
                )
            except subprocess.TimeoutExpired:
                st.error("Run timed out after 10 minutes.")
                st.stop()

        if result.returncode == 0:
            st.success("Model run completed!")
            load_backtest.clear()
            load_metrics.clear()
            load_features.clear()
            load_merged.clear()
            load_config_yaml.clear()
        else:
            st.error("Run failed!")

        with st.expander("Output Log", expanded=result.returncode != 0):
            st.code(result.stdout + "\n" + result.stderr, language="text")

    # Show existing report
    report_path = os.path.join(OUTPUT_DIR, "report.md")
    if os.path.exists(report_path):
        st.divider()
        st.header("Latest Report")
        with open(report_path) as f:
            st.markdown(f.read())
