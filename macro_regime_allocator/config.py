"""
Configuration for the macro regime allocation system.
Two assets: equities and safe rate (risk-free short-term bonds).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Config:
    # ── Date range ──────────────────────────────────────────────────────
    start_date: str = "2005-01-01"
    end_date: str = "2026-03-01"

    # ── Asset proxies ───────────────────────────────────────────────────
    equity_ticker: str = "SPY"

    asset_tickers: Dict[str, str] = field(default_factory=lambda: {
        "equity": "SPY",
    })

    # VIX data for crash detection
    vix_ticker: str = "^VIX"
    vix3m_ticker: str = "^VIX3M"      # 3-month VIX for term structure

    # The "safe" return is the fed funds rate (risk-free short-term rate).
    safe_rate_series: str = "fed_funds"

    # ── FRED series IDs ─────────────────────────────────────────────────
    fred_series: Dict[str, str] = field(default_factory=lambda: {
        "cpi":              "CPIAUCSL",
        "core_cpi":         "CPILFESL",
        "unemployment":     "UNRATE",
        "treasury_10y":     "DGS10",
        "treasury_2y":      "DGS2",
        "fed_funds":        "FEDFUNDS",
        "credit_spread":    "BAMLH0A0HYM2",
        "industrial_prod":  "INDPRO",
    })

    fred_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("FRED_API_KEY")
    )

    # ── Forecast & rebalance ────────────────────────────────────────────
    forecast_horizon_months: int = 3
    rebalance_frequency: str = "M"

    # ── Feature engineering ─────────────────────────────────────────────
    macro_lag_months: int = 1
    zscore_window: int = 60
    momentum_window: int = 3
    volatility_window: int = 3

    # ── Model ───────────────────────────────────────────────────────────
    # "logistic"     = retrain from scratch each step (LogisticRegression)
    # "incremental"  = online updates via SGDClassifier with partial_fit
    model_type: str = "logistic"
    regularization_C: float = 0.5
    sgd_alpha: float = 0.005
    class_weight: Optional[str] = "balanced"
    max_iter: int = 1000

    # Incremental learning settings
    incremental_warmstart: bool = True
    recency_halflife_months: int = 36
    checkpoint_every: int = 12

    # ── Backtest ────────────────────────────────────────────────────────
    window_type: str = "expanding"
    rolling_window_months: int = 120
    min_train_months: int = 36

    # ── Allocation ──────────────────────────────────────────────────────
    # Equity-biased: default to 70% equity, only go defensive on strong signal
    min_weight: float = 0.05             # per-asset floor (allows 95% equity)
    max_weight: float = 0.95             # per-asset cap
    confidence_blend: bool = True
    equal_weight: List[float] = field(default_factory=lambda: [0.70, 0.30])

    # Aggressive sigmoid: amplifies model probability into bigger weight swings
    allocation_steepness: float = 10.0   # higher = sharper transitions

    # Crash-detection overlay: uses current (unlagged) observable market data
    # to force rapid defensive positioning when danger signals fire
    crash_overlay: bool = True
    vix_spike_threshold: float = 8.0     # VIX 1m change above this = danger
    drawdown_defense_threshold: float = -12.0 # equity drawdown % triggers defense
    credit_spike_threshold: float = 1.0  # credit spread 3m widening = danger

    # ── Benchmarks ──────────────────────────────────────────────────────
    static_benchmark_weights: List[float] = field(
        default_factory=lambda: [0.60, 0.40]  # 60/40 equity/safe
    )

    # ── Paths ───────────────────────────────────────────────────────────
    data_dir: str = "data"
    output_dir: str = "outputs"
    plot_dir: str = "outputs/plots"
    model_path: str = "outputs/model.joblib"
    checkpoint_dir: str = "outputs/checkpoints"

    # ── Asset class names (order matters) ───────────────────────────────
    asset_classes: List[str] = field(
        default_factory=lambda: ["equity", "safe"]
    )
    class_labels: Dict[int, str] = field(
        default_factory=lambda: {0: "equity", 1: "safe"}
    )
