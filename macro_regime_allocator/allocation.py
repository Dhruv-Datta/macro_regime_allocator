"""
Convert predicted class probabilities into portfolio weights.

Two mechanisms for aggressive, responsive allocation:

1. Sigmoid mapping: amplifies model probabilities into bigger weight swings.
   A model that's 60% confident in equity → ~80% equity weight (not 62%).

2. Crash overlay: uses CURRENT (unlagged) observable market data to force
   rapid defensive positioning. When VIX spikes, drawdowns deepen, or credit
   widens — cut equity weight immediately, don't wait for the model to catch up.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from config import Config


def sigmoid_weight_map(p_equity: float, cfg: Config) -> float:
    """
    Map P(equity outperforms) through a steep sigmoid centered at 0.5.

    With steepness=10, baseline=60/40:
      P=0.50 → weight≈0.60   (neutral → standard 60/40)
      P=0.55 → weight≈0.73   (modest confidence → meaningful tilt)
      P=0.60 → weight≈0.82   (moderate confidence → strong tilt)
      P=0.40 → weight≈0.37   (model says safe → big defensive move)
      P=0.35 → weight≈0.25   (strong safe signal → go defensive fast)

    The equity-biased baseline shifts the center so neutral (P=0.5)
    maps to ~60% equity instead of 50%.
    """
    steepness = cfg.allocation_steepness

    # Shift center so P=0.5 maps to 70% equity (the baseline bias)
    # We want sigmoid(0) to give 0.70, so we add a bias term
    # sigmoid(bias) = 0.70 → bias = ln(0.70/0.30) ≈ 0.847
    bias = np.log(cfg.equal_weight[0] / cfg.equal_weight[1])

    x = (p_equity - 0.5) * steepness + bias
    equity_weight = 1.0 / (1.0 + np.exp(-x))

    return equity_weight


def crash_overlay(
    equity_weight: float,
    market_data: Optional[Dict[str, float]],
    cfg: Config,
) -> Tuple[float, str]:
    """
    Defensive overlay using current observable market conditions.

    Key design: reacts to ACTIVE deterioration, not static levels.
    A drawdown that's recovering should NOT trigger defense.
    VIX spikes matter, but persistent elevated VIX doesn't.

    Returns adjusted equity weight and a reason string.
    """
    if not cfg.crash_overlay or market_data is None:
        return equity_weight, "none"

    penalties = []
    vix_change = market_data.get("vix_1m_change")
    vix_ts = market_data.get("vix_term_structure")

    # 1. VIX spike: sharp jump = crash unfolding RIGHT NOW
    #    Higher threshold = fewer false positives during normal vol
    if vix_change is not None and vix_change > cfg.vix_spike_threshold:
        severity = min((vix_change - cfg.vix_spike_threshold) / 15.0, 1.0)
        penalties.append(("vix_spike", severity * 0.50))

    # 2. VIX backwardation + rising VIX = full panic mode
    #    Tighter conditions: need stronger backwardation AND bigger VIX move
    if (vix_ts is not None and vix_ts > 1.08
            and vix_change is not None and vix_change > 3.0):
        severity = min((vix_ts - 1.08) * 5.0, 1.0)
        penalties.append(("vix_panic", severity * 0.35))

    # 3. Drawdown accelerating: deep drawdown getting worse this month
    #    Only when combined with VIX stress (avoids staying defensive in
    #    grinding sideways markets). Deeper threshold = fewer false positives.
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
    adjusted = equity_weight * (1.0 - total_penalty)

    reasons = "+".join(name for name, _ in penalties)
    return adjusted, reasons


def confidence_blend(probabilities: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Legacy smooth blend. Kept for comparison but sigmoid is now default.
    """
    baseline = np.array(cfg.equal_weight[:2])
    max_prob = np.max(probabilities)
    raw_strength = (max_prob - 0.5) / 0.5
    blend_strength = np.clip(raw_strength, 0.0, 1.0) ** 0.7
    blended = blend_strength * probabilities + (1.0 - blend_strength) * baseline
    return blended


def apply_caps(weights: np.ndarray, cfg: Config) -> np.ndarray:
    weights = weights.copy()
    weights = np.clip(weights, cfg.min_weight, cfg.max_weight)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = np.array(cfg.equal_weight[:2])
    return weights


def probabilities_to_weights(
    probabilities: np.ndarray,
    cfg: Config,
    market_data: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Full pipeline: model probabilities → sigmoid amplification → crash overlay → caps.

    Returns (raw_proba, final_weights, overlay_reason).
    """
    raw = probabilities.copy()
    p_equity = probabilities[0]

    # Step 1: Sigmoid mapping (aggressive probability → weight)
    equity_weight = sigmoid_weight_map(p_equity, cfg)

    # Step 2: Crash overlay (defensive adjustment from current market data)
    equity_weight, overlay_reason = crash_overlay(equity_weight, market_data, cfg)

    weights = np.array([equity_weight, 1.0 - equity_weight])

    # Step 3: Apply caps
    weights = apply_caps(weights, cfg)

    return raw, weights, overlay_reason


def get_equal_weights(cfg: Config) -> np.ndarray:
    return np.array(cfg.equal_weight)


