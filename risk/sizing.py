"""
Constrained fractional Kelly sizing.
"""

from typing import Dict, Any
import config


def _kelly_fraction(p: float, b: float) -> float:
    q = 1.0 - p
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - q) / b)


def recommend_weight(p_up: float, uncertainty: float,
                     shock_prob: float, consecutive_losses: int = 0) -> Dict[str, Any]:
    base = _kelly_fraction(p_up, config.PAYOFF_RATIO)
    frac = base * config.FRACTIONAL_KELLY

    # Uncertainty penalty
    frac *= max(0.0, 1.0 - config.WEIGHT_UNCERTAINTY_PENALTY * max(0.0, uncertainty))

    # Shock regime penalty
    if shock_prob >= config.SHOCK_REGIME_PROB_THRESHOLD:
        frac *= 0.5

    # Drawdown throttle
    if consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
        frac *= config.DRAWDOWN_THROTTLE_FACTOR

    weight = min(max(frac, config.MIN_RECOMMENDED_WEIGHT), config.MAX_RECOMMENDED_WEIGHT)

    return {
        'recommended_weight': round(float(weight), 4),
        'weight_reason': (
            f"kelly={base:.4f}, uncertainty={uncertainty:.3f}, "
            f"shock_prob={shock_prob:.3f}, losses={consecutive_losses}"
        ),
    }
