"""
Bayesian-style probability fusion utilities.
"""

from typing import Dict
import math
import config


def _clamp_prob(p: float) -> float:
    return min(max(float(p), 1e-4), 1 - 1e-4)


def _logit(p: float) -> float:
    p = _clamp_prob(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def fuse_probabilities(p_price: float, p_sentiment: float, p_regime: float,
                       weights: Dict[str, float] = None) -> Dict[str, float]:
    if weights is None:
        weights = {
            'price': config.FUSION_WEIGHT_PRICE,
            'sentiment': config.FUSION_WEIGHT_SENTIMENT,
            'regime': config.FUSION_WEIGHT_REGIME,
        }

    l = (
        weights['price'] * _logit(p_price) +
        weights['sentiment'] * _logit(p_sentiment) +
        weights['regime'] * _logit(p_regime)
    )
    p = _sigmoid(l)

    disagreement = (abs(p_price - p_sentiment) +
                    abs(p_price - p_regime) +
                    abs(p_sentiment - p_regime)) / 3.0

    return {
        'posterior_p_up': round(float(p), 4),
        'model_disagreement': round(float(disagreement), 4),
        'epistemic_uncertainty': round(float(min(1.0, disagreement * 1.5)), 4),
    }
