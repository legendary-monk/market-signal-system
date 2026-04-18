"""
Physics-inspired market diagnostics:
- kinetic-like energy from normalized return
- entropy from regime probabilities
"""

from typing import Dict, Any
import math


def compute_physics_metrics(last_return: float, vol_estimate: float,
                            regime_probs: Dict[str, float]) -> Dict[str, Any]:
    sigma = max(abs(vol_estimate), 1e-6)
    norm_return = last_return / sigma
    energy = 0.5 * (norm_return ** 2)

    pvals = [
        max(regime_probs.get('p_diffusion', 1 / 3), 1e-12),
        max(regime_probs.get('p_trend', 1 / 3), 1e-12),
        max(regime_probs.get('p_shock', 1 / 3), 1e-12),
    ]
    entropy = -sum(p * math.log(p) for p in pvals)

    return {
        'energy': round(float(energy), 4),
        'entropy': round(float(entropy), 4),
        'normalized_return': round(float(norm_return), 4),
    }
