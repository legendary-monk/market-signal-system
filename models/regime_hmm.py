"""
Lightweight regime model that outputs probabilities for:
LOW_VOL_DIFFUSION, TREND_PERSISTENCE, SHOCK.

This is a pragmatic approximation of a 3-state HMM suitable for zero-cost infra.
"""

from typing import Dict, Any
import math
import numpy as np
import pandas as pd


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def infer_regime_probabilities(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or 'Returns' not in df.columns:
        return {
            'p_diffusion': 0.34,
            'p_trend': 0.33,
            'p_shock': 0.33,
            'entropy': 1.0986,
            'regime': 'UNKNOWN',
        }

    rets = df['Returns'].dropna()
    if len(rets) < 15:
        return {
            'p_diffusion': 0.34,
            'p_trend': 0.33,
            'p_shock': 0.33,
            'entropy': 1.0986,
            'regime': 'UNKNOWN',
        }

    short_mean = float(rets.iloc[-5:].mean())
    short_std = float(rets.iloc[-5:].std() or 0.0)
    long_std = float(rets.iloc[-20:].std() or 1e-6)
    z_vol = short_std / max(long_std, 1e-6)

    # Regime logits
    l_diffusion = 1.0 - abs(short_mean) * 100 - max(0.0, z_vol - 1.0)
    l_trend = abs(short_mean) * 120 - abs(z_vol - 1.0)
    l_shock = (z_vol - 1.0) * 2.0

    probs = _softmax(np.array([l_diffusion, l_trend, l_shock], dtype=float))
    p_diffusion, p_trend, p_shock = [float(p) for p in probs]

    entropy = -sum(p * math.log(max(p, 1e-12)) for p in [p_diffusion, p_trend, p_shock])
    regime_map = ['LOW_VOL_DIFFUSION', 'TREND_PERSISTENCE', 'SHOCK']
    regime = regime_map[int(np.argmax(probs))]

    return {
        'p_diffusion': round(p_diffusion, 4),
        'p_trend': round(p_trend, 4),
        'p_shock': round(p_shock, 4),
        'entropy': round(entropy, 4),
        'regime': regime,
    }
