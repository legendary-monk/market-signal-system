"""
State-space market model (Kalman filter based).
Produces latent drift/acceleration/volatility estimates and 1-day up probability.
"""

from typing import Dict, Any
import math
import numpy as np
import pandas as pd


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_state(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Kalman filter on returns with latent state:
        x_t = [mu_t, a_t, sigma_t]
    Measurement:
        r_t = [1, 0, 0] x_t + v_t
    """
    if df is None or df.empty or 'Returns' not in df.columns:
        return {
            'latent_drift': 0.0,
            'latent_acceleration': 0.0,
            'vol_state': 0.01,
            'state_uncertainty': 1.0,
            'p_up_1d': 0.5,
        }

    returns = df['Returns'].dropna().values.astype(float)
    if len(returns) < 10:
        return {
            'latent_drift': float(np.mean(returns)) if len(returns) > 0 else 0.0,
            'latent_acceleration': 0.0,
            'vol_state': float(np.std(returns) if len(returns) > 1 else 0.01),
            'state_uncertainty': 1.0,
            'p_up_1d': 0.5,
        }

    # Transition matrix (constant acceleration style for drift term)
    F = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 0.9, 0.0],
        [0.0, 0.0, 0.95],
    ])
    H = np.array([[1.0, 0.0, 0.0]])

    Q = np.diag([1e-5, 1e-5, 1e-6])
    R = np.array([[max(np.var(returns), 1e-6)]])

    x = np.array([[returns[0]], [0.0], [np.std(returns)]], dtype=float)
    P = np.eye(3) * 0.1

    for r in returns:
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        y = np.array([[r]]) - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(3) - K @ H) @ P

    latent_drift = float(x[0, 0])
    latent_acceleration = float(x[1, 0])
    vol_state = max(float(abs(x[2, 0])), 1e-6)
    state_uncertainty = float(np.trace(P))

    z = latent_drift / max(vol_state, 1e-6)
    p_up_1d = min(max(_normal_cdf(z), 0.01), 0.99)

    return {
        'latent_drift': round(latent_drift, 6),
        'latent_acceleration': round(latent_acceleration, 6),
        'vol_state': round(vol_state, 6),
        'state_uncertainty': round(state_uncertainty, 6),
        'p_up_1d': round(p_up_1d, 4),
    }
