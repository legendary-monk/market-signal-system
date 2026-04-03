"""
features.py — Market Feature Engineering
==========================================
Responsibility: Compute technical features from raw OHLCV data.
These features become inputs to the signal engine.

Current features computed:
1. Trend Score       — Direction and strength of recent price movement
2. Volatility Regime — Current volatility vs historical baseline
3. Momentum          — Rate of change in returns
4. RSI               — Relative Strength Index (overbought/oversold)

WHY these specific features:
- All computable from daily OHLCV with no paid data
- Interpretable: Each has a clear financial meaning
- Robust: Fail gracefully when data is limited

WHY NOT moving average crossovers, Bollinger Bands, MACD:
- Require longer data histories (more Yahoo Finance calls)
- Introduce more parameters to tune
- Overfitting risk on a simple daily signal system
- Keep it simple first, add complexity only when baseline is validated
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

import config
from logger import get_logger

logger = get_logger(__name__)


# Type alias for our feature output
FeatureDict = Dict[str, Any]


def _compute_trend_score(df: pd.DataFrame) -> Optional[float]:
    """
    Computes a trend score in [-1.0, +1.0] based on linear regression slope
    of closing prices over the TREND_WINDOW.
    
    WHY linear regression slope instead of simple % change:
    - % change is easily distorted by a single extreme day (gap up/down)
    - Regression slope captures the overall direction and ignores single outliers
    - More statistically robust for a noisy price series
    
    Returns:
        float in [-1.0, +1.0]: Positive = uptrend, Negative = downtrend.
        Normalized so ±1.0 represents an unusually strong trend.
    """
    window = config.TREND_WINDOW
    
    if len(df) < window:
        logger.warning("Insufficient data for trend calculation (need %d rows, have %d)",
                       window, len(df))
        return None
    
    recent_closes = df['Close'].iloc[-window:].values
    
    # WHY normalize prices to % of first day:
    # This makes the slope comparable across different absolute price levels.
    # A slope of 10 on Nifty (~22,000) is tiny; we need relative measures.
    baseline = recent_closes[0]
    if baseline == 0:
        return None
    
    normalized = (recent_closes / baseline - 1) * 100  # In percentage terms
    
    # Fit linear regression: y = slope * x + intercept
    x = np.arange(window, dtype=float)
    
    try:
        # np.polyfit is available on all numpy versions, runs fast on mobile
        coeffs = np.polyfit(x, normalized, 1)
        slope = coeffs[0]  # Daily % change implied by the trend line
        
        # Normalize slope to [-1.0, +1.0]:
        # Cap at ±2% daily slope (unusually strong trend = score of ±1.0)
        # WHY 2%: Nifty rarely trends at >2% per day for multiple days.
        max_daily_pct = 2.0
        trend_score = slope / max_daily_pct
        trend_score = max(-1.0, min(1.0, trend_score))
        
        logger.debug("Trend: slope=%.4f%% per day → score=%.4f", slope, trend_score)
        return round(trend_score, 4)
    
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.error("Linear regression failed: %s", e)
        return None


def _compute_volatility_regime(df: pd.DataFrame) -> Optional[str]:
    """
    Classifies current volatility as HIGH, NORMAL, or LOW.
    
    Method: Compare recent N-day realized volatility to longer-term baseline.
    
    WHY realized volatility: We're not using options data (no VIX access).
    Realized vol from daily returns is our best free proxy for current risk.
    
    Returns:
        str: "HIGH", "NORMAL", or "LOW"
        None if insufficient data.
    """
    vol_window = config.VOLATILITY_WINDOW
    baseline_window = max(vol_window * 2, 20)  # WHY 2x: Need context to compare against
    
    if len(df) < baseline_window:
        logger.warning("Insufficient data for volatility (need %d rows, have %d)",
                       baseline_window, len(df))
        return None
    
    returns = df['Returns'].dropna()
    
    # Recent volatility: annualized standard deviation of last N returns
    # WHY annualize (×√252): Makes vol comparable across different periods.
    # 252 = approximate trading days per year.
    recent_vol = returns.iloc[-vol_window:].std() * np.sqrt(252)
    baseline_vol = returns.iloc[-baseline_window:].std() * np.sqrt(252)
    
    if baseline_vol == 0 or np.isnan(baseline_vol):
        logger.warning("Baseline volatility is zero or NaN — cannot classify")
        return None
    
    vol_ratio = recent_vol / baseline_vol
    
    logger.debug(
        "Volatility: recent=%.4f | baseline=%.4f | ratio=%.2f",
        recent_vol, baseline_vol, vol_ratio
    )
    
    if vol_ratio >= config.HIGH_VOLATILITY_MULTIPLIER:
        return "HIGH"
    elif vol_ratio <= (1 / config.HIGH_VOLATILITY_MULTIPLIER):
        return "LOW"
    else:
        return "NORMAL"


def _compute_volatility_score(regime: Optional[str]) -> float:
    """
    Converts volatility regime to a numeric score for the signal engine.
    
    WHY this mapping:
    - HIGH volatility → bearish signal. High vol often coincides with fear/selling.
    - LOW volatility → mildly bullish. Complacency often accompanies rising markets.
    - NORMAL → neutral
    
    This is a simplification — sometimes LOW vol precedes a breakout UP or DOWN.
    That's a limitation of this simple system.
    
    Returns:
        float in [-1.0, +1.0]
    """
    mapping = {
        "HIGH": -0.5,      # Elevated risk environment
        "NORMAL": 0.0,     # No directional signal
        "LOW": 0.3,        # Calm markets, slight bullish lean
        None: 0.0,         # Missing data → neutral
    }
    return mapping.get(regime, 0.0)


def _compute_momentum(df: pd.DataFrame) -> Optional[float]:
    """
    Rate of change in returns: are returns accelerating or decelerating?
    
    WHY momentum: A market with positive returns that are INCREASING is
    stronger than one with positive returns that are DECREASING.
    
    Returns:
        float in [-1.0, +1.0]: Positive = accelerating returns (bullish momentum)
    """
    if len(df) < 6:
        return None
    
    returns = df['Returns'].dropna()
    
    # Average return in most recent 3 trading days vs prior 3 trading days
    recent_avg = returns.iloc[-3:].mean()
    prior_avg = returns.iloc[-6:-3].mean()
    
    # Momentum = how much returns have changed
    momentum_raw = recent_avg - prior_avg
    
    # Normalize: cap at ±2% daily return differential
    max_diff = 0.02
    momentum = momentum_raw / max_diff
    momentum = max(-1.0, min(1.0, momentum))
    
    logger.debug("Momentum: recent_avg=%.4f | prior_avg=%.4f → score=%.4f",
                 recent_avg, prior_avg, momentum)
    
    return round(momentum, 4)


def _compute_rsi(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Relative Strength Index (RSI) — classic momentum oscillator.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss over `period` days.
    
    WHY RSI:
    - RSI > 70: Overbought → potential reversal down → bearish lean
    - RSI < 30: Oversold → potential reversal up → bullish lean
    - RSI 45–55: Neutral
    
    WHY NOT make this a primary signal: RSI works poorly in trending markets.
    In a strong uptrend, RSI stays overbought for weeks. We use it as a
    secondary weight, not a primary driver.
    
    Returns:
        float in [0, 100] or None
    """
    if len(df) < period + 1:
        return None
    
    closes = df['Close'].iloc[-(period + 5):]  # A bit extra for stability
    delta = closes.diff().dropna()
    
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    
    # Simple moving average RSI (Wilder's smoothing is more standard,
    # but SMA is fine for our signal granularity)
    avg_gain = gains.iloc[-period:].mean()
    avg_loss = losses.iloc[-period:].mean()
    
    if avg_loss == 0:
        # No losses in window = extremely overbought
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    logger.debug("RSI(%d): %.2f", period, rsi)
    return round(rsi, 2)


def _rsi_to_signal(rsi: Optional[float]) -> float:
    """
    Converts RSI to a signal score in [-1.0, +1.0].
    
    RSI < 30 → strong bullish (oversold)
    RSI 30–45 → mild bullish
    RSI 45–55 → neutral
    RSI 55–70 → mild bearish
    RSI > 70 → strong bearish (overbought)
    """
    if rsi is None:
        return 0.0
    
    if rsi < 30:
        return 0.8      # Oversold — contrarian bullish signal
    elif rsi < 45:
        return 0.3
    elif rsi > 70:
        return -0.8     # Overbought — contrarian bearish signal
    elif rsi > 55:
        return -0.3
    else:
        return 0.0      # Neutral zone


def compute_features(df: pd.DataFrame) -> FeatureDict:
    """
    Main entry point. Computes all features from the OHLCV DataFrame.
    
    Args:
        df: Output of market_data.fetch_market_data().
    
    Returns:
        FeatureDict with all computed features. Any that couldn't be
        computed will have None values — the signal engine handles this.
    
    Design principle: This function NEVER raises. It always returns a dict,
    possibly with None values. Missing data → neutral contribution to signal.
    """
    if df is None or df.empty:
        logger.error("No market data provided to feature computation")
        return _empty_features()
    
    logger.info("Computing features from %d trading days of data...", len(df))
    
    # Compute each feature independently
    # WHY independent try/except: One feature failing shouldn't kill the others
    
    try:
        trend_score = _compute_trend_score(df)
    except Exception as e:
        logger.error("Trend computation error: %s", e, exc_info=True)
        trend_score = None
    
    try:
        vol_regime = _compute_volatility_regime(df)
        vol_score = _compute_volatility_score(vol_regime)
    except Exception as e:
        logger.error("Volatility computation error: %s", e, exc_info=True)
        vol_regime = None
        vol_score = 0.0
    
    try:
        momentum = _compute_momentum(df)
    except Exception as e:
        logger.error("Momentum computation error: %s", e, exc_info=True)
        momentum = None
    
    try:
        rsi = _compute_rsi(df)
        rsi_signal = _rsi_to_signal(rsi)
    except Exception as e:
        logger.error("RSI computation error: %s", e, exc_info=True)
        rsi = None
        rsi_signal = 0.0
    
    # Latest price info
    latest_close = float(df['Close'].iloc[-1]) if not df.empty else None
    price_change_1d = None
    price_change_5d = None
    
    try:
        if len(df) >= 2:
            price_change_1d = round(
                ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100, 4
            )
        if len(df) >= 6:
            price_change_5d = round(
                ((df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1) * 100, 4
            )
    except Exception as e:
        logger.error("Price change calculation error: %s", e)
    
    features = {
        # Primary signal inputs
        'trend_score': trend_score,           # [-1.0, +1.0]
        'vol_score': vol_score,               # [-1.0, +1.0]
        'rsi_signal': rsi_signal,             # [-1.0, +1.0]
        
        # Human-readable context (for Telegram message)
        'vol_regime': vol_regime,             # "HIGH", "NORMAL", "LOW"
        'rsi': rsi,                           # [0, 100]
        'momentum': momentum,                 # [-1.0, +1.0]
        'latest_close': latest_close,
        'price_change_1d': price_change_1d,   # Percentage
        'price_change_5d': price_change_5d,   # Percentage
        'data_rows': len(df),
    }
    
    logger.info(
        "Features: trend=%.4f | vol_regime=%s | RSI=%.1f | momentum=%.4f",
        trend_score or 0, vol_regime or 'N/A', rsi or 0, momentum or 0
    )
    
    return features


def _empty_features() -> FeatureDict:
    """Returns a zeroed-out feature dict for when data is unavailable."""
    return {
        'trend_score': None,
        'vol_score': 0.0,
        'rsi_signal': 0.0,
        'vol_regime': None,
        'rsi': None,
        'momentum': None,
        'latest_close': None,
        'price_change_1d': None,
        'price_change_5d': None,
        'data_rows': 0,
    }


# ── Quick sanity test ──
if __name__ == "__main__":
    from market_data import fetch_market_data
    
    df = fetch_market_data()
    if df is not None:
        features = compute_features(df)
        print("\nComputed Features:")
        for key, val in features.items():
            print(f"  {key:20s}: {val}")
    else:
        print("Cannot test features — market data unavailable")
