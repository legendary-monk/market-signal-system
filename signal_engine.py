"""
signal_engine.py — Signal Generation Engine
=============================================
Responsibility: Combine sentiment and market features into a final
signal: BULLISH / BEARISH / NEUTRAL with a confidence score.

Architecture Decision: Weighted Linear Combination
---------------------------------------------------
WHY NOT a machine learning model:
- We have no labeled historical data to train on yet
- ML models need backtesting infrastructure we haven't built
- A simple weighted model is TRANSPARENT — you can audit exactly
  why it said BULLISH on any given day
- It's our baseline. Once we have 60+ days of predictions in
  predictions.csv, we can compare vs a trained model.

WHY NOT a purely rules-based system (if sentiment > X: BULLISH):
- Too brittle — threshold needs manual tuning
- Ignores interaction effects between features
- Weighted combination naturally handles mixed signals

Signal Computation:
    raw_score = (w1 × sentiment) + (w2 × trend) + (w3 × volatility)
    
    raw_score ∈ [-1.0, +1.0]
    
    Mapped to [0.0, 1.0] for confidence:
    normalized = (raw_score + 1) / 2
    
    Classification:
    normalized > BULLISH_THRESHOLD → BULLISH
    normalized < BEARISH_THRESHOLD → BEARISH
    else                           → NEUTRAL
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
import math

import config
from logger import get_logger

logger = get_logger(__name__)


# Type alias
SignalResult = Dict[str, Any]


def _safe_score(value: Optional[float], default: float = 0.0) -> float:
    """
    Returns the value if valid, else the default.
    
    WHY: None features (data unavailable) should contribute 0 (neutral)
    to the weighted sum, not crash the computation.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


def _compute_raw_score(
    sentiment_score: float,
    features: Dict[str, Any]
) -> float:
    """
    Computes the raw composite score from all inputs.
    
    The raw score is a weighted sum in [-1.0, +1.0] where:
    +1.0 = maximally bullish signal
    -1.0 = maximally bearish signal
     0.0 = perfectly neutral / conflicting signals
    
    Args:
        sentiment_score: From sentiment.analyze_sentiment() — range [-1, +1]
        features: From features.compute_features()
    
    Returns:
        float in [-1.0, +1.0]
    """
    # Extract feature scores, defaulting to 0 (neutral) if unavailable
    trend_score = _safe_score(features.get('trend_score'))
    vol_score = _safe_score(features.get('vol_score'))
    rsi_signal = _safe_score(features.get('rsi_signal'))
    momentum = _safe_score(features.get('momentum'))
    
    # Track which components have actual data vs defaults
    has_sentiment = sentiment_score != 0.0
    has_market = features.get('data_rows', 0) > 0
    
    if not has_sentiment and not has_market:
        logger.error("No data available for signal computation — returning NEUTRAL")
        return 0.0
    
    # Primary weighted combination using configured weights
    # Note: We split the TREND weight between trend and RSI/momentum sub-signals
    # WHY: Trend alone is the direction; RSI and momentum provide context
    
    # Remap: The market TREND weight is split 60/40 between pure trend and momentum
    # The VOLATILITY weight includes a 50/50 blend of vol regime and RSI-contrarian
    trend_component = (
        0.60 * trend_score +     # Dominant trend direction
        0.40 * momentum          # Momentum quality
    ) * config.WEIGHT_TREND
    
    vol_component = (
        0.50 * vol_score +       # Volatility regime signal
        0.50 * rsi_signal        # RSI contrarian signal
    ) * config.WEIGHT_VOLATILITY
    
    sentiment_component = sentiment_score * config.WEIGHT_SENTIMENT
    
    raw_score = sentiment_component + trend_component + vol_component
    
    # Clamp to valid range
    raw_score = max(-1.0, min(1.0, raw_score))
    
    logger.debug(
        "Score decomposition: sentiment=%.4f×%.2f=%.4f | "
        "trend_comp=%.4f×%.2f=%.4f | vol_comp=%.4f×%.2f=%.4f | "
        "TOTAL=%.4f",
        sentiment_score, config.WEIGHT_SENTIMENT, sentiment_component,
        (0.60 * trend_score + 0.40 * momentum), config.WEIGHT_TREND, trend_component,
        (0.50 * vol_score + 0.50 * rsi_signal), config.WEIGHT_VOLATILITY, vol_component,
        raw_score
    )
    
    return raw_score


def _classify_signal(normalized_score: float) -> str:
    """
    Converts normalized score [0, 1] into BULLISH / BEARISH / NEUTRAL.
    
    WHY normalized (0 to 1) instead of raw (-1 to +1):
    Confidence scores are more intuitively expressed as percentages.
    A score of 0.72 → "72% confident" is clearer than "raw score 0.44".
    """
    if normalized_score >= config.BULLISH_THRESHOLD:
        return "BULLISH"
    elif normalized_score <= config.BEARISH_THRESHOLD:
        return "BEARISH"
    else:
        return "NEUTRAL"


def _compute_confidence(raw_score: float, normalized: float,
                        has_sentiment: bool, has_market: bool) -> float:
    """
    Computes signal confidence — how much we trust this signal.
    
    Confidence is penalized when:
    - Only one data source is available (no market data OR no news)
    - Signals are conflicting (close to 0.5 on the normalized scale)
    
    Returns:
        float in [0.0, 1.0]
    """
    # Base confidence = distance from the neutral midpoint (0.5)
    # WHY: A score of 0.72 is 0.22 above neutral — that distance IS the confidence
    # A score of 0.51 is nearly 50/50 — very low confidence
    base_confidence = abs(normalized - 0.5) * 2  # Scale to [0, 1]
    
    # Apply penalty for missing data sources
    data_penalty = 1.0
    if not has_sentiment:
        data_penalty *= 0.7    # 30% penalty for missing news sentiment
        logger.warning("Confidence penalized 30%: no news sentiment data")
    if not has_market:
        data_penalty *= 0.6    # 40% penalty for missing market data
        logger.warning("Confidence penalized 40%: no market price data")
    
    confidence = base_confidence * data_penalty
    
    # Floor at 5% (never claim 0% confidence — that's meaningless)
    # Ceiling at 95% (never claim certainty — this is a heuristic system)
    confidence = max(0.05, min(0.95, confidence))
    
    return round(confidence, 4)


def _generate_reasoning(
    signal: str,
    sentiment_score: float,
    features: Dict,
    normalized_score: float
) -> list:
    """
    Generates human-readable bullet points explaining WHY this signal was generated.
    
    WHY invest effort in reasoning: Signal alone is useless without justification.
    A BULLISH signal from sentiment=+0.4 and a falling trend is different from
    BULLISH with sentiment=+0.4 AND an upward trend. Context matters.
    
    Returns:
        List of strings, each a reason contributing to the signal.
    """
    reasons = []
    
    # Sentiment reasoning
    if abs(sentiment_score) < 0.05:
        reasons.append("News sentiment is mixed/neutral")
    elif sentiment_score > 0:
        strength = "strongly" if sentiment_score > 0.3 else "mildly"
        reasons.append(f"News sentiment is {strength} positive ({sentiment_score:+.2f})")
    else:
        strength = "strongly" if sentiment_score < -0.3 else "mildly"
        reasons.append(f"News sentiment is {strength} negative ({sentiment_score:+.2f})")
    
    # Trend reasoning
    trend_score = features.get('trend_score')
    price_change_5d = features.get('price_change_5d')
    if trend_score is not None:
        if trend_score > 0.2:
            reasons.append(
                f"5-day price trend is upward ({price_change_5d:+.2f}%)" 
                if price_change_5d else "5-day trend is upward"
            )
        elif trend_score < -0.2:
            reasons.append(
                f"5-day price trend is downward ({price_change_5d:+.2f}%)"
                if price_change_5d else "5-day trend is downward"
            )
        else:
            reasons.append("Price trend is largely flat over 5 days")
    else:
        reasons.append("Market price data unavailable (trend not computed)")
    
    # Volatility reasoning
    vol_regime = features.get('vol_regime')
    if vol_regime == "HIGH":
        reasons.append("Volatility is elevated — elevated risk environment")
    elif vol_regime == "LOW":
        reasons.append("Volatility is low — calm, trending market conditions")
    elif vol_regime == "NORMAL":
        reasons.append("Volatility is within normal range")
    
    # RSI reasoning
    rsi = features.get('rsi')
    if rsi is not None:
        if rsi > 70:
            reasons.append(f"RSI is overbought ({rsi:.0f}) — caution on chasing rally")
        elif rsi < 30:
            reasons.append(f"RSI is oversold ({rsi:.0f}) — possible bounce candidate")
        else:
            reasons.append(f"RSI is neutral ({rsi:.0f})")
    
    return reasons


def generate_signal(
    sentiment_score: float,
    analyzed_articles: list,
    features: Dict[str, Any]
) -> SignalResult:
    """
    Main entry point. Generates the final market signal.
    
    Args:
        sentiment_score: Aggregate sentiment from sentiment.analyze_sentiment()
        analyzed_articles: Per-article sentiment from sentiment.analyze_sentiment()
        features: Dict from features.compute_features()
    
    Returns:
        SignalResult dict containing all signal data for Telegram and validator.
    
    This function should NEVER raise — it always returns a result.
    Worst case: NEUTRAL with 0% confidence and a data error note.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Determine data availability
    has_sentiment = len(analyzed_articles) > 0
    has_market = features.get('data_rows', 0) > 0
    
    try:
        # Core computation
        raw_score = _compute_raw_score(sentiment_score, features)
        
        # Normalize from [-1, +1] to [0, 1] for confidence readability
        normalized_score = (raw_score + 1) / 2
        
        # Classify
        signal = _classify_signal(normalized_score)
        
        # Compute confidence
        confidence = _compute_confidence(raw_score, normalized_score,
                                         has_sentiment, has_market)
        
        # Generate human-readable reasoning
        reasons = _generate_reasoning(signal, sentiment_score, features, normalized_score)
        
        # Article stats for Telegram
        positive_articles = sum(1 for a in analyzed_articles
                                if a.get('sentiment_label') == 'POSITIVE')
        negative_articles = sum(1 for a in analyzed_articles
                                if a.get('sentiment_label') == 'NEGATIVE')
        
        logger.info(
            "SIGNAL GENERATED: %s | confidence=%.4f | "
            "raw_score=%.4f | normalized=%.4f",
            signal, confidence, raw_score, normalized_score
        )
        
        return {
            # Core signal
            'signal': signal,
            'confidence': confidence,
            'confidence_pct': round(confidence * 100, 1),
            
            # Scores (for debugging and validator)
            'raw_score': round(raw_score, 4),
            'normalized_score': round(normalized_score, 4),
            
            # Inputs (for Telegram context)
            'sentiment_score': round(sentiment_score, 4),
            'trend_score': features.get('trend_score'),
            'vol_regime': features.get('vol_regime'),
            'rsi': features.get('rsi'),
            'momentum': features.get('momentum'),
            
            # Price context
            'latest_close': features.get('latest_close'),
            'price_change_1d': features.get('price_change_1d'),
            'price_change_5d': features.get('price_change_5d'),
            
            # Article stats
            'article_count': len(analyzed_articles),
            'positive_articles': positive_articles,
            'negative_articles': negative_articles,
            
            # Reasoning
            'reasons': reasons,
            
            # Metadata
            'timestamp': timestamp,
            'data_quality': {
                'has_sentiment': has_sentiment,
                'has_market': has_market,
            }
        }
    
    except Exception as e:
        logger.error("Critical error in signal generation: %s", e, exc_info=True)
        
        # Emergency fallback — return NEUTRAL with low confidence
        return {
            'signal': 'NEUTRAL',
            'confidence': 0.05,
            'confidence_pct': 5.0,
            'raw_score': 0.0,
            'normalized_score': 0.5,
            'sentiment_score': sentiment_score,
            'trend_score': None,
            'vol_regime': None,
            'rsi': None,
            'momentum': None,
            'latest_close': None,
            'price_change_1d': None,
            'price_change_5d': None,
            'article_count': 0,
            'positive_articles': 0,
            'negative_articles': 0,
            'reasons': [f"Signal engine error: {str(e)[:100]}"],
            'timestamp': timestamp,
            'data_quality': {'has_sentiment': False, 'has_market': False},
        }


# ── Quick sanity test ──
if __name__ == "__main__":
    # Simulate a moderately bullish scenario
    test_sentiment = 0.25
    test_articles = [
        {'sentiment_label': 'POSITIVE'},
        {'sentiment_label': 'POSITIVE'},
        {'sentiment_label': 'NEGATIVE'},
        {'sentiment_label': 'NEUTRAL'},
    ]
    test_features = {
        'trend_score': 0.3,
        'vol_score': 0.0,
        'rsi_signal': -0.1,
        'vol_regime': 'NORMAL',
        'rsi': 58.0,
        'momentum': 0.15,
        'latest_close': 22500.0,
        'price_change_1d': 0.8,
        'price_change_5d': 1.5,
        'data_rows': 25,
    }
    
    result = generate_signal(test_sentiment, test_articles, test_features)
    
    print(f"\nSignal: {result['signal']}")
    print(f"Confidence: {result['confidence_pct']}%")
    print(f"Raw score: {result['raw_score']}")
    print(f"\nReasons:")
    for r in result['reasons']:
        print(f"  • {r}")
