"""
validator.py — Signal Accuracy Tracking
=========================================
Responsibility: Record predictions and compare against actual outcomes
to measure how accurate our signal system is over time.

WHY this matters:
- A signal system without validation is just noise generation
- You CANNOT know if TextBlob + linear regression adds value without measurement
- This module is the scientific rigor layer of the system

Validation Logic:
    On signal day (D): Save signal to predictions.csv
    On next trading day (D+1): Fetch actual D-to-D+1 price change
                               Compare against predicted signal
    
Accuracy Definition:
    BULLISH was correct if next-day close > today's close
    BEARISH was correct if next-day close < today's close
    NEUTRAL: Correct if |price change| < 0.5% (markets were indeed flat)
    
WHY next-day accuracy instead of intraday:
- We generate signals at 9 AM. The market hasn't opened yet.
- Daily close-to-close is the most natural validation horizon.
- Intraday noise makes validation meaningless at this signal granularity.
"""

import csv
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json

import config
from logger import get_logger

logger = get_logger(__name__)

# IST offset
IST_OFFSET = timedelta(hours=5, minutes=30)

# CSV column names — order matters for compatibility with future versions
CSV_COLUMNS = [
    'date',              # Signal date (YYYY-MM-DD in IST)
    'timestamp_utc',     # Full UTC timestamp of signal generation
    'signal',            # BULLISH / BEARISH / NEUTRAL
    'confidence',        # 0.0–1.0
    'raw_score',         # [-1.0, +1.0] composite score
    'sentiment_score',   # [-1.0, +1.0]
    'trend_score',       # [-1.0, +1.0] or empty
    'vol_regime',        # HIGH / NORMAL / LOW or empty
    'rsi',               # 0–100 or empty
    'nifty_close',       # Actual Nifty close on signal day
    'next_close',        # Actual Nifty close on D+1 (filled in later)
    'actual_change_pct', # (next_close - nifty_close) / nifty_close * 100
    'outcome',           # CORRECT / INCORRECT / NEUTRAL_HIT / PENDING
    'article_count',     # Number of articles analyzed
]

# Threshold for calling NEUTRAL prediction correct (±0.5% price change)
NEUTRAL_CORRECT_THRESHOLD = 0.5


def _ensure_csv_exists():
    """
    Creates the predictions CSV file with headers if it doesn't exist.
    
    WHY: First run has no CSV. We create it here rather than in each function
    to keep the file-touching logic in one place.
    """
    if not os.path.exists(config.PREDICTIONS_FILE):
        try:
            with open(config.PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            logger.info("Created new predictions file: %s", config.PREDICTIONS_FILE)
        except IOError as e:
            logger.error("Cannot create predictions file: %s", e)


def save_prediction(signal_result: Dict[str, Any]) -> bool:
    """
    Saves today's signal to predictions.csv.
    
    Args:
        signal_result: Full output from signal_engine.generate_signal()
    
    Returns:
        bool: True if saved successfully.
    
    WHY write one row per day with PENDING outcome:
    The outcome column starts as PENDING and gets filled in the next day
    when we have the actual price movement. This two-phase design means
    we never need to modify old data — we just fill in the blanks.
    """
    _ensure_csv_exists()
    
    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc + IST_OFFSET
    date_str = now_ist.strftime('%Y-%m-%d')
    
    # Check if we already saved a prediction for today
    existing = _read_all_predictions()
    today_predictions = [p for p in existing if p.get('date') == date_str]
    
    if today_predictions:
        logger.warning(
            "Prediction for %s already exists (signal=%s). "
            "Skipping duplicate save.",
            date_str, today_predictions[0].get('signal')
        )
        return True  # WHY True: Not an error, just a duplicate prevention
    
    row = {
        'date': date_str,
        'timestamp_utc': now_utc.isoformat(),
        'signal': signal_result.get('signal', 'NEUTRAL'),
        'confidence': round(signal_result.get('confidence', 0.0), 4),
        'raw_score': round(signal_result.get('raw_score', 0.0), 4),
        'sentiment_score': round(signal_result.get('sentiment_score', 0.0), 4),
        'trend_score': round(signal_result.get('trend_score', 0) or 0, 4),
        'vol_regime': signal_result.get('vol_regime', ''),
        'rsi': round(signal_result.get('rsi', 0) or 0, 2),
        'nifty_close': round(signal_result.get('latest_close', 0) or 0, 2),
        'next_close': '',        # Filled in next day
        'actual_change_pct': '', # Filled in next day
        'outcome': 'PENDING',    # Will be updated next run
        'article_count': signal_result.get('article_count', 0),
    }
    
    try:
        with open(config.PREDICTIONS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)
        
        logger.info("Prediction saved: date=%s signal=%s confidence=%.2f",
                    date_str, row['signal'], row['confidence'])
        return True
    
    except IOError as e:
        logger.error("Failed to save prediction: %s", e)
        return False


def update_pending_outcomes(current_close: Optional[float]) -> int:
    """
    Fills in outcomes for PENDING predictions from previous days.
    
    This function is called at the start of each run with today's Nifty close.
    It looks for yesterday's PENDING prediction and resolves its outcome.
    
    Args:
        current_close: Today's Nifty 50 closing price.
    
    Returns:
        Number of outcomes updated.
    
    WHY this design: We can't know yesterday's outcome until today's market closes.
    The PENDING system elegantly handles this temporal dependency.
    """
    if current_close is None:
        logger.warning("Cannot update outcomes: no current close price available")
        return 0
    
    predictions = _read_all_predictions()
    if not predictions:
        return 0
    
    updated_count = 0
    updated_predictions = []
    
    for pred in predictions:
        if pred.get('outcome') != 'PENDING':
            updated_predictions.append(pred)
            continue
        
        # Try to resolve this pending prediction
        prev_close = pred.get('nifty_close', '')
        if not prev_close:
            # We didn't have price data when this prediction was made
            pred['outcome'] = 'NO_PRICE_DATA'
            updated_predictions.append(pred)
            continue
        
        try:
            prev_close = float(prev_close)
        except (ValueError, TypeError):
            pred['outcome'] = 'INVALID_DATA'
            updated_predictions.append(pred)
            continue
        
        # Compute actual change
        actual_change_pct = ((current_close - prev_close) / prev_close) * 100
        pred['next_close'] = round(current_close, 2)
        pred['actual_change_pct'] = round(actual_change_pct, 4)
        
        # Determine if our prediction was correct
        signal = pred.get('signal', 'NEUTRAL')
        pred['outcome'] = _evaluate_outcome(signal, actual_change_pct)
        
        updated_count += 1
        logger.info(
            "Outcome resolved for %s: signal=%s actual_change=%.2f%% → %s",
            pred.get('date'), signal, actual_change_pct, pred['outcome']
        )
        
        updated_predictions.append(pred)
    
    if updated_count > 0:
        _write_all_predictions(updated_predictions)
    
    return updated_count


def _evaluate_outcome(signal: str, actual_change_pct: float) -> str:
    """
    Determines if the signal prediction was correct given actual price movement.
    
    Args:
        signal: "BULLISH", "BEARISH", or "NEUTRAL"
        actual_change_pct: Actual next-day price change in percentage
    
    Returns:
        "CORRECT", "INCORRECT", "NEUTRAL_HIT", or "NEUTRAL_MISS"
    """
    if signal == 'BULLISH':
        return 'CORRECT' if actual_change_pct > 0 else 'INCORRECT'
    
    elif signal == 'BEARISH':
        return 'CORRECT' if actual_change_pct < 0 else 'INCORRECT'
    
    elif signal == 'NEUTRAL':
        # NEUTRAL is correct if market didn't move much
        if abs(actual_change_pct) < NEUTRAL_CORRECT_THRESHOLD:
            return 'NEUTRAL_HIT'
        else:
            return 'NEUTRAL_MISS'
    
    return 'UNKNOWN'


def _read_all_predictions() -> List[Dict]:
    """
    Reads all predictions from CSV into a list of dicts.
    Returns empty list if file doesn't exist or is corrupt.
    """
    _ensure_csv_exists()
    
    predictions = []
    try:
        with open(config.PREDICTIONS_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append(dict(row))
    except FileNotFoundError:
        pass  # File doesn't exist yet — fine
    except Exception as e:
        logger.error("Error reading predictions file: %s", e)
    
    return predictions


def _write_all_predictions(predictions: List[Dict]) -> bool:
    """
    Overwrites the entire predictions CSV with the given data.
    
    WHY overwrite instead of append-and-delete: Atomic rewrite prevents
    corruption if the process is killed mid-write on mobile.
    """
    try:
        # Write to a temp file first, then rename
        # WHY: Rename is atomic on most filesystems. If we crash mid-write,
        # the original file is preserved.
        temp_path = config.PREDICTIONS_FILE + '.tmp'
        
        with open(temp_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(predictions)
        
        os.replace(temp_path, config.PREDICTIONS_FILE)
        return True
    
    except Exception as e:
        logger.error("Failed to write predictions: %s", e)
        # Clean up temp file if it exists
        try:
            os.remove(config.PREDICTIONS_FILE + '.tmp')
        except FileNotFoundError:
            pass
        return False


def compute_performance_metrics() -> Dict[str, Any]:
    """
    Computes accuracy statistics across all resolved predictions.
    
    Returns:
        Dict with performance metrics including:
        - overall_accuracy: % of BULLISH/BEARISH signals that were correct
        - signal_distribution: How often each signal type was given
        - by_confidence: Accuracy grouped by confidence level
        - streak: Current winning/losing streak
    
    WHY separate NEUTRAL from directional accuracy:
    NEUTRAL signals can't really be "wrong" in the same way.
    We track them separately.
    """
    predictions = _read_all_predictions()
    
    if not predictions:
        return {'error': 'No predictions found', 'total': 0}
    
    resolved = [p for p in predictions if p.get('outcome') not in ('PENDING', '', None)]
    directional = [p for p in resolved if p.get('signal') in ('BULLISH', 'BEARISH')]
    
    if not directional:
        return {
            'error': 'No resolved directional predictions yet',
            'total_saved': len(predictions),
            'pending': sum(1 for p in predictions if p.get('outcome') == 'PENDING'),
        }
    
    # Overall accuracy on directional signals
    correct = sum(1 for p in directional if p.get('outcome') == 'CORRECT')
    total_dir = len(directional)
    accuracy = correct / total_dir if total_dir > 0 else 0
    
    # Signal distribution
    bullish_count = sum(1 for p in predictions if p.get('signal') == 'BULLISH')
    bearish_count = sum(1 for p in predictions if p.get('signal') == 'BEARISH')
    neutral_count = sum(1 for p in predictions if p.get('signal') == 'NEUTRAL')
    
    # Accuracy by confidence band
    high_conf = [p for p in directional if _safe_float(p.get('confidence', 0)) >= 0.70]
    mid_conf = [p for p in directional if 0.50 <= _safe_float(p.get('confidence', 0)) < 0.70]
    low_conf = [p for p in directional if _safe_float(p.get('confidence', 0)) < 0.50]
    
    def band_accuracy(band):
        if not band:
            return None
        correct_in_band = sum(1 for p in band if p.get('outcome') == 'CORRECT')
        return round(100 * correct_in_band / len(band), 1)
    
    # Consecutive streak (most recent first)
    recent = sorted(directional, key=lambda p: p.get('date', ''), reverse=True)
    streak = 0
    streak_type = None
    for p in recent:
        outcome = p.get('outcome')
        if streak_type is None:
            streak_type = 'WIN' if outcome == 'CORRECT' else 'LOSS'
        
        if (streak_type == 'WIN' and outcome == 'CORRECT') or \
           (streak_type == 'LOSS' and outcome == 'INCORRECT'):
            streak += 1
        else:
            break
    
    return {
        'total_saved': len(predictions),
        'total_resolved': len(resolved),
        'directional_total': total_dir,
        'directional_correct': correct,
        'overall_accuracy_pct': round(100 * accuracy, 1),
        'signal_distribution': {
            'BULLISH': bullish_count,
            'BEARISH': bearish_count,
            'NEUTRAL': neutral_count,
        },
        'accuracy_by_confidence': {
            'high_70plus': band_accuracy(high_conf),
            'mid_50to70': band_accuracy(mid_conf),
            'low_below50': band_accuracy(low_conf),
        },
        'current_streak': f"{streak} {streak_type}" if streak_type else "N/A",
        'pending_count': sum(1 for p in predictions if p.get('outcome') == 'PENDING'),
    }


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def print_performance_report():
    """
    Prints a human-readable performance report to console.
    Run this function directly to see how the system is doing.
    """
    metrics = compute_performance_metrics()
    
    if 'error' in metrics:
        print(f"\n⚠️  {metrics['error']}")
        if 'total_saved' in metrics:
            print(f"   Total predictions saved: {metrics['total_saved']}")
        return
    
    print("\n" + "="*50)
    print("📊 MARKET SIGNAL SYSTEM — PERFORMANCE REPORT")
    print("="*50)
    
    print(f"\nTotal predictions saved:    {metrics['total_saved']}")
    print(f"Total resolved:             {metrics['total_resolved']}")
    print(f"Pending (today/recent):     {metrics['pending_count']}")
    
    print(f"\n📈 DIRECTIONAL ACCURACY")
    print(f"   Total signals:   {metrics['directional_total']}")
    print(f"   Correct:         {metrics['directional_correct']}")
    print(f"   Accuracy:        {metrics['overall_accuracy_pct']}%")
    
    print(f"\n📊 SIGNAL DISTRIBUTION")
    dist = metrics['signal_distribution']
    print(f"   BULLISH: {dist['BULLISH']}")
    print(f"   BEARISH: {dist['BEARISH']}")
    print(f"   NEUTRAL: {dist['NEUTRAL']}")
    
    print(f"\n📶 ACCURACY BY CONFIDENCE")
    conf = metrics['accuracy_by_confidence']
    print(f"   High (≥70%): {conf['high_70plus']}%")
    print(f"   Mid  (50-70%): {conf['mid_50to70']}%")
    print(f"   Low  (<50%): {conf['low_below50']}%")
    
    print(f"\n🔥 CURRENT STREAK: {metrics['current_streak']}")
    
    # Context
    total = metrics['directional_total']
    acc = metrics['overall_accuracy_pct']
    if total < 20:
        print(f"\n⚠️  NOTE: Only {total} directional signals resolved.")
        print("   Need ≥20 for statistically meaningful accuracy.")
    elif acc >= 60:
        print(f"\n✅ System is above 60% accuracy. Statistically meaningful edge.")
    elif acc >= 50:
        print(f"\n🟡 System is at coin-flip accuracy. Review signal weights.")
    else:
        print(f"\n🔴 System is below 50%. Consider inverting signals or recalibrating.")
    
    print("="*50)


# ── Run this file directly for the performance report ──
if __name__ == "__main__":
    print_performance_report()
