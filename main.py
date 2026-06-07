"""
main.py — Pipeline Orchestrator
=================================
This is the entry point for the entire Market Signal System.
Run this file weekly to generate and receive your weekly market report.

Pipeline flow:
    1. Validate configuration
    2. Update pending outcomes (if market data available)
    3. Fetch weekly financial news
    4. Analyze weekly news sentiment
    5. Fetch market data (Nifty 50)
    6. Compute market features
    7. Generate weekly signal
    8. Attach the news associated with the biggest weekly moves
    9. Save prediction to CSV
    10. Send to Telegram

Design Principles:
    - Each step is wrapped in try/except so one failure doesn't kill others
    - Partial data is better than no signal (degrade gracefully)
    - Every run produces a Telegram message — even if just an error alert
    - Total runtime should be <60 seconds on Redmi Note 11

Usage:
    python main.py
"""

import sys
import traceback
from datetime import datetime, timezone, timedelta

# ── Import all modules ──
# Import errors should fail fast with Python's native traceback
# so setup issues are explicit during startup.
import config
from logger import get_logger
from news_fetcher import fetch_news
from sentiment import analyze_sentiment, get_sentiment_summary
from market_data import fetch_market_data, get_latest_price
from features import compute_features
from signal_engine import generate_signal
from telegram_bot import send_weekly_report, send_error_alert, test_connection
from validator import save_prediction, update_pending_outcomes, compute_performance_metrics
from weekly_report import build_weekly_movement_news

logger = get_logger(__name__)

IST_OFFSET = timedelta(hours=5, minutes=30)


def run_pipeline() -> bool:
    """
    Executes the full signal generation pipeline.
    
    Returns:
        True if pipeline completed successfully (signal sent).
        False if a critical failure occurred.
    """
    start_time = datetime.now(timezone.utc)
    now_ist = start_time + IST_OFFSET
    
    logger.info("="*60)
    logger.info("MARKET SIGNAL SYSTEM — Weekly run started at %s IST",
                now_ist.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("="*60)
    
    # ─────────────────────────────────────────────────────────
    # STEP 0: Configuration Validation
    # WHY first: Fail fast on misconfiguration before making any
    # network calls or doing any compute.
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 0] Validating configuration...")
    config_errors = config.validate_config()
    if config_errors:
        for err in config_errors:
            logger.error("Config error: %s", err)
        
        # Special case: if Telegram is misconfigured, we can't send anything
        # Just print to console and exit
        print("\n❌ CONFIGURATION ERRORS (check config.py):")
        for err in config_errors:
            print(f"   • {err}")
        print("\nSee BEGINNERS_GUIDE.md Part 2 for setup instructions.")
        return False
    
    logger.info("[Step 0] Configuration OK")
    
    # ─────────────────────────────────────────────────────────
    # STEP 1: Fetch Market Data
    # WHY early: We need the latest close price to update prior
    # PENDING outcomes in the validator. Also if this fails, we know
    # before spending time on news analysis.
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 1] Fetching market data...")
    market_df = None
    features = {}
    
    try:
        market_df = fetch_market_data()
        
        if market_df is not None:
            logger.info("[Step 1] Market data fetched: %d trading days", len(market_df))
        else:
            logger.warning("[Step 1] Market data unavailable — continuing with degraded signal")
    
    except Exception as e:
        logger.error("[Step 1] Market data fetch crashed: %s", e, exc_info=True)
    
    # ─────────────────────────────────────────────────────────
    # STEP 2: Update Pending Outcomes
    # WHY now: Weekly runs still need to resolve any older PENDING
    # prediction rows before saving the latest weekly signal.
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 2] Updating pending prediction outcomes...")
    
    try:
        current_close = get_latest_price(market_df)
        if current_close:
            updated = update_pending_outcomes(current_close)
            if updated > 0:
                logger.info("[Step 2] Updated %d pending outcome(s)", updated)
            else:
                logger.info("[Step 2] No pending outcomes to update")
        else:
            logger.warning("[Step 2] No current close price available for outcome update")
    
    except Exception as e:
        logger.error("[Step 2] Outcome update failed: %s", e, exc_info=True)
        # Non-critical — continue
    
    # ─────────────────────────────────────────────────────────
    # STEP 3: Compute Market Features
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 3] Computing market features...")
    
    try:
        if market_df is not None:
            features = compute_features(market_df)
            logger.info("[Step 3] Features computed: trend=%.4f, vol=%s, RSI=%.1f",
                        features.get('trend_score', 0) or 0,
                        features.get('vol_regime', 'N/A'),
                        features.get('rsi', 0) or 0)
        else:
            logger.warning("[Step 3] Skipping features — no market data")
            features = {}
    
    except Exception as e:
        logger.error("[Step 3] Feature computation crashed: %s", e, exc_info=True)
        features = {}
    
    # ─────────────────────────────────────────────────────────
    # STEP 4: Fetch News
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 4] Fetching weekly financial news...")
    articles = []
    
    try:
        articles = fetch_news()
        logger.info("[Step 4] Fetched %d articles", len(articles))
        
        if not articles:
            logger.warning("[Step 4] No weekly articles fetched — news sentiment will be neutral")
    
    except Exception as e:
        logger.error("[Step 4] News fetch crashed: %s", e, exc_info=True)
    
    # ─────────────────────────────────────────────────────────
    # STEP 5: Analyze Sentiment
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 5] Analyzing weekly news sentiment...")
    sentiment_score = 0.0
    analyzed_articles = []
    
    try:
        if articles:
            sentiment_score, analyzed_articles = analyze_sentiment(articles)
            logger.info("[Step 5] Sentiment: %.4f across %d articles",
                        sentiment_score, len(analyzed_articles))
        else:
            logger.warning("[Step 5] No weekly articles to analyze — using neutral sentiment")
    
    except Exception as e:
        logger.error("[Step 5] Sentiment analysis crashed: %s", e, exc_info=True)
    
    # ─────────────────────────────────────────────────────────
    # STEP 6: Generate Signal
    # WHY in its own try block: Signal engine has emergency fallback
    # to NEUTRAL, but we still want to catch unexpected failures.
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 6] Generating weekly signal...")
    signal_result = None
    
    try:
        signal_result = generate_signal(sentiment_score, analyzed_articles, features)
        
        logger.info(
            "[Step 6] SIGNAL: %s | Confidence: %.1f%% | Raw score: %.4f",
            signal_result['signal'],
            signal_result['confidence_pct'],
            signal_result['raw_score']
        )
    
    except Exception as e:
        logger.error("[Step 6] Signal generation crashed: %s", e, exc_info=True)
        
        # Build emergency signal
        signal_result = {
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
            'latest_close': get_latest_price(market_df),
            'price_change_1d': None,
            'price_change_5d': None,
            'movement_news': [],
            'article_count': len(analyzed_articles),
            'positive_articles': 0,
            'negative_articles': 0,
            'reasons': ["System error in signal engine — defaulting to NEUTRAL"],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_quality': {'has_sentiment': bool(articles), 'has_market': market_df is not None},
        }
    
    # ─────────────────────────────────────────────────────────
    # STEP 7: Attach Weekly Movement-News Context
    # WHY here: It needs both market movement data and analyzed article
    # sentiment so the report can show the largest moves and associated news.
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 7] Building weekly movement-news context...")

    try:
        signal_result['movement_news'] = build_weekly_movement_news(market_df, analyzed_articles)
        logger.info("[Step 7] Movement-news entries: %d", len(signal_result['movement_news']))
    except Exception as e:
        logger.error("[Step 7] Movement-news context failed: %s", e, exc_info=True)
        signal_result['movement_news'] = []

    # ─────────────────────────────────────────────────────────
    # STEP 8: Save Prediction to CSV
    # WHY before Telegram: If Telegram fails, we still have a record.
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 8] Saving prediction...")
    
    try:
        saved = save_prediction(signal_result)
        if saved:
            logger.info("[Step 8] Prediction saved to %s", config.PREDICTIONS_FILE)
        else:
            logger.warning("[Step 8] Failed to save prediction")
    
    except Exception as e:
        logger.error("[Step 8] Save prediction crashed: %s", e, exc_info=True)
    
    # ─────────────────────────────────────────────────────────
    # STEP 9: Send Telegram Message
    # ─────────────────────────────────────────────────────────
    logger.info("[Step 9] Sending weekly Telegram report...")
    telegram_success = False
    
    try:
        telegram_success = send_weekly_report(signal_result)
        
        if telegram_success:
            logger.info("[Step 9] Telegram weekly report sent successfully")
        else:
            logger.error("[Step 9] Telegram weekly report delivery failed")
    
    except Exception as e:
        logger.error("[Step 9] Telegram send crashed: %s", e, exc_info=True)
    
    # ─────────────────────────────────────────────────────────
    # PIPELINE COMPLETE
    # ─────────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    logger.info(
        "Pipeline complete in %.1fs | Signal=%s | Confidence=%.1f%% | Telegram=%s",
        elapsed,
        signal_result['signal'] if signal_result else 'N/A',
        signal_result['confidence_pct'] if signal_result else 0,
        '✓' if telegram_success else '✗'
    )
    
    return telegram_success


def run_test_mode():
    """
    Runs a connection test without generating a real signal.
    Use this first to verify your Telegram setup.
    """
    print("\n" + "="*50)
    print("MARKET SIGNAL SYSTEM — CONNECTION TEST")
    print("="*50)
    
    # Check config
    errors = config.validate_config()
    if errors:
        print("\n❌ Configuration errors found:")
        for e in errors:
            print(f"   • {e}")
        print("\nFix these in config.py before proceeding.")
        return False
    
    print("✓ Configuration looks valid")
    print("\nSending test message to Telegram...")
    
    success = test_connection()
    
    if success:
        print("\n✅ SUCCESS! Check your Telegram for the test message.")
        print("You can now run the weekly pipeline: python main.py")
    else:
        print("\n❌ FAILED! Check config.py and market_signal.log for details.")
    
    return success


def show_performance():
    """Prints the performance report to console."""
    from validator import print_performance_report
    print_performance_report()


# ── Entry point ──
if __name__ == "__main__":
    # Command line argument handling
    # Usage: python main.py [test|performance]
    
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'run'
    
    if mode == 'test':
        # python main.py test — Send a test Telegram message
        run_test_mode()
    
    elif mode == 'performance':
        # python main.py performance — Show accuracy report
        show_performance()
    
    elif mode == 'run':
        # python main.py — Normal signal run
        try:
            success = run_pipeline()
            sys.exit(0 if success else 1)
        
        except KeyboardInterrupt:
            logger.info("Run interrupted by user")
            sys.exit(0)
        
        except Exception as e:
            # This should NOT happen — individual steps are all wrapped.
            # If we're here, something truly unexpected occurred.
            error_msg = f"CRITICAL PIPELINE FAILURE:\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # Try to send an error alert to Telegram
            try:
                send_error_alert(str(e))
            except Exception:
                pass  # If Telegram fails too, we've logged everything
            
            sys.exit(1)
    
    else:
        print(f"Unknown mode: '{mode}'")
        print("Usage: python main.py [run|test|performance]")
        print("  run         — Generate and send this week's report (default)")
        print("  test        — Send a test message to verify Telegram setup")
        print("  performance — Show signal accuracy report")
        sys.exit(1)
