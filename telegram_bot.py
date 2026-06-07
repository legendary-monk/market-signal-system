"""
telegram_bot.py — Telegram Output Module
==========================================
Responsibility: Format the signal result into a human-readable message
and deliver it via Telegram Bot API.

WHY Telegram Bot API (not email, not WhatsApp):
- Free, no third-party library needed (raw HTTPS requests)
- Instant delivery, no spam filters
- No rate limits for reasonable usage (one message/week is well within limits)
- Easy to set up on any device in under 5 minutes
- The Bot API is stable and has no authentication complexity

Design: Uses requests.post() to send messages directly to Telegram's API.
No telegram library dependency — fewer packages, fewer breakages.
"""

import requests
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import config
from logger import get_logger

logger = get_logger(__name__)

# Telegram Bot API base URL
# WHY hardcoded here not in config: This never changes and shouldn't be configurable.
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"

# IST offset for display (UTC+5:30)
IST_OFFSET = timedelta(hours=5, minutes=30)


def _get_signal_emoji(signal: str) -> str:
    """Returns appropriate emoji for the signal type."""
    return {
        'BULLISH': '🟢',
        'BEARISH': '🔴',
        'NEUTRAL': '🟡',
    }.get(signal, '⚪')


def _get_confidence_bar(confidence_pct: float) -> str:
    """
    Creates a simple ASCII progress bar for confidence.
    
    WHY: A visual bar is immediately readable without doing math in your head.
    "▓▓▓▓▓░░░░░" is faster to parse than "50.0%"
    """
    total_blocks = 10
    filled = round(confidence_pct / 10)
    empty = total_blocks - filled
    return '▓' * filled + '░' * empty


def _get_trend_arrow(price_change: Optional[float]) -> str:
    """Returns an arrow emoji based on price change direction."""
    if price_change is None:
        return '↔️'
    elif price_change > 0.5:
        return '↗️'
    elif price_change < -0.5:
        return '↘️'
    else:
        return '↔️'


def _format_message(signal_result: Dict[str, Any]) -> str:
    """
    Formats the signal result into a Telegram-ready markdown message.
    
    WHY Telegram's MarkdownV2: Supports bold, monospace, and other formatting.
    However, MarkdownV2 requires escaping special chars. We use plain text
    to avoid escaping bugs — reliability over aesthetics.
    
    Args:
        signal_result: Full output from signal_engine.generate_signal()
    
    Returns:
        Formatted string ready to send to Telegram.
    """
    signal = signal_result.get('signal', 'NEUTRAL')
    confidence_pct = signal_result.get('confidence_pct', 0.0)
    sentiment_score = signal_result.get('sentiment_score', 0.0)
    latest_close = signal_result.get('latest_close')
    price_change_1d = signal_result.get('price_change_1d')
    price_change_5d = signal_result.get('price_change_5d')
    vol_regime = signal_result.get('vol_regime', 'N/A')
    rsi = signal_result.get('rsi')
    article_count = signal_result.get('article_count', 0)
    positive_articles = signal_result.get('positive_articles', 0)
    negative_articles = signal_result.get('negative_articles', 0)
    reasons = signal_result.get('reasons', [])
    posterior_p_up = signal_result.get('posterior_p_up', 0.5)
    model_disagreement = signal_result.get('model_disagreement', 0.0)
    epistemic_uncertainty = signal_result.get('epistemic_uncertainty', 1.0)
    regime = signal_result.get('regime', 'UNKNOWN')
    regime_diffusion_prob = signal_result.get('regime_diffusion_prob', 0.0)
    regime_trend_prob = signal_result.get('regime_trend_prob', 0.0)
    regime_shock_prob = signal_result.get('regime_shock_prob', 0.0)
    market_energy = signal_result.get('market_energy', 0.0)
    market_entropy = signal_result.get('market_entropy', 0.0)
    recommended_weight = signal_result.get('recommended_weight', 0.0)
    weight_reason = signal_result.get('weight_reason', '')
    has_market = signal_result.get('data_quality', {}).get('has_market', False)
    has_sentiment = signal_result.get('data_quality', {}).get('has_sentiment', False)
    movement_news = signal_result.get('movement_news', [])
    
    # Current time in IST
    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc + IST_OFFSET
    time_str = now_ist.strftime('%d %b %Y, %I:%M %p IST')
    
    # Signal emoji and bar
    emoji = _get_signal_emoji(signal)
    confidence_bar = _get_confidence_bar(confidence_pct)
    trend_arrow = _get_trend_arrow(price_change_5d)
    
    # Build message sections
    lines = []
    
    # ─── Header ───
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("📊 WEEKLY MARKET SIGNAL REPORT")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🕐 {time_str}")
    lines.append("")
    
    # ─── Main Signal ───
    lines.append(f"{emoji}  Signal:  {signal}")
    lines.append(f"📶  Confidence: {confidence_pct:.1f}%")
    lines.append(f"🎯  Posterior P(up): {posterior_p_up:.1%}")
    lines.append(f"     [{confidence_bar}]")
    lines.append(f"⚖️  Recommended Weight: {recommended_weight:.2%}")
    lines.append("")

    # ─── Model Risk Diagnostics ───
    lines.append("🧠 MODEL RISK")
    lines.append(f"   Regime: {regime}")
    lines.append(
        f"   Regime Mix: D {regime_diffusion_prob:.1%} | "
        f"T {regime_trend_prob:.1%} | S {regime_shock_prob:.1%}"
    )
    lines.append(f"   Disagreement: {model_disagreement:.3f}")
    lines.append(f"   Uncertainty: {epistemic_uncertainty:.3f}")
    lines.append(f"   Energy / Entropy: {market_energy:.2f} / {market_entropy:.2f}")
    lines.append("")
    
    # ─── Market Data ───
    if has_market and latest_close:
        lines.append("📈 NIFTY 50 DATA")
        lines.append(f"   Close: {latest_close:,.2f}")
        
        if price_change_1d is not None:
            sign = '+' if price_change_1d >= 0 else ''
            lines.append(f"   Latest Day: {sign}{price_change_1d:.2f}%  {_get_trend_arrow(price_change_1d)}")
        
        if price_change_5d is not None:
            sign = '+' if price_change_5d >= 0 else ''
            lines.append(f"   Week: {sign}{price_change_5d:.2f}%  {trend_arrow}")
        
        if vol_regime:
            vol_map = {'HIGH': '⚡ High', 'NORMAL': '〜 Normal', 'LOW': '🧘 Low'}
            lines.append(f"   Volatility: {vol_map.get(vol_regime, vol_regime)}")
        
        if rsi is not None:
            rsi_note = ""
            if rsi > 70:
                rsi_note = " (Overbought ⚠️)"
            elif rsi < 30:
                rsi_note = " (Oversold ⚠️)"
            lines.append(f"   RSI(14): {rsi:.1f}{rsi_note}")
        
        lines.append("")
    else:
        lines.append("📈 Market data unavailable for this week")
        lines.append("")
    
    # ─── News Sentiment ───
    if has_sentiment:
        sentiment_direction = "Positive 📰" if sentiment_score > 0.1 else \
                              "Negative 📰" if sentiment_score < -0.1 else "Mixed 📰"
        lines.append("📰 NEWS SENTIMENT")
        lines.append(f"   Overall: {sentiment_direction}")
        lines.append(f"   Score: {sentiment_score:+.3f} (range: -1 to +1)")
        lines.append(f"   Articles: {article_count} analyzed")
        lines.append(f"   Positive: {positive_articles} | Negative: {negative_articles}")
        lines.append("")
    else:
        lines.append("📰 Weekly news data unavailable")
        lines.append("")

    # ─── Biggest Weekly Moves + Associated News ───
    if movement_news:
        lines.append("🗞️ BIGGEST WEEKLY MOVES + LIKELY NEWS DRIVERS")
        for item in movement_news:
            move_pct = item.get('move_pct', 0.0)
            sign = '+' if move_pct >= 0 else ''
            direction = 'up' if move_pct >= 0 else 'down'
            close = item.get('close')
            close_text = f" | Close {close:,.2f}" if close else ""
            lines.append(f"   • {item.get('date')}: {sign}{move_pct:.2f}% {direction}{close_text}")
            articles = item.get('articles', [])
            if articles:
                for article in articles[:2]:
                    score = article.get('sentiment_score', 0.0) or 0.0
                    lines.append(
                        f"     - {article.get('title', 'Untitled')} "
                        f"({article.get('sentiment_label', 'NEUTRAL')} {score:+.2f}, "
                        f"{article.get('source', 'unknown source')})"
                    )
            else:
                lines.append("     - No same-day RSS headline was available; avoid assuming causality.")
        lines.append("   Note: items are associated by date/sentiment and are not proof of causality.")
        lines.append("")

    # ─── Signal Reasoning ───
    if reasons:
        lines.append("🔍 KEY FACTORS")
        for reason in reasons[:3]:
            lines.append(f"   • {reason}")
        if weight_reason:
            lines.append(f"   • Position sizing: {weight_reason}")
        lines.append("")

    # ─── Scenario Playbook ───
    lines.append("🧭 SCENARIO PLAYBOOK")
    lines.append("   • Base: keep suggested weight while shock/uncertainty stay contained.")
    if regime_shock_prob >= 0.40 or market_energy >= 2.0:
        lines.append("   • Defensive: shock/energy elevated — cut exposure and prefer neutral.")
    else:
        lines.append("   • Upside: trend persistence with low shock supports full risk budget.")
    lines.append("")
    
    # ─── Data Quality Warning ───
    if not has_market or not has_sentiment:
        lines.append("⚠️  DATA NOTE")
        missing = []
        if not has_market:
            missing.append("market price data")
        if not has_sentiment:
            missing.append("news sentiment")
        lines.append(f"   Missing: {', '.join(missing)}")
        lines.append("   Signal reliability is REDUCED this week.")
        lines.append("")
    
    # ─── Disclaimer ───
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("⚠️  NOT financial advice.")
    lines.append("   Always do your own research.")
    lines.append("   Past signals don't guarantee future accuracy.")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    
    return '\n'.join(lines)


def send_telegram_message(
    message: str,
    token: str = None,
    chat_id: str = None
) -> bool:
    """
    Sends a text message to the configured Telegram chat.
    
    Args:
        message: The formatted text to send.
        token: Bot token (defaults to config.TELEGRAM_TOKEN).
        chat_id: Recipient chat ID (defaults to config.TELEGRAM_CHAT_ID).
    
    Returns:
        True if message sent successfully, False otherwise.
    
    WHY return bool instead of raising: The pipeline should complete its
    run (including saving to predictions.csv) even if Telegram is down.
    """
    token = token or config.TELEGRAM_TOKEN
    chat_id = chat_id or config.TELEGRAM_CHAT_ID
    
    url = TELEGRAM_API_BASE.format(token=token, method="sendMessage")
    
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': None,  # Plain text — no escaping needed
        'disable_web_page_preview': True,
    }
    
    for attempt in range(1, config.REQUEST_RETRIES + 1):
        try:
            response = requests.post(
                url,
                data=payload,
                timeout=config.REQUEST_TIMEOUT
            )
            
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('ok'):
                logger.info("Telegram message sent successfully to chat %s", chat_id)
                return True
            
            # Handle specific Telegram error codes
            error_code = response_data.get('error_code', 0)
            description = response_data.get('description', 'Unknown error')
            
            if error_code == 401:
                logger.error("Telegram: Invalid bot token. Check TELEGRAM_TOKEN in config.py")
                return False  # No point retrying — won't change
            
            elif error_code == 400 and 'chat not found' in description.lower():
                logger.error(
                    "Telegram: Chat ID not found. "
                    "Did you send /start to your bot first? "
                    "Check TELEGRAM_CHAT_ID in config.py"
                )
                return False  # No point retrying
            
            elif error_code == 429:
                # Rate limited by Telegram
                retry_after = response_data.get('parameters', {}).get('retry_after', 30)
                logger.warning("Telegram rate limit hit. Waiting %ds...", retry_after)
                import time
                time.sleep(min(retry_after, 60))  # Cap wait at 60s
                continue
            
            else:
                logger.warning(
                    "Telegram API error %d: %s (attempt %d/%d)",
                    error_code, description, attempt, config.REQUEST_RETRIES
                )
        
        except requests.exceptions.Timeout:
            logger.warning("Telegram request timed out (attempt %d/%d)",
                           attempt, config.REQUEST_RETRIES)
        
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot connect to Telegram (attempt %d/%d). "
                           "Check internet connection.", attempt, config.REQUEST_RETRIES)
        
        except json.JSONDecodeError:
            logger.error("Telegram returned non-JSON response")
            return False
        
        except Exception as e:
            logger.error("Unexpected Telegram error: %s", e, exc_info=True)
            return False
        
        # Wait before retry
        if attempt < config.REQUEST_RETRIES:
            import time
            wait = config.RETRY_DELAY_BASE ** attempt
            time.sleep(wait)
    
    logger.error("Failed to send Telegram message after %d attempts",
                 config.REQUEST_RETRIES)
    return False


def send_weekly_report(signal_result: Dict[str, Any]) -> bool:
    """
    Formats and sends the weekly signal report to one or more Telegram IDs.
    Supports comma-separated IDs in config.TELEGRAM_CHAT_ID.
    """
    try:
        message = _format_message(signal_result)
        
        # Split the IDs by comma in case there are multiple
        # .strip() removes any accidental spaces
        chat_ids = [id.strip() for id in str(config.TELEGRAM_CHAT_ID).split(',')]
        
        success = True
        for cid in chat_ids:
            if not send_telegram_message(message, chat_id=cid):
                logger.error(f"Failed to send to Chat ID: {cid}")
                success = False
        
        return success
    
    except Exception as e:
        logger.error("Failed to format/send signal to Telegram: %s", e,
                     exc_info=True)
        return False
        

def send_error_alert(error_summary: str) -> bool:
    """
    Sends a brief error notification to all configured Telegram chat IDs.
    
    WHY: If main.py crashes, all users need to know it was a system failure 
    rather than just a neutral market day.
    """
    now_ist = datetime.now(timezone.utc) + IST_OFFSET
    time_str = now_ist.strftime('%d %b %Y, %I:%M %p IST')
    
    message = (
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️  SYSTEM ALERT\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {time_str}\n\n"
        "Market Signal System encountered an error:\n\n"
        f"{error_summary[:500]}\n\n"
        "No weekly report generated.\n"
        "Check market_signal.log for details.\n"
        "━━━━━━━━━━━━━━━━━━━━━━"
    )
    
    # Logic to handle multiple IDs
    # Converts "ID1,ID2" into ['ID1', 'ID2']
    chat_ids = [id.strip() for id in str(config.TELEGRAM_CHAT_ID).split(',')]
    
    for cid in chat_ids:
        send_telegram_message(message, chat_id=cid)
    
    return True
    

def test_connection() -> bool:
    """
    Sends a test message to verify bot token and chat ID are correct.
    Call this before the first real run.
    """
    now_ist = datetime.now(timezone.utc) + IST_OFFSET
    test_message = (
        "✅ Market Signal System — Connection Test\n"
        f"Time: {now_ist.strftime('%d %b %Y, %I:%M %p IST')}\n\n"
        "Bot is configured correctly!\n"
        "You will receive your weekly market signal report here."
    )
    
    success = send_telegram_message(test_message)
    if success:
        logger.info("Telegram connection test PASSED")
    else:
        logger.error("Telegram connection test FAILED — check token and chat ID")
    
    return success


# ── Quick sanity test ──
if __name__ == "__main__":
    print("Testing Telegram connection...")
    print("(Make sure TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are set in config.py)")
    
    errors = config.validate_config()
    if errors:
        print("\nConfiguration errors:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nPlease fix config.py before running.")
    else:
        result = test_connection()
        if result:
            print("✓ Test message sent! Check your Telegram.")
        else:
            print("✗ Test failed. Check the log file for details.")
