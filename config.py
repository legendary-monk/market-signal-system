
import os
import logging

# ─────────────────────────────────────────────
# TELEGRAM SETTINGS (REQUIRED — EDIT THESE)
# ─────────────────────

# This reads from GitHub Secrets if available, otherwise uses your manual entry
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', "YOUR_CHAT_ID_HERE")


# ─────────────────────────────────────────────
# MARKET DATA SETTINGS
# ─────────────────────────────────────────────
# Yahoo Finance ticker for Nifty 50 index.
# "^NSEI" is the standard Yahoo Finance symbol for Nifty 50.
MARKET_TICKER = "^NSEI"

# How many calendar days of historical data to pull.
# WHY 45: Enough for weekly reporting plus the 20-trading-day feature baseline.
MARKET_LOOKBACK_DAYS = 45

# ─────────────────────────────────────────────
# NEWS FEED SETTINGS
# ─────────────────────────────────────────────
# Free RSS feeds covering Indian financial markets.
# WHY multiple feeds: Single-source bias. If one feed is down, others compensate.
# WHY RSS: No API key needed, lightweight, reliable.
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://feeds.feedburner.com/ndtvprofit-latest",
    "https://www.livemint.com/rss/markets",
    "https://economictimes.indiatimes.com/prime/money-and-markets/rssfeeds/61703836.cms",
]

# Maximum number of articles to analyze per run.
# WHY 50: Beyond this, returns diminish while compute cost rises.
MAX_ARTICLES = 50

# How old (in hours) an article can be and still count.
# WHY 168: Weekly reports need a full seven-day news window.
MAX_ARTICLE_AGE_HOURS = 168
ALLOW_UNDATED_ARTICLES = False

# Weekly report settings.
# REPORT_LOOKBACK_TRADING_DAYS covers the standard Monday-Friday market week.
REPORT_LOOKBACK_TRADING_DAYS = 5
TOP_MOVEMENT_NEWS = 3

# ─────────────────────────────────────────────
# SIGNAL ENGINE PARAMETERS
# ─────────────────────────────────────────────
# These weights determine how much each factor contributes to the final signal.
# They must sum to 1.0 (100%).
# WHY these weights: Sentiment is noisier than price. Price trend is more reliable.
WEIGHT_SENTIMENT = 0.40      # 40% weight on news sentiment
WEIGHT_TREND = 0.35          # 35% weight on price trend direction
WEIGHT_VOLATILITY = 0.25     # 25% weight on volatility regime

# Confidence thresholds for signal classification.
# WHY 0.60: Below this, our signal is essentially a coin flip — better to say NEUTRAL.
BULLISH_THRESHOLD = 0.60     # Score above this → BULLISH
BEARISH_THRESHOLD = 0.40     # Score below this → BEARISH
# Between 0.40 and 0.60 → NEUTRAL

# ─────────────────────────────────────────────
# FEATURE COMPUTATION PARAMETERS
# ─────────────────────────────────────────────
# Rolling window (in trading days) for trend calculation.
# WHY 5: One trading week — captures short-term momentum.
TREND_WINDOW = 5

# Rolling window for volatility (standard deviation of returns).
# WHY 10: Two trading weeks — smooths out daily noise in volatility.
VOLATILITY_WINDOW = 10

# How much the current volatility must exceed the 20-day average
# to be classified as "HIGH volatility".
# WHY 1.5x: Statistically meaningful elevation above baseline.
HIGH_VOLATILITY_MULTIPLIER = 1.5

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
# All output files go in the same folder as the scripts.
# WHY relative paths: Portability — works on any Android regardless of username.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "market_signal.log")
PREDICTIONS_FILE = os.path.join(BASE_DIR, "predictions.csv")

# ─────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────
# WHY both file AND console: File persists across runs for debugging.
# Console gives real-time feedback when running manually.
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─────────────────────────────────────────────
# NETWORK SETTINGS
# ─────────────────────────────────────────────
# Timeout in seconds for all HTTP requests.
# WHY 15: Mobile networks can be slow. 15s is patient but not endless.
REQUEST_TIMEOUT = 15

# Number of retries on network failure before giving up.
REQUEST_RETRIES = 3

# Delay (seconds) between retries.
# WHY exponential: Avoids hammering a struggling server.
RETRY_DELAY_BASE = 2   # Will be multiplied: 2s, 4s, 8s

# ─────────────────────────────────────────────
# ADVANCED MODEL / RISK SETTINGS
# ─────────────────────────────────────────────
# Bayesian fusion baseline weights (log-odds blend)
FUSION_WEIGHT_PRICE = 0.45
FUSION_WEIGHT_SENTIMENT = 0.30
FUSION_WEIGHT_REGIME = 0.25

# Exponential smoothing for online weight adaptation from observed outcomes
FUSION_LEARNING_RATE = 0.05

# Regime and physics thresholds
SHOCK_REGIME_PROB_THRESHOLD = 0.55
HIGH_ENERGY_THRESHOLD = 2.0
HIGH_ENTROPY_THRESHOLD = 0.95

# Risk sizing constraints
PAYOFF_RATIO = 1.0              # Assumed symmetric payoff
FRACTIONAL_KELLY = 0.20         # Conservative Kelly scaling
MAX_RECOMMENDED_WEIGHT = 0.20   # 20% max allocation
MIN_RECOMMENDED_WEIGHT = 0.0
WEIGHT_UNCERTAINTY_PENALTY = 0.60
DRAWDOWN_THROTTLE_FACTOR = 0.70
MAX_CONSECUTIVE_LOSSES = 3

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def validate_config():
    """
    Checks that required settings are filled in.
    Called at startup so the system fails fast with a clear error
    rather than crashing mysteriously later.
    """
    errors = []
    
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        errors.append("TELEGRAM_TOKEN not set in config.py")
    
    if TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        errors.append("TELEGRAM_CHAT_ID not set in config.py")
    
    if not (0.0 < BULLISH_THRESHOLD <= 1.0):
        errors.append("BULLISH_THRESHOLD must be between 0 and 1")
    
    total_weight = WEIGHT_SENTIMENT + WEIGHT_TREND + WEIGHT_VOLATILITY
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Signal weights must sum to 1.0, currently sum to {total_weight:.2f}")

    if MAX_ARTICLE_AGE_HOURS < 24:
        errors.append("MAX_ARTICLE_AGE_HOURS should cover at least one day")

    if REPORT_LOOKBACK_TRADING_DAYS < 1:
        errors.append("REPORT_LOOKBACK_TRADING_DAYS must be at least 1")

    if TOP_MOVEMENT_NEWS < 1:
        errors.append("TOP_MOVEMENT_NEWS must be at least 1")

    fusion_sum = FUSION_WEIGHT_PRICE + FUSION_WEIGHT_SENTIMENT + FUSION_WEIGHT_REGIME
    if abs(fusion_sum - 1.0) > 0.01:
        errors.append(f"Fusion weights must sum to 1.0, currently sum to {fusion_sum:.2f}")
    
    return errors
  
