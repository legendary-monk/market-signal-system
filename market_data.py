"""
market_data.py — Market Price Data Fetcher
============================================
Responsibility: Fetch historical OHLCV data for the Nifty 50 index
from Yahoo Finance and return it as a clean pandas DataFrame.

WHY Yahoo Finance (yfinance):
- Free, no API key, covers Nifty 50 (^NSEI)
- Returns OHLCV data in standard format
- Well-maintained Python library
- Limitation: Rate limits exist. If run too frequently, Yahoo
  may temporarily block. We run once daily — well within limits.

WHY pandas DataFrame as output:
- All downstream computations (features.py) use vectorized pandas operations
- Easy to inspect, debug, and extend
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import time

import config
from logger import get_logger

logger = get_logger(__name__)


def _validate_dataframe(df: pd.DataFrame, ticker: str) -> Tuple[bool, str]:
    """
    Validates that the fetched DataFrame has what we need.
    
    WHY separate validation: Corrupted data that passes silently causes
    wrong signals. Explicit validation fails fast with a clear error.
    
    Returns:
        (is_valid: bool, error_message: str)
    """
    if df is None or df.empty:
        return False, f"No data returned for {ticker}"
    
    # Check required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c in df.columns]
    # Note: yfinance sometimes returns lowercase column names
    # We'll handle both cases below
    
    if len(df) < 5:
        return False, f"Insufficient data: only {len(df)} rows (need ≥5 for trend)"
    
    # Check for excessive NaN in Close prices
    nan_ratio = df['Close'].isna().mean()
    if nan_ratio > 0.3:
        return False, f"Too many missing Close prices: {nan_ratio:.0%} NaN"
    
    return True, ""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names to Title Case (Open, High, Low, Close, Volume).
    
    WHY: yfinance sometimes returns lowercase columns depending on version.
    Downstream code uses Title Case. This adapter prevents silent KeyErrors.
    """
    df.columns = [c.capitalize() if c.lower() in 
                  ['open', 'high', 'low', 'close', 'volume', 'adj close']
                  else c for c in df.columns]
    
    # Rename 'Adj close' to 'Adj_Close' if present
    if 'Adj close' in df.columns:
        df = df.rename(columns={'Adj close': 'Adj_Close'})
    
    return df


def fetch_market_data(
    ticker: str = None,
    lookback_days: int = None
) -> Optional[pd.DataFrame]:
    """
    Fetches historical OHLCV data for the given ticker.
    
    Args:
        ticker: Yahoo Finance symbol (e.g., "^NSEI"). Defaults to config.
        lookback_days: Calendar days of history to fetch. Defaults to config.
    
    Returns:
        pd.DataFrame with columns [Open, High, Low, Close, Volume, Returns]
        indexed by date (DatetimeIndex, UTC-naive).
        Returns None if data cannot be fetched.
    
    WHY return None instead of raising: The signal engine checks for None
    and gracefully degrades (uses sentiment-only signal). The system stays
    alive even when Yahoo Finance is temporarily unavailable.
    """
    ticker = ticker or config.MARKET_TICKER
    lookback_days = lookback_days or config.MARKET_LOOKBACK_DAYS
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info("Fetching %s data from %s to %s...",
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'))
    
    for attempt in range(1, config.REQUEST_RETRIES + 1):
        try:
            # WHY auto_adjust=True: Adjusts for splits and dividends.
            # For index trackers like Nifty, this doesn't matter much,
            # but it's good practice for any OHLCV analysis.
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                auto_adjust=True,
                actions=False  # WHY False: We don't need dividend/split events
            )
            
            # yfinance returns an empty DataFrame (not None) on failure
            if df is None or df.empty:
                logger.warning(
                    "yfinance returned empty data for %s (attempt %d/%d). "
                    "This often happens outside trading hours or on holidays.",
                    ticker, attempt, config.REQUEST_RETRIES
                )
                
                if attempt < config.REQUEST_RETRIES:
                    wait = config.RETRY_DELAY_BASE ** attempt
                    logger.info("Waiting %ds before retry...", wait)
                    time.sleep(wait)
                continue
            
            # Normalize column names for consistency
            df = _normalize_columns(df)
            
            # Drop timezone info from index for cleaner downstream handling
            # WHY: Mixed-timezone DataFrames cause subtle bugs in pandas operations
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Validate
            is_valid, error_msg = _validate_dataframe(df, ticker)
            if not is_valid:
                logger.error("Data validation failed: %s", error_msg)
                return None
            
            # Add daily returns column
            # WHY pct_change: Returns are stationary; raw prices aren't.
            # All volatility and trend features are built on returns, not price.
            df['Returns'] = df['Close'].pct_change()
            
            # Drop the first row (NaN return from pct_change)
            df = df.dropna(subset=['Returns'])
            
            logger.info(
                "Successfully fetched %d trading days of %s data. "
                "Latest close: %.2f",
                len(df), ticker, df['Close'].iloc[-1]
            )
            
            return df
            
        except Exception as e:
            logger.error(
                "Unexpected error fetching %s (attempt %d/%d): %s",
                ticker, attempt, config.REQUEST_RETRIES, e,
                exc_info=True
            )
            
            if attempt < config.REQUEST_RETRIES:
                wait = config.RETRY_DELAY_BASE ** attempt
                time.sleep(wait)
    
    logger.error("All %d attempts to fetch %s data failed. Returning None.",
                 config.REQUEST_RETRIES, ticker)
    return None


def get_latest_price(df: pd.DataFrame) -> Optional[float]:
    """
    Safely extracts the most recent closing price from the DataFrame.
    
    Args:
        df: Output of fetch_market_data().
    
    Returns:
        float or None.
    
    WHY a dedicated function: Simple but important. iloc[-1] on an empty
    DataFrame raises IndexError. This wraps that safely.
    """
    if df is None or df.empty:
        return None
    
    try:
        return float(df['Close'].iloc[-1])
    except (IndexError, KeyError, ValueError) as e:
        logger.error("Could not extract latest price: %s", e)
        return None


def get_price_change_pct(df: pd.DataFrame, days: int = 1) -> Optional[float]:
    """
    Computes the percentage price change over the last N trading days.
    
    Args:
        df: Output of fetch_market_data().
        days: Number of trading days to look back.
    
    Returns:
        Percentage change as a float (e.g., 1.5 means +1.5%).
        Returns None if insufficient data.
    """
    if df is None or len(df) < days + 1:
        logger.warning("Insufficient data for %d-day price change calculation", days)
        return None
    
    try:
        latest = df['Close'].iloc[-1]
        past = df['Close'].iloc[-(days + 1)]
        
        if past == 0:
            return None
        
        return round(((latest - past) / past) * 100, 4)
    
    except (IndexError, KeyError, ZeroDivisionError) as e:
        logger.error("Error computing price change: %s", e)
        return None


# ── Quick sanity test ──
if __name__ == "__main__":
    df = fetch_market_data()
    
    if df is not None:
        print(f"\nData shape: {df.shape}")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"\nLatest closing price (Nifty 50): {get_latest_price(df):.2f}")
        print(f"1-day change: {get_price_change_pct(df, 1):+.2f}%")
        print(f"5-day change: {get_price_change_pct(df, 5):+.2f}%")
        print(f"\nTail of data:")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']].tail())
    else:
        print("Failed to fetch data.")
