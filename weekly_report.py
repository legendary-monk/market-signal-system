"""
weekly_report.py — Weekly Movement and News Attribution
========================================================
Builds the weekly context block for the Telegram report.

The module links the largest NIFTY 50 close-to-close moves of the week with
same-day RSS headlines. It intentionally describes links as "associated" or
"likely drivers" because RSS headlines and price data alone cannot prove
causality.
"""

from datetime import datetime, date
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

import config
from logger import get_logger

logger = get_logger(__name__)


def _parse_article_date(article: Dict[str, Any]) -> Optional[date]:
    """Returns the publication date for an article, if available."""
    published = article.get('published')
    if not published:
        return None

    try:
        return datetime.fromisoformat(str(published).replace('Z', '+00:00')).date()
    except ValueError:
        logger.debug("Could not parse article date: %s", published)
        return None


def _source_name(source: str) -> str:
    """Converts a feed URL into a compact source label."""
    if not source:
        return "unknown source"

    host = urlparse(source).netloc or source
    return host.replace('www.', '')


def _article_direction_matches_move(article: Dict[str, Any], move_pct: float) -> bool:
    """Returns True when article sentiment direction aligns with price move."""
    score = article.get('final_score', 0.0) or 0.0
    return (move_pct >= 0 and score >= 0) or (move_pct < 0 and score <= 0)


def _rank_articles_for_move(
    articles: List[Dict[str, Any]],
    move_date: date,
    move_pct: float,
) -> List[Dict[str, Any]]:
    """Ranks same-day articles by directional agreement and sentiment strength."""
    same_day = [a for a in articles if _parse_article_date(a) == move_date]
    if not same_day:
        return []

    return sorted(
        same_day,
        key=lambda article: (
            _article_direction_matches_move(article, move_pct),
            abs(article.get('final_score', 0.0) or 0.0),
        ),
        reverse=True,
    )


def build_weekly_movement_news(
    market_df: Optional[pd.DataFrame],
    analyzed_articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Finds the biggest weekly market moves and attaches likely associated news.

    Returns a list of dicts with date, move_pct, close, direction, and up to
    TOP_MOVEMENT_NEWS same-day articles. Empty list means the weekly attribution
    block should be omitted or marked unavailable.
    """
    if market_df is None or market_df.empty or 'Returns' not in market_df.columns:
        logger.warning("Weekly movement-news attribution skipped: market data unavailable")
        return []

    weekly_df = market_df.tail(config.REPORT_LOOKBACK_TRADING_DAYS).copy()
    if weekly_df.empty:
        return []

    weekly_df = weekly_df.dropna(subset=['Returns'])
    weekly_df['move_pct'] = weekly_df['Returns'] * 100
    weekly_df['abs_move_pct'] = weekly_df['move_pct'].abs()
    weekly_df = weekly_df.sort_values('abs_move_pct', ascending=False)

    movement_news = []
    for idx, row in weekly_df.head(config.TOP_MOVEMENT_NEWS).iterrows():
        move_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
        move_pct = round(float(row['move_pct']), 2)
        ranked_articles = _rank_articles_for_move(analyzed_articles, move_date, move_pct)

        articles = []
        for article in ranked_articles[:config.TOP_MOVEMENT_NEWS]:
            articles.append({
                'title': article.get('title', 'Untitled')[:140],
                'sentiment_label': article.get('sentiment_label', 'NEUTRAL'),
                'sentiment_score': article.get('final_score', 0.0),
                'source': _source_name(article.get('source', '')),
            })

        movement_news.append({
            'date': move_date.isoformat(),
            'move_pct': move_pct,
            'close': round(float(row['Close']), 2) if 'Close' in row else None,
            'direction': 'up' if move_pct >= 0 else 'down',
            'articles': articles,
        })

    logger.info("Weekly movement-news attribution built for %d moves", len(movement_news))
    return movement_news
