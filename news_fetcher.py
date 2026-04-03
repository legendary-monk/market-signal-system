"""
news_fetcher.py — RSS News Ingestion
=======================================
Responsibility: Download and parse financial news from RSS feeds.
Returns a clean list of article dicts for downstream sentiment analysis.

WHY RSS feeds specifically:
- No API keys required
- Lightweight XML parsing (no JS rendering needed)
- Reliable for major Indian financial outlets
- Works on mobile with minimal compute
"""

import feedparser
import requests
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import config
from logger import get_logger

logger = get_logger(__name__)


def _is_article_fresh(entry: feedparser.FeedParserDict) -> bool:
    """
    Checks if an article was published within MAX_ARTICLE_AGE_HOURS.
    
    WHY filter by age: Sentiment from 3-day-old news is stale.
    A BEARISH article from Monday shouldn't affect Wednesday's signal.
    
    Args:
        entry: A feedparser entry object.
    
    Returns:
        bool: True if article is recent enough.
    """
    # feedparser normalizes dates into published_parsed (a time.struct_time)
    if not hasattr(entry, 'published_parsed') or entry.published_parsed is None:
        # WHY include undated articles: Some RSS feeds (especially Indian outlets)
        # are inconsistent with date metadata. We don't want to silently drop them.
        # Better to include with a warning.
        logger.debug("Article has no date metadata — including anyway: %s",
                     getattr(entry, 'title', 'Unknown'))
        return True
    
    # Convert to timezone-aware UTC datetime
    pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.MAX_ARTICLE_AGE_HOURS)
    
    return pub_time >= cutoff


def _parse_entry(entry: feedparser.FeedParserDict, source: str) -> Optional[Dict]:
    """
    Extracts relevant fields from a feedparser entry into a clean dict.
    
    WHY normalize to a dict: Downstream modules shouldn't depend on
    feedparser's internal object structure. Decoupling matters.
    
    Args:
        entry: Raw feedparser entry.
        source: URL of the feed this entry came from.
    
    Returns:
        dict with keys: title, summary, text, published, source
        or None if the entry has no usable text content.
    """
    title = getattr(entry, 'title', '').strip()
    summary = getattr(entry, 'summary', '').strip()
    
    # Some feeds put content in 'content' key instead of 'summary'
    content = ''
    if hasattr(entry, 'content') and entry.content:
        content = entry.content[0].get('value', '').strip()
    
    # We want title + summary/content for sentiment analysis
    # WHY concatenate: Title alone is too short for reliable sentiment.
    # Summary alone sometimes misses the key framing in the headline.
    text = f"{title}. {summary or content}".strip('. ')
    
    if len(text) < 20:
        # Too short to be meaningful — likely a malformed entry
        logger.debug("Skipping article with insufficient text: %s", title)
        return None
    
    # Get published date as ISO string (human-readable for logging/storage)
    published = None
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        try:
            published = datetime(*entry.published_parsed[:6]).isoformat()
        except (TypeError, ValueError):
            published = None
    
    return {
        'title': title,
        'summary': summary,
        'text': text,         # Primary field for sentiment analysis
        'published': published,
        'source': source,
    }


def _fetch_single_feed(feed_url: str) -> List[Dict]:
    """
    Fetches and parses one RSS feed with retry logic.
    
    WHY retry: Mobile networks on Redmi Note 11 can drop packets.
    A transient 3-second timeout shouldn't kill the whole run.
    
    Args:
        feed_url: URL of the RSS feed.
    
    Returns:
        List of article dicts, possibly empty on failure.
    """
    articles = []
    
    for attempt in range(1, config.REQUEST_RETRIES + 1):
        try:
            logger.debug("Fetching feed [attempt %d/%d]: %s",
                         attempt, config.REQUEST_RETRIES, feed_url)
            
            # WHY requests instead of feedparser directly:
            # feedparser's built-in HTTP client doesn't support timeout.
            # Hanging forever on a slow Indian server is unacceptable.
            response = requests.get(
                feed_url,
                timeout=config.REQUEST_TIMEOUT,
                headers={
                    # WHY a User-Agent: Some servers block requests with no UA,
                    # treating them as bots (ironic, but common).
                    'User-Agent': 'Mozilla/5.0 (compatible; MarketSignalBot/1.0)'
                }
            )
            response.raise_for_status()  # Raises on 4xx/5xx HTTP errors
            
            # Parse the XML content feedparser received
            feed = feedparser.parse(response.content)
            
            if feed.bozo and not feed.entries:
                # 'bozo' flag = feedparser found XML errors but may have recovered.
                # WHY check entries too: Bozo doesn't always mean total failure.
                logger.warning("Feed has XML errors and no entries: %s", feed_url)
                break
            
            # Process each entry in this feed
            feed_article_count = 0
            for entry in feed.entries:
                if not _is_article_fresh(entry):
                    continue
                
                parsed = _parse_entry(entry, feed_url)
                if parsed:
                    articles.append(parsed)
                    feed_article_count += 1
            
            logger.info("Fetched %d fresh articles from %s",
                        feed_article_count, feed_url)
            break  # Success — don't retry
            
        except requests.exceptions.Timeout:
            logger.warning("Timeout on feed %s (attempt %d/%d)",
                           feed_url, attempt, config.REQUEST_RETRIES)
            
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error on feed %s (attempt %d/%d)",
                           feed_url, attempt, config.REQUEST_RETRIES)
            
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP error %s on feed %s", e, feed_url)
            break  # WHY break on HTTP errors: Retrying a 404 is pointless.
            
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", feed_url, e,
                         exc_info=True)
            break
        
        # Exponential backoff between retries
        if attempt < config.REQUEST_RETRIES:
            delay = config.RETRY_DELAY_BASE ** attempt
            logger.debug("Waiting %ds before retry...", delay)
            time.sleep(delay)
    
    return articles


def fetch_news() -> List[Dict]:
    """
    Main entry point. Fetches articles from all configured RSS feeds.
    
    Returns:
        List of article dicts, deduplicated, capped at MAX_ARTICLES.
        Returns empty list (not exception) if all feeds fail.
    
    WHY return empty list on failure: The signal engine handles missing
    data gracefully. Raising here would crash the whole pipeline.
    """
    logger.info("Starting news fetch from %d feeds...", len(config.RSS_FEEDS))
    
    all_articles = []
    failed_feeds = 0
    
    for feed_url in config.RSS_FEEDS:
        articles = _fetch_single_feed(feed_url)
        all_articles.extend(articles)
        
        if not articles:
            failed_feeds += 1
        
        # Brief pause between feed requests
        # WHY: Avoid looking like a DDoS attack to news servers
        time.sleep(1)
    
    if failed_feeds == len(config.RSS_FEEDS):
        logger.error("ALL feeds failed. No news data available.")
        return []
    
    # Deduplicate by title to remove cross-posted articles
    # WHY deduplicate: Economic Times syndicates to multiple outlets.
    # Counting the same article twice inflates sentiment score.
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        # Use first 60 chars of title as fingerprint (handles minor variations)
        fingerprint = article['title'][:60].lower().strip()
        if fingerprint not in seen_titles:
            seen_titles.add(fingerprint)
            unique_articles.append(article)
    
    # Cap total articles
    result = unique_articles[:config.MAX_ARTICLES]
    
    logger.info("News fetch complete: %d unique articles (from %d total, %d feeds failed)",
                len(result), len(all_articles), failed_feeds)
    
    return result


# ── Quick sanity test (run this file directly to check feeds) ──
if __name__ == "__main__":
    articles = fetch_news()
    print(f"\nTotal articles fetched: {len(articles)}")
    if articles:
        print("\nSample article:")
        print(f"  Title: {articles[0]['title']}")
        print(f"  Published: {articles[0]['published']}")
        print(f"  Text preview: {articles[0]['text'][:100]}...")
