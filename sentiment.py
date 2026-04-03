"""
sentiment.py — News Sentiment Analysis
========================================
Responsibility: Given a list of articles, produce an aggregate sentiment
score in [-1.0, +1.0] and per-article breakdown.

WHY TextBlob instead of a transformer model (like BERT):
- TextBlob runs on CPU in ~1ms per article. BERT needs 500ms+ per article
  and requires >2GB RAM — completely impractical on Redmi Note 11.
- TextBlob's lexicon-based approach is surprisingly effective for
  financial headlines which tend to use explicit positive/negative words.
- Limitation: It misses sarcasm, context-dependent meaning, and
  domain-specific financial jargon. Acknowledged trade-off.

WHY domain keyword boosting:
- TextBlob's general English lexicon doesn't know that "rate hike"
  is bearish or "FII inflows" is bullish for Indian markets.
- We manually boost/penalize scores for known financial terms.
"""

from textblob import TextBlob
from typing import List, Dict, Tuple
import re

from logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# DOMAIN-SPECIFIC FINANCIAL LEXICON
# ─────────────────────────────────────────────
# These are terms TextBlob's general lexicon handles poorly.
# Scores are in [-1.0, +1.0] representing bearish to bullish sentiment.
#
# WHY include these: A general sentiment model sees "rate hike" as neutral
# (it contains no obviously negative words), but in Indian market context,
# RBI rate hikes typically cause Nifty to fall.

BULLISH_TERMS = {
    # Strong bullish signals
    "rate cut": 0.6,
    "fii inflows": 0.6,
    "foreign inflows": 0.5,
    "gdp growth": 0.5,
    "record high": 0.5,
    "all time high": 0.6,
    "bull run": 0.7,
    "earnings beat": 0.6,
    "profit surge": 0.6,
    "recovery": 0.3,
    "upgrade": 0.4,
    "rally": 0.4,
    "buying": 0.3,
    "gains": 0.3,
    "positive outlook": 0.5,
    "rbi accommodative": 0.5,
    "infrastructure push": 0.4,
    "capex boost": 0.4,
    "strong iip": 0.4,
    "exports rise": 0.4,
    "sensex surges": 0.5,
    "nifty surges": 0.5,
    "market up": 0.4,
}

BEARISH_TERMS = {
    # Strong bearish signals
    "rate hike": -0.6,
    "fii outflows": -0.6,
    "foreign outflows": -0.5,
    "recession": -0.7,
    "crash": -0.8,
    "selloff": -0.6,
    "sell-off": -0.6,
    "bear market": -0.7,
    "earnings miss": -0.6,
    "profit warning": -0.7,
    "downgrade": -0.5,
    "geopolitical tensions": -0.4,
    "inflation surge": -0.5,
    "deficit widens": -0.4,
    "current account deficit": -0.3,
    "rupee falls": -0.4,
    "rupee weakens": -0.4,
    "npa rises": -0.5,
    "market crash": -0.8,
    "global headwinds": -0.4,
    "uncertainty": -0.3,
    "slowdown": -0.4,
    "contraction": -0.5,
    "sensex drops": -0.5,
    "nifty drops": -0.5,
    "market down": -0.4,
}

# Combine for lookup
ALL_DOMAIN_TERMS = {**BULLISH_TERMS, **BEARISH_TERMS}


def _preprocess_text(text: str) -> str:
    """
    Cleans text before sentiment analysis.
    
    WHY: RSS feed text often contains HTML entities (&amp;, &lt;),
    special characters, and extra whitespace that confuse TextBlob.
    """
    # Remove HTML tags (some feeds include them in summaries)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Decode common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace(
        '&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
    
    # Remove URLs (they add no sentiment signal)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.lower()


def _get_domain_boost(text: str) -> float:
    """
    Scans text for domain-specific financial terms and returns a boost score.
    
    Returns:
        float: Net boost in [-1.0, +1.0]. Positive = bullish boost,
               negative = bearish boost.
    
    WHY cap at ±1.0: Prevents a single term from dominating.
    """
    boost = 0.0
    matched_terms = []
    
    for term, score in ALL_DOMAIN_TERMS.items():
        if term in text:
            boost += score
            matched_terms.append((term, score))
    
    if matched_terms:
        logger.debug("Domain terms matched: %s", matched_terms)
    
    # Normalize: if multiple terms matched, scale down to avoid overflow
    # WHY: 5 bearish terms shouldn't produce -3.0 — cap meaningful range
    if abs(boost) > 1.0:
        boost = boost / abs(boost)  # Clamp to [-1.0, +1.0]
    
    return boost


def _analyze_single_article(article: Dict) -> Dict:
    """
    Runs sentiment analysis on a single article.
    
    Returns the article dict enriched with:
    - 'textblob_score': Raw TextBlob polarity [-1, +1]
    - 'domain_boost': Financial domain adjustment [-1, +1]
    - 'final_score': Weighted combination
    - 'sentiment_label': POSITIVE / NEGATIVE / NEUTRAL
    """
    raw_text = article.get('text', '')
    
    if not raw_text:
        return {**article, 'textblob_score': 0.0, 'domain_boost': 0.0,
                'final_score': 0.0, 'sentiment_label': 'NEUTRAL'}
    
    clean_text = _preprocess_text(raw_text)
    
    # TextBlob polarity analysis
    # WHY try/except: TextBlob rarely but silently crashes on certain unicode
    try:
        blob = TextBlob(clean_text)
        textblob_score = blob.sentiment.polarity  # [-1.0, +1.0]
    except Exception as e:
        logger.warning("TextBlob failed on article '%s': %s",
                       article.get('title', 'unknown')[:40], e)
        textblob_score = 0.0
    
    # Domain keyword boost
    domain_boost = _get_domain_boost(clean_text)
    
    # Combine TextBlob (60% weight) + domain boost (40% weight)
    # WHY these weights: TextBlob is broader but noisier.
    # Domain terms are precise but sparse. Balance them.
    final_score = (0.60 * textblob_score) + (0.40 * domain_boost)
    
    # Clamp to valid range
    final_score = max(-1.0, min(1.0, final_score))
    
    # Label for human readability
    if final_score > 0.10:
        label = "POSITIVE"
    elif final_score < -0.10:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    
    # WHY ±0.10 threshold: TextBlob returns near-zero for genuinely neutral text.
    # A strict 0.0 cutoff would miss slightly tilted sentiment.
    
    return {
        **article,
        'textblob_score': round(textblob_score, 4),
        'domain_boost': round(domain_boost, 4),
        'final_score': round(final_score, 4),
        'sentiment_label': label,
    }


def analyze_sentiment(articles: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    Main entry point. Analyzes a list of articles and returns
    aggregate sentiment.
    
    Args:
        articles: List of article dicts from news_fetcher.
    
    Returns:
        Tuple of:
        - aggregate_score (float): Overall sentiment [-1.0, +1.0]
        - analyzed_articles (list): Articles enriched with sentiment fields
    
    WHY return both: The signal engine needs the score. The validator
    and Telegram formatter may want article-level detail.
    """
    if not articles:
        logger.warning("No articles provided to sentiment analyzer. Returning 0.")
        return 0.0, []
    
    logger.info("Analyzing sentiment for %d articles...", len(articles))
    
    analyzed = []
    for article in articles:
        result = _analyze_single_article(article)
        analyzed.append(result)
    
    # Compute aggregate score
    # WHY simple mean: Weighted by nothing — all articles get equal say.
    # A more sophisticated system would weight by source credibility or recency.
    # That's a future improvement. Simple is robust.
    scores = [a['final_score'] for a in analyzed]
    aggregate = sum(scores) / len(scores)
    
    # Count sentiment distribution (useful for signal confidence)
    positive_count = sum(1 for a in analyzed if a['sentiment_label'] == 'POSITIVE')
    negative_count = sum(1 for a in analyzed if a['sentiment_label'] == 'NEGATIVE')
    neutral_count = sum(1 for a in analyzed if a['sentiment_label'] == 'NEUTRAL')
    
    logger.info(
        "Sentiment analysis complete: aggregate=%.4f | "
        "POSITIVE=%d NEGATIVE=%d NEUTRAL=%d",
        aggregate, positive_count, negative_count, neutral_count
    )
    
    return round(aggregate, 4), analyzed


def get_sentiment_summary(aggregate_score: float, articles: List[Dict]) -> Dict:
    """
    Produces a human-readable summary dict for use in Telegram message.
    
    Args:
        aggregate_score: Output of analyze_sentiment()[0]
        articles: Analyzed articles from analyze_sentiment()[1]
    
    Returns:
        Dict with display-ready fields.
    """
    total = len(articles)
    if total == 0:
        return {
            'score': 0.0,
            'label': 'UNAVAILABLE',
            'positive_pct': 0,
            'negative_pct': 0,
            'article_count': 0,
        }
    
    positive = sum(1 for a in articles if a['sentiment_label'] == 'POSITIVE')
    negative = sum(1 for a in articles if a['sentiment_label'] == 'NEGATIVE')
    
    if aggregate_score > 0.10:
        label = "Positive 📰"
    elif aggregate_score < -0.10:
        label = "Negative 📰"
    else:
        label = "Mixed/Neutral 📰"
    
    return {
        'score': aggregate_score,
        'label': label,
        'positive_pct': round(100 * positive / total),
        'negative_pct': round(100 * negative / total),
        'article_count': total,
    }


# ── Quick sanity test ──
if __name__ == "__main__":
    sample_articles = [
        {'title': 'Nifty hits record high as FII inflows surge',
         'text': 'Nifty hits record high as FII inflows surge on positive GDP data.',
         'published': None, 'source': 'test'},
        {'title': 'Market crashes on rate hike fears',
         'text': 'Market crashes as RBI signals rate hike amid inflation surge.',
         'published': None, 'source': 'test'},
        {'title': 'Markets trade flat amid global uncertainty',
         'text': 'Markets trade flat amid global uncertainty and mixed cues.',
         'published': None, 'source': 'test'},
    ]
    
    score, analyzed = analyze_sentiment(sample_articles)
    print(f"\nAggregate Sentiment Score: {score}")
    print("\nPer-article breakdown:")
    for a in analyzed:
        print(f"  [{a['sentiment_label']:8s}] score={a['final_score']:+.4f} | {a['title'][:60]}")
