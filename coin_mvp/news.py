from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any


DEFAULT_NEWS_QUERIES = (
    "bitcoin OR crypto market",
    "upbit crypto korea",
    "ethereum bitcoin regulation",
)

POSITIVE_TERMS = {
    "approve",
    "approval",
    "bull",
    "bullish",
    "breakout",
    "etf inflow",
    "inflow",
    "rally",
    "recover",
    "surge",
    "institutional",
    "adoption",
    "accumulate",
}

NEGATIVE_TERMS = {
    "hack",
    "exploit",
    "outflow",
    "lawsuit",
    "ban",
    "crackdown",
    "sec sues",
    "liquidation",
    "selloff",
    "plunge",
    "fear",
    "fraud",
    "bankruptcy",
}


@dataclass(frozen=True)
class NewsSignal:
    sentiment_score: float
    headline_count: int
    risk_headline_count: int
    positive_headline_count: int
    latest_headlines: list[str]
    source: str = "google-news-rss"
    fetched_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fetch_crypto_news_signal(timeout_seconds: int = 4, max_items: int = 12) -> NewsSignal:
    headlines: list[tuple[datetime, str]] = []
    for query in DEFAULT_NEWS_QUERIES:
        try:
            headlines.extend(fetch_google_news_rss(query, timeout_seconds=timeout_seconds))
        except Exception:
            continue
    deduped = dedupe_headlines(headlines)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=36)
    fresh = [(published_at, title) for published_at, title in deduped if published_at >= cutoff]
    selected = fresh[:max_items] if fresh else deduped[:max_items]
    return score_headlines([title for _, title in selected])


def fetch_google_news_rss(query: str, timeout_seconds: int) -> list[tuple[datetime, str]]:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
    )
    url = f"https://news.google.com/rss/search?{params}"
    request = urllib.request.Request(url, headers={"Accept": "application/rss+xml", "User-Agent": "coin-paper-simulation/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    items = []
    for item in root.findall(".//item"):
        title = clean_headline(item.findtext("title") or "")
        if not title:
            continue
        published_at = parse_rss_datetime(item.findtext("pubDate") or "")
        items.append((published_at, title))
    items.sort(key=lambda value: value[0], reverse=True)
    return items


def parse_rss_datetime(value: str) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def dedupe_headlines(headlines: list[tuple[datetime, str]]) -> list[tuple[datetime, str]]:
    seen: set[str] = set()
    result = []
    for published_at, title in sorted(headlines, key=lambda value: value[0], reverse=True):
        key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        result.append((published_at, title))
    return result


def clean_headline(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if " - " in value:
        return value.rsplit(" - ", 1)[0].strip()
    return value


def score_headlines(headlines: list[str]) -> NewsSignal:
    positive = 0
    negative = 0
    for headline in headlines:
        normalized = headline.lower()
        if any(term in normalized for term in POSITIVE_TERMS):
            positive += 1
        if any(term in normalized for term in NEGATIVE_TERMS):
            negative += 1
    total = max(1, len(headlines))
    raw_score = (positive - negative) / total
    return NewsSignal(
        sentiment_score=max(-1.0, min(1.0, raw_score)),
        headline_count=len(headlines),
        risk_headline_count=negative,
        positive_headline_count=positive,
        latest_headlines=headlines[:5],
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
