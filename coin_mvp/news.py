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

COMMUNITY_SOURCES = (
    "https://gall.dcinside.com/board/lists/?id=bitcoins_new1",
    "https://bitman.kr/",
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
    "반등",
    "돌파",
    "상승",
    "급등",
    "매수",
    "호재",
    "불장",
    "간다",
    "펌핑",
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
    "하락",
    "급락",
    "손절",
    "손실",
    "숏",
    "악재",
    "패닉",
    "물림",
    "청산",
    "나락",
    "망",
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


def fetch_community_signal(timeout_seconds: int = 4, max_items: int = 20) -> NewsSignal:
    titles: list[str] = []
    for url in COMMUNITY_SOURCES:
        try:
            titles.extend(fetch_community_titles(url, timeout_seconds=timeout_seconds))
        except Exception:
            continue
    selected = dedupe_text(titles)[:max_items]
    signal = score_headlines(selected)
    return NewsSignal(
        sentiment_score=signal.sentiment_score,
        headline_count=signal.headline_count,
        risk_headline_count=signal.risk_headline_count,
        positive_headline_count=signal.positive_headline_count,
        latest_headlines=signal.latest_headlines,
        source="dcinside-bitcoin-gallery+bitman",
        fetched_at=signal.fetched_at,
    )


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


def fetch_community_titles(url: str, timeout_seconds: int) -> list[str]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html",
            "User-Agent": "Mozilla/5.0 coin-paper-simulation/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    html = decode_html(payload)
    candidates = extract_dcinside_post_titles(html)
    if not candidates:
        candidates = re.findall(r"<a[^>]+href=[\"'][^\"']*(?:board/view|/view|post|article)[^\"']*[\"'][^>]*>(.*?)</a>", html, flags=re.I | re.S)
    if not candidates:
        candidates = re.findall(r"title=[\"']([^\"']{2,80})[\"']", html, flags=re.I)
    titles = []
    for candidate in candidates:
        title = clean_html_text(candidate)
        if 2 <= len(title) <= 80 and not title.lower().startswith(("http", "javascript")):
            titles.append(title)
    return titles


def extract_dcinside_post_titles(html: str) -> list[str]:
    rows = re.findall(r"<tr[^>]+class=[\"'][^\"']*ub-content[^\"']*[\"'][^>]*>(.*?)</tr>", html, flags=re.I | re.S)
    titles: list[str] = []
    for row in rows:
        number_match = re.search(r"<td[^>]+class=[\"'][^\"']*gall_num[^\"']*[\"'][^>]*>(.*?)</td>", row, flags=re.I | re.S)
        if not number_match:
            continue
        number_text = clean_html_text(number_match.group(1))
        if not number_text.isdigit():
            continue
        title_match = re.search(r"<td[^>]+class=[\"'][^\"']*gall_tit[^\"']*[\"'][^>]*>(.*?)</td>", row, flags=re.I | re.S)
        if not title_match:
            continue
        title = clean_html_text(title_match.group(1))
        if title:
            titles.append(title)
    return titles


def decode_html(payload: bytes) -> str:
    best = ""
    best_score = -1
    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            text = payload.decode(encoding)
        except UnicodeDecodeError:
            continue
        sample = text[:4000]
        hangul_count = sum(1 for char in sample if "\uac00" <= char <= "\ud7a3")
        replacement_penalty = sample.count("\ufffd") * 5
        mojibake_penalty = sample.count("����") * 10
        score = hangul_count - replacement_penalty - mojibake_penalty
        if score > best_score:
            best = text
            best_score = score
    return best or payload.decode("utf-8", errors="ignore")


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


def dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = re.sub(r"\s+", " ", value.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def clean_headline(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if " - " in value:
        return value.rsplit(" - ", 1)[0].strip()
    return value


def clean_html_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = (
        value.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", value).strip()


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
