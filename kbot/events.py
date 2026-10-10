import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import httpx

FNG_URL = "https://api.alternative.me/fng/"
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]

# (keyword, score, description)
KEYWORDS = [
    ("cpi", 3, "US CPI inflation data"),
    ("fomc", 3, "FOMC / Fed decision"),
    ("rate decision", 3, "central bank rate decision"),
    ("fed", 2, "Fed-related news"),
    ("nonfarm", 3, "US jobs report"),
    ("payrolls", 3, "US jobs report"),
    ("etf", 1, "ETF flow news"),
    ("sec", 2, "SEC action"),
    ("breach", 2, "security incident"),
    ("attack", 1, "security incident"),
    ("outage", 1, "service outage"),
    ("liquidat", 1, "liquidation cascade"),
    ("lawsuit", 2, "legal action"),
    ("binance", 1, "exchange news"),
    ("coinbase", 1, "exchange news"),
]


@dataclass
class EventContext:
    fear_greed: Optional[int] = None
    fear_greed_label: str = ""
    headlines: list = field(default_factory=list)
    risk_score: int = 0
    matched: list = field(default_factory=list)

    def to_prompt_text(self) -> str:
        lines = []
        if self.fear_greed is not None:
            lines.append(f"fear_greed={self.fear_greed} ({self.fear_greed_label})")
        if self.headlines:
            lines.append("recent_headlines:")
            for h in self.headlines[:5]:
                lines.append(f"- {h}")
        if self.matched:
            lines.append("risk_flags: " + ", ".join(self.matched))
        return "\n".join(lines) if lines else "no macro context"


async def _rss_titles(http: httpx.AsyncClient, url: str, limit: int = 8) -> list:
    try:
        r = await http.get(url)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        titles = []
        for item in root.iter("item"):
            t = item.findtext("title")
            if t and t.strip():
                titles.append(t.strip()[:160])
            if len(titles) >= limit:
                break
        return titles
    except (httpx.HTTPError, ET.ParseError, ValueError):
        return []


async def fetch(http: httpx.AsyncClient) -> EventContext:
    ctx = EventContext()

    try:
        r = await http.get(FNG_URL, params={"limit": 1})
        data = (r.json().get("data") or [{}])[0]
        if r.status_code == 200 and data.get("value"):
            ctx.fear_greed = int(data["value"])
            ctx.fear_greed_label = data.get("value_classification", "")
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        pass

    results = await asyncio.gather(*[_rss_titles(http, u) for u in RSS_FEEDS])
    deduped = []
    seen_titles = set()
    for titles in results:
        for t in titles:
            key = t.lower()
            if key not in seen_titles:
                seen_titles.add(key)
                deduped.append(t)
    ctx.headlines = deduped[:10]

    seen = set()
    for h in ctx.headlines:
        low = h.lower()
        for kw, score, desc in KEYWORDS:
            if kw in low and desc not in seen:
                seen.add(desc)
                ctx.risk_score += score
                ctx.matched.append(desc)
    if ctx.fear_greed is not None and ctx.fear_greed <= 20:
        ctx.risk_score += 1
        ctx.matched.append("extreme fear")
    elif ctx.fear_greed is not None and ctx.fear_greed >= 80:
        ctx.risk_score += 1
        ctx.matched.append("extreme greed")
    return ctx
