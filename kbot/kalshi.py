from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

BASE = "https://api.elections.kalshi.com/trade-api/v2"

SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
    "DOGE": "KXDOGE15M",
    "HYPE": "KXHYPE15M",
    "BNB": "KXBNB15M",
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _price(m: dict, dollars_key: str, cents_key: str) -> Optional[float]:
    v = _f(m.get(dollars_key))
    if v is not None:
        return v
    c = _f(m.get(cents_key))
    return c / 100.0 if c is not None else None


@dataclass
class Level:
    price: float
    size: float


@dataclass
class Snapshot:
    asset: str
    ticker: str
    close_time: datetime
    open_time: Optional[datetime]
    strike: Optional[float]
    rules: str
    volume: float
    open_interest: float
    up_levels: list  # ask side for UP (derived from NO bids)
    down_levels: list  # ask side for DOWN (derived from YES bids)
    up_bid: Optional[float]
    down_bid: Optional[float]
    last_price: Optional[float]
    fetched_at: datetime

    @property
    def minutes_left(self) -> float:
        return (self.close_time - now_utc()).total_seconds() / 60.0

    @property
    def up_ask(self) -> Optional[float]:
        return self.up_levels[0].price if self.up_levels else None

    @property
    def down_ask(self) -> Optional[float]:
        return self.down_levels[0].price if self.down_levels else None

    @property
    def mid_up(self) -> Optional[float]:
        if self.up_ask is None or self.up_bid is None:
            return None
        return (self.up_ask + self.up_bid) / 2

    @property
    def spread(self) -> Optional[float]:
        if self.up_ask is None or self.up_bid is None:
            return None
        return max(0.0, self.up_ask - self.up_bid)

    def imbalance(self, depth: int = 3) -> Optional[float]:
        if len(self.up_levels) < depth or len(self.down_levels) < depth:
            return None
        up_sz = sum(l.size for l in self.up_levels[:depth])
        down_sz = sum(l.size for l in self.down_levels[:depth])
        total = up_sz + down_sz
        return (down_sz - up_sz) / total if total > 0 else None

    def fill_estimate(self, side: str, contracts: float) -> Optional[dict]:
        levels = self.up_levels if side == "UP" else self.down_levels
        if not levels:
            return None
        remaining, cost, filled = contracts, 0.0, 0.0
        for lvl in levels:
            take = min(remaining, lvl.size)
            cost += take * lvl.price
            filled += take
            remaining -= take
            if remaining <= 0:
                break
        if filled <= 0:
            return None
        return {"avg": cost / filled, "filled": filled, "partial": remaining > 0}


class Kalshi:
    def __init__(self, http: httpx.AsyncClient, depth: int = 10):
        self.http = http
        self.depth = depth

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        try:
            r = await self.http.get(BASE + path, params=params)
            if r.status_code != 200:
                return None
            return r.json()
        except (httpx.HTTPError, ValueError):
            return None

    async def snapshot(self, asset: str) -> Optional[Snapshot]:
        data = await self._get(
            "/markets", {"series_ticker": SERIES[asset], "status": "open", "limit": 10}
        )
        markets = (data or {}).get("markets") or []
        live = []
        for m in markets:
            try:
                ct = parse_iso(m["close_time"])
            except (KeyError, ValueError, TypeError):
                continue
            if (ct - now_utc()).total_seconds() > 0:
                live.append((ct, m))
        if not live:
            return None
        ct, m = min(live, key=lambda x: x[0])

        up_ask = _price(m, "yes_ask_dollars", "yes_ask")
        down_ask = _price(m, "no_ask_dollars", "no_ask")
        up_bid = _price(m, "yes_bid_dollars", "yes_bid")
        down_bid = _price(m, "no_bid_dollars", "no_bid")

        snap = Snapshot(
            asset=asset,
            ticker=m["ticker"],
            close_time=ct,
            open_time=None,
            strike=_f(m.get("floor_strike")) or _f(m.get("cap_strike")),
            rules=(m.get("rules_primary") or "")[:500],
            volume=_f(m.get("volume_fp")) or 0.0,
            open_interest=_f(m.get("open_interest_fp")) or 0.0,
            up_levels=[Level(up_ask, 0.0)] if up_ask is not None else [],
            down_levels=[Level(down_ask, 0.0)] if down_ask is not None else [],
            up_bid=up_bid,
            down_bid=down_bid,
            last_price=_f(m.get("last_price_dollars")),
            fetched_at=now_utc(),
        )

        book = await self._get(f"/markets/{snap.ticker}/orderbook", {"depth": self.depth})
        ob = (book or {}).get("orderbook_fp") or (book or {}).get("orderbook") or {}
        no_lv = self._levels(ob, "no")
        yes_lv = self._levels(ob, "yes")
        if no_lv or yes_lv:
            # YES ask = 1 - NO bid; DOWN ask = 1 - YES bid. Best (lowest ask) first.
            snap.up_levels = [Level(round(1 - p, 4), s) for p, s in sorted(no_lv, key=lambda x: -x[0])]
            snap.down_levels = [Level(round(1 - p, 4), s) for p, s in sorted(yes_lv, key=lambda x: -x[0])]
            snap.up_bid = max((p for p, _ in yes_lv), default=None)
            snap.down_bid = max((p for p, _ in no_lv), default=None)
        return snap

    @staticmethod
    def _levels(ob: dict, side: str) -> list:
        out = []
        for key, scale in ((f"{side}_dollars", 1.0), (side, 100.0)):
            lv = ob.get(key)
            if isinstance(lv, list) and lv:
                for lvl in lv:
                    if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                        p, s = _f(lvl[0]), _f(lvl[1])
                        if p is not None:
                            out.append((p / scale, s or 0.0))
                if out:
                    break
        return out

    async def trades(self, ticker: str, limit: int = 25) -> list:
        data = await self._get("/markets/trades", {"ticker": ticker, "limit": limit})
        out = []
        for t in (data or {}).get("trades") or []:
            out.append({
                "ts": t.get("created_time"),
                "yes_price": _price(t, "yes_price_dollars", "yes_price"),
                "count": _f(t.get("count_fp")) or _f(t.get("count")) or 0.0,
                "taker_side": t.get("taker_side"),
            })
        return out

    async def result(self, ticker: str) -> Optional[str]:
        data = await self._get(f"/markets/{ticker}")
        res = ((data or {}).get("market") or {}).get("result")
        if isinstance(res, str) and res.lower() in ("yes", "no"):
            return res.lower()
        return None
