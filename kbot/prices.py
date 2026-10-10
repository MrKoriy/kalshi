import statistics
from dataclasses import dataclass, field
from typing import Optional

import httpx

COINBASE = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com"
BINANCE = "https://api.binance.com"
BYBIT = "https://api.bybit.com"

UA = {"User-Agent": "kalshi-ai-bot/0.2"}

# alternate symbols per venue
KRAKEN_PAIR = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "XRP": "XRPUSD", "DOGE": "DOGEUSD"}
BYBIT_LINEAR = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
                "DOGE": "DOGEUSDT", "BNB": "BNBUSDT"}


@dataclass
class Candle:
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class SpotBundle:
    asset: str
    prices: dict = field(default_factory=dict)   # venue -> last price
    candles: dict = field(default_factory=dict)  # venue -> [Candle]

    @property
    def best_price(self) -> Optional[float]:
        for venue in ("coinbase", "kraken", "binance", "bybit"):
            if venue in self.prices:
                return self.prices[venue]
        return None

    @property
    def divergence(self) -> Optional[float]:
        cb, bn = self.prices.get("coinbase"), self.prices.get("binance")
        if cb and bn and cb > 0:
            return (bn - cb) / cb
        return None

    @property
    def best_candles(self) -> list:
        for venue in ("coinbase", "kraken", "binance", "bybit"):
            if len(self.candles.get(venue) or []) >= 5:
                return self.candles[venue]
        return []


async def _coinbase(http, asset, n):
    r = await http.get(f"{COINBASE}/products/{asset}-USD/candles", params={"granularity": 60})
    if r.status_code != 200:
        return None, []
    rows = r.json()[:n][::-1]
    cs = [Candle(int(t), float(o), float(h), float(l), float(c), float(v))
          for t, l, h, o, c, v in (row[:6] for row in rows)]
    return (cs[-1].c if cs else None), cs


async def _kraken(http, asset, n):
    pair = KRAKEN_PAIR.get(asset)
    if not pair:
        return None, []
    r = await http.get(f"{KRAKEN}/0/public/OHLC", params={"pair": pair, "interval": 1})
    if r.status_code != 200:
        return None, []
    result = r.json().get("result") or {}
    rows = result.get(pair) or (next((v for k, v in result.items() if k != "last"), []) or [])
    rows = rows[-n:]
    cs = [Candle(int(t), float(o), float(h), float(l), float(c), float(v))
          for t, o, h, l, c, v, *_ in rows]
    return (cs[-1].c if cs else None), cs


async def _binance(http, asset, n):
    symbol = f"{asset}USDT"
    r = await http.get(f"{BINANCE}/api/v3/klines",
                       params={"symbol": symbol, "interval": "1m", "limit": n})
    if r.status_code != 200:
        return None, []
    cs = [Candle(int(row[0]) // 1000, float(row[1]), float(row[2]), float(row[3]),
                 float(row[4]), float(row[5])) for row in r.json()]
    return (cs[-1].c if cs else None), cs


async def _bybit(http, asset, n):
    symbol = BYBIT_LINEAR.get(asset, f"{asset}USDT")
    r = await http.get(f"{BYBIT}/v5/market/kline",
                       params={"category": "spot", "symbol": symbol, "interval": "1", "limit": n})
    if r.status_code != 200 or r.json().get("retCode") != 0:
        return None, []
    rows = (r.json().get("result") or {}).get("list") or []
    cs = [Candle(int(row[0]) // 1000, float(row[1]), float(row[2]), float(row[3]),
                 float(row[4]), float(row[5])) for row in reversed(rows)]
    return (cs[-1].c if cs else None), cs


async def spot_bundle(http: httpx.AsyncClient, asset: str, n: int = 30) -> SpotBundle:
    bundle = SpotBundle(asset=asset)
    import asyncio
    results = await asyncio.gather(
        _safe(_coinbase(http, asset, n)),
        _safe(_kraken(http, asset, n)),
        _safe(_binance(http, asset, n)),
        _safe(_bybit(http, asset, n)),
    )
    for venue, res in zip(("coinbase", "kraken", "binance", "bybit"), results):
        if res is None:
            continue
        price, cs = res
        if price is not None:
            bundle.prices[venue] = price
        if cs:
            bundle.candles[venue] = cs
    return bundle


async def _safe(coro):
    try:
        return await coro
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return None


def realized_vol(cs: list, halflife: int = 10) -> Optional[float]:
    """EWMA annualized vol from 1-minute candles; returns per-minute sigma."""
    if len(cs) < 3:
        return None
    rets = []
    for a, b in zip(cs, cs[1:]):
        if a.c > 0 and b.c > 0:
            import math
            rets.append(math.log(b.c / a.c))
    if len(rets) < 2:
        return None
    w = [0.5 ** (i / halflife) for i in range(len(rets) - 1, -1, -1)]
    sw = sum(w)
    mean = sum(r * wi for r, wi in zip(rets, w)) / sw
    var = sum(wi * (r - mean) ** 2 for r, wi in zip(rets, w)) / sw
    return var ** 0.5


def atr(cs: list, n: int = 14) -> Optional[float]:
    if len(cs) < n + 1:
        return None
    trs = []
    for a, b in zip(cs[-n - 1:], cs[-n:]):
        trs.append(max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c)))
    return statistics.fmean(trs) if trs else None
