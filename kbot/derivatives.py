from dataclasses import dataclass, field
from typing import Optional

import httpx

BYBIT = "https://api.bybit.com"
BINANCE_FAPI = "https://fapi.binance.com"

SYMBOLS = {"BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "HYPE"}


@dataclass
class Derivs:
    asset: str
    funding: Optional[float] = None          # current funding rate, per 8h
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    open_interest: Optional[float] = None    # in contracts
    basis: Optional[float] = None            # (mark - index) / index
    source: str = ""
    extra: dict = field(default_factory=dict)


async def fetch(http: httpx.AsyncClient, asset: str) -> Derivs:
    if asset not in SYMBOLS:
        return Derivs(asset=asset)
    symbol = f"{asset}USDT"
    d = Derivs(asset=asset)

    try:
        r = await http.get(f"{BYBIT}/v5/market/tickers",
                           params={"category": "linear", "symbol": symbol})
        item = ((r.json().get("result") or {}).get("list") or [{}])[0]
        if r.status_code == 200 and item:
            d.source = "bybit"
            d.funding = _f(item.get("fundingRate"))
            d.mark_price = _f(item.get("markPrice"))
            d.index_price = _f(item.get("indexPrice"))
            d.open_interest = _f(item.get("openInterest"))
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        pass

    if d.funding is None:
        try:
            r = await http.get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", params={"symbol": symbol})
            j = r.json()
            if r.status_code == 200 and isinstance(j, dict):
                d.source = "binance"
                d.funding = _f(j.get("lastFundingRate"))
                d.mark_price = _f(j.get("markPrice"))
                d.index_price = _f(j.get("indexPrice"))
                r2 = await http.get(f"{BINANCE_FAPI}/fapi/v1/openInterest", params={"symbol": symbol})
                if r2.status_code == 200:
                    d.open_interest = _f(r2.json().get("openInterest"))
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass

    if d.mark_price and d.index_price and d.index_price > 0:
        d.basis = (d.mark_price - d.index_price) / d.index_price
    return d


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
