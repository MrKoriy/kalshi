"""Historical backtest of the replayable strategies over settled Kalshi 15m markets.

Replays, per market, at decision time (close - DECIDE_BEFORE minutes):
  - spot + sigma_min from Coinbase 1m candles (same realized_vol as live)
  - momentum mom_5m from the same candles
  - market prices from Kalshi 1m candlesticks (yes_ask -> up_ask, 1-yes_bid -> down_ask)
  - outcome from the market result
Strategies: vol_only (same decide_side + fees as live), momentum, always_up, always_down.
LLM/Jev voters cannot be backtested (no historical calls), so the full ensemble is not
replayed here.

Usage: python -m kbot.hist --days 2 --assets BTC,ETH,SOL
"""

import argparse
import asyncio
import json
import math
from datetime import datetime, timedelta, timezone

import httpx

from .kalshi import BASE, SERIES, parse_iso
from .prices import COINBASE, Candle, realized_vol
from .quant import fair_prob_up
from .ensemble import decide_side
from .ledger import kalshi_fee, fee_per_contract
from .config import load

SIZE = 300.0


async def fetch_markets(http, series: str, since: datetime) -> list:
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = await http.get(f"{BASE}/markets", params=params)
        if r.status_code != 200:
            break
        j = r.json()
        ms = j.get("markets") or []
        stop = False
        for m in ms:
            try:
                ct = parse_iso(m["close_time"])
            except Exception:
                continue
            if ct < since:
                stop = True
                break
            if m.get("result") in ("yes", "no") and m.get("floor_strike"):
                out.append({"ticker": m["ticker"], "close": ct,
                            "strike": float(m["floor_strike"]), "result": m["result"]})
        cursor = j.get("cursor")
        if stop or not cursor or not ms:
            break
    return out


async def fetch_market_book(http, series: str, ticker: str, close: datetime,
                            decide_before: int):
    """Kalshi 1m candlesticks -> (up_ask, down_ask) at decision time."""
    t_dec = close - timedelta(minutes=decide_before)
    s = int((close - timedelta(minutes=decide_before + 5)).timestamp())
    e = int(close.timestamp())
    try:
        r = await http.get(f"{BASE}/series/{series}/markets/{ticker}/candlesticks",
                           params={"start_ts": s, "end_ts": e, "period_interval": 1})
        if r.status_code != 200:
            return None
        cs = r.json().get("candlesticks") or []
    except Exception:
        return None
    best = None
    for c in cs:
        ts = c.get("end_period_ts")
        if ts and ts <= int(t_dec.timestamp()) + 60:
            if best is None or ts > best["end_period_ts"]:
                best = c
    if not best:
        return None
    def _f(d):
        try:
            return float(d.get("close_dollars"))
        except (TypeError, ValueError, AttributeError):
            return None
    ya = _f(best.get("yes_ask") or {})
    yb = _f(best.get("yes_bid") or {})
    if ya is None or yb is None or not (0 < ya < 1) or not (0 < yb < 1):
        return None
    return {"up_ask": ya, "down_ask": round(1 - yb, 4)}


async def fetch_coinbase(http, asset: str, start: datetime, end: datetime) -> list:
    """1m candles, oldest first."""
    out = []
    t = start
    while t < end:
        t2 = min(t + timedelta(minutes=300), end)
        try:
            r = await http.get(f"{COINBASE}/products/{asset}-USD/candles",
                               params={"granularity": 60,
                                       "start": t.isoformat(), "end": t2.isoformat()})
            if r.status_code == 200:
                for row in r.json():
                    out.append(Candle(t=int(row[0]), l=float(row[1]), h=float(row[2]),
                                      o=float(row[3]), c=float(row[4]), v=float(row[5])))
        except Exception:
            pass
        t = t2
        await asyncio.sleep(0.2)
    out.sort(key=lambda c: c.t)
    return out


def candles_before(cs: list, ts: int, n: int) -> list:
    return [c for c in cs if c.t <= ts][-n:]


async def run(args):
    cfg = load(require_llm=False)
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    sem = asyncio.Semaphore(8)
    stats = {s: {"n": 0, "w": 0, "pnl": 0.0, "staked": 0.0}
             for s in ("vol_only", "momentum", "always_up", "always_down")}
    brier_model, brier_market, scored = 0.0, 0.0, 0
    skipped = {"book": 0, "spot": 0}

    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "kalshi-ai-bot/hist"}) as http:
        for asset in args.assets.split(","):
            asset = asset.strip()
            series = SERIES[asset]
            markets = await fetch_markets(http, series, since)
            markets.sort(key=lambda m: m["close"])
            print(f"{asset}: {len(markets)} settled markets in window", flush=True)
            if not markets:
                continue
            cs = await fetch_coinbase(http, asset,
                                      markets[0]["close"] - timedelta(minutes=60),
                                      markets[-1]["close"])
            print(f"{asset}: {len(cs)} spot candles", flush=True)

            async def one(m):
                nonlocal brier_model, brier_market, scored
                t_dec = m["close"] - timedelta(minutes=args.decide_before)
                async with sem:
                    book = await fetch_market_book(http, series, m["ticker"], m["close"],
                                                   args.decide_before)
                if not book:
                    skipped["book"] += 1
                    return
                hist = candles_before(cs, int(t_dec.timestamp()), 35)
                if len(hist) < 10:
                    skipped["spot"] += 1
                    return
                spot = hist[-1].c
                sigma = realized_vol(hist)
                if not sigma:
                    skipped["spot"] += 1
                    return
                mom5 = (hist[-1].c / hist[-6].c - 1) if len(hist) >= 6 else None
                p = fair_prob_up(spot, m["strike"], args.decide_before,
                                 sigma * cfg.sigma_mult)
                up_ask, down_ask = book["up_ask"], book["down_ask"]
                up_won = m["result"] == "yes"

                brier_model += (p - up_won) ** 2
                brier_market += (up_ask - up_won) ** 2
                scored += 1

                def settle(side, price):
                    won = up_won if side == "UP" else not up_won
                    fee = kalshi_fee(SIZE, price)
                    return ((1.0 if won else 0.0) - price) * SIZE - fee

                picks = {"always_up": ("UP", up_ask), "always_down": ("DOWN", down_ask)}
                if mom5 is not None:
                    mside = "UP" if mom5 > 0 else "DOWN"
                    if cfg.momentum_fade:
                        mside = "DOWN" if mside == "UP" else "UP"
                    picks["momentum"] = (mside, up_ask if mside == "UP" else down_ask)
                pick = decide_side(p, up_ask, down_ask, fee_per_contract(up_ask),
                                   fee_per_contract(down_ask), cfg)
                if pick:
                    picks["vol_only"] = (pick["side"], pick["price"])
                for name, (side, price) in picks.items():
                    pnl = settle(side, price)
                    st = stats[name]
                    st["n"] += 1
                    st["w"] += pnl > 0
                    st["pnl"] += pnl
                    st["staked"] += price * SIZE

            for i in range(0, len(markets), 20):
                await asyncio.gather(*[one(m) for m in markets[i:i + 20]])
                print(f"  {asset}: {min(i + 20, len(markets))}/{len(markets)}", flush=True)

    print(f"\nskipped: no book {skipped['book']}, no spot {skipped['spot']}")
    print(f"\nmodel vs market over {scored} scored markets:")
    if scored:
        print(f"  brier vol-model : {brier_model / scored:.4f}")
        print(f"  brier market ask: {brier_market / scored:.4f}")
    print(f"\n{'strategy':12s} {'trades':>6s} {'wins':>7s} {'pnl':>10s} {'roi':>8s}")
    for name, st in sorted(stats.items(), key=lambda kv: -kv[1]["pnl"]):
        if not st["n"]:
            continue
        roi = 100 * st["pnl"] / st["staked"] if st["staked"] else 0.0
        print(f"{name:12s} {st['n']:6d} {st['w']:3d}/{st['n']:<3d} {st['pnl']:+10.2f} {roi:+7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,DOGE")
    ap.add_argument("--decide-before", type=int, default=14,
                    help="minutes before close when the decision is made")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
