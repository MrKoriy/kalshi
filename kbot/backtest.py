import json
import sqlite3
from collections import defaultdict

from .quant import fair_prob_up
from .ledger import fee_per_contract


def load(db_path: str, ticker: str) -> list:
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT ts, data FROM snapshots WHERE ticker=? ORDER BY ts", (ticker,)
    ).fetchall()
    out = []
    for ts, data in rows:
        d = json.loads(data)
        d["ts"] = ts
        out.append(d)
    return out


def replay(db_path: str, min_edge: float = 0.06, size: float = 300.0) -> None:
    """Replay the vol-model strategy over recorded snapshots, no lookahead:
    decision uses only data available at that snapshot."""
    db = sqlite3.connect(db_path)
    tickers = [r[0] for r in db.execute("SELECT DISTINCT ticker FROM snapshots")]
    results = defaultdict(list)
    for ticker in tickers:
        snaps = load(db_path, ticker)
        if len(snaps) < 3:
            continue
        decided = False
        for i, s in enumerate(snaps):
            if decided or s["minutes_left"] > 14.0 or s["minutes_left"] < 1.0:
                continue
            if s.get("spot") and s.get("strike") and s.get("sigma_min"):
                p = fair_prob_up(s["spot"], s["strike"], s["minutes_left"], s["sigma_min"])
                up_ask = s.get("up_ask")
                down_ask = s.get("down_ask")
                if up_ask is None or down_ask is None:
                    continue
                ev_up = p - up_ask - fee_per_contract(up_ask)
                ev_dn = (1 - p) - down_ask - fee_per_contract(down_ask)
                side = None
                if ev_up >= min_edge:
                    side, price, pw = "UP", up_ask, p
                elif ev_dn >= min_edge:
                    side, price, pw = "DOWN", down_ask, 1 - p
                if side:
                    decided = True
                    # outcome is resolved later from Kalshi; here we just log the pick
                    results[ticker].append({"side": side, "price": price, "p": pw,
                                            "minutes_left": s["minutes_left"]})
    picks_flat = [(tk, p) for tk, ps in results.items() for p in ps]
    print(f"hypothetical trades: {len(picks_flat)} across {len(results)} markets")
    if not picks_flat:
        print("no entries replayed (snapshots need sigma_min — recorded after the fix)")
        return

    # resolve outcomes from Kalshi
    import httpx
    from .kalshi import BASE
    from .ledger import kalshi_fee
    pnl_sum, wins, n = 0.0, 0, 0
    with httpx.Client(timeout=30) as http:
        for ticker, pick in picks_flat:
            try:
                r = http.get(f"{BASE}/markets/{ticker}")
                res = ((r.json() or {}).get("market") or {}).get("result")
            except Exception:
                res = None
            if res not in ("yes", "no"):
                print(f"  {ticker}: unresolved, skipped")
                continue
            up = res == "yes"
            won = up if pick["side"] == "UP" else not up
            fee = kalshi_fee(size, pick["price"])
            pnl = ((1.0 if won else 0.0) - pick["price"]) * size - fee
            pnl_sum += pnl
            wins += 1 if pnl > 0 else 0
            n += 1
            print(f"  {ticker} {pick['side']:4s} @ {pick['price']:.3f} "
                  f"p={pick['p']:.3f} -> {res.upper():3s} {'WIN ' if won else 'LOSS'} {pnl:+.2f}")
    if n:
        staked = sum(p["price"] * size for _, p in picks_flat[:n])
        print(f"\nresolved {n}: wins {wins}/{n}  pnl {pnl_sum:+.2f}  "
              f"roi {100 * pnl_sum / staked:+.1f}% on staked {staked:.0f}")


if __name__ == "__main__":
    import sys
    replay(sys.argv[1] if len(sys.argv) > 1 else "data/bot.db")
