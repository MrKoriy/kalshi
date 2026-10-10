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
    for ticker, picks in results.items():
        print(ticker, picks)
    print(f"\ntotal hypothetical trades: {sum(len(v) for v in results.values())}")
    print("Note: outcomes are not resolved here; this only replays entry logic.")


if __name__ == "__main__":
    import sys
    replay(sys.argv[1] if len(sys.argv) > 1 else "data/bot.db")
