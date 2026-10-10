import sqlite3
from collections import defaultdict

from . import calibration


def main(db_path: str = "data/bot.db"):
    db = sqlite3.connect(db_path)
    print("=== strategy PnL ===")
    rows = db.execute(
        """SELECT strategy, COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl)
           FROM settlements GROUP BY strategy"""
    ).fetchall()
    for strat, n, wins, pnl in rows:
        print(f"{strat:15s} n={n:4d} wins={wins:4d} ({(wins or 0)/n*100:.1f}%) pnl={pnl:+.2f}")

    print("\n=== calibration ===")
    for source in ("ensemble", "llm_solo", "vol_only"):
        print(calibration.bucket_report(db, source))
        print()

    print("=== recent trades ===")
    for r in db.execute(
        """SELECT t.ts, t.strategy, t.asset, t.side, t.entry, t.size, s.pnl, s.result
           FROM settlements s JOIN trades t ON t.id = s.trade_id
           ORDER BY s.id DESC LIMIT 15"""
    ):
        print(r)


if __name__ == "__main__":
    main()
