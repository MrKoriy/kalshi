import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    asset TEXT NOT NULL,
    ticker TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snap_ticker ON snapshots(ticker, ts);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    strategy TEXT NOT NULL,
    asset TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    entry REAL NOT NULL,
    size REAL NOT NULL,
    fees REAL NOT NULL,
    p_win REAL,
    model TEXT,
    reason TEXT,
    data TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE TABLE IF NOT EXISTS settlements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    trade_id INTEGER,
    strategy TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    result TEXT NOT NULL,
    pnl REAL NOT NULL,
    data TEXT
);
CREATE TABLE IF NOT EXISTS calib (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    p REAL NOT NULL,
    outcome INTEGER NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Recorder:
    def __init__(self, path: str, enabled: bool = True):
        self.enabled = enabled
        self.db: Optional[sqlite3.Connection] = None
        if not enabled:
            return
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def snapshot(self, asset: str, ticker: str, data: dict) -> None:
        if not self.db:
            return
        self.db.execute(
            "INSERT INTO snapshots(ts, asset, ticker, data) VALUES (?,?,?,?)",
            (utcnow(), asset, ticker, json.dumps(data, default=str)),
        )
        self.db.commit()

    def trade(self, strategy: str, asset: str, ticker: str, side: str, entry: float,
              size: float, fees: float, p_win: Optional[float], model: str,
              reason: str, extra: Optional[dict] = None) -> int:
        if not self.db:
            return -1
        cur = self.db.execute(
            "INSERT INTO trades(ts, strategy, asset, ticker, side, entry, size, fees, p_win, model, reason, data)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (utcnow(), strategy, asset, ticker, side, entry, size, fees, p_win, model, reason,
             json.dumps(extra or {}, default=str)),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def settle(self, trade_id: int, strategy: str, ticker: str, side: str, result: str,
               pnl: float, extra: Optional[dict] = None) -> None:
        if not self.db:
            return
        self.db.execute(
            "INSERT INTO settlements(ts, trade_id, strategy, ticker, side, result, pnl, data)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (utcnow(), trade_id, strategy, ticker, side, result, pnl,
             json.dumps(extra or {}, default=str)),
        )
        self.db.commit()

    def calibrate(self, source: str, p: float, outcome: int) -> None:
        if not self.db:
            return
        self.db.execute("INSERT INTO calib(ts, source, p, outcome) VALUES (?,?,?,?)",
                        (utcnow(), source, p, outcome))
        self.db.commit()

    def unsettled(self) -> list:
        """Trades with no settlement row (orphaned by a process restart)."""
        if not self.db:
            return []
        cur = self.db.execute(
            """SELECT t.id, t.strategy, t.asset, t.ticker, t.side, t.entry, t.size,
                      t.fees, t.p_win, t.model, t.reason, t.ts
               FROM trades t LEFT JOIN settlements s ON s.trade_id = t.id
               WHERE s.id IS NULL"""
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def has_open(self, strategy: str, ticker: str) -> bool:
        """True if this strategy already has an unsettled trade on this ticker."""
        if not self.db:
            return False
        row = self.db.execute(
            """SELECT 1 FROM trades t LEFT JOIN settlements s ON s.trade_id = t.id
               WHERE s.id IS NULL AND t.strategy=? AND t.ticker=? LIMIT 1""",
            (strategy, ticker),
        ).fetchone()
        return row is not None

    def recent_memory(self, asset: str, days: int, limit: int = 10) -> list:
        """Last settled trades for prompt memory: (side, entry, pnl, result)."""
        if not self.db:
            return []
        rows = self.db.execute(
            """SELECT t.side, t.entry, s.pnl, s.result
               FROM settlements s JOIN trades t ON t.id = s.trade_id
               WHERE s.strategy='ensemble' AND t.asset=?
               ORDER BY s.id DESC LIMIT ?""",
            (asset, limit),
        ).fetchall()
        return rows

    def pnl_history(self, strategy: str = "ensemble") -> list:
        if not self.db:
            return []
        rows = self.db.execute(
            "SELECT ts, pnl FROM settlements WHERE strategy=? ORDER BY id", (strategy,)
        ).fetchall()
        return rows
