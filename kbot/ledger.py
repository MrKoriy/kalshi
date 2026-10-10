import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from .config import Config
from .kalshi import now_utc
from .recorder import Recorder


def kalshi_fee(contracts: float, price: float) -> float:
    return math.ceil(0.07 * contracts * price * (1 - price) * 100 - 1e-9) / 100.0


def fee_per_contract(price: float) -> float:
    return 0.07 * price * (1 - price)


@dataclass
class Trade:
    strategy: str
    asset: str
    ticker: str
    close_time: datetime
    side: str
    entry: float
    size: float
    fees: float
    p_win: Optional[float]
    model: str
    reason: str
    opened_at: str
    trade_id: int = -1


class StrategyLedger:
    def __init__(self, name: str):
        self.name = name
        self.pnl = 0.0
        self.wins = 0
        self.closed = 0
        self.open_cost = 0.0
        self.pending: list = []

    def halted(self, cfg: Config) -> bool:
        if cfg.max_daily_profit > 0 and self.pnl >= cfg.max_daily_profit:
            return True
        if cfg.max_daily_loss < 0 and self.pnl <= cfg.max_daily_loss:
            return True
        return False


class Ledger:
    """One primary ('ensemble') strategy plus any number of shadow strategies."""

    def __init__(self, cfg: Config, recorder: Recorder):
        self.cfg = cfg
        self.rec = recorder
        self.primary = StrategyLedger("ensemble")
        self.shadows = {name: StrategyLedger(name) for name in cfg.shadow_strategies}
        self.stop = False

    def all(self) -> list:
        return [self.primary] + list(self.shadows.values())

    def get(self, name: str) -> StrategyLedger:
        return self.primary if name == "ensemble" else self.shadows[name]

    def total_open_cost(self) -> float:
        return sum(s.open_cost for s in self.all())

    def open(self, strategy: str, asset: str, snap, side: str, entry: float,
             size: float, p_win: Optional[float], model: str, reason: str) -> Optional[Trade]:
        s = self.get(strategy)
        if s.halted(self.cfg):
            return None
        if self.rec.has_open(strategy, snap.ticker):
            return None
        if strategy == "ensemble" and self.total_open_cost() + entry * size > self.cfg.max_exposure:
            return None
        fees = kalshi_fee(size, entry) if self.cfg.simulate_fees else 0.0
        t = Trade(
            strategy=strategy, asset=asset, ticker=snap.ticker, close_time=snap.close_time,
            side=side, entry=entry, size=size, fees=fees, p_win=p_win,
            model=model, reason=reason, opened_at=now_utc().isoformat(),
        )
        t.trade_id = self.rec.trade(strategy, asset, snap.ticker, side, entry, size,
                                    fees, p_win, model, reason,
                                    {"obey": self.cfg.obey_model})
        s.pending.append(t)
        s.open_cost += entry * size
        return t

    async def settle(self, kc, log, calibrate_cb=None) -> None:
        for s in self.all():
            for t in list(s.pending):
                if now_utc() < t.close_time:
                    continue
                res = await kc.result(t.ticker)
                if res is None:
                    continue
                up_won = res == "yes"
                won = up_won if t.side == "UP" else not up_won
                payout = 1.0 if won else 0.0
                pnl = (payout - t.entry) * t.size - t.fees
                s.pnl += pnl
                s.closed += 1
                s.wins += 1 if pnl > 0 else 0
                s.open_cost -= t.entry * t.size
                s.pending.remove(t)
                self.rec.settle(t.trade_id, s.name, t.ticker, t.side, res, pnl,
                                {"p_win": t.p_win, "model": t.model})
                if t.p_win is not None:
                    outcome = 1 if won else 0
                    self.rec.calibrate(s.name, t.p_win, outcome)
                    if calibrate_cb:
                        calibrate_cb(s.name, t.p_win, outcome)
                log(f"[{s.name}] {t.asset} {t.ticker} settled {res.upper()} | {t.side} "
                    f"{'WIN' if pnl > 0 else 'LOSS'} | pnl {pnl:+.2f} | total {s.pnl:+.2f} "
                    f"| wins {s.wins}/{s.closed}")

    def summary(self) -> str:
        parts = []
        for s in self.all():
            parts.append(f"{s.name}: pnl {s.pnl:+.2f} wins {s.wins}/{s.closed} open {len(s.pending)}")
        return " | ".join(parts)
