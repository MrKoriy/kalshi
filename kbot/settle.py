import asyncio
import logging

from .kalshi import Kalshi
from .ledger import Ledger
from .recorder import Recorder
from . import calibration

log = logging.getLogger("settle")


class Calibrator:
    """Collects (p, outcome) rows and refits Platt scaling periodically."""

    def __init__(self, rec: Recorder):
        self.rec = rec
        self.coeffs = {}  # source -> (a, b)

    def add(self, source: str, p: float, outcome: int) -> None:
        pass  # rows are already stored by Ledger; refit reads from DB

    def refit(self) -> None:
        if not self.rec.db:
            return
        rows = self.rec.db.execute("SELECT source, p, outcome FROM calib").fetchall()
        by_source = {}
        for source, p, o in rows:
            by_source.setdefault(source, []).append((p, o))
        for source, rs in by_source.items():
            c = calibration.platt_coeffs(rs)
            if c:
                self.coeffs[source] = c

    def apply(self, source: str, p: float) -> float:
        c = self.coeffs.get(source)
        return calibration.platt_apply(p, *c) if c else p


async def settle_loop(cfg, kc: Kalshi, ledger: Ledger, calibrator: Calibrator):
    while not ledger.stop:
        try:
            await ledger.settle(kc, log.info, calibrate_cb=calibrator.add)
            calibrator.refit()
            prim = ledger.primary
            if prim.halted(cfg) and not prim.pending and all(
                not s.pending for s in ledger.shadows.values()
            ):
                log.info("Daily limit reached and nothing pending, stopping. %s",
                         ledger.summary())
                ledger.stop = True
        except Exception:
            log.exception("settle loop error")
        await asyncio.sleep(3)
