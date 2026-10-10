import math
from dataclasses import dataclass, field
from typing import Optional

from .config import Config


@dataclass
class Vote:
    source: str           # "vol", "llm:<model>", "clef"
    p_up: Optional[float]
    weight: float
    meta: dict = field(default_factory=dict)


@dataclass
class EnsembleResult:
    p_up: Optional[float]
    votes: list = field(default_factory=list)
    used_weight: float = 0.0

    @property
    def p_down(self) -> Optional[float]:
        return None if self.p_up is None else 1.0 - self.p_up


def combine(cfg: Config, votes: list) -> EnsembleResult:
    """Weighted average of available probability votes."""
    base = {"vol": cfg.w_vol, "llm": cfg.w_llm, "clef": cfg.w_clef, "jev": cfg.w_jev}
    weighted = 0.0
    wsum = 0.0
    used = []
    for v in votes:
        if v.p_up is None or not (0.0 <= v.p_up <= 1.0):
            continue
        group = v.source.split(":")[0]
        w = base.get(group, 0.0) * v.weight
        if w <= 0:
            continue
        # shrink extreme votes slightly toward 0.5 (regularization)
        p = 0.5 + (v.p_up - 0.5) * 0.98
        weighted += w * p
        wsum += w
        used.append(v)
    if wsum <= 0:
        return EnsembleResult(p_up=None, votes=used)
    return EnsembleResult(p_up=weighted / wsum, votes=used, used_weight=wsum)


def decide_side(p_up: float, up_price: float, down_price: float,
                fee_up: float, fee_down: float, cfg: Config,
                extra_edge: float = 0.0) -> Optional[dict]:
    """Pick side by expected value after fees, subject to edge bounds."""
    min_edge = cfg.min_edge + extra_edge
    cands = []
    if 0 < up_price < 1:
        cands.append(("UP", p_up, up_price, fee_up))
    if 0 < down_price < 1:
        cands.append(("DOWN", 1 - p_up, down_price, fee_down))
    best = None
    for side, p, price, fee in cands:
        ev = p - price - fee
        if ev < min_edge or ev > cfg.max_edge:
            continue
        if best is None or ev > best["ev"]:
            best = {"side": side, "p": p, "price": price, "fee": fee, "ev": ev}
    return best
