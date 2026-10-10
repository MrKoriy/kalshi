import math
from dataclasses import dataclass, field
from typing import Optional

from .kalshi import Snapshot
from .prices import SpotBundle, realized_vol, atr
from .derivatives import Derivs


@dataclass
class Features:
    asset: str
    ticker: str
    minutes_left: float
    spot: Optional[float]
    strike: Optional[float]
    dist_pct: Optional[float]          # (spot - strike) / strike
    sigma_min: Optional[float]         # per-minute EWMA vol
    atr1h: Optional[float]
    mom_1m: Optional[float]
    mom_5m: Optional[float]
    mom_15m: Optional[float]
    hour_ret: Optional[float]          # ~last 60 min from candles
    venue_divergence: Optional[float]  # binance vs coinbase
    funding: Optional[float]
    basis: Optional[float]
    oi: Optional[float]
    spread: Optional[float]
    imbalance: Optional[float]
    last_kalshi_price: Optional[float]
    up_mid: Optional[float]
    notes: list = field(default_factory=list)


def _mom(cs: list, n: int) -> Optional[float]:
    if len(cs) < n + 1:
        return None
    a, b = cs[-n - 1].c, cs[-1].c
    return (b - a) / a if a > 0 else None


def build(asset: str, snap: Snapshot, bundle: SpotBundle, derivs: Optional[Derivs]) -> Features:
    cs = bundle.best_candles
    spot = bundle.best_price
    strike = snap.strike
    dist = None
    if spot is not None and strike:
        dist = (spot - strike) / strike

    f = Features(
        asset=asset,
        ticker=snap.ticker,
        minutes_left=snap.minutes_left,
        spot=spot,
        strike=strike,
        dist_pct=dist,
        sigma_min=realized_vol(cs) if cs else None,
        atr1h=atr(cs) if cs else None,
        mom_1m=_mom(cs, 1),
        mom_5m=_mom(cs, 5),
        mom_15m=_mom(cs, 15),
        hour_ret=_mom(cs, min(60, len(cs) - 1)) if len(cs) > 5 else None,
        venue_divergence=bundle.divergence,
        funding=derivs.funding if derivs else None,
        basis=derivs.basis if derivs else None,
        oi=derivs.open_interest if derivs else None,
        spread=snap.spread,
        imbalance=snap.imbalance(),
        last_kalshi_price=snap.last_price,
        up_mid=snap.mid_up,
    )

    if f.sigma_min is None:
        f.notes.append("no_vol")
    if spot is None:
        f.notes.append("no_spot")
    if strike is None:
        f.notes.append("no_strike")
    if len(bundle.prices) >= 2 and f.venue_divergence is not None and abs(f.venue_divergence) > 0.004:
        f.notes.append("venue_divergence_high")
    if f.spread is not None and f.spread > 0.06:
        f.notes.append("wide_spread")
    return f


def to_prompt_text(f: Features) -> str:
    def pf(x, d=4):
        return "n/a" if x is None else f"{x:+.{d}f}" if isinstance(x, float) else str(x)

    def pp(x, d=3):
        return "n/a" if x is None else f"{x * 100:.{d}f}%"

    return "\n".join([
        f"asset={f.asset} ticker={f.ticker} minutes_left={f.minutes_left:.2f}",
        f"spot={f.spot} strike={f.strike} dist_pct={pp(f.dist_pct)}",
        f"sigma_min={pf(f.sigma_min, 6)} atr1h={pf(f.atr1h, 2)}",
        f"mom_1m={pp(f.mom_1m)} mom_5m={pp(f.mom_5m)} mom_15m={pp(f.mom_15m)} hour_ret={pp(f.hour_ret)}",
        f"venue_divergence={pp(f.venue_divergence)}",
        f"funding={pf(f.funding, 6)} basis={pp(f.basis)} open_interest={f.oi}",
        f"kalshi: spread={pf(f.spread)} imbalance={pf(f.imbalance)} "
        f"last_price={pf(f.last_kalshi_price)} up_mid={pf(f.up_mid)}",
    ])
