import math
from statistics import NormalDist

NORM = NormalDist()


def fair_prob_up(spot: float, strike: float, minutes_left: float, sigma_min: float,
                 avg_seconds: float = 60.0) -> float:
    """P(avg price over last `avg_seconds` before close >= strike).

    Kalshi 15m crypto markets resolve on the CF Benchmarks RTI averaged over the
    final 60 seconds vs the average over the 60 seconds at open (strike). The
    averaging shrinks end-of-window variance; we approximate by using effective
    time t = max(minutes_left, avg/2 converted to minutes) and full drift.
    """
    if spot <= 0 or strike <= 0 or sigma_min <= 0:
        return 0.5
    t = max(minutes_left, (avg_seconds / 60.0) / 2.0)
    sigma_t = sigma_min * math.sqrt(t)
    z = math.log(spot / strike) / sigma_t
    return NORM.cdf(z)


def kelly_contracts(p: float, price: float, bankroll: float, fraction: float) -> float:
    """Contracts for a binary paying 1, bought at `price` with win prob p."""
    if not 0 < price < 1 or not 0 <= p <= 1 or bankroll <= 0:
        return 0.0
    b = (1 - price) / price
    q = 1 - p
    f = (b * p - q) / b
    if f <= 0:
        return 0.0
    stake = bankroll * f * fraction
    return stake / price


def edge_ok(p: float, price: float, fee_rate_per_contract: float, min_edge: float,
            max_edge: float) -> bool:
    ev = p - price - fee_rate_per_contract
    return min_edge <= ev <= max_edge


def ev_per_contract(p: float, price: float) -> float:
    return p - price
