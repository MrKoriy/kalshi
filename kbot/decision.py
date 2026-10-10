"""Shared SystemOne-style decision-model voter (Jev API / Clef API compatible)."""

from dataclasses import dataclass
from typing import Optional

import httpx

from .features import Features
from .events import EventContext

QUESTIONS = {
    "direction": {
        "type": "choice",
        "instructions": (
            "Will this 15-minute Kalshi market resolve UP (CF Benchmarks reference "
            "average at close >= average at open/strike)?"
        ),
        "criteria": {
            "up": "Resolves UP",
            "down": "Resolves DOWN",
            "unclear": "No reliable signal either way",
        },
    },
    "regime": {
        "type": "choice",
        "instructions": "What is the current micro regime?",
        "criteria": {
            "trending": "Directional move with follow-through",
            "choppy": "Mean-reverting, low conviction",
            "spiky": "High vol, jump risk",
        },
    },
    "event_risk": {
        "type": "noul",
        "instructions": "Is there elevated near-term event risk (macro release, incident)?",
    },
}


@dataclass
class DecisionVote:
    p_up: Optional[float]
    regime: str
    event_risk: Optional[float]
    raw: dict
    error: str = ""


def build_state(f: Features, ev_ctx: EventContext) -> dict:
    return {
        "market": {
            "asset": f.asset, "ticker": f.ticker,
            "minutes_left": round(f.minutes_left, 3),
            "spot": f.spot, "strike": f.strike,
            "dist_pct": f.dist_pct,
        },
        "momentum": {"m1": f.mom_1m, "m5": f.mom_5m, "m15": f.mom_15m, "hour": f.hour_ret},
        "volatility": {"sigma_min": f.sigma_min, "atr1h": f.atr1h},
        "kalshi": {"spread": f.spread, "imbalance": f.imbalance,
                   "last_price": f.last_kalshi_price, "up_mid": f.up_mid},
        "derivatives": {"funding": f.funding, "basis": f.basis, "open_interest": f.oi},
        "macro": {"fear_greed": ev_ctx.fear_greed, "risk_score": ev_ctx.risk_score,
                  "headlines": ev_ctx.headlines[:5]},
    }


def parse_answers(data: dict) -> DecisionVote:
    answers = data.get("answers") or {}
    p_up = None
    regime = ""
    event_risk = None

    d = answers.get("direction") or {}
    probs = d.get("probabilities") or {}
    if probs:
        pu = probs.get("up")
        pd = probs.get("down")
        if pu is not None and pd is not None and (float(pu) + float(pd)) > 0:
            p_up = float(pu) / (float(pu) + float(pd))
        elif d.get("choice") == "up":
            p_up = float(d.get("confidence") or 0.6)
        elif d.get("choice") == "down":
            p_up = 1.0 - float(d.get("confidence") or 0.6)

    rg = answers.get("regime") or {}
    regime = str(rg.get("choice") or "")

    er = answers.get("event_risk") or {}
    # Jev returns {"noul": p}, Clef returns {"probability": p}
    raw_p = er.get("noul", er.get("probability"))
    if raw_p is not None:
        try:
            event_risk = float(raw_p)
        except (TypeError, ValueError):
            event_risk = None

    return DecisionVote(p_up=p_up, regime=regime, event_risk=event_risk, raw=answers)


async def decide(base_url: str, model: str, api_key: str, timeout: float,
                 http: httpx.AsyncClient, f: Features, ev_ctx: EventContext,
                 questions: Optional[dict] = None) -> DecisionVote:
    if not base_url or not model:
        return DecisionVote(p_up=None, regime="", event_risk=None, raw={}, error="disabled")
    body = {
        "model": model,
        "state": build_state(f, ev_ctx),
        "questions": questions or QUESTIONS,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = await http.post(f"{base_url}/v1/systemone", json=body, headers=headers,
                            timeout=timeout)
        if r.status_code != 200:
            return DecisionVote(p_up=None, regime="", event_risk=None, raw={},
                                error=f"http_{r.status_code}")
        return parse_answers(r.json())
    except (httpx.HTTPError, ValueError):
        return DecisionVote(p_up=None, regime="", event_risk=None, raw={},
                            error="request_failed")
