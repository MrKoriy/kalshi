import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .config import Config
from .features import Features, to_prompt_text
from .events import EventContext

SYSTEM = (
    "You forecast 15-minute crypto up/down prediction markets on Kalshi. "
    "The market resolves UP if the CF Benchmarks reference-rate average over the final "
    "60 seconds at close is at or above the average from the 60 seconds at open (the strike). "
    "You are given live features: distance to strike, remaining time, momentum, realized "
    "volatility, derivatives data, Kalshi microstructure, and macro context. "
    "Return a calibrated probability that the market resolves UP, and only trade when the "
    "probability clearly differs from the market price. Be honest about uncertainty."
)

USER_TMPL = """{features}

{macro}

Fair-probability estimate from a lognormal vol model: p_fair={p_fair}
Market prices: UP ask={up_ask} DOWN ask={down_ask}

Recent bot outcomes for this asset (memory):
{memory}

{instructions}

Reply with exactly one JSON object on one line:
{{"p_up": <float 0-1>, "confidence": <float 0-1>, "reason": "<one short sentence>"}}"""


@dataclass
class ProbVote:
    model: str
    p_up: Optional[float]
    confidence: float
    reason: str = ""
    error: str = ""


def _parse(text: str) -> Optional[dict]:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    p = obj.get("p_up")
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= p <= 1.0:
        return None
    conf = obj.get("confidence", 0.5)
    try:
        conf = min(max(float(conf), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.5
    return {"p_up": p, "confidence": conf, "reason": str(obj.get("reason", ""))[:240]}


class LLMEnsemble:
    def __init__(self, cfg: Config, http: httpx.AsyncClient):
        self.cfg = cfg
        self.http = http

    async def _one(self, model: str, user: str) -> ProbVote:
        temp = self.cfg.llm_temperature
        # kimi-k3 via this gateway rejects 0 < temperature < 0.6
        if model.startswith("kimi-k3") and 0.0 < temp < 0.6:
            temp = 0.6
        body = {
            "model": model,
            "temperature": temp,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.cfg.llm_api_key}"}
        delay = 1.0
        for attempt in range(3):
            try:
                r = await self.http.post(
                    f"{self.cfg.llm_base_url}/chat/completions",
                    json=body, headers=headers, timeout=self.cfg.llm_timeout,
                )
                if r.status_code == 503 and attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if r.status_code != 200:
                    continue
                text = r.json()["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, ValueError, KeyError, IndexError):
                if attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                break
            parsed = _parse(text)
            if parsed:
                return ProbVote(model=model, p_up=parsed["p_up"],
                                confidence=parsed["confidence"], reason=parsed["reason"])
            if attempt == 2 and not parsed:
                last_text = text
                return ProbVote(model=model, p_up=None, confidence=0.0,
                                error=f"no_valid_response: {last_text[:140]}")
        return ProbVote(model=model, p_up=None, confidence=0.0, error="no_valid_response")

    async def decide(self, f: Features, ev_ctx: EventContext, p_fair: Optional[float],
                     memory_rows: list) -> list:
        if not self.cfg.llm_api_key or not self.cfg.llm_models:
            return []
        mem = "\n".join(
            f"- {side} @ {entry:.2f} -> {'win' if pnl > 0 else 'loss'} ({result})"
            for side, entry, pnl, result in memory_rows
        ) or "none yet"
        user = USER_TMPL.format(
            features=to_prompt_text(f),
            macro=ev_ctx.to_prompt_text(),
            p_fair="n/a" if p_fair is None else f"{p_fair:.4f}",
            up_ask=f.up_mid if f.up_mid is not None else "n/a",
            down_ask="n/a" if f.up_mid is None else f"{1 - f.up_mid:.4f}",
            memory=mem,
            instructions=self.cfg.instructions or "(no extra instructions)",
        )
        tasks = [self._one(m, user) for m in self.cfg.llm_models]
        return await asyncio.gather(*tasks)
