"""Cloudflare Clef voter — thin wrapper over the shared SystemOne client."""

from . import decision
from .config import Config
from .decision import DecisionVote as ClefVote, QUESTIONS
from .features import Features
from .events import EventContext


def enabled(cfg: Config) -> bool:
    return bool(cfg.clef_base_url and cfg.clef_model)


async def decide(cfg: Config, http, f: Features, ev_ctx: EventContext) -> ClefVote:
    return await decision.decide(cfg.clef_base_url, cfg.clef_model, "",
                                 cfg.clef_timeout, http, f, ev_ctx)
