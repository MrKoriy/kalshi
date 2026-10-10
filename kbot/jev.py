"""TypeSafe Jev voter — hosted SystemOne API."""

from . import decision
from .config import Config
from .decision import DecisionVote as JevVote
from .features import Features
from .events import EventContext

DEFAULT_BASE = "https://api.typesafe.ai"


def enabled(cfg: Config) -> bool:
    return bool(cfg.jev_api_key and cfg.jev_model)


async def decide(cfg: Config, http, f: Features, ev_ctx: EventContext) -> JevVote:
    base = cfg.jev_base_url or DEFAULT_BASE
    return await decision.decide(base, cfg.jev_model, cfg.jev_api_key,
                                 cfg.jev_timeout, http, f, ev_ctx)
