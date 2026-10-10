import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_ASSETS = {"BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE", "BNB"}
SHADOW_CHOICES = {"llm_solo", "clef_solo", "jev_solo", "vol_only", "always_up", "always_down", "momentum"}


@dataclass(frozen=True)
class Config:
    assets: list
    obey_model: bool
    time_delay: float
    min_edge: float
    max_edge: float
    kelly_fraction: float
    order_size: float
    max_exposure: float
    max_daily_profit: float
    max_daily_loss: float
    w_vol: float
    w_llm: float
    w_clef: float
    w_jev: float
    simulate_fees: bool
    book_depth: int
    poll_seconds: float
    event_edge_bonus: float
    sigma_mult: float
    momentum_fade: bool
    trades_file: str
    db_path: str
    recorder: bool
    snapshot_seconds: float
    memory_days: int
    shadow_strategies: list
    llm_base_url: str
    llm_api_key: str
    llm_models: list
    llm_temperature: float
    llm_timeout: float
    instructions: str
    clef_base_url: str
    clef_model: str
    clef_timeout: float
    jev_api_key: str
    jev_model: str
    jev_base_url: str
    jev_timeout: float
    bankroll: float = 1000.0


def _yes(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().upper() in ("YES", "TRUE", "1")


def _csv(name: str, default: str) -> list:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def load(require_llm: bool = True) -> Config:
    errors = []

    def num(name, default):
        try:
            return float(os.getenv(name, default))
        except ValueError:
            errors.append(f"{name} must be a number")
            return 0.0

    assets = [a.upper() for a in _csv("ASSETS", "BTC")]
    bad = [a for a in assets if a not in SUPPORTED_ASSETS]
    if not assets or bad:
        errors.append(f"ASSETS invalid {bad or ''}; supported: {sorted(SUPPORTED_ASSETS)}")

    shadows = [s.lower() for s in _csv("SHADOW_STRATEGIES", "llm_solo,vol_only,always_up,momentum")]
    bad = [s for s in shadows if s not in SHADOW_CHOICES]
    if bad:
        errors.append(f"SHADOW_STRATEGIES invalid {bad}; supported: {sorted(SHADOW_CHOICES)}")

    models = _csv("LLM_MODELS", os.getenv("LLM_MODEL", ""))
    key = os.getenv("LLM_API_KEY", "").strip()
    jev_key = os.getenv("JEV_API_KEY", "").strip()
    clef_ready = bool(os.getenv("CLEF_BASE_URL", "").strip() and os.getenv("CLEF_MODEL", "").strip())
    if require_llm and not key and not jev_key and not clef_ready:
        errors.append("Need at least one voter: LLM_API_KEY, JEV_API_KEY or CLEF_BASE_URL+CLEF_MODEL")

    order_size = num("ORDER_SIZE", "300")
    time_delay = num("TIME_DELAY", "15")
    min_edge = num("MIN_EDGE", "0.06")
    max_edge = num("MAX_EDGE", "0.35")
    kelly = num("KELLY_FRACTION", "0.25")
    max_profit = num("MAX_DAILY_PROFIT", "400")
    max_loss = num("MAX_DAILY_LOSS", "-200")
    poll = num("POLL_SECONDS", "2")
    max_exposure = num("MAX_EXPOSURE", "1000")

    if order_size <= 0:
        errors.append("ORDER_SIZE must be > 0")
    if not 0 < time_delay <= 15:
        errors.append("TIME_DELAY must be in (0, 15]")
    if not 0 <= min_edge < max_edge < 1:
        errors.append("need 0 <= MIN_EDGE < MAX_EDGE < 1")
    if kelly < 0:
        errors.append("KELLY_FRACTION must be >= 0")
    if max_profit < 0 or max_loss > 0:
        errors.append("MAX_DAILY_PROFIT >= 0 and MAX_DAILY_LOSS <= 0")

    w_vol, w_llm, w_clef = num("W_VOL", "0.45"), num("W_LLM", "0.4"), num("W_CLEF", "0.15")
    w_jev = num("W_JEV", "0.15")
    if w_vol + w_llm + w_clef + w_jev <= 0:
        errors.append("ensemble weights sum to zero")

    if errors:
        raise SystemExit("Config errors:\n - " + "\n - ".join(errors))

    return Config(
        assets=assets,
        obey_model=_yes("OBEY_MODEL", "YES"),
        time_delay=time_delay,
        min_edge=min_edge,
        max_edge=max_edge,
        kelly_fraction=kelly,
        order_size=order_size,
        max_exposure=max_exposure,
        max_daily_profit=max_profit,
        max_daily_loss=max_loss,
        w_vol=w_vol,
        w_llm=w_llm,
        w_clef=w_clef,
        w_jev=w_jev,
        simulate_fees=_yes("SIMULATE_FEES", "YES"),
        book_depth=max(1, int(num("BOOK_DEPTH", "10"))),
        poll_seconds=max(poll, 0.5),
        event_edge_bonus=num("EVENT_EDGE_BONUS", "0.03"),
        # realized 15m moves run ~1.23-1.30x our EWMA sigma with fat tails (measured
        # on 7d Binance 1m BTC/ETH): scale sigma so p_fair isn't overconfident
        sigma_mult=num("SIGMA_MULT", "1.25"),
        # mom_5m predicts next-15m direction only ~48% of the time (n~10k): fade it
        momentum_fade=_yes("MOMENTUM_FADE", "YES"),
        trades_file=os.getenv("TRADES_FILE", "trades.jsonl"),
        db_path=os.getenv("DB_PATH", "data/bot.db"),
        recorder=_yes("RECORDER", "YES"),
        snapshot_seconds=num("SNAPSHOT_SECONDS", "8"),
        memory_days=max(1, int(num("MEMORY_DAYS", "7"))),
        shadow_strategies=shadows,
        llm_base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        llm_api_key=key,
        llm_models=models,
        llm_temperature=num("LLM_TEMPERATURE", "0.2"),
        llm_timeout=num("LLM_TIMEOUT", "45"),
        instructions=os.getenv("AI_INSTRUCTIONS", "").strip(),
        clef_base_url=os.getenv("CLEF_BASE_URL", "").rstrip("/"),
        clef_model=os.getenv("CLEF_MODEL", "").strip(),
        clef_timeout=num("CLEF_TIMEOUT", "10"),
        jev_api_key=os.getenv("JEV_API_KEY", "").strip(),
        jev_model=os.getenv("JEV_MODEL", "jev-latest").strip(),
        jev_base_url=os.getenv("JEV_BASE_URL", "https://api.typesafe.ai").rstrip("/"),
        jev_timeout=num("JEV_TIMEOUT", "10"),
    )
