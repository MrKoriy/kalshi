# kalshi-ai-bot

Python 3.11+ paper-trading bot for Kalshi 15-minute crypto up/down markets.

For each market it collects a feature snapshot (multi-venue spot, volatility, derivatives,
Kalshi order book, macro context), asks an LLM ensemble and optionally Cloudflare Clef for a
calibrated UP probability, combines that with a fair-probability vol model, and only simulates
an entry when the ensemble edge over the ask clears fees and `MIN_EDGE`. Fills are estimated
by walking the order book, positions hold to resolution, and every trade is recorded in
SQLite for stats, calibration and backtesting. No live orders are sent.

## Resolution rule

Kalshi 15m crypto markets resolve on the CF Benchmarks RTI average over the final 60 seconds
at close vs the average over the 60 seconds at open (the strike is that open average, exposed
as `floor_strike`). The vol model and prompts account for this averaging.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill LLM_BASE_URL, LLM_API_KEY, LLM_MODELS
python -m kbot --check # data-only smoke test, no LLM call
python -m kbot         # run the bot (Ctrl+C to stop, prints summary)
```

Stats and calibration: `python -m kbot.stats` — backtest entry logic: `python -m kbot.backtest`

## Clef

[Clef](https://huggingface.co/Cloudflare/clef) (or `clef-flash`) is a decision model: state +
typed questions in, per-option probabilities out. It is used as an ensemble voter plus a
regime/event-risk check. It needs a GPU server; run it via SGLang and point `CLEF_BASE_URL`
at it, e.g.:

```bash
docker run --gpus all -p 30000:30000 lmsysorg/sglang:dev-clef \
  sglang serve --model-path Cloudflare/clef-flash --host 0.0.0.0 --port 30000
```

`CLEF_BASE_URL=http://your-host:30000` in `.env`. Leave empty to disable.

## Jev

[Jev](https://typesafe.ai/) is TypeSafe's hosted System One decision model — same typed
questions API as Clef (Clef is API-compatible). It is hosted, so no GPU is needed: get a key
at <https://console.typesafe.ai/keys> and set `JEV_API_KEY`. It joins the ensemble as the
`jev` voter (`W_JEV` weight) and can run standalone as the `jev_solo` shadow strategy.
Leave `JEV_API_KEY` empty to disable.

## Settings (.env)

See `.env.example`. Key knobs:

- `MIN_EDGE` / `MAX_EDGE` — ensemble EV-after-fees bounds to enter
- `KELLY_FRACTION` — fractional Kelly sizing (0 = fixed `ORDER_SIZE`)
- `W_VOL` / `W_LLM` / `W_CLEF` / `W_JEV` — ensemble weights
- `SHADOW_STRATEGIES` — comma list of baselines trading the same markets paper-only
  (`llm_solo`, `clef_solo`, `jev_solo`, `vol_only`, `always_up`, `always_down`, `momentum`)
- `RECORDER` — write periodic snapshots to `DB_PATH` for offline backtesting
- `EVENT_EDGE_BONUS` — extra edge required when macro risk flags fire

## Layout

- `kbot/kalshi.py` — public Kalshi API (markets, order book, trades, result)
- `kbot/prices.py` — Coinbase / Kraken / Binance / Bybit spot + 1m candles
- `kbot/derivatives.py` — funding, basis, open interest
- `kbot/events.py` — Fear & Greed, news headlines, risk flags
- `kbot/features.py` — feature builder + prompt text
- `kbot/quant.py` — fair probability, Kelly, EV
- `kbot/llm.py` — LLM ensemble (probability output)
- `kbot/decision.py` — shared SystemOne client (state build, answer parsing)
- `kbot/clef.py` — Clef voter over `/v1/systemone`
- `kbot/jev.py` — Jev voter (hosted TypeSafe API)
- `kbot/ensemble.py` — vote combining and side selection
- `kbot/ledger.py` — per-strategy paper accounting
- `kbot/settle.py` — settlement polling + Platt calibration
- `kbot/recorder.py` — SQLite store (snapshots, trades, settlements, calibration rows)
- `kbot/bot.py` — main loops, shadow mode, `--check`
- `kbot/stats.py`, `kbot/backtest.py` — reporting and offline replay
