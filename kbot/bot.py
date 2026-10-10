import asyncio
import logging

import httpx

from . import config as cfgmod
from . import prices, derivatives, events, features, quant, ensemble, clef, jev
from .kalshi import Kalshi, now_utc
from .ledger import Ledger, StrategyLedger, Trade, fee_per_contract
from .llm import LLMEnsemble
from .recorder import Recorder
from .settle import Calibrator

log = logging.getLogger("bot")

SHADOW_NAMES = {"llm_solo", "clef_solo", "vol_only", "always_up", "always_down", "momentum"}


class AssetState:
    def __init__(self):
        self.ticker = None
        self.phase = "WAIT"
        self.votes = []
        self.p_fair = None
        self.ev_ctx = None
        self.shadows_done = set()
        self.jev_vote = None


async def snapshot_loop(cfg, kc, rec, spot_cache, derivs_cache, stop):
    """Records Kalshi + spot snapshots every cfg.snapshot_seconds for offline backtest."""
    while not stop["flag"]:
        for asset in cfg.assets:
            try:
                snap = await kc.snapshot(asset)
                if snap is None:
                    continue
                bundle = spot_cache.get(asset)
                d = derivs_cache.get(asset)
                sigma = prices.realized_vol(bundle.best_candles) if bundle else None
                rec.snapshot(asset, snap.ticker, {
                    "minutes_left": snap.minutes_left,
                    "strike": snap.strike,
                    "sigma_min": sigma,
                    "up_ask": snap.up_ask, "down_ask": snap.down_ask,
                    "up_bid": snap.up_bid, "down_bid": snap.down_bid,
                    "spread": snap.spread, "mid_up": snap.mid_up,
                    "volume": snap.volume, "open_interest": snap.open_interest,
                    "spot": bundle.best_price if bundle else None,
                    "funding": d.funding if d else None,
                    "up_levels": [[l.price, l.size] for l in snap.up_levels],
                    "down_levels": [[l.price, l.size] for l in snap.down_levels],
                })
            except Exception:
                log.exception("recorder snapshot error")
        await asyncio.sleep(cfg.snapshot_seconds)


async def asset_loop(asset, cfg, kc, http, llm_ens, ledger, rec, calibrator,
                     spot_cache, derivs_cache, ev_cache, stop):
    st = AssetState()
    while not stop["flag"]:
        try:
            snap = await kc.snapshot(asset)
            if snap is None:
                await asyncio.sleep(cfg.poll_seconds)
                continue

            if snap.ticker != st.ticker:
                st = AssetState()
                st.ticker = snap.ticker
                log.info("%s new session %s | closes %s", asset, snap.ticker,
                         snap.close_time.strftime("%H:%M:%S"))

            mins = snap.minutes_left
            if st.phase == "WAIT":
                if ledger.primary.halted(cfg):
                    st.phase = "HALTED"
                    log.info("%s daily PnL limit reached, no new trades", asset)
                elif mins <= cfg.time_delay:
                    if mins < 0.5:
                        st.phase = "DONE"
                        log.info("%s too close to expiry, skipping", asset)
                    else:
                        st.phase = "DECIDE"
                        log.info("%s gathering signals with %.2f min left", asset, mins)

            if st.phase == "DECIDE":
                bundle = await prices.spot_bundle(http, asset)
                spot_cache[asset] = bundle
                derivs = await derivatives.fetch(http, asset)
                derivs_cache[asset] = derivs
                ev_ctx = ev_cache.get("ctx")
                if ev_ctx is None:
                    ev_ctx = await events.fetch(http)
                    ev_cache["ctx"] = ev_ctx
                st.ev_ctx = ev_ctx

                f = features.build(asset, snap, bundle, derivs)
                st.p_fair = None
                if f.spot and f.strike and f.sigma_min:
                    st.p_fair = quant.fair_prob_up(f.spot, f.strike, mins, f.sigma_min)

                # votes
                st.votes = []
                if st.p_fair is not None:
                    st.votes.append(ensemble.Vote("vol", st.p_fair, 1.0))
                memory = rec.recent_memory(asset, cfg.memory_days)
                llm_votes = await llm_ens.decide(f, ev_ctx, st.p_fair, memory)
                for v in llm_votes:
                    w = max(0.2, v.confidence)
                    st.votes.append(ensemble.Vote(f"llm:{v.model}", v.p_up, w,
                                                  {"reason": v.reason, "error": v.error}))
                cv = await clef.decide(cfg, http, f, ev_ctx)
                if cv.p_up is not None:
                    st.votes.append(ensemble.Vote("clef", cv.p_up, 1.0,
                                                  {"regime": cv.regime,
                                                   "event_risk": cv.event_risk}))
                jv = await jev.decide(cfg, http, f, ev_ctx)
                if jv.p_up is not None:
                    st.votes.append(ensemble.Vote("jev", jv.p_up, 1.0,
                                                  {"regime": jv.regime,
                                                   "event_risk": jv.event_risk}))
                st.jev_vote = jv

                res = ensemble.combine(cfg, st.votes)
                if res.p_up is not None:
                    res.p_up = calibrator.apply("ensemble", res.p_up)
                if res.p_up is None:
                    st.phase = "DONE"
                    log.warning("%s no usable votes, skipping session", asset)
                    continue

                extra = cfg.event_edge_bonus if ev_ctx.risk_score >= 3 else 0.0
                fee_up = fee_per_contract(snap.up_ask or 0.5)
                fee_dn = fee_per_contract(snap.down_ask or 0.5)
                pick = ensemble.decide_side(res.p_up, snap.up_ask or 9.9,
                                            snap.down_ask or 9.9,
                                            fee_up, fee_dn, cfg, extra)
                if pick is None:
                    st.phase = "DONE"
                    log.info("%s no edge: p=%.3f up_ask=%s down_ask=%s risk=%d, SKIP",
                             asset, res.p_up, snap.up_ask, snap.down_ask, ev_ctx.risk_score)
                else:
                    side = pick["side"]
                    if not cfg.obey_model:
                        side = "DOWN" if side == "UP" else "UP"
                    size = cfg.order_size
                    if cfg.kelly_fraction > 0:
                        size = quant.kelly_contracts(pick["p"], pick["price"],
                                                     cfg.bankroll, cfg.kelly_fraction)
                        size = max(1.0, min(size, cfg.order_size * 3))
                    fill = snap.fill_estimate(side, size)
                    if fill is None or fill["partial"]:
                        log.info("%s cannot fill %g %s contracts, SKIP", asset, size, side)
                        st.phase = "DONE"
                    else:
                        t = ledger.open("ensemble", asset, snap, side, fill["avg"], size,
                                        pick["p"], "ensemble",
                                        "; ".join(v.meta.get("reason", "") for v in st.votes if v.meta.get("reason"))[:240])
                        st.phase = "HOLD"
                        if t:
                            log.info("%s PAPER %s x%.1f @ %.3f (edge %.3f, p %.3f, votes %d)",
                                     asset, side, size, fill["avg"], pick["ev"], pick["p"],
                                     len(res.votes))
                        else:
                            st.phase = "DONE"
                # shadow strategies see the same market state
                await run_shadows(asset, cfg, snap, f, st, llm_votes, cv, ledger, res,
                                  st.jev_vote)

            if st.phase == "HOLD":
                # keep position; shadows already placed
                pass

        except Exception:
            log.exception("%s loop error", asset)
        await asyncio.sleep(cfg.poll_seconds)


async def run_shadows(asset, cfg, snap, f, st, llm_votes, cv, ledger, ens_res, jv=None):
    async def place(name, side, entry, p_win, reason):
        if entry is None or not 0 < entry < 1:
            return
        fill = snap.fill_estimate(side, cfg.order_size)
        if fill is None or fill["partial"]:
            return
        t = ledger.open(name, asset, snap, side, fill["avg"], cfg.order_size, p_win,
                        name, reason)
        if t:
            log.info("[shadow:%s] %s %s x%.0f @ %.3f | %s", name, asset, side,
                     cfg.order_size, fill["avg"], reason[:120])

    fee_up = fee_per_contract(snap.up_ask or 0.5)
    fee_dn = fee_per_contract(snap.down_ask or 0.5)
    for name in cfg.shadow_strategies:
        if name in st.shadows_done:
            continue
        st.shadows_done.add(name)
        if name == "always_up":
            await place(name, "UP", snap.up_ask, 0.5, "baseline always UP")
        elif name == "always_down":
            await place(name, "DOWN", snap.down_ask, 0.5, "baseline always DOWN")
        elif name == "momentum" and f.mom_5m is not None:
            side = "UP" if f.mom_5m > 0 else "DOWN"
            await place(name, side, snap.up_ask if side == "UP" else snap.down_ask,
                        0.5, f"mom_5m={f.mom_5m:+.4%}")
        elif name == "vol_only" and st.p_fair is not None:
            pick = ensemble.decide_side(st.p_fair, snap.up_ask or 9.9,
                                        snap.down_ask or 9.9, fee_up, fee_dn, cfg)
            if pick:
                await place(name, pick["side"], pick["price"], pick["p"],
                            f"vol model edge {pick['ev']:.3f}")
        elif name == "llm_solo":
            usable = [v for v in llm_votes if v.p_up is not None]
            if usable:
                p = sum(v.p_up for v in usable) / len(usable)
                pick = ensemble.decide_side(p, snap.up_ask or 9.9,
                                            snap.down_ask or 9.9, fee_up, fee_dn, cfg)
                if pick:
                    await place(name, pick["side"], pick["price"], pick["p"],
                                "llm ensemble")
        elif name == "clef_solo" and cv.p_up is not None:
            pick = ensemble.decide_side(cv.p_up, snap.up_ask or 9.9,
                                        snap.down_ask or 9.9, fee_up, fee_dn, cfg)
            if pick:
                await place(name, pick["side"], pick["price"], pick["p"],
                            f"clef regime={cv.regime}")
        elif name == "jev_solo" and jv is not None and jv.p_up is not None:
            pick = ensemble.decide_side(jv.p_up, snap.up_ask or 9.9,
                                        snap.down_ask or 9.9, fee_up, fee_dn, cfg)
            if pick:
                await place(name, pick["side"], pick["price"], pick["p"],
                            f"jev regime={jv.regime}")


async def run(cfg):
    rec = Recorder(cfg.db_path, enabled=cfg.recorder)
    ledger = Ledger(cfg, rec)
    calibrator = Calibrator(rec)

    # recover trades orphaned by a previous process restart so they still settle
    orphans = rec.unsettled()
    for o in orphans:
        if o["strategy"] != "ensemble" and o["strategy"] not in ledger.shadows:
            ledger.shadows[o["strategy"]] = StrategyLedger(o["strategy"])
        s = ledger.get(o["strategy"])
        t = Trade(strategy=o["strategy"], asset=o["asset"], ticker=o["ticker"],
                  close_time=now_utc(), side=o["side"], entry=o["entry"],
                  size=o["size"], fees=o["fees"], p_win=o["p_win"],
                  model=o["model"] or "", reason=o["reason"] or "",
                  opened_at=o["ts"], trade_id=o["id"])
        s.pending.append(t)
        s.open_cost += o["entry"] * o["size"]
    if orphans:
        log.warning("recovered %d unsettled trades from previous runs", len(orphans))
    stop = {"flag": False}
    spot_cache, derivs_cache, ev_cache = {}, {}, {}

    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "kalshi-ai-bot/0.2"}) as http:
        kc = Kalshi(http, depth=cfg.book_depth)
        llm_ens = LLMEnsemble(cfg, http)
        log.info("PAPER MODE | assets %s | models %s | clef %s | shadows %s",
                 ",".join(cfg.assets), ",".join(cfg.llm_models),
                 cfg.clef_model or "off", ",".join(cfg.shadow_strategies))

        from .settle import settle_loop
        tasks = [asset_loop(a, cfg, kc, http, llm_ens, ledger, rec, calibrator,
                            spot_cache, derivs_cache, ev_cache, stop)
                 for a in cfg.assets]
        tasks.append(settle_loop(cfg, kc, ledger, calibrator))
        if cfg.recorder:
            tasks.append(snapshot_loop(cfg, kc, rec, spot_cache, derivs_cache, stop))
        try:
            await asyncio.gather(*tasks)
        finally:
            stop["flag"] = True
            log.info("Summary | %s", ledger.summary())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fetch data once, no LLM call")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = cfgmod.load(require_llm=not args.check)
    try:
        if args.check:
            asyncio.run(_check(cfg))
        else:
            asyncio.run(run(cfg))
    except KeyboardInterrupt:
        pass


async def _check(cfg):
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "kalshi-ai-bot/0.2"}) as http:
        kc = Kalshi(http, depth=cfg.book_depth)
        ev = await events.fetch(http)
        print("macro:", ev.to_prompt_text())
        for a in cfg.assets:
            snap = await kc.snapshot(a)
            bundle = await prices.spot_bundle(http, a, 5)
            d = await derivatives.fetch(http, a)
            print(f"\n{a}: {snap}")
            print("  venues:", bundle.prices)
            print("  derivs:", d)


if __name__ == "__main__":
    main()
