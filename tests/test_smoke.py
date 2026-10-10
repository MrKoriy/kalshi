import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kbot import quant, kalshi, ensemble, ledger, config as cfgmod  # noqa: E402


def test_fair_prob():
    p = quant.fair_prob_up(100.0, 100.0, 10.0, 0.001)
    assert abs(p - 0.5) < 0.01
    p2 = quant.fair_prob_up(100.0, 99.0, 10.0, 0.001)
    assert p2 > 0.9


def test_kelly():
    c = quant.kelly_contracts(0.6, 0.5, 1000.0, 0.25)
    assert c > 0
    assert quant.kelly_contracts(0.5, 0.5, 1000.0, 0.25) == 0.0


def test_fee():
    assert ledger.fee_per_contract(0.5) == 0.0175
    assert ledger.fee_per_contract(0.9) < 0.01


def test_fill():
    levels = [kalshi.Level(0.5, 100), kalshi.Level(0.51, 100)]
    snap = kalshi.Snapshot(
        asset="BTC", ticker="T", close_time=kalshi.now_utc(), open_time=None,
        strike=100.0, rules="", volume=0, open_interest=0,
        up_levels=levels, down_levels=[], up_bid=0.49, down_bid=None,
        last_price=0.5, fetched_at=kalshi.now_utc(),
    )
    f = snap.fill_estimate("UP", 150)
    assert abs(f["avg"] - (100 * 0.5 + 50 * 0.51) / 150) < 1e-9
    assert not f["partial"]
    f2 = snap.fill_estimate("UP", 500)
    assert f2["partial"]


def test_ensemble_combine():
    cfg = cfgmod.Config(
        assets=["BTC"], obey_model=True, time_delay=15, min_edge=0.06, max_edge=0.35,
        kelly_fraction=0.25, order_size=300, max_exposure=1000,
        max_daily_profit=400, max_daily_loss=-200,
        w_vol=0.45, w_llm=0.4, w_clef=0.15, w_jev=0.15,
        simulate_fees=True, book_depth=10, poll_seconds=2, event_edge_bonus=0.03,
        trades_file="t", db_path=":memory:", recorder=False, snapshot_seconds=8,
        memory_days=7, shadow_strategies=[], llm_base_url="x", llm_api_key="k",
        llm_models=["m"], llm_temperature=0.2, llm_timeout=45, instructions="",
        clef_base_url="", clef_model="", clef_timeout=10, jev_api_key="", jev_model="", jev_base_url="", jev_timeout=10,
        sigma_mult=1.25, momentum_fade=True,
    )
    votes = [
        ensemble.Vote("vol", 0.6, 1.0),
        ensemble.Vote("llm:a", 0.7, 1.0),
        ensemble.Vote("llm:b", None, 1.0),
        ensemble.Vote("clef", 0.4, 1.0),
    ]
    res = ensemble.combine(cfg, votes)
    assert res.p_up is not None and 0.5 < res.p_up < 0.7


def test_decide_side():
    cfg = cfgmod.Config(
        assets=["BTC"], obey_model=True, time_delay=15, min_edge=0.05, max_edge=0.35,
        kelly_fraction=0.25, order_size=300, max_exposure=1000,
        max_daily_profit=400, max_daily_loss=-200,
        w_vol=0.45, w_llm=0.4, w_clef=0.15, w_jev=0.15,
        simulate_fees=True, book_depth=10, poll_seconds=2, event_edge_bonus=0.03,
        trades_file="t", db_path=":memory:", recorder=False, snapshot_seconds=8,
        memory_days=7, shadow_strategies=[], llm_base_url="x", llm_api_key="k",
        llm_models=["m"], llm_temperature=0.2, llm_timeout=45, instructions="",
        clef_base_url="", clef_model="", clef_timeout=10, jev_api_key="", jev_model="", jev_base_url="", jev_timeout=10,
        sigma_mult=1.25, momentum_fade=True,
    )
    pick = ensemble.decide_side(0.6, 0.5, 0.52, 0.0175, 0.0175, cfg)
    assert pick and pick["side"] == "UP"
    pick2 = ensemble.decide_side(0.5, 0.5, 0.52, 0.0175, 0.0175, cfg)
    assert pick2 is None
