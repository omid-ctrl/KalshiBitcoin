"""Tests for the strategy and risk layers.

The five things that have to be true before this bot is allowed near money:

1. The Kelly formula is the Kelly formula (checked against values computed by hand).
2. Fractional Kelly scales linearly (a quarter-Kelly bet is a quarter the size).
3. The correlated-ladder correction genuinely SHRINKS the position versus naive
   per-strike sizing — the whole point of the ladder maths.
4. No taker trade is ever emitted below the fee hurdle, at any price, ever.
5. The kill switch LATCHES: once tripped it stays tripped until someone resets it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from kalshi_btc.config import RiskLimits
from kalshi_btc.core.fees import min_taker_edge, taker_fee_per_contract
from kalshi_btc.core.types import Action, Liquidity, Side
from kalshi_btc.risk.killswitch import HaltReason, KillSwitch, TradingHalted
from kalshi_btc.risk.limits import Position, RiskManager, worst_case_pnl
from kalshi_btc.strategy.edge import (
    DEFAULT_DF,
    MIN_STRIKES_FOR_SPOT,
    FairValueEngine,
    LadderQuote,
    SettlementWindow,
    estimate_spot_from_ladder,
    implied_spot_from_ladder,
    student_t_df_for_excess_kurtosis,
    unit_density,
    unit_tail_inverse,
)
from kalshi_btc.strategy.quoting import MakerQuoter, adverse_move_std, residual_std_after
from kalshi_btc.strategy.sizing import (
    SizingCandidate,
    kelly_fraction_binary,
    kelly_fraction_for,
    ladder_covariance,
    naive_contracts,
    size_ladder,
)

SIGMA = 0.00466 / math.sqrt(60.0)  # measured 0.466%/hour, expressed per minute


def _limits(**kw) -> RiskLimits:
    """RiskLimits with generous caps so a test measures the MATHS, not the clamps."""
    base = {
        "bankroll": Decimal("10000"),
        "max_contracts_per_order": 100_000,
        "max_position_per_strike": 100_000,
        "max_loss_per_event": Decimal("1000000"),
        "max_loss_per_day": Decimal("1000000"),
        "kelly_fraction": Decimal("0.25"),
    }
    base.update(kw)
    return RiskLimits(**base)


def _consistent_ladder(
    engine: FairValueEngine, spot: float, mtc: float, strikes: tuple[int, ...] = (
        117_800, 117_900, 118_000, 118_100, 118_200
    )
) -> list[LadderQuote]:
    """A ladder priced FROM a known spot and snapped to the 1-cent grid, 1 cent wide.

    This is what a healthy KXBTCD book looks like: every strike agrees on the same level
    to within a tick, and the spread is locked at the minimum.
    """
    out = []
    for k in strikes:
        p = engine.fair_quote(Decimal(k), spot, SIGMA, mtc).prob_above
        mid = Decimal(str(round(p, 2)))
        if not (Decimal("0.03") <= mid <= Decimal("0.97")):
            continue
        out.append(
            LadderQuote(
                ticker=f"KXBTCD-26JUL2819-T{k}.00",
                strike=Decimal(k),
                yes_bid=mid - Decimal("0.005"),
                yes_ask=mid + Decimal("0.005"),
                yes_bid_size=Decimal("100"),
                yes_ask_size=Decimal("100"),
            )
        )
    return out


def _ladder(spot: float, probs: dict[float, tuple[str, str]]) -> list[LadderQuote]:
    return [
        LadderQuote(
            ticker=f"KXBTCD-26JUL2819-T{k:.2f}",
            strike=Decimal(str(k)),
            yes_bid=Decimal(b),
            yes_ask=Decimal(a),
            yes_bid_size=Decimal("100"),
            yes_ask_size=Decimal("100"),
        )
        for k, (b, a) in probs.items()
    ]


# ======================================================================================
# 1 & 2. Kelly
# ======================================================================================
def test_kelly_formula_against_hand_computed_values():
    """f* = (q - p) / (1 - p). Each case worked out by hand."""
    # p = 0.50, q = 0.60  ->  (0.60 - 0.50) / 0.50 = 0.20
    assert kelly_fraction_binary(0.50, 0.60) == pytest.approx(0.20)
    # p = 0.25, q = 0.50  ->  (0.50 - 0.25) / 0.75 = 1/3
    assert kelly_fraction_binary(0.25, 0.50) == pytest.approx(1.0 / 3.0)
    # p = 0.80, q = 0.90  ->  (0.90 - 0.80) / 0.20 = 0.50
    assert kelly_fraction_binary(0.80, 0.90) == pytest.approx(0.50)
    # p = 0.10, q = 0.12  ->  (0.12 - 0.10) / 0.90 = 0.0222...
    assert kelly_fraction_binary(0.10, 0.12) == pytest.approx(0.02222222, abs=1e-8)


def test_kelly_edge_cases():
    """No edge means no bet; a certainty means the whole bankroll; never negative."""
    assert kelly_fraction_binary(0.42, 0.42) == 0.0
    assert kelly_fraction_binary(0.42, 1.0) == pytest.approx(1.0)
    assert kelly_fraction_binary(0.60, 0.40) == 0.0  # adverse: refuse, do not invert
    assert kelly_fraction_binary(0.0, 0.5) == 0.0
    assert kelly_fraction_binary(1.0, 0.5) == 0.0


def test_kelly_maximises_log_growth_numerically():
    """The closed form must be the argmax of the actual growth function."""
    p, q = 0.35, 0.45
    f_star = kelly_fraction_binary(p, q)

    def growth(f: float) -> float:
        b = (1.0 - p) / p
        return q * math.log(1.0 + f * b) + (1.0 - q) * math.log(1.0 - f)

    grid = [i / 10000.0 for i in range(1, 9000)]
    best = max(grid, key=growth)
    assert best == pytest.approx(f_star, abs=1e-3)


def test_selling_yes_is_buying_no():
    """Selling YES at p with prob q must equal buying NO at 1-p with prob 1-q."""
    f_sell, p_eff = kelly_fraction_for(Action.SELL, Decimal("0.60"), 0.40)
    assert p_eff == pytest.approx(0.60)
    assert f_sell == pytest.approx(kelly_fraction_binary(0.40, 0.60))
    # Hand check: buying NO at 0.40 with q_no = 0.60 -> (0.60-0.40)/0.60 = 1/3
    assert f_sell == pytest.approx(1.0 / 3.0)


def test_fees_move_the_effective_price():
    """A taker fee makes a buy strictly less attractive, by exactly the fee."""
    price, q = Decimal("0.50"), 0.60
    fee = taker_fee_per_contract(price)
    f_free, _ = kelly_fraction_for(Action.BUY, price, q, Decimal("0"))
    f_fee, p_eff = kelly_fraction_for(Action.BUY, price, q, fee)
    assert p_eff == pytest.approx(float(price + fee))
    assert f_fee < f_free
    assert f_fee == pytest.approx(kelly_fraction_binary(float(price + fee), q))


@pytest.mark.parametrize("fraction", [1.0, 0.5, 0.25, 0.1])
def test_fractional_kelly_scales_linearly(fraction):
    """Quarter-Kelly must be exactly a quarter of full Kelly, not 'about a quarter'."""
    c = SizingCandidate(
        ticker="KXBTCD-26JUL2819-T118000.00",
        strike=Decimal("118000"),
        action=Action.BUY,
        price=Decimal("0.40"),
        fair_prob=0.50,
        liquidity=Liquidity.MAKER,
    )
    full, _, _ = naive_contracts(c, Decimal("10000"), kelly_fraction=1.0)
    part, _, _ = naive_contracts(c, Decimal("10000"), kelly_fraction=fraction)
    assert part == pytest.approx(full * fraction)


def test_fractional_kelly_flows_through_size_ladder():
    """The setting, not just the helper, has to scale the final contract count."""
    c = SizingCandidate(
        ticker="KXBTCD-26JUL2819-T118000.00",
        strike=Decimal("118000"),
        action=Action.BUY,
        price=Decimal("0.40"),
        fair_prob=0.50,
        liquidity=Liquidity.MAKER,
    )
    quarter = size_ladder([c], _limits(), kelly_fraction=0.25).joint_total
    full = size_ladder([c], _limits(), kelly_fraction=1.0).joint_total
    assert quarter == pytest.approx(full * 0.25, rel=0.01)


# ======================================================================================
# 3. The correlated-ladder correction
# ======================================================================================
def test_ladder_covariance_is_exact_for_nested_indicators():
    """Cov(1{S>k_i}, 1{S>k_j}) = min(q_i,q_j) - q_i q_j, and the diagonal is q(1-q)."""
    q = [0.8, 0.5, 0.2]
    cov = ladder_covariance(q)
    for i, qi in enumerate(q):
        assert cov[i, i] == pytest.approx(qi * (1 - qi))
    assert cov[0, 1] == pytest.approx(0.5 - 0.8 * 0.5)  # = 0.5 * (1 - 0.8)
    assert cov[0, 2] == pytest.approx(0.2 - 0.8 * 0.2)
    assert cov[1, 2] == pytest.approx(0.2 - 0.5 * 0.2)
    assert cov[2, 0] == pytest.approx(cov[0, 2])  # symmetric


def _correlated_candidates(n: int) -> list[SizingCandidate]:
    """n adjacent strikes, all cheap by the same 4 cents. Highly correlated by design."""
    out = []
    for i in range(n):
        q = 0.60 - 0.05 * i
        out.append(
            SizingCandidate(
                ticker=f"KXBTCD-26JUL2819-T{118000 + 100 * i}.00",
                strike=Decimal(str(118000 + 100 * i)),
                action=Action.BUY,
                price=Decimal(str(round(q - 0.04, 2))),
                fair_prob=q,
                liquidity=Liquidity.MAKER,
            )
        )
    return out


def test_correlated_ladder_correction_reduces_total_size():
    """THE test. Naive per-strike Kelly over-bets a ladder; the correction must shrink it."""
    sizing = size_ladder(_correlated_candidates(6), _limits())
    assert sizing.naive_total > 0
    assert sizing.joint_total < sizing.naive_total
    assert sizing.ladder_scale < 1.0
    assert sizing.shrink < 1.0


def test_shrinkage_grows_with_the_number_of_correlated_strikes():
    """More nested strikes on the same underlying means a harder haircut, not a softer one."""
    scales = [size_ladder(_correlated_candidates(n), _limits()).ladder_scale for n in (2, 4, 8)]
    assert scales[0] > scales[1] > scales[2]


def test_single_strike_is_left_alone():
    """With one bet there is nothing to correlate; the quadratic scale must be ~1."""
    sizing = size_ladder(_correlated_candidates(1), _limits())
    assert sizing.ladder_scale == pytest.approx(1.0, rel=0.05)
    assert sizing.joint_total == pytest.approx(sizing.naive_total, rel=0.05)


def test_identical_strikes_total_equals_one_strike():
    """N copies of the same bet must total roughly ONE bet, not N."""
    one = SizingCandidate(
        ticker="KXBTCD-26JUL2819-T118000.00",
        strike=Decimal("118000"),
        action=Action.BUY,
        price=Decimal("0.46"),
        fair_prob=0.50,
        liquidity=Liquidity.MAKER,
    )
    solo = size_ladder([one], _limits()).joint_total
    # Five markets with identical fair probs are perfectly correlated (Cov = q(1-q) for
    # every pair), so the joint total must collapse back to the single-bet size.
    clones = [
        SizingCandidate(
            ticker=f"KXBTCD-26JUL2819-T{118000 + i}.00",
            strike=Decimal(str(118000 + i)),
            action=one.action,
            price=one.price,
            fair_prob=one.fair_prob,
            liquidity=one.liquidity,
        )
        for i in range(5)
    ]
    many = size_ladder(clones, _limits()).joint_total
    assert many == pytest.approx(solo, rel=0.05)


def test_hard_limits_bind_after_kelly():
    """Kelly says how much is optimal; the risk config says how much is allowed."""
    sizing = size_ladder(_correlated_candidates(4), _limits(max_contracts_per_order=3))
    assert all(o.contracts <= 3 for o in sizing.orders)
    assert any("per_order" in o.clamps for o in sizing.orders)


def test_existing_position_limits_further_buying():
    c = SizingCandidate(
        ticker="KXBTCD-26JUL2819-T118000.00",
        strike=Decimal("118000"),
        action=Action.BUY,
        price=Decimal("0.40"),
        fair_prob=0.60,
        liquidity=Liquidity.MAKER,
        existing_position=24,
    )
    sizing = size_ladder([c], _limits(max_position_per_strike=25))
    assert sizing.orders[0].contracts <= 1
    assert "per_strike" in sizing.orders[0].clamps


# ======================================================================================
# 4. The fee hurdle
# ======================================================================================
@pytest.mark.parametrize(
    "bid,ask",
    [("0.01", "0.02"), ("0.10", "0.11"), ("0.49", "0.50"), ("0.50", "0.51"), ("0.98", "0.99")],
)
def test_no_taker_trade_below_the_fee_hurdle(bid, ask):
    """Sweep fair value across the whole 0..1 range at every price level.

    Any strike the engine marks takeable must clear min_taker_edge at the traded price,
    and any strike it refuses must genuinely be below it. This is the gate that stands
    between the bot and paying 1.75c to capture 0.5c.
    """
    engine = FairValueEngine()
    q = LadderQuote(
        ticker="KXBTCD-26JUL2819-T118000.00",
        strike=Decimal("118000"),
        yes_bid=Decimal(bid),
        yes_ask=Decimal(ask),
    )
    for i in range(0, 101):
        fair = Decimal(i) / 100
        # Drive fair value directly rather than via spot, so the sweep is exhaustive.
        edge_buy = fair - q.yes_ask
        edge_sell = q.yes_bid - fair
        takeable_buy = edge_buy >= engine.hurdle(q.yes_ask)
        takeable_sell = edge_sell >= engine.hurdle(q.yes_bid)
        if takeable_buy:
            assert edge_buy >= min_taker_edge(q.yes_ask)
        if takeable_sell:
            assert edge_sell >= min_taker_edge(q.yes_bid)
        # Never both directions at once — that would be an arbitrage against ourselves.
        assert not (takeable_buy and takeable_sell)


def test_engine_marks_nothing_takeable_when_fair_equals_mid():
    """A market we agree with is a market we do not trade."""
    engine = FairValueEngine()
    quotes = _consistent_ladder(engine, 118_000.0, 30.0)
    spot = implied_spot_from_ladder(quotes, SIGMA, 30.0)
    assert spot is not None
    ladder = engine.evaluate(
        quotes, spot=spot, sigma_per_minute=SIGMA, minutes_to_close=30.0, spot_source="ladder"
    )
    for e in ladder.edges:
        assert not e.any_takeable, e.describe()
        assert "below hurdle" in e.reason


def test_taker_hurdle_is_strictly_above_the_bare_fee():
    """min_taker_edge must never be cheaper than the fee it is supposed to cover."""
    engine = FairValueEngine()
    for cents in range(1, 100):
        p = Decimal(cents) / 100
        assert engine.hurdle(p) > taker_fee_per_contract(p)


def test_decided_strikes_are_never_takeable():
    """Once residual sigma has collapsed, 'edge' is arithmetic on a settled outcome."""
    engine = FairValueEngine()
    quotes = _ladder(118000.0, {118000.0: ("0.10", "0.11")})
    ladder = engine.evaluate(
        quotes,
        spot=118500.0,
        sigma_per_minute=SIGMA,
        minutes_to_close=0.02,  # ~1 second left
        spot_source="brti",
    )
    assert not ladder.edges[0].any_takeable
    assert "decided" in ladder.edges[0].reason


# ======================================================================================
# 5. The kill switch
# ======================================================================================
def test_killswitch_latches_and_needs_an_explicit_reset():
    ks = KillSwitch(max_feed_age_s=5.0)
    now = datetime.now(UTC)

    assert ks.check_feed_age(now, now) is True
    assert not ks.halted

    # Feed goes stale -> halted.
    assert ks.check_feed_age(now - timedelta(seconds=30), now) is False
    assert ks.halted
    assert ks.halts[0].reason is HaltReason.STALE_FEED
    assert "30.0s" in ks.reason

    # A GOOD tick arriving must NOT clear the halt. This is the whole point.
    assert ks.check_feed_age(now, now) is False
    assert ks.halted

    with pytest.raises(TradingHalted):
        ks.require_live("place an order")

    assert ks.reset("test") == 1
    assert not ks.halted
    ks.require_live()  # no longer raises


def test_killswitch_trips_on_every_documented_condition():
    now = datetime.now(UTC)
    for check, expected in [
        (lambda k: k.check_feed_age(None, now), HaltReason.STALE_FEED),
        (lambda k: k.check_venue_agreement(118000, 118500), HaltReason.VENUE_DISAGREEMENT),
        (lambda k: k.check_websocket(False), HaltReason.WS_DISCONNECT),
        (
            lambda k: k.check_reconciliation({"A": 3}, {"A": 1}),
            HaltReason.RECONCILE_MISMATCH,
        ),
        (lambda k: k.check_daily_loss(Decimal("-60"), Decimal("50")), HaltReason.DAILY_LOSS),
    ]:
        ks = KillSwitch()
        assert check(ks) is False
        assert ks.halted
        assert ks.halts[0].reason is expected
        assert ks.halts[0].detail  # every halt carries a human-readable reason


def test_killswitch_records_every_halt_not_just_the_first():
    ks = KillSwitch()
    ks.check_websocket(False)
    ks.check_venue_agreement(1, 1000)
    assert len(ks.halts) == 2
    assert ks.counts() == {"ws_disconnect": 1, "venue_disagreement": 1}


def test_halted_riskmanager_rejects_every_order():
    rm = RiskManager(risk=_limits())
    rm.killswitch.trip(HaltReason.MANUAL, "operator pulled the plug")
    d = rm.check_order(
        ticker="KXBTCD-26JUL2819-T118000.00",
        action=Action.BUY,
        contracts=1,
        price=Decimal("0.40"),
        liquidity=Liquidity.MAKER,
    )
    assert not d
    assert "halted" in d.reason


# ======================================================================================
# Risk limits
# ======================================================================================
def test_worst_case_pnl_is_exact_for_a_hedged_ladder():
    """Long 117,900 / short 118,000 can lose at most the net premium, not both legs."""
    lo = Position("KXBTCD-x-T117900.00", Decimal("117900"))
    hi = Position("KXBTCD-x-T118000.00", Decimal("118000"))
    lo.apply(Action.BUY, 10, Decimal("0.60"), Decimal("0"))   # pay $6
    hi.apply(Action.SELL, 10, Decimal("0.40"), Decimal("0"))  # receive $4
    # cash = -6 + 4 = -2
    # S below both:   -2
    # S between:      -2 + 10 = +8
    # S above both:   -2 + 10 - 10 = -2
    assert worst_case_pnl([lo, hi]) == Decimal("-2")
    # Naively summing premium at risk would have said 6 + 6 = $12.


def test_per_event_loss_limit_blocks_the_order_that_would_breach_it():
    rm = RiskManager(risk=_limits(max_loss_per_event=Decimal("5"), max_contracts_per_order=100))
    d = rm.check_order(
        ticker="KXBTCD-26JUL2819-T118000.00",
        action=Action.BUY,
        contracts=100,
        price=Decimal("0.50"),
        liquidity=Liquidity.TAKER,
    )
    # 100 contracts at 50c risks $50 plus fees; the limit is $5, so the clip must shrink.
    assert d.contracts < 100
    worst = -(Decimal(d.contracts) * Decimal("0.50"))
    assert worst >= -Decimal("5")


def test_settlement_realises_the_right_side():
    rm = RiskManager(risk=_limits())
    rm.record_fill(
        ticker="KXBTCD-26JUL2819-T118000.00",
        action=Action.BUY,
        contracts=10,
        price=Decimal("0.40"),
        liquidity=Liquidity.MAKER,
    )
    realised = rm.settle_event("KXBTCD-26JUL2819", Decimal("118500"))
    assert realised == Decimal("6")  # paid $4, received $10
    assert rm.position_map() == {}


def test_maker_fills_are_free_and_taker_fills_are_not():
    rm = RiskManager(risk=_limits())
    assert rm.fee_for(Liquidity.MAKER, Decimal("0.50"), 100) == Decimal("0")
    assert rm.fee_for(Liquidity.TAKER, Decimal("0.50"), 100) == Decimal("1.75")


# ======================================================================================
# Fair value engine
# ======================================================================================
def test_student_t_df_matches_measured_kurtosis():
    """df is pinned by the measurement, not chosen: excess kurtosis 6/(df-4) = 12.88."""
    assert student_t_df_for_excess_kurtosis(12.88) == pytest.approx(4.0 + 6.0 / 12.88)
    assert 6.0 / (DEFAULT_DF - 4.0) == pytest.approx(12.88)


def test_unit_tail_inverse_inverts_the_pricer_tail():
    from kalshi_btc.model.pricing import _tail_prob

    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        for dist, df in (("normal", 4.0), ("t", DEFAULT_DF)):
            z = unit_tail_inverse(p, dist, df)
            assert _tail_prob(z, dist, df) == pytest.approx(p, abs=1e-8)


def test_unit_density_integrates_to_one():
    import numpy as np

    zs = np.linspace(-40, 40, 400_001)
    for dist, df in (("normal", 4.0), ("t", DEFAULT_DF)):
        dens = np.array([unit_density(float(z), dist, df) for z in zs])
        assert np.trapezoid(dens, zs) == pytest.approx(1.0, abs=2e-3)


def test_fat_tails_price_the_wings_above_gaussian():
    """The entire reason for the Student-t default."""
    gauss = FairValueEngine(dist="normal")
    fat = FairValueEngine(dist="t", df=DEFAULT_DF)
    spot, mtc = 118_000.0, 30.0
    far = Decimal("119500")  # deep out of the money
    p_gauss = gauss.fair_quote(far, spot, SIGMA, mtc).prob_above
    p_fat = fat.fair_quote(far, spot, SIGMA, mtc).prob_above
    assert p_fat > p_gauss
    # ...and correspondingly less mass just outside the money.
    near = Decimal("118200")
    assert fat.fair_quote(near, spot, SIGMA, mtc).prob_above < (
        gauss.fair_quote(near, spot, SIGMA, mtc).prob_above
    )


def test_in_window_pricer_uses_the_locked_in_average():
    """Inside the window the KNOWN ticks are what move the price, not the current spot.

    Note what is NOT different: `settlement_std_dollars` already knows only a handful of
    ticks remain random, so both paths report a similar residual sigma. The value the
    in-window pricer adds is the LEVEL — 55 ticks that printed well above the strike are a
    constant we have already banked, and pricing 50/50 because spot happens to sit on the
    strike would be leaving that information on the table.
    """
    engine = FairValueEngine()
    spot, strike = 118_000.0, Decimal("118000")
    # 55 of the 60 settlement ticks have printed at 118,060 — the average is already
    # locked well above the strike whatever the last five do.
    window = SettlementWindow(known_sum=118_060.0 * 55, known_ticks=55, spot_now=spot)
    with_win = engine.fair_quote(strike, spot, SIGMA, 0.083, window=window)
    without = engine.fair_quote(strike, spot, SIGMA, 0.083)

    assert without.prob_above == pytest.approx(0.5, abs=0.01)  # blind to the window
    assert with_win.prob_above > 0.99  # knows the settlement is effectively decided
    assert with_win.minutes_to_close == pytest.approx(5.0 / 60.0)


def test_in_window_residual_collapses_as_ticks_print():
    """Each printed tick permanently removes variance; it can never come back."""
    engine = FairValueEngine()
    spot = 118_000.0
    stds = [
        engine.fair_quote(
            Decimal("118000"), spot, SIGMA, (60 - n) / 60.0,
            window=SettlementWindow(spot * n, n, spot),
        ).residual_std
        for n in (10, 30, 50, 58)
    ]
    assert stds == sorted(stds, reverse=True)
    assert stds[-1] < stds[0] / 10.0


def test_implied_spot_recovers_a_known_spot():
    """Price a ladder from a known spot, then invert it: we must get the spot back."""
    engine = FairValueEngine()
    spot, mtc = 118_137.0, 25.0
    strikes = [117_900, 118_000, 118_100, 118_200, 118_300]
    quotes = []
    for k in strikes:
        p = engine.fair_quote(Decimal(k), spot, SIGMA, mtc).prob_above
        # Round to the 1-cent grid the venue actually quotes on, both sides of the mid.
        mid = Decimal(str(round(p, 2)))
        if not (Decimal("0.03") <= mid <= Decimal("0.97")):
            continue
        quotes.append(
            LadderQuote(
                ticker=f"KXBTCD-x-T{k}.00",
                strike=Decimal(k),
                yes_bid=mid - Decimal("0.005"),
                yes_ask=mid + Decimal("0.005"),
            )
        )
    recovered = implied_spot_from_ladder(quotes, SIGMA, mtc)
    # A 1-cent tick on a ~$200 residual sigma is worth a few dollars of spot.
    assert recovered == pytest.approx(spot, abs=15.0)


def test_implied_spot_is_none_when_every_strike_is_pinned():
    quotes = _ladder(118000.0, {117000.0: ("0.99", "1.00"), 119000.0: ("0.00", "0.01")})
    assert implied_spot_from_ladder(quotes, SIGMA, 30.0) is None


def test_spot_needs_enough_strikes_to_be_identified():
    """A one-strike ladder restates its own mid; the 'edge' it implies is noise.

    This is the guard the first live paper session earned: every 25-cent edge it reported
    came from a cycle where the book had collapsed to one or two quotable strikes.
    """
    engine = FairValueEngine()
    full = _consistent_ladder(engine, 118_000.0, 30.0)
    assert len(full) >= MIN_STRIKES_FOR_SPOT

    for n in range(MIN_STRIKES_FOR_SPOT):
        est = estimate_spot_from_ladder(full[:n], SIGMA, 30.0)
        assert not est.usable
        assert est.value is None
        assert "quotable strike" in est.reason or "closed" in est.reason

    ok = estimate_spot_from_ladder(full, SIGMA, 30.0)
    assert ok.usable and ok.value is not None
    assert ok.dispersion_std < 0.75


def test_a_ladder_that_disagrees_with_itself_is_rejected():
    """Strikes that imply wildly different levels mean a stale book, not an opportunity."""
    quotes = _ladder(
        118_000.0,
        # These mids are mutually inconsistent: 0.50 and 0.45 are one tick of level apart
        # in reality, but 0.05 at the next strike is hundreds of dollars away.
        {
            117_900.0: ("0.495", "0.505"),
            118_000.0: ("0.445", "0.455"),
            118_100.0: ("0.045", "0.055"),
        },
    )
    est = estimate_spot_from_ladder(quotes, SIGMA, 30.0)
    assert not est.usable
    assert "disagrees with itself" in est.reason
    assert implied_spot_from_ladder(quotes, SIGMA, 30.0) is None


def test_an_abnormally_wide_strike_is_not_quotable():
    """KXBTCD spreads are locked at 1 cent. A 14-cent spread is a broken book."""
    tight = LadderQuote("T", Decimal("118000"), Decimal("0.50"), Decimal("0.51"))
    wide = LadderQuote("T", Decimal("118000"), Decimal("0.50"), Decimal("0.64"))
    assert tight.is_quotable
    assert not wide.is_quotable
    assert wide.spread == Decimal("0.14")


# ======================================================================================
# Quoting and adverse selection
# ======================================================================================
def test_adverse_move_explodes_as_the_window_closes():
    """The pull rule's driving quantity must blow up near expiry, not drift."""
    spot = 118_000.0
    early = adverse_move_std(0.0, *residual_std_after(SIGMA, spot, 45.0, 2.0))
    late = adverse_move_std(0.0, *residual_std_after(SIGMA, spot, 0.5, 2.0))
    last = adverse_move_std(0.0, *residual_std_after(SIGMA, spot, 0.05, 2.0))
    assert early < late < last
    assert last > 0.05  # over five cents per 2s refresh in the final seconds


def test_adverse_move_shrinks_away_from_the_money():
    """Wing strikes have a small digital delta, which is why they are quotable at all."""
    spot = 118_000.0
    std = residual_std_after(SIGMA, spot, 40.0, 2.0)
    atm = adverse_move_std(0.0, *std)
    wing = adverse_move_std(2.0, *std)
    assert wing < atm / 10.0


def test_at_the_money_is_unquotable_at_a_two_second_cadence():
    """A real, uncomfortable finding, pinned so nobody 'fixes' it by loosening the guard.

    An at-the-money digital moves ~1.4c of fair value per two seconds with 45 minutes to
    run. A locked 1-cent spread pays 0.5c. At this reprice cadence there is no version of
    quoting the money that makes money, and the guard must say so.
    """
    engine = FairValueEngine()
    quoter = MakerQuoter(risk=_limits(), refresh_seconds=2.0)
    quotes = _ladder(118000.0, {118000.0: ("0.49", "0.50")})
    ladder = engine.evaluate(
        quotes, spot=118_000.0, sigma_per_minute=SIGMA, minutes_to_close=45.0,
        spot_source="brti", event_ticker="KXBTCD-26JUL2819",
    )
    plan = quoter.plan(ladder)
    assert plan.n_quotes == 0
    assert "adverse selection" in plan.pulls[0].reason


def test_quoter_pulls_inside_the_hard_window():
    """No resting quotes into the settlement print, whatever the model thinks."""
    engine = FairValueEngine()
    quoter = MakerQuoter(risk=_limits())
    quotes = _ladder(118000.0, {118000.0: ("0.49", "0.50")})
    ladder = engine.evaluate(
        quotes, spot=118_000.0, sigma_per_minute=SIGMA, minutes_to_close=0.2,
        spot_source="brti", event_ticker="KXBTCD-26JUL2819",
    )
    plan = quoter.plan(ladder)
    assert plan.n_quotes == 0
    assert plan.pulls and "hard pull window" in plan.pulls[0].reason


def _wing_ladder(
    engine: FairValueEngine, spot: float, strike: int, mtc: float
) -> list[LadderQuote]:
    """A wing strike quoted 1 cent wide, straddling our own fair value."""
    fair = engine.fair_quote(Decimal(strike), spot, SIGMA, mtc).prob_above
    bid = (Decimal(str(round(fair, 4))) - Decimal("0.005")).quantize(Decimal("0.01"))
    return [
        LadderQuote(
            ticker=f"KXBTCD-26JUL2819-T{strike}.00",
            strike=Decimal(strike),
            yes_bid=bid,
            yes_ask=bid + Decimal("0.01"),
            yes_bid_size=Decimal("50"),
            yes_ask_size=Decimal("50"),
        )
    ]


def test_quoter_quotes_the_wings_where_the_delta_is_small():
    """Where the guard permits it, we must actually put quotes up."""
    engine = FairValueEngine()
    quoter = MakerQuoter(risk=_limits(max_contracts_per_order=5))
    spot, mtc = 118_000.0, 40.0
    quotes = _wing_ladder(engine, spot, 118_800, mtc)  # ~1.8 sigma out
    ladder = engine.evaluate(
        quotes, spot=spot, sigma_per_minute=SIGMA, minutes_to_close=mtc,
        spot_source="brti", event_ticker="KXBTCD-26JUL2819",
    )
    assert abs(ladder.edges[0].z_score) > 1.5
    plan = quoter.plan(ladder)
    assert plan.n_quotes >= 1
    for i in plan.intents:
        assert i.liquidity is Liquidity.MAKER
        assert i.price == i.price.quantize(Decimal("0.01"))  # on the tick grid
        assert Decimal("0.01") <= i.price <= Decimal("0.99")
        # A maker quote may never cross: our bid stays at or below the touch, our ask at
        # or above it. Crossing would be rejected as post_only at the venue.
        if i.action is Action.BUY:
            assert i.price <= quotes[0].yes_bid
        else:
            assert i.price >= quotes[0].yes_ask


def test_inventory_skew_withdraws_the_side_we_are_already_long():
    engine = FairValueEngine()
    quoter = MakerQuoter(risk=_limits(max_position_per_strike=10), max_skew_ticks=2)
    quotes = _wing_ladder(engine, 118_000.0, 118_800, 40.0)
    ladder = engine.evaluate(
        quotes, spot=118_000.0, sigma_per_minute=SIGMA, minutes_to_close=40.0,
        spot_source="brti", event_ticker="KXBTCD-26JUL2819",
    )
    ticker = quotes[0].ticker

    def actions(pos: int) -> set[Action]:
        return {i.action for i in quoter.plan(ladder, positions={ticker: pos}).intents}

    flat, long, short = actions(0), actions(10), actions(-10)
    assert flat  # there is something to withdraw in the first place
    # Long inventory withdraws the buy side and never adds one.
    assert Action.BUY not in long
    # Short inventory withdraws the sell side and never adds one.
    assert Action.SELL not in short
    # At the position cap the skew is one-directional: we quote only the flattening side.
    assert long <= {Action.SELL}
    assert short <= {Action.BUY}


def test_quoter_ignores_a_hopeless_queue():
    """Joining the back of a 2,000-contract queue is decoration, not a quote."""
    engine = FairValueEngine()
    base = _wing_ladder(engine, 118_000.0, 118_800, 40.0)[0]
    deep = LadderQuote(
        ticker=base.ticker, strike=base.strike, yes_bid=base.yes_bid, yes_ask=base.yes_ask,
        yes_bid_size=Decimal("2000"), yes_ask_size=Decimal("2000"),
    )
    ladder = engine.evaluate(
        [deep], spot=118_000.0, sigma_per_minute=SIGMA, minutes_to_close=40.0, spot_source="brti"
    )
    assert MakerQuoter(risk=_limits(), max_queue_ahead=Decimal("10")).plan(ladder).n_quotes == 0
    assert MakerQuoter(risk=_limits(), max_queue_ahead=Decimal("5000")).plan(ladder).n_quotes > 0


# ======================================================================================
# Paper fill simulation
# ======================================================================================
def test_maker_fill_needs_the_queue_to_clear_first():
    """Front-of-queue fills are the classic paper-trading lie. Make sure we do not tell it."""
    from kalshi_btc.core.types import MarketSnapshot
    from kalshi_btc.runner.paper import FillSimulator
    from kalshi_btc.strategy.quoting import QuoteIntent, QuotePlan

    now = datetime.now(UTC)
    sim = FillSimulator()
    intent = QuoteIntent(
        ticker="KXBTCD-x-T118000.00",
        strike=Decimal("118000"),
        side=Side.YES,
        action=Action.BUY,
        price=Decimal("0.49"),
        contracts=5,
        fair=Decimal("0.50"),
        edge=Decimal("0.01"),
        queue_ahead=Decimal("400"),
        reason="join bid",
    )
    sim.reconcile(QuotePlan(intents=[intent]), now)

    def snap(volume: str, bid: str = "0.49", ask: str = "0.50") -> MarketSnapshot:
        return MarketSnapshot(
            ticker=intent.ticker, strike=Decimal("118000"),
            yes_bid=Decimal(bid), yes_ask=Decimal(ask),
            yes_bid_size=Decimal("400"), yes_ask_size=Decimal("400"),
            volume=Decimal(volume), open_interest=Decimal("0"),
            open_time=now, close_time=now, status="active",
        )

    sim.on_market_update({intent.ticker: snap("1000")}, now)  # first sight: baseline
    # 200 contracts print; only half is assumed to hit our side, so 100 of the 400-deep
    # queue clears and we get nothing.
    key = (intent.ticker, "buy", Decimal("0.49"))
    assert sim.on_market_update({intent.ticker: snap("1200")}, now) == []
    assert sim.resting[key].queue_ahead == Decimal("300")
    assert sim.maker_fills == 0

    # Another 1,000 prints: 500 to our side clears the remaining 300 and fills 5.
    fills = sim.on_market_update({intent.ticker: snap("2200")}, now)
    assert len(fills) == 1
    assert fills[0].contracts == 5
    assert fills[0].liquidity is Liquidity.MAKER
    assert fills[0].fee == Decimal("0")  # maker fees are ZERO on KXBTCD


def test_repricing_loses_queue_position():
    from kalshi_btc.runner.paper import FillSimulator
    from kalshi_btc.strategy.quoting import QuoteIntent, QuotePlan

    now = datetime.now(UTC)
    sim = FillSimulator()

    def intent(price: str) -> QuoteIntent:
        return QuoteIntent(
            ticker="KXBTCD-x-T118000.00", strike=Decimal("118000"), side=Side.YES,
            action=Action.BUY, price=Decimal(price), contracts=1, fair=Decimal("0.50"),
            edge=Decimal("0.01"), queue_ahead=Decimal("300"), reason="join bid",
        )

    sim.reconcile(QuotePlan(intents=[intent("0.49")]), now)
    sim.resting[("KXBTCD-x-T118000.00", "buy", Decimal("0.49"))].queue_ahead = Decimal("10")
    # Same price -> keep priority.
    sim.reconcile(QuotePlan(intents=[intent("0.49")]), now)
    assert sim.resting[("KXBTCD-x-T118000.00", "buy", Decimal("0.49"))].queue_ahead == Decimal("10")
    # New price -> back of a fresh queue.
    sim.reconcile(QuotePlan(intents=[intent("0.48")]), now)
    moved = sim.resting[("KXBTCD-x-T118000.00", "buy", Decimal("0.48"))]
    assert moved.queue_ahead == Decimal("300")
    assert sim.cancels == 1


def test_taker_fill_pays_the_venue_fee_and_respects_available_size():
    from kalshi_btc.runner.paper import FillSimulator

    sim = FillSimulator()
    fill = sim.take(
        ticker="KXBTCD-x-T118000.00", action=Action.BUY, price=Decimal("0.50"),
        contracts=10, available=Decimal("4"), now=datetime.now(UTC), reason="test",
    )
    assert fill is not None
    assert fill.contracts == 4  # capped by resting size, not by our appetite
    assert fill.liquidity is Liquidity.TAKER
    assert fill.fee == Decimal("0.07")  # 0.07 * 4 * 0.5 * 0.5 = 0.07 exactly

    assert sim.take(
        ticker="x", action=Action.BUY, price=Decimal("0.50"), contracts=10,
        available=Decimal("0"), now=datetime.now(UTC), reason="test",
    ) is None


def test_run_paper_signature_matches_what_the_cli_passes():
    """`cli._run_entrypoint` RAISES on a dropped kwarg, so this is a real contract."""
    import inspect

    from kalshi_btc.runner.paper import run_paper

    params = inspect.signature(run_paper).parameters
    for required in ("settings", "duration_s", "hours"):
        assert required in params, f"cli passes {required!r} and would raise without it"
    assert params["duration_s"].default is None
    assert params["hours"].default is None
