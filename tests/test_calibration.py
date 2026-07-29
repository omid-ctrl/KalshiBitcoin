"""Verify the scoring rules against hand-computed values and the skill score against truth.

Two kinds of test here:

1. ARITHMETIC. Brier, log loss and the reliability curve are computed by hand in the
   test and asserted exactly. A scoring rule with an off-by-one in its denominator would
   silently reshape every conclusion the project draws, and it is cheap to pin down.

2. BEHAVIOUR OF THE SKILL SCORE. The contract is precise: exactly 0 when the model IS
   the market, strictly positive when the model is strictly better, strictly negative
   when worse. Plus the anti-lookahead machinery, which is asserted rather than trusted -
   including a test that literally inspects what arguments the model was called with.

The final test runs the real Asian-settlement pricer against a simulated point-in-time
market maker on simulated KXBTCD events, which is the project's whole thesis expressed
as an assertion.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy import stats

from kalshi_btc.model.calibration import (
    DEFAULT_MINUTE_EDGES,
    LOG_LOSS_EPSILON,
    Calibrator,
    LadderRecord,
    Observation,
    brier_score,
    build_observations,
    log_loss,
    reliability_curve,
    skill_score,
)
from kalshi_btc.model.pricing import TICKS, annual_to_per_minute, price_above

SEED = 20260728
SIGMA_MIN = annual_to_per_minute(0.436)
SPOT = 63_800.0
T0 = datetime(2026, 6, 1, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Scoring rules: hand-computed
# --------------------------------------------------------------------------------------
def test_brier_score_matches_hand_computation():
    # ((0.8-1)^2 + (0.3-0)^2) / 2 = (0.04 + 0.09) / 2 = 0.065
    assert brier_score([0.8, 0.3], [1, 0]) == pytest.approx(0.065)


def test_brier_score_endpoints():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0
    assert brier_score([0.5] * 8, [1, 0, 1, 0, 1, 0, 1, 0]) == pytest.approx(0.25)


def test_log_loss_matches_hand_computation():
    # -(ln 0.8 + ln 0.7) / 2
    expected = -(math.log(0.8) + math.log(0.7)) / 2.0
    assert log_loss([0.8, 0.3], [1, 0]) == pytest.approx(expected)
    assert expected == pytest.approx(0.2899092476)


def test_log_loss_of_a_coin_flip_is_ln_two():
    assert log_loss([0.5] * 6, [1, 0, 1, 0, 1, 0]) == pytest.approx(math.log(2.0))


def test_log_loss_clips_instead_of_returning_infinity():
    """One confidently wrong forecast must not annihilate the whole sample."""
    value = log_loss([0.0], [1])
    assert math.isfinite(value)
    assert value == pytest.approx(-math.log(LOG_LOSS_EPSILON))
    assert log_loss([1.0], [0]) == pytest.approx(-math.log(LOG_LOSS_EPSILON))


def test_log_loss_penalises_confident_errors_harder_than_brier():
    """The property that justifies reporting both.

    Two forecasters with IDENTICAL Brier scores: one consistently mediocre, one that gets
    a 99.5-cent contract completely wrong and compensates by being nearly right elsewhere.
    Brier cannot tell them apart. Log loss rates the bold one 2.2x worse - and the bold
    one is the one that blows up a book.
    """
    timid = [math.sqrt(0.5)] * 2
    bold = [0.995, math.sqrt(1.0 - 0.995**2)]
    outcomes = [0, 0]

    assert brier_score(timid, outcomes) == pytest.approx(0.5)
    assert brier_score(bold, outcomes) == pytest.approx(0.5)
    assert log_loss(bold, outcomes) > 2.0 * log_loss(timid, outcomes)


def test_scoring_rules_reject_bad_input():
    with pytest.raises(ValueError, match="same shape"):
        brier_score([0.5, 0.5], [1])
    with pytest.raises(ValueError, match="empty"):
        brier_score([], [])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.5], [1])
    with pytest.raises(ValueError, match="binary"):
        brier_score([0.5], [0.5])
    with pytest.raises(ValueError, match="eps"):
        log_loss([0.5], [1], eps=0.9)


# --------------------------------------------------------------------------------------
# Skill score
# --------------------------------------------------------------------------------------
def test_skill_score_is_zero_when_model_equals_benchmark():
    assert skill_score(0.0731, 0.0731) == 0.0


def test_skill_score_sign_and_magnitude():
    assert skill_score(0.05, 0.10) == pytest.approx(0.5)
    assert skill_score(0.20, 0.10) == pytest.approx(-1.0)
    assert skill_score(0.0, 0.10) == 1.0


def test_skill_score_against_a_perfect_benchmark():
    """You cannot beat a benchmark with zero Brier, and we must not divide by zero."""
    assert skill_score(0.0, 0.0) == 0.0
    assert skill_score(0.01, 0.0) == -math.inf


# --------------------------------------------------------------------------------------
# Reliability curve
# --------------------------------------------------------------------------------------
def test_reliability_curve_bins_and_centers():
    centers, freq, counts = reliability_curve([0.05, 0.15, 0.15, 0.95], [0, 1, 0, 1], bins=10)
    assert centers.tolist() == pytest.approx(
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    )
    assert counts.tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    assert freq[0] == 0.0
    assert freq[1] == pytest.approx(0.5)
    assert freq[9] == 1.0
    # Empty bins are NaN with a zero count, not a silent zero.
    assert np.isnan(freq[2:9]).all()


def test_reliability_curve_top_bin_is_closed():
    """p == 1.0 must land in the last bin, not fall off the end."""
    _, freq, counts = reliability_curve([1.0, 0.0], [1, 0], bins=4)
    assert counts.tolist() == [1, 0, 0, 1]
    assert freq[3] == 1.0
    assert freq[0] == 0.0


def test_reliability_curve_traces_the_diagonal_for_a_calibrated_forecaster():
    rng = np.random.default_rng(SEED)
    p = rng.uniform(0.0, 1.0, size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    centers, freq, counts = reliability_curve(p, y, bins=10)
    assert (counts > 1000).all()
    assert np.allclose(freq, centers, atol=0.01)


def test_reliability_curve_detects_an_overconfident_forecaster():
    """Push forecasts away from 0.5 without changing the truth; the curve must bend."""
    rng = np.random.default_rng(SEED)
    truth = rng.uniform(0.2, 0.8, size=100_000)
    y = (rng.uniform(size=truth.size) < truth).astype(float)
    overconfident = np.clip(0.5 + 2.0 * (truth - 0.5), 0.0, 1.0)
    centers, freq, counts = reliability_curve(overconfident, y, bins=10)
    populated = counts > 500
    # Above 0.5 the model says more than happens; below 0.5 it says less.
    high = populated & (centers > 0.6)
    low = populated & (centers < 0.4)
    assert (freq[high] < centers[high]).all()
    assert (freq[low] > centers[low]).all()


def test_reliability_curve_rejects_zero_bins():
    with pytest.raises(ValueError, match="bins"):
        reliability_curve([0.5], [1], bins=0)


# --------------------------------------------------------------------------------------
# Fixtures for the Calibrator
# --------------------------------------------------------------------------------------
def make_observations(
    model_probs, market_probs, *, minutes=None, event_prefix="EV", close=T0
) -> list[Observation]:
    """Build observations directly, one synthetic event per index."""
    n = len(model_probs)
    mins = [30.0] * n if minutes is None else minutes
    return [
        Observation(
            ts=close - timedelta(minutes=mins[i]),
            event_ticker=f"{event_prefix}{i}",
            ticker=f"{event_prefix}{i}-T100",
            strike=100.0,
            minutes_to_close=mins[i],
            model_prob=float(model_probs[i]),
            market_prob=float(market_probs[i]),
            close_time=close,
        )
        for i in range(n)
    ]


def settlements_for(observations, outcomes) -> dict[str, float]:
    """Settlement above the strike encodes YES, below encodes NO."""
    return {
        o.event_ticker: (o.strike + 10.0 if y else o.strike - 10.0)
        for o, y in zip(observations, outcomes)
    }


# --------------------------------------------------------------------------------------
# Calibrator: the skill-score contract
# --------------------------------------------------------------------------------------
def test_skill_is_exactly_zero_when_the_model_is_the_market():
    """The single most important guard: no accidental edge from bookkeeping."""
    rng = np.random.default_rng(SEED)
    p = rng.uniform(0.05, 0.95, size=2_000)
    y = (rng.uniform(size=p.size) < p).astype(float)

    obs = make_observations(p, p)
    result = Calibrator().score(obs, settlements_for(obs, y))

    assert result.skill == 0.0
    assert result.model.brier == result.market.brier
    assert result.model.log_loss == result.market.log_loss
    assert not result.beats_market
    assert result.n_observations == 2_000


def test_skill_is_positive_when_the_model_is_strictly_better():
    """Model sees the true probability; the market sees it shrunk toward a coin flip."""
    rng = np.random.default_rng(SEED)
    truth = rng.uniform(0.05, 0.95, size=5_000)
    y = (rng.uniform(size=truth.size) < truth).astype(float)
    market = 0.5 + 0.6 * (truth - 0.5)  # systematically underconfident

    obs = make_observations(truth, market)
    result = Calibrator().score(obs, settlements_for(obs, y))

    assert result.skill > 0.05
    assert result.beats_market
    assert result.model.brier < result.market.brier
    assert result.model.log_loss < result.market.log_loss


def test_skill_is_negative_when_the_model_is_worse():
    rng = np.random.default_rng(SEED)
    truth = rng.uniform(0.05, 0.95, size=5_000)
    y = (rng.uniform(size=truth.size) < truth).astype(float)
    noisy = np.clip(truth + rng.normal(0.0, 0.20, size=truth.size), 0.0, 1.0)

    obs = make_observations(noisy, truth)
    result = Calibrator().score(obs, settlements_for(obs, y))
    assert result.skill < 0.0
    assert not result.beats_market


def test_scoreset_reports_bias_and_base_rate():
    obs = make_observations([0.9] * 100, [0.5] * 100)
    result = Calibrator().score(obs, settlements_for(obs, [1] * 40 + [0] * 60))
    assert result.model.base_rate == pytest.approx(0.4)
    assert result.model.mean_prob == pytest.approx(0.9)
    assert result.model.bias == pytest.approx(0.5)
    assert result.market.bias == pytest.approx(0.1)


# --------------------------------------------------------------------------------------
# Outcomes come from settlement, and only from settlement
# --------------------------------------------------------------------------------------
def test_outcome_uses_strict_greater_than_against_the_strike():
    """strike_type is 'greater': settlement == strike settles NO, not YES."""
    close = T0
    obs = [
        Observation(
            ts=close - timedelta(minutes=10),
            event_ticker="EV",
            ticker="EV-T63800",
            strike=strike,
            minutes_to_close=10.0,
            model_prob=1.0,
            market_prob=0.5,
            close_time=close,
        )
        for strike in (63_799.99, 63_800.00, 63_800.01)
    ]
    # All three share one event, so give them distinct events to score independently.
    obs = [
        Observation(**{**o.__dict__, "event_ticker": f"EV{i}"}) for i, o in enumerate(obs)
    ]
    settlements = {f"EV{i}": 63_800.00 for i in range(3)}
    result = Calibrator(min_bucket_n=1).score(obs, settlements)
    # Model says 1.0 everywhere: Brier 0 for the strike it clears, 1 for the other two.
    assert result.model.brier == pytest.approx(2.0 / 3.0)
    assert result.model.base_rate == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------------------
# Anti-lookahead
# --------------------------------------------------------------------------------------
def test_build_observations_never_hands_the_model_future_information():
    """Inspect the actual call arguments. The settlement must not be reachable."""
    seen: list[dict] = []

    def spy(*, ts, spot, strike, minutes_to_close):
        seen.append(
            {"ts": ts, "spot": spot, "strike": strike, "minutes_to_close": minutes_to_close}
        )
        return 0.5

    close = T0
    records = [
        LadderRecord(
            ts=close - timedelta(minutes=m),
            event_ticker="EV",
            ticker="EV-T63800",
            strike=63_800.0,
            spot=SPOT,
            market_mid=0.5,
            close_time=close,
        )
        for m in (30.0, 10.0, 1.0)
    ]
    obs = build_observations(records, spy)

    assert len(obs) == 3
    assert {k for call in seen for k in call} == {"ts", "spot", "strike", "minutes_to_close"}
    for call in seen:
        assert call["ts"] < close, "the model must never be positioned at or past close"
        assert call["minutes_to_close"] > 0.0
    assert [c["minutes_to_close"] for c in seen] == pytest.approx([30.0, 10.0, 1.0])


def test_build_observations_skips_records_at_or_past_close():
    close = T0
    records = [
        LadderRecord(close, "EV", "EV-T1", 1.0, SPOT, 0.5, close),
        LadderRecord(close + timedelta(minutes=5), "EV", "EV-T1", 1.0, SPOT, 0.5, close),
        LadderRecord(close - timedelta(minutes=5), "EV", "EV-T1", 1.0, SPOT, 0.5, close),
    ]
    obs = build_observations(records, lambda **kw: 0.5)
    assert len(obs) == 1
    assert obs[0].minutes_to_close == pytest.approx(5.0)


def test_calibrator_drops_observations_at_or_after_close():
    """Second line of defence, for observations built outside `build_observations`."""
    close = T0
    good = Observation(close - timedelta(minutes=5), "A", "A-T1", 1.0, 5.0, 0.9, 0.5, close)
    at_close = Observation(close, "B", "B-T1", 1.0, 0.0, 0.9, 0.5, close)
    after = Observation(close + timedelta(minutes=1), "C", "C-T1", 1.0, -1.0, 0.9, 0.5, close)

    cal = Calibrator(min_bucket_n=1)
    result = cal.score([good, at_close, after], {"A": 2.0, "B": 2.0, "C": 2.0})
    assert result.n_observations == 1
    assert result.n_dropped == 2
    assert result.dropped_reasons["timestamp_at_or_after_close"] == 2


def test_train_cutoff_enforces_an_out_of_sample_score():
    """Everything at or before the cutoff is discarded, so the fit cannot score itself."""
    close = T0 + timedelta(hours=10)
    obs = [
        Observation(
            ts=T0 + timedelta(hours=i),
            event_ticker=f"EV{i}",
            ticker=f"EV{i}-T1",
            strike=1.0,
            minutes_to_close=60.0,
            model_prob=0.7,
            market_prob=0.5,
            close_time=T0 + timedelta(hours=i + 1),
        )
        for i in range(10)
    ]
    settlements = {f"EV{i}": 2.0 for i in range(10)}
    cutoff = T0 + timedelta(hours=4)

    cal = Calibrator(min_bucket_n=1)
    full = cal.score(obs, settlements)
    assert full.n_observations == 10
    assert not full.is_out_of_sample
    assert "IN-SAMPLE" in full.headline()

    split = cal.score(obs, settlements, train_cutoff=cutoff)
    assert split.n_observations == 5  # hours 5..9
    assert split.dropped_reasons["at_or_before_train_cutoff"] == 5
    assert split.is_out_of_sample
    assert split.first_ts == T0 + timedelta(hours=5)
    assert "out-of-sample" in split.headline()


def test_missing_and_bad_settlements_are_reported_not_silently_ignored():
    obs = make_observations([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    settlements = {obs[0].event_ticker: 200.0, obs[1].event_ticker: float("nan")}
    result = Calibrator(min_bucket_n=1).score(obs, settlements)
    assert result.n_observations == 1
    assert result.dropped_reasons == {"bad_settlement": 1, "no_settlement": 1}
    assert result.n_dropped == 2


def test_out_of_range_probabilities_are_dropped_with_a_reason():
    close = T0
    obs = [
        Observation(close - timedelta(minutes=5), "A", "A-T1", 1.0, 5.0, 1.4, 0.5, close),
        Observation(close - timedelta(minutes=5), "B", "B-T1", 1.0, 5.0, 0.5, -0.2, close),
        Observation(close - timedelta(minutes=5), "C", "C-T1", 1.0, 5.0, 0.5, 0.5, close),
    ]
    result = Calibrator(min_bucket_n=1).score(obs, {"A": 2.0, "B": 2.0, "C": 2.0})
    assert result.dropped_reasons == {"model_prob_out_of_range": 1, "market_prob_out_of_range": 1}
    assert result.n_observations == 1


def test_scoring_nothing_raises_rather_than_returning_a_flattering_empty_result():
    obs = make_observations([0.5], [0.5])
    with pytest.raises(ValueError, match="no scorable observations"):
        Calibrator().score(obs, {})


# --------------------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------------------
def test_buckets_partition_the_sample_exactly_once():
    rng = np.random.default_rng(SEED)
    n = 3_000
    p = rng.uniform(0.02, 0.98, size=n)
    y = (rng.uniform(size=n) < p).astype(float)
    minutes = rng.uniform(0.05, 59.0, size=n)

    obs = make_observations(p, p, minutes=minutes.tolist())
    result = Calibrator().score(obs, settlements_for(obs, y))

    assert sum(b.n for b in result.by_minutes_to_close) == n
    assert sum(b.n for b in result.by_price_bucket) == n
    assert [b.low for b in result.by_minutes_to_close] == list(DEFAULT_MINUTE_EDGES[:-1])


def test_edge_ends_up_in_the_bucket_where_it_was_generated():
    """Give the model an edge ONLY in the final minute; the buckets must localise it."""
    rng = np.random.default_rng(SEED)
    n = 6_000
    truth = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(size=n) < truth).astype(float)
    minutes = rng.choice([0.5, 20.0], size=n)

    endgame = minutes < 1.0
    market = np.where(endgame, 0.5 + 0.3 * (truth - 0.5), truth)

    obs = make_observations(truth, market, minutes=minutes.tolist())
    result = Calibrator().score(obs, settlements_for(obs, y))

    by_label = {b.label: b for b in result.by_minutes_to_close}
    assert by_label["0-1min"].skill > 0.10, "edge must show up in the final minute"
    assert by_label["15-30min"].skill == pytest.approx(0.0, abs=1e-9)
    assert all(b.reliable for b in result.by_minutes_to_close if b.n > 0)


def test_price_buckets_use_the_market_mid_not_the_model():
    """Bucketing key must be the benchmark's own price - it is known at ts and neutral."""
    close = T0
    obs = [
        Observation(close - timedelta(minutes=5), f"E{i}", f"E{i}-T1", 1.0, 5.0, 0.95, 0.05, close)
        for i in range(50)
    ]
    result = Calibrator(min_bucket_n=1).score(obs, {f"E{i}": 2.0 for i in range(50)})
    assert [b.label for b in result.by_price_bucket] == ["0-0.1"]
    assert result.by_price_bucket[0].n == 50


def test_thin_buckets_are_flagged_unreliable():
    obs = make_observations([0.5] * 5, [0.5] * 5)
    result = Calibrator(min_bucket_n=30).score(obs, settlements_for(obs, [1, 0, 1, 0, 1]))
    assert all(not b.reliable for b in result.by_minutes_to_close)
    assert result.by_minutes_to_close[0].n == 5


# --------------------------------------------------------------------------------------
# Result plumbing for report/
# --------------------------------------------------------------------------------------
def test_result_serialises_to_plain_types():
    rng = np.random.default_rng(SEED)
    p = rng.uniform(0.05, 0.95, size=500)
    y = (rng.uniform(size=p.size) < p).astype(float)
    obs = make_observations(p, p, minutes=rng.uniform(0.1, 59.0, size=500).tolist())
    result = Calibrator().score(obs, settlements_for(obs, y), train_cutoff=T0 - timedelta(days=1))

    d = result.as_dict()
    assert d["skill"] == 0.0
    assert isinstance(d["train_cutoff"], str)
    assert set(d["model"]) == {"n", "brier", "log_loss", "mean_prob", "base_rate", "bias"}
    assert len(d["model_reliability"]["centers"]) == 10
    assert all("reliable" in b for b in d["by_minutes_to_close"])

    import json

    json.dumps(d)  # must not raise on numpy scalars


def test_reliability_curve_populated_drops_empty_bins():
    obs = make_observations([0.05] * 20, [0.95] * 20)
    result = Calibrator(min_bucket_n=1).score(obs, settlements_for(obs, [1] * 10 + [0] * 10))
    populated = result.model_reliability.populated()
    assert len(populated) == 1
    center, freq, count = populated[0]
    assert center == pytest.approx(0.05)
    assert freq == pytest.approx(0.5)
    assert count == 20


# --------------------------------------------------------------------------------------
# End to end: the Asian pricer versus a point-in-time market maker
# --------------------------------------------------------------------------------------
def simulate_events(n_events: int, rng: np.random.Generator):
    """Simulate KXBTCD hours and quote them from both models.

    Truth: BTC is a random walk and the settlement is the mean of the final 60 one-second
    ticks. Our model prices that correctly. The simulated market maker prices a
    point-in-time digital with std sigma*sqrt(tau), which is the naive mistake the whole
    project is built around, then rounds to the cent as a real book does.
    """
    sigma_sec = SIGMA_MIN * SPOT / math.sqrt(60.0)
    sample_minutes = [45.0, 20.0, 8.0, 3.0, 1.5, 0.5]
    records: list[LadderRecord] = []
    settlements: dict[str, float] = {}

    for e in range(n_events):
        close = T0 + timedelta(hours=e)
        path = SPOT + np.cumsum(rng.normal(0.0, sigma_sec, size=3600))
        settlement = float(path[-TICKS:].mean())
        event = f"KXBTCD-SIM{e:05d}"
        settlements[event] = settlement

        for mtc in sample_minutes:
            idx = 3600 - int(round(mtc * 60.0))
            spot = float(path[idx - 1])
            base = round(spot / 100.0) * 100.0 - 0.01
            for offset in (-200.0, -100.0, 0.0, 100.0, 200.0):
                strike = base + offset
                # Point-in-time market maker: no averaging correction anywhere.
                naive_std = SIGMA_MIN * spot * math.sqrt(mtc)
                mid = float(stats.norm.sf((strike - spot) / naive_std))
                mid = min(max(round(mid, 2), 0.01), 0.99)
                records.append(
                    LadderRecord(
                        ts=close - timedelta(minutes=mtc),
                        event_ticker=event,
                        ticker=f"{event}-T{strike}",
                        strike=strike,
                        spot=spot,
                        market_mid=mid,
                        close_time=close,
                    )
                )
    return records, settlements


def test_asian_pricer_beats_a_point_in_time_market_maker():
    """The project thesis, expressed as a skill score.

    If this ever goes negative, either the pricer or the calibrator is broken - there is
    no third possibility, because the data generating process here IS the Asian settlement
    the pricer assumes.

    Note how SMALL the aggregate is: about +0.011 across the whole ladder, against +0.33
    in the final minute (see the next test). That dilution is the honest headline. Most
    captured observations sit 20+ minutes from close where the averaging correction is
    worth almost nothing, so an aggregate skill score is a bad way to size a position and
    a good way to talk yourself out of a real edge.
    """
    rng = np.random.default_rng(SEED)
    records, settlements = simulate_events(400, rng)

    def asian(*, ts, spot, strike, minutes_to_close):
        return price_above(spot, strike, SIGMA_MIN, minutes_to_close).prob_above

    obs = build_observations(records, asian)
    result = Calibrator().score(obs, settlements)

    assert result.n_observations == len(records)
    assert result.n_events == 400
    assert result.skill > 0.005, result.headline()
    assert result.model.brier < result.market.brier
    assert result.model.log_loss < result.market.log_loss


def test_the_edge_concentrates_in_the_final_minutes():
    """Skill must be largest where the averaging destroys the most variance."""
    rng = np.random.default_rng(SEED)
    records, settlements = simulate_events(400, rng)

    def asian(*, ts, spot, strike, minutes_to_close):
        return price_above(spot, strike, SIGMA_MIN, minutes_to_close).prob_above

    result = Calibrator().score(build_observations(records, asian), settlements)
    by_label = {b.label: b for b in result.by_minutes_to_close}

    endgame = by_label["0-1min"].skill
    far = by_label["30-infmin"].skill
    assert endgame > far, f"endgame {endgame:.4f} should beat far {far:.4f}"
    assert endgame > 0.15, "the final-minute edge is the whole strategy"
    # An hour out, the averaging correction is worth ~0.7% of the variance and our
    # advantage over a point-in-time quote is genuinely nil. Saying so is the point.
    assert abs(far) < 0.01
    # Monotone-ish decay as the window recedes.
    assert by_label["1-5min"].skill > by_label["5-15min"].skill


def test_our_model_is_well_calibrated_where_the_market_maker_is_not():
    """Reliability, not just aggregate score: our curve should hug the diagonal.

    1200 events, not 400, and the reason is worth recording. `simulate_events` draws the
    truth with SIGMA_MIN and we price with the same SIGMA_MIN, so the model here is
    EXACTLY correctly specified and every deviation from the diagonal is sampling noise.
    At 400 events that noise routinely breaches the 0.05 bar below: measured across 8
    seeds the worst max-deviation was 0.056, and the bar held for the committed seed
    purely by luck (2 of 8 seeds failed). The bar was never measuring model quality.

    The noise is bigger than a naive binomial estimate suggests because the 30
    observations of one event (5 strikes x 6 sample times) all resolve against a single
    settlement, so bin counts of ~600 carry far fewer than 600 independent draws.

    Tripling the event count shrinks the noise by ~sqrt(3) and takes the worst
    max-deviation across those same 8 seeds to 0.036 - comfortably inside 0.05, so the
    assertion now fails when the model is wrong rather than when the seed is unkind.
    """
    rng = np.random.default_rng(SEED)
    records, settlements = simulate_events(1200, rng)

    def asian(*, ts, spot, strike, minutes_to_close):
        return price_above(spot, strike, SIGMA_MIN, minutes_to_close).prob_above

    result = Calibrator(bins=10).score(build_observations(records, asian), settlements)

    def max_deviation(curve):
        pts = [(c, f) for c, f, n in curve.populated() if n >= 200]
        assert pts, "need populated bins to judge calibration"
        return max(abs(f - c) for c, f in pts)

    assert max_deviation(result.model_reliability) < 0.05
    assert max_deviation(result.model_reliability) < max_deviation(result.market_reliability)
