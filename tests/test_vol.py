"""Validate every volatility estimator against synthetic data with a KNOWN sigma.

The pattern throughout: generate a path whose true per-minute volatility we chose
ourselves, hand it to the estimator, and assert the estimator gives that number back.
An estimator that cannot recover a sigma it was handed on a silver platter has no
business setting fair values on real money.

Tolerances are set from the estimators' own sampling error, not from whatever made the
test pass. Where an estimator is knowingly biased (Parkinson on discretely sampled bars)
the test asserts the bias exists and is small, rather than pretending it does not.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kalshi_btc.model.pricing import TICKS, annual_to_per_minute, per_minute_to_annual
from kalshi_btc.model.vol import (
    DEFAULT_SIGMA_PER_MINUTE,
    EWMAVol,
    HARRV,
    SeasonalProfile,
    VolClamp,
    VolModel,
    bipower_variation,
    garman_klass_variance,
    jump_ratio,
    log_returns,
    median_realized_variance,
    parkinson_variance,
    realized_variance,
    rogers_satchell_variance,
    settlement_return_variance_offset_minutes,
    yang_zhang_variance,
)

SEED = 20260728
TRUE_SIGMA_MIN = annual_to_per_minute(0.436)  # 0.466%/hour, the measured KXBTCD level
SPOT = 63_800.0


# --------------------------------------------------------------------------------------
# Synthetic path generators
# --------------------------------------------------------------------------------------
def gbm_path(
    n_seconds: int,
    sigma_min: float,
    rng: np.random.Generator,
    *,
    drift_per_minute: float = 0.0,
    spot: float = SPOT,
) -> np.ndarray:
    """One-second GBM price path with a known per-minute log-return volatility."""
    sigma_sec = sigma_min / math.sqrt(60.0)
    mu_sec = drift_per_minute / 60.0
    steps = rng.normal(mu_sec, sigma_sec, size=n_seconds)
    return spot * np.exp(np.cumsum(steps))


def ohlc_bars(path: np.ndarray, seconds_per_bar: int) -> tuple[np.ndarray, ...]:
    """Carve a one-second path into OHLC bars. Bars are contiguous, so there are no gaps."""
    n = (path.size // seconds_per_bar) * seconds_per_bar
    m = path[:n].reshape(-1, seconds_per_bar)
    return m[:, 0], m.max(axis=1), m.min(axis=1), m[:, -1]


def settlement_series(
    n_hours: int, sigma_min: float, rng: np.random.Generator, *, spot: float = SPOT
) -> tuple[list[datetime], np.ndarray]:
    """Simulate hourly KXBTCD settlements: the mean of 60 one-second ticks before close.

    Only the window ticks and the jump between windows are simulated - for a random walk
    that is exact, and it avoids materialising millions of intermediate seconds.
    """
    sigma_sec = sigma_min / math.sqrt(60.0)
    gap_seconds = 3600 - (TICKS - 1)  # last tick of hour h-1 -> first tick of hour h

    gaps = rng.normal(0.0, sigma_sec * math.sqrt(gap_seconds), size=n_hours)
    within = rng.normal(0.0, sigma_sec, size=(n_hours, TICKS - 1))

    # Value at each window's FIRST tick, chained through the gaps.
    window_totals = within.sum(axis=1)
    ends = np.cumsum(gaps + window_totals)
    starts = ends - window_totals

    logp = starts[:, None] + np.concatenate(
        [np.zeros((n_hours, 1)), np.cumsum(within, axis=1)], axis=1
    )
    settlements = (spot * np.exp(logp)).mean(axis=1)

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    times = [t0 + timedelta(hours=h) for h in range(n_hours)]
    return times, settlements


# --------------------------------------------------------------------------------------
# The settlement-averaging variance correction
# --------------------------------------------------------------------------------------
def test_settlement_offset_matches_exact_tick_algebra():
    """Var(B - A) = D - offset, computed directly from the tick covariance matrix."""
    offset = settlement_return_variance_offset_minutes()
    # Window A ends at t=60s, window B at t=360s, i.e. D = 5 minutes apart.
    ta = np.arange(1.0, 61.0)
    tb = ta + 300.0
    var_a = np.minimum.outer(ta, ta).sum() / TICKS**2
    var_b = np.minimum.outer(tb, tb).sum() / TICKS**2
    cov = np.minimum.outer(ta, tb).sum() / TICKS**2
    exact_seconds = var_a + var_b - 2.0 * cov
    assert (300.0 - exact_seconds) / 60.0 == pytest.approx(offset, rel=1e-9)
    assert offset * 60.0 == pytest.approx(19.994, abs=0.01)


def test_settlement_offset_matches_monte_carlo():
    """The correction is real: simulated settlement returns lose ~20 seconds of variance."""
    rng = np.random.default_rng(SEED)
    times, s = settlement_series(120_000, TRUE_SIGMA_MIN, rng)
    r = log_returns(s)
    offset = settlement_return_variance_offset_minutes()
    # Variance per unit of EFFECTIVE time should equal sigma^2 once corrected.
    corrected = float(np.mean(r**2)) / (60.0 - offset)
    naive = float(np.mean(r**2)) / 60.0
    assert math.sqrt(corrected) == pytest.approx(TRUE_SIGMA_MIN, rel=0.01)
    # The naive divisor understates sigma, which is the dangerous direction.
    assert math.sqrt(naive) < math.sqrt(corrected)


# --------------------------------------------------------------------------------------
# Realized variance
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("interval_minutes", [1.0, 5.0, 15.0])
def test_realized_variance_recovers_known_sigma(interval_minutes):
    """Sampling frequency must not change the recovered per-minute sigma."""
    rng = np.random.default_rng(SEED)
    step = int(interval_minutes * 60)
    path = gbm_path(step * 20_000, TRUE_SIGMA_MIN, rng)
    r = log_returns(path[::step])
    assert math.sqrt(realized_variance(r, interval_minutes)) == pytest.approx(
        TRUE_SIGMA_MIN, rel=0.02
    )


def test_realized_variance_is_blind_to_drift_at_our_horizon():
    """A drift big enough to matter over a day is invisible over a minute.

    E[r^2] = sigma^2 + mu^2, so the drift contamination is (mu/sigma)^2. At 2e-5/minute
    (2.9% per day, a hard trending day) that ratio is 0.04 and the inflation is 0.16% -
    below the sampling noise. This is exactly why `realized_variance` does not demean:
    the correction is smaller than the degree of freedom it would cost.
    """
    rng = np.random.default_rng(SEED)
    trend = gbm_path(600_000, TRUE_SIGMA_MIN, rng, drift_per_minute=2e-5)
    rng = np.random.default_rng(SEED)
    flat = gbm_path(600_000, TRUE_SIGMA_MIN, rng, drift_per_minute=0.0)

    with_drift = math.sqrt(realized_variance(log_returns(trend[::60]), 1.0))
    without = math.sqrt(realized_variance(log_returns(flat[::60]), 1.0))
    assert with_drift == pytest.approx(TRUE_SIGMA_MIN, rel=0.03)
    assert with_drift == pytest.approx(without, rel=0.005)


# --------------------------------------------------------------------------------------
# Range-based estimators
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "estimator",
    [parkinson_variance, garman_klass_variance, rogers_satchell_variance, yang_zhang_variance],
)
def test_range_estimators_recover_known_sigma(estimator):
    """All four must return the sigma we generated, within their discretisation bias.

    The bias is real and it is DOWNWARD: with 900 one-second observations per bar the
    sampled high/low still understates the continuous extremes by 2-4%. That is a
    property of range estimators on sampled data, not a bug, so the test asserts the
    magnitude rather than pretending it is zero.
    """
    rng = np.random.default_rng(SEED)
    bar_seconds, n_bars = 900, 3_000  # 15-minute bars
    path = gbm_path(bar_seconds * n_bars, TRUE_SIGMA_MIN, rng)
    o, h, lo, c = ohlc_bars(path, bar_seconds)
    bar_minutes = bar_seconds / 60.0

    if estimator is parkinson_variance:
        var = estimator(h, lo, bar_minutes)
    else:
        var = estimator(o, h, lo, c, bar_minutes)
    assert math.sqrt(var) == pytest.approx(TRUE_SIGMA_MIN, rel=0.05)


@pytest.mark.parametrize("bar_seconds", [120, 3600])
def test_range_estimator_discretisation_bias_is_downward_and_shrinks(bar_seconds):
    """Finer sampling within the bar must move the estimate UP toward the truth."""
    rng = np.random.default_rng(SEED)
    path = gbm_path(3600 * 900, TRUE_SIGMA_MIN, rng)
    o, h, lo, c = ohlc_bars(path, bar_seconds)
    ratio = math.sqrt(yang_zhang_variance(o, h, lo, c, bar_seconds / 60.0)) / TRUE_SIGMA_MIN
    assert ratio < 1.02, "range estimators do not overshoot on sampled data"
    # 120s bars see 120 sub-steps; 3600s bars see 3600 and land closer to the truth.
    assert ratio > (0.90 if bar_seconds == 120 else 0.95)


def test_range_estimators_are_more_efficient_than_close_to_close():
    """The whole point of using OHLC: far less sampling noise for the same bars.

    Repeated independent samples; we compare the spread of each estimator's estimates.
    """
    rng = np.random.default_rng(SEED)
    bar_seconds, n_bars, trials = 120, 60, 200

    cc, park, yz = [], [], []
    for _ in range(trials):
        path = gbm_path(bar_seconds * n_bars, TRUE_SIGMA_MIN, rng)
        o, h, lo, c = ohlc_bars(path, bar_seconds)
        bm = bar_seconds / 60.0
        cc.append(realized_variance(log_returns(c), bm))
        park.append(parkinson_variance(h, lo, bm))
        yz.append(yang_zhang_variance(o, h, lo, c, bm))

    rel = lambda xs: float(np.std(xs) / np.mean(xs))  # noqa: E731
    assert rel(park) < rel(cc) / 1.5, "Parkinson should be clearly tighter than close-to-close"
    assert rel(yz) < rel(cc) / 1.5, "Yang-Zhang should be clearly tighter than close-to-close"


def test_garman_klass_is_inflated_by_drift_but_rogers_satchell_is_not():
    """The reason Rogers-Satchell exists, demonstrated on a hard trend.

    Drift is set so that mu*T is 2.5x sigma*sqrt(T) over a 10-minute bar - a violent but
    not absurd move. Close-to-close more than doubles, Garman-Klass inflates ~28%, and
    Rogers-Satchell and Yang-Zhang barely move.
    """
    rng = np.random.default_rng(SEED)
    bar_seconds, n_bars = 600, 4_000
    path = gbm_path(bar_seconds * n_bars, TRUE_SIGMA_MIN, rng, drift_per_minute=4e-4)
    o, h, lo, c = ohlc_bars(path, bar_seconds)
    bm = bar_seconds / 60.0

    gk = math.sqrt(garman_klass_variance(o, h, lo, c, bm))
    rs = math.sqrt(rogers_satchell_variance(o, h, lo, c, bm))
    yz = math.sqrt(yang_zhang_variance(o, h, lo, c, bm))
    cc = math.sqrt(realized_variance(log_returns(c), bm))

    assert rs == pytest.approx(TRUE_SIGMA_MIN, rel=0.08)
    assert yz == pytest.approx(TRUE_SIGMA_MIN, rel=0.08)
    assert gk > 1.2 * rs, "Garman-Klass must be inflated by the drift Rogers-Satchell ignores"
    assert cc > 2.0 * rs, "close-to-close is the worst of all under a trend"


def test_yang_zhang_captures_gap_variance_the_others_miss():
    """Insert real gaps between bars. Only Yang-Zhang has a term for them."""
    rng = np.random.default_rng(SEED)
    bar_seconds, n_bars = 300, 3_000
    path = gbm_path(bar_seconds * n_bars, TRUE_SIGMA_MIN, rng)
    o, h, lo, c = ohlc_bars(path, bar_seconds)
    bm = bar_seconds / 60.0

    # Shift each bar by an independent gap: total variance per bar roughly doubles.
    gap_sigma = TRUE_SIGMA_MIN * math.sqrt(bm)
    shifts = np.exp(np.cumsum(rng.normal(0.0, gap_sigma, size=o.size)))
    o, h, lo, c = o * shifts, h * shifts, lo * shifts, c * shifts

    yz = math.sqrt(yang_zhang_variance(o, h, lo, c, bm))
    rs = math.sqrt(rogers_satchell_variance(o, h, lo, c, bm))
    expected = TRUE_SIGMA_MIN * math.sqrt(2.0)  # gap variance == within-bar variance

    assert yz == pytest.approx(expected, rel=0.10)
    assert rs < yz, "Rogers-Satchell is gap-blind and must understate the total"


def test_ohlc_validation_rejects_ragged_and_non_positive_input():
    with pytest.raises(ValueError, match="same length"):
        rogers_satchell_variance([1.0, 2.0], [2.0, 3.0, 4.0], [0.5, 1.0], [1.5, 2.5])
    with pytest.raises(ValueError, match="non-positive"):
        parkinson_variance([1.0, 0.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="at least 3 bars"):
        yang_zhang_variance([1, 1], [2, 2], [0.5, 0.5], [1.5, 1.5])


# --------------------------------------------------------------------------------------
# Jump-robust estimators
# --------------------------------------------------------------------------------------
def test_bipower_and_medrv_recover_known_sigma_without_jumps():
    """Both must be unbiased on a pure diffusion, or their jump robustness is worthless."""
    rng = np.random.default_rng(SEED)
    r = log_returns(gbm_path(60 * 40_000, TRUE_SIGMA_MIN, rng)[::60])
    assert math.sqrt(bipower_variation(r, 1.0)) == pytest.approx(TRUE_SIGMA_MIN, rel=0.03)
    assert math.sqrt(median_realized_variance(r, 1.0)) == pytest.approx(TRUE_SIGMA_MIN, rel=0.03)


def test_jumps_inflate_rv_but_not_the_jump_robust_estimators():
    """A handful of jumps must not be allowed to reprice the whole ladder.

    This is the concrete failure mode: one liquidation cascade doubles close-to-close RV,
    the pricer widens every strike toward 50 cents, and the away-from-the-money edge - the
    only edge that survives the fee - disappears for the rest of the session.
    """
    rng = np.random.default_rng(SEED)
    r = log_returns(gbm_path(60 * 40_000, TRUE_SIGMA_MIN, rng)[::60])

    contaminated = r.copy()
    idx = rng.choice(r.size, size=r.size // 400, replace=False)  # 0.25% of minutes
    contaminated[idx] += rng.choice([-1.0, 1.0], size=idx.size) * 15.0 * TRUE_SIGMA_MIN

    rv = math.sqrt(realized_variance(contaminated, 1.0))
    bv = math.sqrt(bipower_variation(contaminated, 1.0))
    medrv = math.sqrt(median_realized_variance(contaminated, 1.0))

    assert rv > TRUE_SIGMA_MIN * 1.15, "the jumps must actually contaminate RV"
    assert bv == pytest.approx(TRUE_SIGMA_MIN, rel=0.08)
    assert medrv == pytest.approx(TRUE_SIGMA_MIN, rel=0.05)
    # MedRV discards an isolated jump outright; bipower only damps it.
    assert abs(medrv - TRUE_SIGMA_MIN) < abs(bv - TRUE_SIGMA_MIN)
    assert jump_ratio(contaminated, 1.0) > 0.25


def test_medrv_handles_stale_prints_better_than_bipower():
    """A stalled BRTI feed prints the same value repeatedly: a run of ZERO returns.

    Zeros are bipower's other weak spot - each one annihilates the two adjacent products
    it appears in. Both estimators are biased down (correctly: there genuinely is less
    variance in a stalled tape), but bipower overshoots the correction by ~2x.
    """
    rng = np.random.default_rng(SEED)
    r = log_returns(gbm_path(60 * 40_000, TRUE_SIGMA_MIN, rng)[::60])

    stale_fraction = 0.30
    contaminated = r.copy()
    idx = rng.choice(r.size, size=int(r.size * stale_fraction), replace=False)
    contaminated[idx] = 0.0

    # With a fraction f of returns zeroed the remaining variance is (1-f)*sigma^2.
    target = TRUE_SIGMA_MIN * math.sqrt(1.0 - stale_fraction)
    bv = math.sqrt(bipower_variation(contaminated, 1.0))
    medrv = math.sqrt(median_realized_variance(contaminated, 1.0))

    assert bv < target * 0.90, "bipower must visibly undershoot on stale prints"
    assert medrv > bv
    assert abs(medrv - target) < abs(bv - target)


def test_jump_ratio_is_near_zero_on_a_clean_diffusion():
    rng = np.random.default_rng(SEED)
    r = log_returns(gbm_path(60 * 40_000, TRUE_SIGMA_MIN, rng)[::60])
    assert jump_ratio(r, 1.0) < 0.1


# --------------------------------------------------------------------------------------
# HAR-RV
# --------------------------------------------------------------------------------------
def test_har_recovers_a_constant_variance_level():
    """Flat truth in, flat truth out - including the log-space retransformation."""
    rng = np.random.default_rng(SEED)
    true_var = TRUE_SIGMA_MIN**2
    # Chi-square(1) noise: exactly what a series of single squared returns looks like.
    series = true_var * rng.chisquare(df=1, size=4_000)
    har = HARRV(lags=(1, 24, 168)).fit(series)
    assert har.forecast(1) == pytest.approx(true_var, rel=0.10)
    assert har.forecast(24) == pytest.approx(true_var, rel=0.10)


def test_har_log_fit_is_not_destroyed_by_retransformation_bias():
    """The Duan smearing correction, in isolation.

    Applying the textbook lognormal factor exp(s^2/2) to log-chi-square residuals
    overstates variance by ~3x. If this test starts failing, someone has 'simplified'
    `_invlink` back to the wrong formula.
    """
    rng = np.random.default_rng(SEED)
    true_var = TRUE_SIGMA_MIN**2
    series = true_var * rng.chisquare(df=1, size=6_000)
    har = HARRV(lags=(1, 5, 22)).fit(series)

    wrong = math.exp(0.5 * har.residual_variance)
    assert wrong > 2.0 * har.smearing, "the lognormal factor should be visibly too large here"
    assert har.forecast(1) == pytest.approx(true_var, rel=0.10)


def test_har_tracks_a_persistent_volatility_regime():
    """A genuinely autocorrelated variance series must produce a state-dependent forecast."""
    rng = np.random.default_rng(SEED)
    n = 4_000
    # AR(1) in log variance - the standard stylised model of volatility persistence.
    logv = np.zeros(n)
    for t in range(1, n):
        logv[t] = 0.97 * logv[t - 1] + rng.normal(0.0, 0.15)
    series = (TRUE_SIGMA_MIN**2) * np.exp(logv)

    har = HARRV(lags=(1, 24, 168)).fit(series)
    assert har.r_squared > 0.3, "HAR must explain a persistent series"

    calm = series[: 168 + 5].copy()
    calm[-50:] = series.mean() * 0.25
    hot = calm.copy()
    hot[-50:] = series.mean() * 4.0
    assert har.forecast(1, hot) > har.forecast(1, calm)


def test_har_multi_step_forecast_mean_reverts():
    """Iterated forecasts should decay from the current state toward the unconditional mean."""
    rng = np.random.default_rng(SEED)
    n = 3_000
    logv = np.zeros(n)
    for t in range(1, n):
        logv[t] = 0.9 * logv[t - 1] + rng.normal(0.0, 0.2)
    series = (TRUE_SIGMA_MIN**2) * np.exp(logv)
    har = HARRV(lags=(1, 24, 168)).fit(series)

    hot = series[: 168 + 5].copy()
    hot[-30:] = series.mean() * 6.0
    near = har.forecast(1, hot)
    far = har.forecast(48, hot)
    assert far < near, "a long-horizon average must sit below a hot one-step forecast"


def test_har_forecast_is_always_positive():
    rng = np.random.default_rng(SEED)
    series = np.abs(rng.normal(0.0, 1e-8, size=1_000))
    har = HARRV(lags=(1, 5, 22)).fit(series)
    assert har.forecast(1) > 0.0


def test_har_refuses_to_fit_on_insufficient_history():
    har = HARRV(lags=(1, 24, 168))
    with pytest.raises(ValueError, match="needs >="):
        har.fit(np.ones(50))
    with pytest.raises(RuntimeError, match="before fit"):
        har.forecast(1)


def test_har_design_matrix_is_strictly_causal():
    """Row t must be built only from observations <= t; the target is t+1.

    A single off-by-one here silently leaks the answer into the regressors and makes
    every downstream backtest look brilliant.
    """
    har = HARRV(lags=(1, 2, 3), use_log=False)
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x, y = har._design(series)
    # start index = max(lag)-1 = 2, so the first row uses indices 0..2 and targets index 3.
    assert x[0].tolist() == [1.0, 3.0, 2.5, 2.0]  # [const, rv[2], mean(rv[1:3]), mean(rv[0:3])]
    assert y[0] == 4.0
    assert y.tolist() == [4.0, 5.0, 6.0]


# --------------------------------------------------------------------------------------
# EWMA
# --------------------------------------------------------------------------------------
def test_ewma_recovers_a_constant_variance_level():
    rng = np.random.default_rng(SEED)
    true_var = TRUE_SIGMA_MIN**2
    series = true_var * rng.chisquare(df=1, size=20_000)
    ewma = EWMAVol(lam=0.99).fit(series)
    assert math.sqrt(ewma.forecast()) == pytest.approx(TRUE_SIGMA_MIN, rel=0.15)


def test_ewma_reacts_to_a_regime_shift_faster_than_the_sample_mean():
    """The reason EWMA is in the blend at all."""
    rng = np.random.default_rng(SEED)
    calm = (TRUE_SIGMA_MIN**2) * rng.chisquare(df=1, size=2_000)
    storm = (3.0 * TRUE_SIGMA_MIN) ** 2 * rng.chisquare(df=1, size=60)
    series = np.concatenate([calm, storm])

    ewma = EWMAVol(lam=0.94).fit(series)
    assert ewma.forecast() > 3.0 * float(np.mean(calm))
    assert ewma.forecast() > float(np.mean(series))


def test_ewma_forecast_is_flat_in_horizon():
    """IGARCH: shocks never decay. Documented behaviour, asserted so it stays documented."""
    ewma = EWMAVol().fit(np.full(100, 4e-7))
    assert ewma.forecast(1) == ewma.forecast(50)


def test_ewma_rejects_bad_input():
    with pytest.raises(ValueError, match="lambda"):
        EWMAVol(lam=1.0)
    with pytest.raises(ValueError, match="negative"):
        EWMAVol().fit([1.0, -1.0])
    with pytest.raises(RuntimeError, match="before fit"):
        EWMAVol().forecast()


def test_ewma_update_matches_batch_fit():
    ewma = EWMAVol(lam=0.9).fit(np.full(200, 1e-7))
    incremental = ewma.variance
    for _ in range(5):
        incremental = ewma.update(4e-7)
    assert incremental == pytest.approx(ewma.forecast())
    assert incremental > 1e-7


# --------------------------------------------------------------------------------------
# Seasonality
# --------------------------------------------------------------------------------------
def test_seasonal_profile_recovers_a_known_diurnal_pattern():
    """Plant a 2x bump at 14:00Z and a 0.5x trough at 04:00Z; get them back."""
    rng = np.random.default_rng(SEED)
    n_hours = 24 * 400
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    times = [t0 + timedelta(hours=h) for h in range(n_hours)]

    true_factor = np.ones(24)
    true_factor[14] = 2.0
    true_factor[4] = 0.5
    true_factor /= math.exp(float(np.mean(np.log(true_factor))))  # geometric mean 1

    hours = np.array([t.hour for t in times])
    sigmas = TRUE_SIGMA_MIN * true_factor[hours] * np.abs(rng.normal(size=n_hours))

    prof = SeasonalProfile().fit(times, sigmas)
    assert prof.hour_factors[14] == pytest.approx(true_factor[14], rel=0.12)
    assert prof.hour_factors[4] == pytest.approx(true_factor[4], rel=0.12)
    assert int(np.argmax(prof.hour_factors)) == 14
    assert int(np.argmin(prof.hour_factors)) == 4


def test_seasonal_profile_recovers_a_weekend_effect():
    rng = np.random.default_rng(SEED)
    n_hours = 24 * 500
    t0 = datetime(2026, 1, 5, tzinfo=UTC)  # a Monday
    times = [t0 + timedelta(hours=h) for h in range(n_hours)]
    dows = np.array([t.weekday() for t in times])

    quiet = np.isin(dows, (5, 6))  # Saturday, Sunday
    sigmas = TRUE_SIGMA_MIN * np.where(quiet, 0.5, 1.0) * np.abs(rng.normal(size=n_hours))

    prof = SeasonalProfile().fit(times, sigmas)
    weekend = float(np.mean(prof.dow_factors[5:]))
    weekday = float(np.mean(prof.dow_factors[:5]))
    assert weekend < weekday * 0.75


def test_seasonal_profile_is_a_pure_redistribution():
    """Applying the profile must not change the overall level of sigma - only its shape."""
    rng = np.random.default_rng(SEED)
    n_hours = 24 * 200
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    times = [t0 + timedelta(hours=h) for h in range(n_hours)]
    hours = np.array([t.hour for t in times])
    shape = 1.0 + 0.8 * np.sin(np.arange(24) / 24.0 * 2.0 * math.pi)
    sigmas = TRUE_SIGMA_MIN * shape[hours] * np.abs(rng.normal(size=n_hours))

    prof = SeasonalProfile().fit(times, sigmas)
    all_factors = np.array([prof.factor(t) for t in times])
    assert float(np.exp(np.mean(np.log(all_factors)))) == pytest.approx(1.0, rel=0.02)


def test_deseasonalize_reseasonalize_roundtrip():
    rng = np.random.default_rng(SEED)
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    times = [t0 + timedelta(hours=h) for h in range(24 * 100)]
    sigmas = TRUE_SIGMA_MIN * np.abs(rng.normal(size=len(times)))
    prof = SeasonalProfile().fit(times, sigmas)
    back = prof.reseasonalize(times, prof.deseasonalize(times, sigmas))
    assert np.allclose(back, sigmas)


def test_seasonal_thin_cells_are_shrunk_toward_neutral():
    """Two observations in an hour must not brand that hour as 5x volatile forever."""
    t0 = datetime(2026, 1, 1, 9, tzinfo=UTC)
    times = [t0, t0 + timedelta(hours=1)] + [
        datetime(2026, 1, 2, tzinfo=UTC) + timedelta(hours=h) for h in range(24 * 30)
    ]
    sigmas = [50.0 * TRUE_SIGMA_MIN, 50.0 * TRUE_SIGMA_MIN] + [TRUE_SIGMA_MIN] * (24 * 30)
    prof = SeasonalProfile(prior_count=20.0).fit(times, np.array(sigmas))
    # 9:00 has one extreme extra observation among ~31; shrinkage must keep it sane.
    assert prof.hour_factors[9] < 3.0


def test_naive_timestamps_are_treated_as_utc():
    naive = datetime(2026, 1, 1, 14, 0, 0)
    aware = datetime(2026, 1, 1, 14, 0, 0, tzinfo=UTC)
    prof = SeasonalProfile()
    prof.hour_factors[14] = 1.7
    assert prof.factor(naive) == prof.factor(aware) == pytest.approx(1.7)


# --------------------------------------------------------------------------------------
# The VolModel facade
# --------------------------------------------------------------------------------------
def test_vol_model_recovers_sigma_from_the_settlement_series():
    """THE headline test: calibrate on realised settlements, get the true sigma back.

    The settlements are generated with the real KXBTCD structure - each one is the mean
    of 60 one-second ticks ending at the close - so this exercises the averaging
    correction end to end, not just the forecasters.
    """
    rng = np.random.default_rng(SEED)
    times, settlements = settlement_series(6_000, TRUE_SIGMA_MIN, rng)

    model = VolModel().fit_settlements(times, settlements)
    assert model.n_returns == 5_999
    assert model.unconditional_sigma == pytest.approx(TRUE_SIGMA_MIN, rel=0.03)

    sigma = model.sigma_per_minute(times[-1] + timedelta(hours=1))
    assert sigma == pytest.approx(TRUE_SIGMA_MIN, rel=0.15)
    assert per_minute_to_annual(sigma) == pytest.approx(0.436, rel=0.15)


def test_vol_model_is_not_fooled_by_the_averaging_correction():
    """Omitting the correction would bias sigma DOWN. Confirm we are on the right side."""
    rng = np.random.default_rng(SEED)
    times, settlements = settlement_series(20_000, TRUE_SIGMA_MIN, rng)
    model = VolModel().fit_settlements(times, settlements)

    r = log_returns(settlements)
    naive = math.sqrt(float(np.mean(r**2)) / 60.0)  # no averaging correction
    assert naive < TRUE_SIGMA_MIN
    assert abs(model.unconditional_sigma - TRUE_SIGMA_MIN) < abs(naive - TRUE_SIGMA_MIN)


def test_vol_model_tracks_a_settlement_regime_shift():
    """Double the true vol for the recent stretch; the forecast must follow."""
    rng = np.random.default_rng(SEED)
    calm_times, calm = settlement_series(2_000, TRUE_SIGMA_MIN, rng)
    hot_times, hot = settlement_series(400, 3.0 * TRUE_SIGMA_MIN, rng, spot=float(calm[-1]))
    shift = calm_times[-1] + timedelta(hours=1) - hot_times[0]
    times = calm_times + [t + shift for t in hot_times]
    settlements = np.concatenate([calm, hot])

    model = VolModel().fit_settlements(times, settlements)
    quiet = VolModel().fit_settlements(calm_times, calm)
    when = times[-1] + timedelta(hours=1)
    assert model.sigma_per_minute(when) > 1.5 * quiet.sigma_per_minute(when)


def test_vol_model_applies_the_seasonal_factor_at_query_time():
    rng = np.random.default_rng(SEED)
    n = 4_000
    times, settlements = settlement_series(n, TRUE_SIGMA_MIN, rng)
    # Re-scale settlement returns so 14:00Z closes are 2.5x as volatile.
    r = log_returns(settlements)
    hours = np.array([t.hour for t in times[1:]])
    r = r * np.where(hours == 14, 2.5, 1.0)
    settlements = settlements[0] * np.exp(np.concatenate([[0.0], np.cumsum(r)]))

    model = VolModel().fit_settlements(times, settlements)
    base = datetime(2026, 6, 1, tzinfo=UTC)
    assert model.seasonal.factor(base.replace(hour=14)) > 1.5
    assert model.sigma_per_minute(base.replace(hour=14)) > model.sigma_per_minute(
        base.replace(hour=3)
    )


def test_vol_model_falls_back_to_ewma_when_har_cannot_fit():
    """A short capture must still produce a usable sigma, not an exception."""
    rng = np.random.default_rng(SEED)
    times, settlements = settlement_series(60, TRUE_SIGMA_MIN, rng)
    model = VolModel().fit_settlements(times, settlements)
    assert not model.har.is_fitted
    assert model.ewma.is_fitted
    fc = model.forecast(times[-1])
    assert fc.har_sigma is None
    assert fc.blend_weight == 0.0
    assert fc.sigma_per_minute > 0.0


def test_vol_model_drops_capture_gaps_instead_of_stretching_them():
    """A three-day hole is an outage, not a three-day return."""
    rng = np.random.default_rng(SEED)
    times, settlements = settlement_series(500, TRUE_SIGMA_MIN, rng)
    times = list(times)
    # Punch a 5-day hole in the middle of the series.
    for i in range(250, len(times)):
        times[i] = times[i] + timedelta(days=5)

    model = VolModel().fit_settlements(times, settlements, max_gap_minutes=180.0)
    assert model.n_returns == 498, "exactly the one straddling return should be dropped"
    assert model.unconditional_sigma == pytest.approx(TRUE_SIGMA_MIN, rel=0.12)


def test_unfitted_model_returns_the_fallback_sigma():
    model = VolModel()
    fc = model.forecast(datetime(2026, 5, 1, tzinfo=UTC))
    assert fc.sigma_per_minute == pytest.approx(DEFAULT_SIGMA_PER_MINUTE)
    assert fc.har_sigma is None and fc.ewma_sigma is None
    assert not fc.clamped


def test_constant_model_is_constant():
    model = VolModel.constant(7e-4)
    t = datetime(2026, 5, 1, tzinfo=UTC)
    assert model.sigma_per_minute(t) == pytest.approx(7e-4)
    assert model.sigma_per_minute(t + timedelta(hours=9)) == pytest.approx(7e-4)


# --------------------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [(1e-9, 1e-4), (1.0, 2.5e-3), (float("nan"), 1e-4), (-1.0, 1e-4), (5e-4, 5e-4)],
)
def test_clamp_band(raw, expected):
    clamp = VolClamp()
    value, clamped = clamp.apply(raw)
    assert value == pytest.approx(expected)
    assert clamped is (raw != expected)


def test_clamping_is_visible_in_the_forecast_and_the_counter():
    """A clamped forecast must announce itself - a silent clamp is a silent mispricing."""
    rng = np.random.default_rng(SEED)
    times, settlements = settlement_series(400, TRUE_SIGMA_MIN, rng)
    model = VolModel(clamp=VolClamp(low=1e-3, high=2e-3))
    model.fit_settlements(times, settlements)

    fc = model.forecast(times[-1])
    assert fc.clamped is True
    assert fc.sigma_per_minute == pytest.approx(1e-3)
    assert fc.raw_sigma < fc.sigma_per_minute, "the raw value must stay visible"
    assert model.clamp_hits == 1
    assert "CLAMPED" in fc.describe()


def test_clamp_describe_reports_annualised_bounds():
    text = VolClamp().describe()
    assert "annualised" in text
    assert "per minute" in text


def test_forecast_describe_is_informative():
    rng = np.random.default_rng(SEED)
    times, settlements = settlement_series(1_000, TRUE_SIGMA_MIN, rng)
    model = VolModel().fit_settlements(times, settlements)
    text = model.forecast(times[-1]).describe()
    assert "sigma=" in text and "ann" in text and "seasonal=" in text
    assert "VolModel" in model.describe()


def test_blend_weight_is_validated():
    with pytest.raises(ValueError, match="blend_weight"):
        VolModel(blend_weight=1.5)


def test_fit_settlements_rejects_degenerate_input():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    times = [t0, t0 + timedelta(hours=1)]
    with pytest.raises(ValueError, match="same length"):
        VolModel().fit_settlements(times, [1.0])
    with pytest.raises(ValueError, match="non-positive"):
        VolModel().fit_settlements(times, [1.0, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        VolModel().fit_settlements([t0], [1.0])
    with pytest.raises(ValueError, match="no usable"):
        # Every pair is inside the averaging window, so no return has a valid horizon.
        VolModel().fit_settlements(
            [t0 + timedelta(seconds=10 * i) for i in range(4)], [1.0, 1.1, 1.0, 1.1]
        )
