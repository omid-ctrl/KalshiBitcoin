"""Validate the Asian-settlement pricer against Monte Carlo simulation.

These tests are the foundation of the whole project: if the settlement variance model
is wrong, every fair value the bot computes is wrong. They simulate the actual discrete
tick structure (60 one-second BRTI samples ending exactly at close) rather than trusting
a closed form.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kalshi_btc.model.pricing import (
    TICKS,
    annual_to_per_minute,
    per_minute_to_annual,
    price_above,
    price_above_in_window,
    residual_std_in_window,
    settlement_std_dollars,
)

SEED = 12345


def simulate_settlements(
    spot: float, sigma_min_frac: float, minutes_to_close: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate the settlement average: mean of the final 60 one-second ticks."""
    sigma_sec = sigma_min_frac * spot / math.sqrt(60.0)
    total_secs = int(round(minutes_to_close * 60))
    steps = rng.normal(0.0, sigma_sec, size=(n, total_secs))
    path = spot + np.cumsum(steps, axis=1)
    # Ticks land at 59, 58, ..., 0 seconds before close -> the final 60 samples.
    return path[:, -TICKS:].mean(axis=1)


def test_vol_unit_roundtrip():
    assert per_minute_to_annual(annual_to_per_minute(0.368)) == pytest.approx(0.368)


@pytest.mark.parametrize("minutes", [2.0, 5.0, 20.0, 60.0])
def test_settlement_std_matches_monte_carlo(minutes):
    """The exact discrete variance formula must match simulation."""
    rng = np.random.default_rng(SEED)
    spot, sigma = 63_800.0, annual_to_per_minute(0.368)
    sims = simulate_settlements(spot, sigma, minutes, 60_000, rng)
    assert sims.std(ddof=1) == pytest.approx(
        settlement_std_dollars(sigma, spot, minutes), rel=0.03
    )


def test_averaging_reduces_variance_versus_point_in_time():
    """Var(settlement) ~= sigma^2 * (tau - 2/3), NOT sigma^2 * tau.

    This is the always-on edge: the average shaves ~40 seconds of variance off every
    quote for the entire hour.
    """
    spot, sigma, tau = 63_800.0, annual_to_per_minute(0.368), 20.0
    asian = settlement_std_dollars(sigma, spot, tau)
    point_in_time = sigma * spot * math.sqrt(tau)
    assert asian < point_in_time
    # Variance-matched horizon should land near tau - 2/3 (exact discrete value differs
    # slightly from the continuous 1/3 factor).
    tau_eff = (asian / (sigma * spot)) ** 2
    assert tau_eff == pytest.approx(tau - 2 / 3, abs=0.02)


@pytest.mark.parametrize("elapsed,expected_ratio", [(0, 1.73), (30, 3.46), (45, 6.93)])
def test_in_window_uncertainty_collapse(elapsed, expected_ratio):
    """A point-in-time model overstates residual uncertainty by these factors.

    residual_std = sigma * (m/60)^1.5 / sqrt(3)   vs   naive sigma * sqrt(m/60)
    """
    spot, sigma = 63_800.0, annual_to_per_minute(0.368)
    m = TICKS - elapsed
    asian = residual_std_in_window(sigma, spot, elapsed)
    naive = sigma * spot * math.sqrt(m / 60.0)
    assert naive / asian == pytest.approx(expected_ratio, rel=0.05)


def test_in_window_pricing_matches_monte_carlo():
    """The in-window pricer must reproduce simulated settle-above probabilities."""
    rng = np.random.default_rng(SEED)
    spot, sigma = 63_800.0, annual_to_per_minute(0.368)
    known_ticks, n = 30, 80_000
    # Pretend the first 30 ticks all printed at spot.
    known_sum = spot * known_ticks

    sigma_sec = sigma * spot / math.sqrt(60.0)
    m = TICKS - known_ticks
    future = spot + np.cumsum(rng.normal(0.0, sigma_sec, size=(n, m)), axis=1)
    settle = (known_sum + future.sum(axis=1)) / TICKS

    for strike in (spot - 20.0, spot, spot + 10.0, spot + 30.0):
        empirical = float((settle > strike).mean())
        model = price_above_in_window(strike, known_sum, known_ticks, spot, sigma).prob_above
        assert model == pytest.approx(empirical, abs=0.012), f"strike={strike}"


def test_known_ticks_lock_in_the_answer():
    """With all 60 ticks known the price is a certainty, not a probability."""
    spot = 63_800.0
    q_yes = price_above_in_window(63_700.0, spot * TICKS, TICKS, spot, 0.0005)
    q_no = price_above_in_window(63_900.0, spot * TICKS, TICKS, spot, 0.0005)
    assert q_yes.prob_above == 1.0
    assert q_no.prob_above == 0.0
    assert q_yes.residual_std == 0.0


def test_late_spike_cannot_rescue_a_locked_in_average():
    """The averaging dampens LATE spikes - the economic heart of the contract.

    55 ticks are locked in $100 below the strike (deficit 55 x $100 = $5,500). Spot then
    spikes $150 above the strike with only 5 ticks left (surplus at most 5 x $150 = $750).
    A point-in-time model sees spot > strike and says "certain YES". It is certainly NO.

    This is precisely the situation a naive bot gets destroyed in.
    """
    strike, spot_low, spiked = 63_800.0, 63_700.0, 63_950.0
    known_ticks = 55
    q = price_above_in_window(
        strike, spot_low * known_ticks, known_ticks, spiked, annual_to_per_minute(0.368)
    )
    assert q.prob_above < 0.02, "late spike must not rescue a locked-in deficit"


def test_early_spike_with_many_ticks_left_does_carry_the_average():
    """The mirror case: the same spike EARLY in the window genuinely does win.

    29 ticks at a $100 deficit (-$2,900) but 31 ticks left at a $150 surplus (+$4,650).
    Net positive, so this should settle YES with high probability. Together with the test
    above this pins down that the model weights by ticks remaining, not by spot alone.
    """
    strike, spot_low, spiked = 63_800.0, 63_700.0, 63_950.0
    known_ticks = 29
    q = price_above_in_window(
        strike, spot_low * known_ticks, known_ticks, spiked, annual_to_per_minute(0.368)
    )
    assert q.prob_above > 0.95


def test_fat_tails_raise_far_strike_probabilities():
    """Student-t must price tails above the Gaussian - measured excess kurtosis is 3.66."""
    spot, sigma, tau = 63_800.0, annual_to_per_minute(0.368), 30.0
    far = spot + 3.0 * settlement_std_dollars(sigma, spot, tau)
    normal = price_above(spot, far, sigma, tau, dist="normal").prob_above
    fat = price_above(spot, far, sigma, tau, dist="t", df=4.0).prob_above
    assert fat > normal


def test_monotonicity_across_the_ladder():
    """P(above K) must strictly decrease in K - the no-arbitrage condition we checked live."""
    spot, sigma, tau = 63_800.0, annual_to_per_minute(0.368), 25.0
    strikes = np.arange(63_000.0, 64_600.0, 100.0)
    probs = [price_above(spot, float(k), sigma, tau).prob_above for k in strikes]
    assert all(a >= b for a, b in zip(probs, probs[1:]))


def test_zero_time_to_close_is_degenerate():
    spot, sigma = 63_800.0, annual_to_per_minute(0.368)
    assert price_above(spot, 63_700.0, sigma, 0.0).prob_above == 1.0
    assert price_above(spot, 63_900.0, sigma, 0.0).prob_above == 0.0
