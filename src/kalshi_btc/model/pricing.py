"""Asian-settlement digital pricer for Kalshi KXBTCD hourly BTC markets.

THE CENTRAL FACT
----------------
KXBTCD does NOT settle on a spot price at the top of the hour. Per Kalshi's contract
terms, the Expiration Value is:

    "a simple average of the CF Bitcoin Real-Time Index (BRTI) for the minute
     (60 seconds) prior to <time>"

and "The Last Trading Time will be <time>" - so trading stays open THROUGHOUT the
averaging window. This is an Asian (arithmetic-average) settled digital, not a
point-in-time coin flip.

Per Kalshi's AsyncAPI, the accumulation window is (close_ts - 60000, close_ts], with
second-indexed counts :01 -> 1 ... close tick -> 60. So the 60 ticks land at 59, 58,
..., 1, 0 seconds before close.

WHY IT MATTERS
--------------
Averaging destroys variance. Two consequences, both verified by Monte Carlo in
tests/test_pricing.py:

1. Outside the window, Var(settlement) ~= sigma_min^2 * (tau - 2/3), i.e. the average
   shaves ~40 seconds of variance off every quote, all hour long.

2. Inside the window, with s seconds elapsed and m = 60 - s remaining,
   residual_std ~= sigma_min * (m/60)^1.5 / sqrt(3).

That second one is the thesis. A point-in-time model overstates remaining uncertainty
by 1.73x when the window opens, 3.46x at the halfway mark, and ~21x with five seconds
left. Anyone pricing this as a coin flip is badly wrong exactly when it matters most.

This module implements the EXACT discrete-tick version of both, not the continuous
approximation, because with only 60 ticks the difference is measurable (the exact
in-window factor is 0.3417 vs the continuous 1/3).

MODELLING CHOICE
----------------
Prices are modelled as arithmetic Brownian motion in dollars over the horizon. At these
horizons (<= 1 hour, ~0.4% moves) the arithmetic-vs-geometric distinction is immaterial,
and an arithmetic average of a geometric process has no closed form anyway. Drift is
taken as zero: over one hour, any plausible drift is orders of magnitude smaller than
sigma * sqrt(tau).

Fat tails are supported via a Student-t innovation (see `dist="t"`). Measured excess
kurtosis on the realised KXBTCD settlement series was 3.66, so the Gaussian default is
a benchmark, not a recommendation for live trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats

TICKS = 60  # 60 one-second BRTI samples
SECONDS_PER_MINUTE = 60.0
MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0


# --------------------------------------------------------------------------------------
# Volatility unit conversions
# --------------------------------------------------------------------------------------
def annual_to_per_minute(annual_vol: float) -> float:
    """Convert an annualised vol (e.g. 0.368 for 36.8%) to a per-minute fraction."""
    return annual_vol / math.sqrt(MINUTES_PER_YEAR)


def per_minute_to_annual(sigma_min: float) -> float:
    return sigma_min * math.sqrt(MINUTES_PER_YEAR)


def tick_offsets_before_close() -> np.ndarray:
    """Seconds before close at which each settlement tick lands: [59, 58, ..., 1, 0]."""
    return np.arange(TICKS - 1, -1, -1, dtype=float)


@lru_cache(maxsize=4096)
def _sum_min_matrix(u_key: tuple[float, ...]) -> float:
    """sum_{i,j} min(u_i, u_j) for tick times u (seconds from now). Exact, cached."""
    u = np.asarray(u_key, dtype=float)
    return float(np.minimum.outer(u, u).sum())


# --------------------------------------------------------------------------------------
# Variance of the settlement average
# --------------------------------------------------------------------------------------
def settlement_std_dollars(sigma_min_frac: float, spot: float, minutes_to_close: float) -> float:
    """Std dev (in dollars) of the final settlement average, seen from `minutes_to_close`.

    Uses the exact discrete tick structure. For minutes_to_close > 1 this converges to
    the familiar sigma * sqrt(tau - 2/3).
    """
    if minutes_to_close <= 0:
        return 0.0
    sigma_dollars_per_min = sigma_min_frac * spot
    sigma_sec = sigma_dollars_per_min / math.sqrt(SECONDS_PER_MINUTE)

    tau_sec = minutes_to_close * SECONDS_PER_MINUTE
    # Time from now until each tick; ticks already past contribute no future variance.
    u = tau_sec - tick_offsets_before_close()
    u = np.clip(u, 0.0, None)
    total = _sum_min_matrix(tuple(u.tolist()))
    return sigma_sec * math.sqrt(total) / TICKS


def effective_minutes(minutes_to_close: float) -> float:
    """The 'point-in-time equivalent' horizon: variance-matched tau.

    Returns tau_eff such that sigma*sqrt(tau_eff) equals the true settlement std.
    Approaches tau - 2/3 for tau > 1. Handy for intuition and for sanity checks.
    """
    if minutes_to_close <= 0:
        return 0.0
    # With sigma=1/min and spot=1, the returned std is in units of sigma*sqrt(minutes),
    # so squaring it recovers the variance-matched horizon directly.
    std = settlement_std_dollars(1.0, 1.0, minutes_to_close)
    return std**2


def residual_std_in_window(sigma_min_frac: float, spot: float, seconds_elapsed: float) -> float:
    """Std dev (dollars) of the settlement still outstanding, `seconds_elapsed` into the window.

    The closed form is sigma * (m/60)^1.5 / sqrt(3) with m = 60 - s; this returns the
    exact discrete equivalent.
    """
    remaining_minutes = max(0.0, (TICKS - seconds_elapsed) / SECONDS_PER_MINUTE)
    if remaining_minutes <= 0:
        return 0.0
    return settlement_std_dollars(sigma_min_frac, spot, remaining_minutes)


# --------------------------------------------------------------------------------------
# Digital pricing
# --------------------------------------------------------------------------------------
def _tail_prob(z: float, dist: str, df: float) -> float:
    """P(X > z) for a zero-mean UNIT-VARIANCE innovation."""
    if dist == "normal":
        return float(stats.norm.sf(z))
    if dist == "t":
        if df <= 2:
            raise ValueError("Student-t needs df > 2 for finite variance")
        # Rescale so the t has unit variance: Var(t_df) = df/(df-2)
        return float(stats.t.sf(z * math.sqrt(df / (df - 2.0)), df))
    raise ValueError(f"unknown dist {dist!r}")


@dataclass(frozen=True)
class Quote:
    """A model fair value for one strike."""

    strike: float
    prob_above: float
    residual_std: float
    minutes_to_close: float

    @property
    def fair_cents(self) -> float:
        return self.prob_above * 100.0


def price_above(
    spot: float,
    strike: float,
    sigma_min_frac: float,
    minutes_to_close: float,
    *,
    dist: str = "normal",
    df: float = 4.0,
) -> Quote:
    """P(settlement average > strike), seen from `minutes_to_close` before close.

    `spot` should be the current BRTI value (or best proxy). Strikes are quoted as
    e.g. 63999.99 with strike_type="greater", so "above" means settlement > strike.
    """
    std = settlement_std_dollars(sigma_min_frac, spot, minutes_to_close)
    if std <= 0:
        p = 1.0 if spot > strike else 0.0
        return Quote(strike, p, 0.0, minutes_to_close)
    z = (strike - spot) / std
    return Quote(strike, _tail_prob(z, dist, df), std, minutes_to_close)


def price_above_in_window(
    strike: float,
    known_sum: float,
    known_ticks: int,
    spot_now: float,
    sigma_min_frac: float,
    *,
    dist: str = "normal",
    df: float = 4.0,
) -> Quote:
    """P(settlement > strike) DURING the 60-second averaging window.

    This is the high-value path. Once the window opens, part of the settlement value is
    already locked in as a known constant, and only the remaining ticks are random.

    Args:
        known_sum:   sum of the BRTI ticks already observed in this window
        known_ticks: how many ticks that sum covers (1..60)
        spot_now:    current BRTI value (the martingale forecast for every future tick)

    We need sum_future > 60*strike - known_sum, where sum_future has mean
    m*spot_now and variance sigma_sec^2 * sum_{i,j in future} min(u_i, u_j).
    """
    m = TICKS - known_ticks
    if m <= 0:
        settled = known_sum / TICKS
        return Quote(strike, 1.0 if settled > strike else 0.0, 0.0, 0.0)

    sigma_dollars_per_min = sigma_min_frac * spot_now
    sigma_sec = sigma_dollars_per_min / math.sqrt(SECONDS_PER_MINUTE)

    # Remaining ticks land 1, 2, ..., m seconds from now.
    u = np.arange(1.0, m + 1.0)
    std_sum = sigma_sec * math.sqrt(_sum_min_matrix(tuple(u.tolist())))

    threshold = TICKS * strike - known_sum
    mean_future = m * spot_now
    if std_sum <= 0:
        p = 1.0 if mean_future > threshold else 0.0
        return Quote(strike, p, 0.0, m / SECONDS_PER_MINUTE)

    z = (threshold - mean_future) / std_sum
    return Quote(
        strike,
        _tail_prob(z, dist, df),
        std_sum / TICKS,
        m / SECONDS_PER_MINUTE,
    )


def implied_vol_from_ladder(
    spot: float,
    strikes: np.ndarray,
    mids: np.ndarray,
    minutes_to_close: float,
    *,
    lo: float = 1e-5,
    hi: float = 0.02,
) -> float:
    """Fit a single per-minute sigma to an observed strike ladder (least squares).

    Useful for measuring what vol the market maker is actually using, and as a sanity
    check that our pricer sits in the same universe as the market. Note that with only
    2-3 non-degenerate strikes this is weakly identified - treat the output as
    indicative, not as a signal.
    """

    def sse(sig: float) -> float:
        model = np.array(
            [price_above(spot, float(k), sig, minutes_to_close).prob_above for k in strikes]
        )
        return float(((model - mids) ** 2).sum())

    # Golden-section-ish ternary search; sse is unimodal in sigma here.
    for _ in range(200):
        a = lo + (hi - lo) / 3.0
        b = hi - (hi - lo) / 3.0
        if sse(a) < sse(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2.0
