"""Volatility estimation and forecasting for the 1-minute-to-1-hour horizon.

WHY THIS MODULE EXISTS
----------------------
`pricing.py` takes `sigma_min_frac` as an input and turns it into a fair value. That
single number is the only free parameter in the whole pricer, so every cent of edge the
bot thinks it has is really a claim about volatility. Get sigma wrong by 10% and a
30-delta strike moves ~3 cents - larger than the entire spread we are trying to capture.

THE CALIBRATION BASIS IS THE SETTLEMENT SERIES, NOT SPOT
--------------------------------------------------------
KXBTCD does not settle on spot. It settles on the 60-second BRTI average. So the
quantity we actually need to forecast is the volatility of the SETTLEMENT process, and
the honest way to estimate it is from settlement-to-settlement log returns
(`expiration_value`, which Kalshi hands us for free on every settled market).

There is a subtlety that matters and that a naive fit gets wrong. Both endpoints of a
settlement-to-settlement return are 60-second averages, so the averaging removes
variance from BOTH ends and adds back covariance:

    Var(ln S_2 - ln S_1) = sigma_min^2 * (dt_minutes - 1/3)

The exact discrete constant (0.33325 minutes, i.e. ~20 seconds) is derived in
`settlement_return_variance_offset_minutes()` from the same tick geometry `pricing.py`
uses, not hardcoded. Dividing hourly settlement returns by sqrt(60) instead of
sqrt(60 - 1/3) biases sigma DOWN by 0.28%. Small, but free to get right, and the sign of
the bias is the dangerous direction (it makes us think we have more edge than we do).

ESTIMATOR MENU AND THE TRADEOFFS
--------------------------------
All estimators here return PER-MINUTE VARIANCE of log returns, so they are directly
comparable and directly usable by `pricing.price_above`.

Close-to-close realized variance
    Unbiased, assumption-free, but the least efficient of the lot: with n bars its
    relative standard error is ~1/sqrt(2n). It is the benchmark everything else is
    measured against.

Parkinson (high/low)
    ~5x more efficient than close-to-close because the range uses the whole path.
    Biased DOWN on discretely sampled data (you never observe the true continuous
    high/low) and it ignores drift and overnight gaps entirely.

Garman-Klass (OHLC)
    ~7x efficient. Assumes zero drift; a real drift inflates it. Also gap-blind.

Rogers-Satchell (OHLC)
    Slightly less efficient than Garman-Klass but DRIFT-INDEPENDENT by construction,
    which matters on a tape that trends for hours at a time.

Yang-Zhang (OHLC + gaps)
    Combines an overnight (gap) term, an open-to-close term and Rogers-Satchell.
    Handles drift AND gaps, ~8x efficient, minimum-variance in the class. It is the
    default here. On a 24/7 crypto tape the "gap" term captures the discontinuity
    between our sampled bars rather than a literal session break, which is exactly what
    we want when the capture process has holes in it.

Bipower variation / median realized variance
    Jump-robust. Close-to-close RV lumps the continuous diffusion and jumps together;
    for a ONE-HOUR digital that conflation is expensive. A single 0.5% jump inside an
    hour blows up RV and makes the pricer quote every strike near 50 cents for the rest
    of the day, killing exactly the away-from-the-money edge we are hunting. Bipower and
    MedRV estimate only the continuous component. Neither survives two jumps inside the
    same window - that is a real limitation of both, not a bipower-only one - but MedRV
    has the smaller finite-sample bias under isolated jumps and is markedly less damaged
    by ZERO returns, which is what a stalled BRTI print looks like. Both are measured in
    tests/test_vol.py rather than asserted here.

FORECASTING
-----------
HAR-RV (Corsi 2009) is the workhorse: regress next-period RV on short/medium/long
moving averages of past RV. It reproduces the long-memory behaviour of volatility with
three regressors and an ordinary least squares fit. EWMA/RiskMetrics is the robust
fallback and blending partner - it cannot break, it needs no design matrix, and it
reacts fast when the regime shifts.

INTRADAY SEASONALITY
--------------------
BTC volatility has a genuine diurnal pattern (US equity open, the CME 4pm settle, the
Asia handover) and a weekday/weekend pattern. Ignoring it biases EVERY hourly quote in
a predictable direction: it overprices the quiet Sunday-morning hours and underprices
the 13:30-15:00 UTC window. We estimate a multiplicative hour-of-day x day-of-week
factor on the volatility scale, deseasonalize before fitting HAR/EWMA (so the forecaster
sees a stationary series), and reseasonalize at query time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Sequence

import numpy as np

from kalshi_btc.model.pricing import (
    MINUTES_PER_YEAR,
    effective_minutes,
    per_minute_to_annual,
    tick_offsets_before_close,
)

log = logging.getLogger(__name__)

SECONDS_PER_MINUTE = 60.0

# Measured on the realised KXBTCD settlement series (`expiration_value`): 0.466% per hour
# = 43.6% annualised, over 1,596 hourly settlement-to-settlement returns spanning
# 2026-05-22 to 2026-07-29. Caveat: ~2 months of a single broad volatility regime, so
# treat it as a starting prior that the fitted model should override, not as a constant.
DEFAULT_SIGMA_PER_MINUTE = 0.00466 / math.sqrt(60.0)

# RiskMetrics daily decay. We run it on hourly observations, where a slower decay is
# appropriate than the 0.94 used on daily bars, but 0.94 remains a sane default that
# gives a ~16-observation effective memory.
RISKMETRICS_LAMBDA = 0.94


# --------------------------------------------------------------------------------------
# The settlement-return variance correction
# --------------------------------------------------------------------------------------
def settlement_return_variance_offset_minutes() -> float:
    """Minutes of variance removed by averaging BOTH endpoints of a settlement return.

    Let A and B be 60-second averages of the same Brownian path, ending D minutes apart
    (D >= 1 so the windows are disjoint). Writing `c` for the variance deduction of a
    single average relative to its endpoint time (that is `tau - effective_minutes(tau)`,
    the tau - 2/3 result from `pricing.py`) and `mbar` for the mean tick offset:

        Var(A) = T1 - c        Var(B) = T1 + D - c        Cov(A, B) = T1 - mbar

    because every tick of A precedes every tick of B, so min(t_a, t_b) = t_a always.
    Therefore Var(B - A) = D - 2c + 2*mbar.

    Returns ~0.33325 minutes (about 20 seconds). Derived from the tick geometry rather
    than hardcoded so it stays consistent if the tick model is ever revised.
    """
    # Any tau comfortably above 1 minute gives the asymptotic deduction.
    tau = 100.0
    c = tau - effective_minutes(tau)
    mean_offset_minutes = float(tick_offsets_before_close().mean()) / SECONDS_PER_MINUTE
    return 2.0 * c - 2.0 * mean_offset_minutes


# --------------------------------------------------------------------------------------
# Return helpers
# --------------------------------------------------------------------------------------
def log_returns(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    """Consecutive log returns. Non-positive prices are a data bug, not a regime."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("need at least two prices to form a return")
    if np.any(p <= 0):
        raise ValueError("non-positive price in series")
    return np.diff(np.log(p))


def _per_minute(bar_variance: float, bar_minutes: float) -> float:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    return bar_variance / bar_minutes


# --------------------------------------------------------------------------------------
# Realized variance estimators
# --------------------------------------------------------------------------------------
def realized_variance(
    returns: Sequence[float] | np.ndarray, interval_minutes: float = 1.0
) -> float:
    """Per-minute variance from k-minute log returns.

    mean(r^2) estimates k * sigma_min^2, so dividing by k recovers the per-minute figure
    regardless of sampling frequency. We use the raw second moment rather than the
    demeaned sample variance: over a one-hour horizon any plausible drift is orders of
    magnitude below sigma*sqrt(tau), and demeaning throws away a degree of freedom for
    nothing.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        raise ValueError("empty return series")
    return _per_minute(float(np.mean(r**2)), interval_minutes)


def parkinson_variance(
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    bar_minutes: float = 1.0,
) -> float:
    """Per-minute variance from the high-low range. ~5x the efficiency of close-to-close.

    Biased down on discretely sampled bars (the observed range understates the true
    continuous range) and blind to both drift and gaps.
    """
    h, lo = _ohlc_arrays(high=high, low=low)
    rng = np.log(h / lo)
    return _per_minute(float(np.mean(rng**2) / (4.0 * math.log(2.0))), bar_minutes)


def garman_klass_variance(
    open_: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    bar_minutes: float = 1.0,
) -> float:
    """Per-minute variance from OHLC. ~7x efficient, but assumes zero drift.

    Can return a negative estimate on tiny samples where the drift term dominates; we
    clamp at zero and say so, because a negative variance downstream is a crash, not a
    signal.
    """
    o, h, lo, c = _ohlc_arrays(open_=open_, high=high, low=low, close=close)
    rng = np.log(h / lo)
    body = np.log(c / o)
    v = float(np.mean(0.5 * rng**2 - (2.0 * math.log(2.0) - 1.0) * body**2))
    if v < 0.0:
        log.warning("garman_klass_variance went negative (%.3e); clamping to zero", v)
        v = 0.0
    return _per_minute(v, bar_minutes)


def rogers_satchell_variance(
    open_: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    bar_minutes: float = 1.0,
) -> float:
    """Per-minute variance from OHLC, DRIFT-INDEPENDENT by construction.

    Slightly less efficient than Garman-Klass but it does not care that BTC trended for
    six hours straight, which Garman-Klass very much does.
    """
    o, h, lo, c = _ohlc_arrays(open_=open_, high=high, low=low, close=close)
    v = float(np.mean(np.log(h / c) * np.log(h / o) + np.log(lo / c) * np.log(lo / o)))
    return _per_minute(max(v, 0.0), bar_minutes)


def yang_zhang_variance(
    open_: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    bar_minutes: float = 1.0,
) -> float:
    """Per-minute variance from OHLC with an explicit gap term. The default estimator.

    V_yz = V_overnight + k*V_open_to_close + (1-k)*V_rogers_satchell
    with k = 0.34 / (1.34 + (n+1)/(n-1)) chosen to minimise estimator variance.

    Needs n >= 3 bars because the gap term consumes the first bar and the variances use
    an (n-1) denominator. On a 24/7 tape the "overnight" term is the jump between the
    close of one captured bar and the open of the next; when capture is continuous it is
    identically zero and Yang-Zhang degrades gracefully to a k-weighted blend of the
    other two, which remains unbiased.
    """
    o, h, lo, c = _ohlc_arrays(open_=open_, high=high, low=low, close=close)
    n = o.size
    if n < 3:
        raise ValueError("yang_zhang_variance needs at least 3 bars")

    overnight = np.log(o[1:] / c[:-1])
    body = np.log(c / o)[1:]
    u = np.log(h / o)[1:]
    d = np.log(lo / o)[1:]
    m = overnight.size  # = n - 1 usable bars

    v_o = float(np.sum((overnight - overnight.mean()) ** 2) / (m - 1))
    v_c = float(np.sum((body - body.mean()) ** 2) / (m - 1))
    v_rs = float(np.mean(u * (u - body) + d * (d - body)))

    k = 0.34 / (1.34 + (m + 1.0) / (m - 1.0))
    v = v_o + k * v_c + (1.0 - k) * max(v_rs, 0.0)
    return _per_minute(max(v, 0.0), bar_minutes)


def _ohlc_arrays(**series: Sequence[float] | np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = []
    length: int | None = None
    for name, raw in series.items():
        a = np.asarray(raw, dtype=float)
        if a.ndim != 1 or a.size == 0:
            raise ValueError(f"{name} must be a non-empty 1-D series")
        if np.any(a <= 0):
            raise ValueError(f"non-positive value in {name}")
        if length is None:
            length = a.size
        elif a.size != length:
            raise ValueError("OHLC series must be the same length")
        arrays.append(a)
    return tuple(arrays)


# --------------------------------------------------------------------------------------
# Jump-robust estimators
# --------------------------------------------------------------------------------------
_BIPOWER_MU1 = math.sqrt(2.0 / math.pi)  # E|Z| for a standard normal


def bipower_variation(
    returns: Sequence[float] | np.ndarray, interval_minutes: float = 1.0
) -> float:
    """Per-minute variance of the CONTINUOUS component only (Barndorff-Nielsen/Shephard).

    Products of ADJACENT absolute returns: a lone jump enters exactly two terms linearly
    instead of one term quadratically, so its contribution vanishes asymptotically. The
    (n/(n-1)) factor corrects for using n-1 products to estimate n periods of variance.

    Two failure modes, both real. Consecutive jumps prop each other up, so a liquidation
    cascade is exactly the event bipower handles worst. And a ZERO return kills the two
    products it appears in, biasing the estimate down: with 30% stale prints bipower runs
    ~16% low (measured in tests/test_vol.py). `median_realized_variance` is the better
    choice when either is a live concern.
    """
    r = np.abs(np.asarray(returns, dtype=float))
    n = r.size
    if n < 2:
        raise ValueError("bipower_variation needs at least 2 returns")
    bv = float(np.sum(r[1:] * r[:-1])) / (_BIPOWER_MU1**2) * (n / (n - 1.0))
    return _per_minute(bv / n, interval_minutes)


# Scaling constant for the median of three independent half-normals.
_MEDRV_SCALE = math.pi / (6.0 - 4.0 * math.sqrt(3.0) + math.pi)


def median_realized_variance(
    returns: Sequence[float] | np.ndarray, interval_minutes: float = 1.0
) -> float:
    """Per-minute continuous variance via rolling medians of three absolute returns.

    A single jump inside a window of three is discarded by the median outright, which
    gives MedRV a smaller finite-sample bias than bipower under isolated jumps (~2% vs
    ~5% in the measured case). It is also far less damaged by stale prints: with 30% zero
    returns MedRV runs ~9% low against bipower's ~16%.

    It does NOT survive two jumps inside the same window of three - nothing with a
    three-point median does. Slightly less efficient than bipower in the no-jump case;
    that is what the robustness costs.
    """
    r = np.abs(np.asarray(returns, dtype=float))
    n = r.size
    if n < 3:
        raise ValueError("median_realized_variance needs at least 3 returns")
    windows = np.stack([r[:-2], r[1:-1], r[2:]], axis=1)
    meds = np.median(windows, axis=1)
    medrv = _MEDRV_SCALE * (n / (n - 2.0)) * float(np.sum(meds**2))
    return _per_minute(medrv / n, interval_minutes)


def jump_ratio(returns: Sequence[float] | np.ndarray, interval_minutes: float = 1.0) -> float:
    """Fraction of realized variance attributable to jumps, in [0, 1].

    Above ~0.3 the recent tape is jump-dominated and the diffusion-based pricer should be
    trusted less (or the Student-t innovation switched on).
    """
    rv = realized_variance(returns, interval_minutes)
    if rv <= 0:
        return 0.0
    continuous = median_realized_variance(returns, interval_minutes)
    return float(min(max(1.0 - continuous / rv, 0.0), 1.0))


# --------------------------------------------------------------------------------------
# HAR-RV (Corsi)
# --------------------------------------------------------------------------------------
@dataclass
class HARRV:
    """Heterogeneous AutoRegressive model of realized variance.

    RV_{t+1} = b0 + b_s*RV_t^(short) + b_m*RV_t^(medium) + b_l*RV_t^(long) + e

    where RV^(l) is the trailing mean of the last l observations. The economic story is
    that traders operating on different horizons each respond to volatility measured over
    their own horizon, and superposing three of them reproduces the long-memory decay of
    volatility without a fractionally integrated model.

    `lags` defaults to (1, 24, 168): with one observation per hourly settlement that is
    hour / day / week, the natural analogue of Corsi's day / week / month on daily bars.

    Fitted in LOG space by default. Variance is positive and right-skewed; regressing the
    level lets one volatility spike dominate the least squares fit and can produce
    negative forecasts.

    RETRANSFORMATION BIAS. Fitting log RV and exponentiating gives the conditional MEDIAN,
    not the mean, and we need the mean. The textbook fix is the lognormal correction
    exp(mu + s^2/2), which is WRONG HERE and badly so: when each RV observation is a
    single squared return, the residuals are log-chi-square, not normal. Var(log chi2_1)
    is pi^2/2 = 4.93, so the lognormal factor would be exp(2.47) = 11.8x against a true
    required factor of 3.56x - a 3x overstatement of volatility that would make the
    pricer quote everything at 50 cents.

    We use Duan's smearing estimator instead: multiply by the sample mean of exp(residual).
    That is consistent for E[RV] whatever the residual distribution, and it collapses to
    the lognormal factor when the residuals really are normal.
    """

    lags: tuple[int, ...] = (1, 24, 168)
    use_log: bool = True
    floor: float = 1e-14  # variance floor before taking logs

    coefficients: np.ndarray | None = field(default=None, init=False)
    residual_variance: float = field(default=0.0, init=False)
    smearing: float = field(default=1.0, init=False)
    r_squared: float = field(default=float("nan"), init=False)
    n_observations: int = field(default=0, init=False)
    _history: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def min_observations(self) -> int:
        """Shortest series that yields at least one usable regression row."""
        return max(self.lags) + 2

    @property
    def is_fitted(self) -> bool:
        return self.coefficients is not None

    def fit(self, series: Sequence[float] | np.ndarray) -> HARRV:
        """Fit on a series of per-period realized VARIANCE (not volatility)."""
        rv = np.asarray(series, dtype=float)
        if rv.ndim != 1:
            raise ValueError("HAR-RV expects a 1-D variance series")
        if rv.size < self.min_observations:
            raise ValueError(
                f"HAR-RV with lags {self.lags} needs >= {self.min_observations} "
                f"observations, got {rv.size}"
            )
        if np.any(rv < 0):
            raise ValueError("negative variance in HAR-RV input")

        rv = np.maximum(rv, self.floor)
        x, y = self._design(rv)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)

        resid = y - x @ coef
        dof = max(x.shape[0] - x.shape[1], 1)
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        self.coefficients = coef
        self.residual_variance = float(resid @ resid) / dof
        # Duan smearing: consistent for E[RV] under ANY residual distribution.
        self.smearing = float(np.mean(np.exp(resid))) if self.use_log else 1.0
        self.r_squared = float("nan") if ss_tot == 0 else 1.0 - float(resid @ resid) / ss_tot
        self.n_observations = x.shape[0]
        self._history = rv
        return self

    def _features(self, rv: np.ndarray, t: int) -> list[float]:
        """Trailing-mean regressors evaluated using data up to and including index t."""
        return [float(np.mean(rv[t - lag + 1 : t + 1])) for lag in self.lags]

    def _design(self, rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        start = max(self.lags) - 1
        rows: list[list[float]] = []
        targets: list[float] = []
        # Target at t+1 uses only features through t: strictly causal by construction.
        for t in range(start, rv.size - 1):
            feats = self._features(rv, t)
            rows.append([1.0] + [self._link(f) for f in feats])
            targets.append(self._link(float(rv[t + 1])))
        return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)

    def _link(self, v: float) -> float:
        return math.log(max(v, self.floor)) if self.use_log else v

    def _invlink(self, v: float) -> float:
        if not self.use_log:
            # Least squares on levels can undershoot zero on a quiet stretch; a negative
            # variance is a crash downstream, so floor it.
            return max(v, self.floor)
        return max(math.exp(v) * self.smearing, self.floor)

    def forecast(
        self, horizon: int = 1, series: Sequence[float] | np.ndarray | None = None
    ) -> float:
        """Mean expected per-period variance over the next `horizon` periods.

        Iterated (not direct) multi-step: each step's point forecast is appended to the
        history and the regressors are recomputed, which is what gives HAR its slow
        mean-reversion. horizon=1 is the plain one-step forecast.
        """
        if not self.is_fitted:
            raise RuntimeError("HARRV.forecast called before fit")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        rv = (
            self._history
            if series is None
            else np.maximum(np.asarray(series, dtype=float), self.floor)
        )
        if rv is None or rv.size < max(self.lags):
            raise ValueError("not enough history to build HAR regressors")

        assert self.coefficients is not None
        path = rv.astype(float).copy()
        out: list[float] = []
        for _ in range(horizon):
            feats = self._features(path, path.size - 1)
            x = np.array([1.0] + [self._link(f) for f in feats])
            step = self._invlink(float(x @ self.coefficients))
            out.append(step)
            path = np.append(path, step)
        return float(np.mean(out))


# --------------------------------------------------------------------------------------
# EWMA / RiskMetrics
# --------------------------------------------------------------------------------------
@dataclass
class EWMAVol:
    """Exponentially weighted variance - the estimator that cannot break.

    sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}

    No design matrix, no minimum sample size, no way to produce a negative number.
    Its forecast is flat in horizon (an EWMA is an IGARCH, so shocks never decay), which
    makes it a bad long-horizon model and a good short-horizon one. At our horizon that
    is a feature.
    """

    lam: float = RISKMETRICS_LAMBDA
    variance: float | None = field(default=None, init=False)
    n_observations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.lam < 1.0:
            raise ValueError("EWMA lambda must lie in (0, 1)")

    @property
    def is_fitted(self) -> bool:
        return self.variance is not None

    @property
    def effective_memory(self) -> float:
        """Approximate number of observations carrying the weight: 1/(1-lam)."""
        return 1.0 / (1.0 - self.lam)

    def fit(self, per_period_variance: Sequence[float] | np.ndarray) -> EWMAVol:
        """Seed on the sample mean, then recurse through the series.

        Takes per-period VARIANCE (e.g. squared standardised returns) so it composes with
        the same deseasonalized series HAR-RV is fitted on.
        """
        v = np.asarray(per_period_variance, dtype=float)
        if v.size == 0:
            raise ValueError("empty variance series")
        if np.any(v < 0):
            raise ValueError("negative variance in EWMA input")
        self.variance = float(v.mean())
        for x in v:
            self.variance = self.lam * self.variance + (1.0 - self.lam) * float(x)
        self.n_observations = int(v.size)
        return self

    def update(self, period_variance: float) -> float:
        """Fold one new observation in. Returns the updated variance."""
        if period_variance < 0:
            raise ValueError("negative variance in EWMA update")
        self.variance = (
            float(period_variance)
            if self.variance is None
            else self.lam * self.variance + (1.0 - self.lam) * float(period_variance)
        )
        self.n_observations += 1
        return self.variance

    def forecast(self, horizon: int = 1) -> float:
        """Flat across horizon - that is the RiskMetrics random-walk assumption."""
        if self.variance is None:
            raise RuntimeError("EWMAVol.forecast called before fit")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return self.variance


# --------------------------------------------------------------------------------------
# Intraday / weekly seasonality
# --------------------------------------------------------------------------------------
@dataclass
class SeasonalProfile:
    """Multiplicative hour-of-day x day-of-week scaling on the VOLATILITY scale.

    factor(t) = f_hour[hour(t)] * f_dow[weekday(t)], both normalised to geometric mean 1
    so that applying the profile leaves the overall level of sigma unchanged and only
    redistributes it.

    Fitted on |return|-derived sigma proxies with a MEDIAN, not a mean: a single 3-sigma
    hour would otherwise permanently mark that hour-of-day as volatile. Cells are shrunk
    toward 1 in proportion to their count, so a bucket seen twice barely moves.

    Times are interpreted in UTC. That is deliberate - the diurnal pattern is anchored to
    global session boundaries (CME 4pm ET settle, Asia handover), which are fixed in UTC
    up to daylight saving, not to whatever the host machine's timezone happens to be.
    """

    prior_count: float = 20.0
    hour_factors: np.ndarray = field(default_factory=lambda: np.ones(24))
    dow_factors: np.ndarray = field(default_factory=lambda: np.ones(7))
    hour_counts: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=int))
    dow_counts: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=int))
    is_fitted: bool = False

    def fit(
        self, timestamps: Sequence[datetime], sigmas: Sequence[float] | np.ndarray
    ) -> SeasonalProfile:
        """Fit on per-observation sigma proxies aligned with `timestamps`."""
        s = np.asarray(sigmas, dtype=float)
        if len(timestamps) != s.size:
            raise ValueError("timestamps and sigmas must be the same length")
        if s.size == 0:
            raise ValueError("empty seasonality input")
        if np.any(s < 0):
            raise ValueError("negative sigma in seasonality input")

        hours = np.array([as_utc(t).hour for t in timestamps])
        dows = np.array([as_utc(t).weekday() for t in timestamps])

        base = float(np.median(s))
        if base <= 0:
            # Degenerate tape (all-zero returns); a flat profile is the honest answer.
            self.is_fitted = True
            return self

        self.hour_factors, self.hour_counts = self._cell_factors(hours, s / base, 24)
        # Estimate day-of-week on hour-adjusted residuals so the two do not double count.
        residual = s / base / self.hour_factors[hours]
        self.dow_factors, self.dow_counts = self._cell_factors(dows, residual, 7)
        self.is_fitted = True
        return self

    def _cell_factors(
        self, keys: np.ndarray, values: np.ndarray, n_cells: int
    ) -> tuple[np.ndarray, np.ndarray]:
        factors = np.ones(n_cells)
        counts = np.zeros(n_cells, dtype=int)
        for cell in range(n_cells):
            mask = keys == cell
            n = int(mask.sum())
            counts[cell] = n
            if n == 0:
                continue
            raw = float(np.median(values[mask]))
            if raw <= 0:
                continue
            # Shrink toward 1 by observation count: thin cells stay near neutral.
            factors[cell] = (n * raw + self.prior_count) / (n + self.prior_count)
        # Renormalise to geometric mean 1 over OBSERVED cells only, so the profile is
        # purely a redistribution and never shifts the overall level of sigma.
        observed = counts > 0
        if observed.any():
            weights = counts[observed].astype(float)
            gm = math.exp(float(np.sum(weights * np.log(factors[observed])) / weights.sum()))
            if gm > 0:
                factors = factors / gm
        return factors, counts

    def factor(self, when: datetime) -> float:
        """Volatility multiplier for `when`. Variance scales by the SQUARE of this."""
        t = as_utc(when)
        return float(self.hour_factors[t.hour] * self.dow_factors[t.weekday()])

    def deseasonalize(
        self, timestamps: Sequence[datetime], sigmas: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        f = np.array([self.factor(t) for t in timestamps])
        return np.asarray(sigmas, dtype=float) / f

    def reseasonalize(
        self, timestamps: Sequence[datetime], sigmas: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        f = np.array([self.factor(t) for t in timestamps])
        return np.asarray(sigmas, dtype=float) * f

    def describe(self) -> str:
        peak = int(np.argmax(self.hour_factors))
        trough = int(np.argmin(self.hour_factors))
        return (
            f"seasonality peak {peak:02d}:00Z x{self.hour_factors[peak]:.2f}, "
            f"trough {trough:02d}:00Z x{self.hour_factors[trough]:.2f}, "
            f"n={int(self.hour_counts.sum())}"
        )


def as_utc(t: datetime) -> datetime:
    """Naive datetimes are treated as UTC - everything in this project is UTC."""
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


# --------------------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VolClamp:
    """Hard band on the per-minute sigma the pricer is allowed to see.

    Defaults are roughly 0.2x and 4x the measured 43.6% annualised level. A forecast
    outside this band means the fit is broken or the tape is doing something the model
    was never calibrated for; in both cases the correct action is to clamp LOUDLY and let
    the operator see it, not to quote off it silently.
    """

    low: float = 1.0e-4  # ~7.2% annualised
    high: float = 2.5e-3  # ~182% annualised

    def apply(self, sigma: float) -> tuple[float, bool]:
        if not math.isfinite(sigma) or sigma <= 0:
            return self.low, True
        if sigma < self.low:
            return self.low, True
        if sigma > self.high:
            return self.high, True
        return sigma, False

    def describe(self) -> str:
        lo_ann = per_minute_to_annual(self.low)
        hi_ann = per_minute_to_annual(self.high)
        return (
            f"sigma band [{self.low:.2e}, {self.high:.2e}] per minute "
            f"= [{lo_ann:.1%}, {hi_ann:.1%}] annualised"
        )


@dataclass(frozen=True)
class VolForecast:
    """A sigma forecast with every intermediate step exposed.

    Nothing here is decoration: when the bot takes a bad trade the first question is
    always "what did the vol model think and why", and a bare float cannot answer it.
    """

    when: datetime
    sigma_per_minute: float
    raw_sigma: float
    har_sigma: float | None
    ewma_sigma: float | None
    blend_weight: float
    seasonal_factor: float
    clamped: bool

    @property
    def annualised(self) -> float:
        return per_minute_to_annual(self.sigma_per_minute)

    @property
    def per_hour(self) -> float:
        return self.sigma_per_minute * math.sqrt(60.0)

    def describe(self) -> str:
        parts = [
            f"sigma={self.sigma_per_minute:.3e}/min ({self.annualised:.1%} ann, "
            f"{self.per_hour:.3%}/hr)",
            f"har={self.har_sigma:.3e}" if self.har_sigma is not None else "har=n/a",
            f"ewma={self.ewma_sigma:.3e}" if self.ewma_sigma is not None else "ewma=n/a",
            f"w={self.blend_weight:.2f}",
            f"seasonal=x{self.seasonal_factor:.2f}",
        ]
        if self.clamped:
            parts.append(f"CLAMPED (raw {self.raw_sigma:.3e})")
        return " ".join(parts)


# --------------------------------------------------------------------------------------
# The facade
# --------------------------------------------------------------------------------------
@dataclass
class VolModel:
    """Blended HAR-RV + EWMA volatility model with a seasonal overlay.

    THE ONE THING TO KNOW: calibrate this on the SETTLEMENT series, not on spot bars.
    `fit_settlements` consumes exactly what `KalshiClient.get_settled_markets` returns
    (`expiration_value` per settled hourly market) and corrects for the fact that both
    endpoints of a settlement-to-settlement return are themselves 60-second averages.

    Fitting on spot bars is not merely less convenient - it measures a different process.
    Spot vol overstates the settlement vol by the averaging deduction, which is worth
    ~0.3% of sigma on hourly returns and considerably more if you sample finer.
    """

    har: HARRV = field(default_factory=HARRV)
    ewma: EWMAVol = field(default_factory=EWMAVol)
    seasonal: SeasonalProfile = field(default_factory=SeasonalProfile)
    clamp: VolClamp = field(default_factory=VolClamp)
    # Weight on HAR-RV. HAR is the better unconditional forecaster; EWMA is the better
    # regime-shift detector. 0.6/0.4 keeps HAR in charge without letting a stale
    # regression ignore a tape that has just changed character.
    blend_weight: float = 0.6
    fallback_sigma: float = DEFAULT_SIGMA_PER_MINUTE

    n_returns: int = field(default=0, init=False)
    unconditional_sigma: float = field(default=DEFAULT_SIGMA_PER_MINUTE, init=False)
    clamp_hits: int = field(default=0, init=False)
    last_forecast: VolForecast | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must lie in [0, 1]")
        self.unconditional_sigma = self.fallback_sigma

    # -- construction ------------------------------------------------------------------
    @classmethod
    def constant(cls, sigma_per_minute: float) -> VolModel:
        """A model that always returns one number. Useful for tests and for `kbtc doctor`."""
        m = cls()
        m.fallback_sigma = sigma_per_minute
        m.unconditional_sigma = sigma_per_minute
        return m

    def fit_settlements(
        self,
        times: Sequence[datetime],
        settlements: Sequence[float] | np.ndarray,
        *,
        max_gap_minutes: float = 24 * 60.0,
    ) -> VolModel:
        """Calibrate on realised settlement values (`expiration_value`).

        Args:
            times: close time of each settled market, ascending. Naive => UTC.
            settlements: the realised 60-second BRTI averages.
            max_gap_minutes: pairs further apart than this are dropped rather than
                scaled. A three-day hole in the capture is a data outage, and pretending
                it is one long return imports the outage's variance into the model.

        Each usable pair contributes a per-minute volatility proxy

            sigma_hat = |ln(S_i / S_{i-1})| / sqrt(dt - offset)

        where `offset` is the settlement-averaging correction above. Note the deliberate
        ABSENCE of a sqrt(pi/2) rescaling. That factor makes E[sigma_hat] equal sigma, but
        it also makes E[sigma_hat^2] equal (pi/2)*sigma^2 - a 25% overstatement of
        variance, which is the quantity HAR and EWMA actually consume. Left unscaled,
        sigma_hat^2 is exactly unbiased for the per-minute variance, and the seasonal
        profile is unaffected because it normalises to geometric mean 1 and any constant
        scale factor cancels out of it.
        """
        s = np.asarray(settlements, dtype=float)
        if len(times) != s.size:
            raise ValueError("times and settlements must be the same length")
        if s.size < 2:
            raise ValueError("need at least two settlements to form a return")
        if np.any(s <= 0):
            raise ValueError("non-positive settlement value")

        ts = [as_utc(t) for t in times]
        offset = settlement_return_variance_offset_minutes()

        stamps: list[datetime] = []
        sigmas: list[float] = []
        for i in range(1, s.size):
            dt_min = (ts[i] - ts[i - 1]).total_seconds() / SECONDS_PER_MINUTE
            # A return needs a strictly positive effective horizon after the averaging
            # deduction, otherwise the two windows overlap and the algebra is invalid.
            if dt_min <= offset or dt_min > max_gap_minutes:
                continue
            r = math.log(s[i] / s[i - 1])
            sigmas.append(abs(r) / math.sqrt(dt_min - offset))
            stamps.append(ts[i])

        if len(sigmas) < 2:
            raise ValueError("no usable settlement returns after gap filtering")

        sig = np.asarray(sigmas, dtype=float)
        self.n_returns = sig.size
        self.unconditional_sigma = float(np.sqrt(np.mean(sig**2)))

        # 1. Seasonality first, on the raw sigma proxies.
        self.seasonal.fit(stamps, sig)
        # 2. Deseasonalize, then fit the forecasters on a stationary variance series.
        deseason_var = self.seasonal.deseasonalize(stamps, sig) ** 2
        self.ewma.fit(deseason_var)
        if deseason_var.size >= self.har.min_observations:
            self.har.fit(deseason_var)
        else:
            log.info(
                "HAR-RV skipped: %d observations < %d required for lags %s; "
                "falling back to EWMA only",
                deseason_var.size,
                self.har.min_observations,
                self.har.lags,
            )
        return self

    # -- query -------------------------------------------------------------------------
    def sigma_per_minute(self, now: datetime) -> float:
        """The number `pricing.price_above` wants. Always finite, always inside the band."""
        return self.forecast(now).sigma_per_minute

    def forecast(self, now: datetime, horizon: int = 1) -> VolForecast:
        """Full forecast with components exposed, for logging and the HTML report."""
        har_var: float | None = None
        ewma_var: float | None = None

        if self.har.is_fitted:
            try:
                har_var = self.har.forecast(horizon)
            except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                log.warning("HAR-RV forecast failed (%s); dropping to EWMA", exc)
        if self.ewma.is_fitted:
            ewma_var = self.ewma.forecast(horizon)

        # Blend on the VARIANCE scale - variances are what add.
        if har_var is not None and ewma_var is not None:
            w = self.blend_weight
            blended = w * har_var + (1.0 - w) * ewma_var
        elif har_var is not None:
            w, blended = 1.0, har_var
        elif ewma_var is not None:
            w, blended = 0.0, ewma_var
        else:
            w, blended = 0.0, self.fallback_sigma**2

        factor = self.seasonal.factor(now) if self.seasonal.is_fitted else 1.0
        raw = math.sqrt(max(blended, 0.0)) * factor
        sigma, clamped = self.clamp.apply(raw)
        if clamped:
            self.clamp_hits += 1
            log.warning(
                "vol forecast clamped at %s: raw=%.3e -> %.3e (%s)",
                as_utc(now).isoformat(),
                raw,
                sigma,
                self.clamp.describe(),
            )

        fc = VolForecast(
            when=as_utc(now),
            sigma_per_minute=sigma,
            raw_sigma=raw,
            har_sigma=math.sqrt(har_var) if har_var is not None else None,
            ewma_sigma=math.sqrt(ewma_var) if ewma_var is not None else None,
            blend_weight=w,
            seasonal_factor=factor,
            clamped=clamped,
        )
        self.last_forecast = fc
        return fc

    def describe(self) -> str:
        bits = [
            f"VolModel n_returns={self.n_returns}",
            f"unconditional={self.unconditional_sigma:.3e}/min "
            f"({per_minute_to_annual(self.unconditional_sigma):.1%} ann)",
            f"har={'fitted' if self.har.is_fitted else 'unfitted'}",
            f"ewma={'fitted' if self.ewma.is_fitted else 'unfitted'}",
            self.seasonal.describe() if self.seasonal.is_fitted else "seasonality=flat",
            self.clamp.describe(),
            f"clamp_hits={self.clamp_hits}",
        ]
        return " | ".join(bits)


__all__ = [
    "MINUTES_PER_YEAR",
    "DEFAULT_SIGMA_PER_MINUTE",
    "EWMAVol",
    "HARRV",
    "SeasonalProfile",
    "as_utc",
    "VolClamp",
    "VolForecast",
    "VolModel",
    "bipower_variation",
    "garman_klass_variance",
    "jump_ratio",
    "log_returns",
    "median_realized_variance",
    "parkinson_variance",
    "realized_variance",
    "rogers_satchell_variance",
    "settlement_return_variance_offset_minutes",
    "yang_zhang_variance",
]
