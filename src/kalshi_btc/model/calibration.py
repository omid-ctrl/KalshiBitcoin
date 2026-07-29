"""Honest scoring of the pricing model against the market.

WHY THIS MODULE EXISTS
----------------------
It is trivially easy to build a model that looks good. Point it at captured data, tune
sigma until the Brier score drops, and declare victory. That number is meaningless on
its own for one reason: the market is already a very good forecaster. A Brier score of
0.08 sounds excellent right up until you notice the market mid scores 0.07 over the same
sample, at which point the model is a liability that pays fees for the privilege of being
worse.

So the headline metric here is not a score, it is a SKILL SCORE against the market mid
as the benchmark forecast:

    skill = 1 - brier_model / brier_market

Positive means we beat the market maker. Zero means we are the market maker with extra
steps. Negative means stop trading. Everything else in this module exists to stop that
number from being a lie.

THREE WAYS THIS NUMBER GETS FAKED, AND WHAT WE DO ABOUT THEM
------------------------------------------------------------
1. LOOKAHEAD IN THE PROBABILITY.
   A probability must use only information available at its own timestamp. We enforce
   this structurally rather than by convention: `build_observations` calls the model
   through a `ProbabilityModel` callable that is handed only (ts, spot, strike,
   minutes_to_close). The realised settlement is not a parameter of that call and is not
   in scope when it is made, so it CANNOT leak into the probability. The Calibrator then
   re-checks every record it is given (`_screen`) and drops anything whose timestamp is
   at or after its own close time.

2. LOOKAHEAD IN THE MODEL PARAMETERS.
   Subtler and more common. If `VolModel` was fitted on settlements that include the
   hours being scored, the model has seen its own answers. The Calibrator cannot detect
   this from the observations alone, so it takes an explicit `train_cutoff` and drops
   every observation at or before it. Pass the timestamp of the last settlement used to
   fit the vol model and the split is clean by construction. Scoring without a cutoff is
   allowed - it is the right thing to do for an in-sample diagnostic - but the result
   records `train_cutoff=None` so the report can say so out loud.

3. SURVIVORSHIP IN THE SAMPLE.
   Scoring only the strikes where the model was confident inflates everything. We score
   every observation supplied and report the drop count and reason for each one, so a
   filter that quietly removed 60% of the sample is visible in the output rather than
   buried in a list comprehension.

BUCKETING
---------
The aggregate skill score hides the shape of the edge. Ours is concentrated (a) in the
final minutes, where the Asian-settlement variance collapse is largest and a
point-in-time market maker is most wrong, and (b) away from the money, where the fee is
smallest relative to the edge. We therefore bucket by minutes-to-close and by price.

Both bucketing keys - minutes-to-close and the MARKET MID - are known at the observation
timestamp, so bucketing introduces no lookahead. Bucketing by the realised outcome, or by
the model's own probability, would; the first is obvious, the second sorts observations
by how confident we were, which is a different and equally misleading picture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from kalshi_btc.model.vol import as_utc

# Probabilities are clipped away from {0, 1} before taking logs. A model that says 0.0
# and is wrong once would otherwise score an infinite loss and destroy the whole sample.
# 1e-6 corresponds to a worst-case per-observation loss of ~13.8 nats.
LOG_LOSS_EPSILON = 1e-6

# Default bucket edges. Minutes-to-close edges follow the market's own structure: the
# final minute IS the settlement window, 1-5 is the endgame, 5-15 is where the averaging
# correction first bites, and beyond 15 the contract is close to a plain digital.
DEFAULT_MINUTE_EDGES: tuple[float, ...] = (0.0, 1.0, 5.0, 15.0, 30.0, math.inf)
DEFAULT_PRICE_EDGES: tuple[float, ...] = (0.0, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 1.0)


# --------------------------------------------------------------------------------------
# Proper scoring rules
# --------------------------------------------------------------------------------------
def _as_prob_outcome(
    probs: Sequence[float] | np.ndarray, outcomes: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and outcomes must have the same shape")
    if p.size == 0:
        raise ValueError("empty scoring input")
    if np.any(~np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("outcomes must be binary 0/1")
    return p, y


def brier_score(
    probs: Sequence[float] | np.ndarray, outcomes: Sequence[float] | np.ndarray
) -> float:
    """Mean squared error of a probabilistic forecast. Lower is better; 0.25 = coin flip.

    Strictly proper, bounded, and - unlike log loss - finite even when the forecast is
    confidently wrong, which is why it is the headline metric here.
    """
    p, y = _as_prob_outcome(probs, outcomes)
    return float(np.mean((p - y) ** 2))


def log_loss(
    probs: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    eps: float = LOG_LOSS_EPSILON,
) -> float:
    """Mean negative log likelihood, in nats. Lower is better; ln(2) = 0.693 = coin flip.

    Punishes confident errors far harder than Brier does, which is the right emphasis for
    a book that can be wiped out by one 95-cent contract settling worthless. Clipped to
    [eps, 1-eps] so a single 0-or-1 forecast cannot return infinity.
    """
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    p, y = _as_prob_outcome(probs, outcomes)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def skill_score(brier_model: float, brier_benchmark: float) -> float:
    """1 - brier_model/brier_benchmark. Positive means the model beats the benchmark.

    A perfect benchmark (Brier 0) cannot be improved upon: we return 0.0 if the model is
    also perfect and -inf otherwise, rather than dividing by zero.
    """
    if brier_benchmark == 0.0:
        return 0.0 if brier_model == 0.0 else -math.inf
    return 1.0 - brier_model / brier_benchmark


def reliability_curve(
    probs: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin forecasts and compare each bin's mean forecast to its realised frequency.

    Returns (bin_centers, empirical_freq, counts). Empty bins get NaN frequency and a
    zero count so the caller can drop them from a plot without guessing.

    A perfectly calibrated model traces the diagonal. Points ABOVE the diagonal mean the
    model is underconfident at that level (it says 30, it happens 40% of the time), which
    on a binary market is a direct buy signal at that price.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    p, y = _as_prob_outcome(probs, outcomes)

    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    # Half-open bins [lo, hi) with the top bin closed so p == 1.0 lands in it.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)

    freq = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        counts[b] = n
        if n:
            freq[b] = float(y[mask].mean())
    return centers, freq, counts


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
class ProbabilityModel(Protocol):
    """A pricer, restricted to information available at the observation timestamp.

    The signature is the enforcement mechanism. There is no settlement parameter, so a
    model implementing this protocol cannot peek even by accident.
    """

    def __call__(
        self, *, ts: datetime, spot: float, strike: float, minutes_to_close: float
    ) -> float: ...


@dataclass(frozen=True)
class LadderRecord:
    """One captured strike quote, as recorded by `kbtc capture`.

    Everything here is observable at `ts`. `spot` is the BRTI value from the
    `cfbenchmarks_value` stream at that moment (the `avg_60s_data` rolling average, or
    the windowed average once inside the final minute).
    """

    ts: datetime
    event_ticker: str
    ticker: str
    strike: float
    spot: float
    market_mid: float
    close_time: datetime

    @property
    def minutes_to_close(self) -> float:
        return (as_utc(self.close_time) - as_utc(self.ts)).total_seconds() / 60.0


@dataclass(frozen=True)
class Observation:
    """A scored-ready record: a model probability, a market probability, and an outcome.

    `outcome` is filled in by the Calibrator from the settlement map, never by the model.
    """

    ts: datetime
    event_ticker: str
    ticker: str
    strike: float
    minutes_to_close: float
    model_prob: float
    market_prob: float
    close_time: datetime


def build_observations(
    records: Iterable[LadderRecord],
    model: ProbabilityModel,
) -> list[Observation]:
    """Price every captured record with `model`, passing only same-timestamp data.

    This is the anti-lookahead choke point. The settlement map is not an argument to this
    function; the outcome is attached later, in the Calibrator, from data the model never
    touched. Records already at or past their close time are skipped - there is no
    forecast to make once the market has settled.
    """
    out: list[Observation] = []
    for rec in records:
        mtc = rec.minutes_to_close
        if mtc <= 0.0:
            continue
        p = float(
            model(ts=as_utc(rec.ts), spot=rec.spot, strike=rec.strike, minutes_to_close=mtc)
        )
        out.append(
            Observation(
                ts=as_utc(rec.ts),
                event_ticker=rec.event_ticker,
                ticker=rec.ticker,
                strike=rec.strike,
                minutes_to_close=mtc,
                model_prob=p,
                market_prob=rec.market_mid,
                close_time=as_utc(rec.close_time),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreSet:
    """Every headline number for one forecaster over one slice of the sample."""

    n: int
    brier: float
    log_loss: float
    mean_prob: float
    base_rate: float

    @property
    def bias(self) -> float:
        """Mean forecast minus realised frequency. Positive = systematically too bullish."""
        return self.mean_prob - self.base_rate

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "mean_prob": self.mean_prob,
            "base_rate": self.base_rate,
            "bias": self.bias,
        }


@dataclass(frozen=True)
class ReliabilityCurve:
    """Reliability diagram data, ready for the HTML report."""

    centers: tuple[float, ...]
    frequencies: tuple[float, ...]  # NaN where the bin is empty
    counts: tuple[int, ...]

    @classmethod
    def build(
        cls,
        probs: Sequence[float] | np.ndarray,
        outcomes: Sequence[float] | np.ndarray,
        bins: int,
    ) -> ReliabilityCurve:
        c, f, n = reliability_curve(probs, outcomes, bins)
        return cls(tuple(c.tolist()), tuple(f.tolist()), tuple(int(x) for x in n))

    def populated(self) -> list[tuple[float, float, int]]:
        """Only the bins that actually contain observations."""
        return [
            (c, f, n)
            for c, f, n in zip(self.centers, self.frequencies, self.counts)
            if n > 0 and not math.isnan(f)
        ]

    def as_dict(self) -> dict[str, list]:
        return {
            "centers": list(self.centers),
            "frequencies": list(self.frequencies),
            "counts": list(self.counts),
        }


@dataclass(frozen=True)
class BucketScore:
    """Model vs market over one bucket, plus the skill score for that bucket."""

    label: str
    low: float
    high: float
    model: ScoreSet
    market: ScoreSet
    skill: float
    # False when the bucket is too thin for its skill number to mean anything. The report
    # should grey these out; a 4-observation bucket can show +0.9 skill by luck alone.
    reliable: bool

    @property
    def n(self) -> int:
        return self.model.n

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "skill": self.skill,
            "reliable": self.reliable,
            "model": self.model.as_dict(),
            "market": self.market.as_dict(),
        }


@dataclass(frozen=True)
class CalibrationResult:
    """Everything `kbtc calibrate` and `report/` need, in one structure.

    `skill` is the headline. Read `n_dropped` and `dropped_reasons` before believing it.
    """

    n_observations: int
    n_events: int
    n_dropped: int
    dropped_reasons: Mapping[str, int]
    train_cutoff: datetime | None
    first_ts: datetime | None
    last_ts: datetime | None

    model: ScoreSet
    market: ScoreSet
    skill: float

    model_reliability: ReliabilityCurve
    market_reliability: ReliabilityCurve

    by_minutes_to_close: tuple[BucketScore, ...]
    by_price_bucket: tuple[BucketScore, ...]

    @property
    def beats_market(self) -> bool:
        return self.skill > 0.0

    @property
    def is_out_of_sample(self) -> bool:
        """False means the vol model may have been fitted on the hours being scored."""
        return self.train_cutoff is not None

    def headline(self) -> str:
        window = "out-of-sample" if self.is_out_of_sample else "IN-SAMPLE (not a backtest)"
        return (
            f"skill {self.skill:+.4f} vs market "
            f"(model brier {self.model.brier:.5f}, market brier {self.market.brier:.5f}) "
            f"over n={self.n_observations} {window}"
        )

    def as_dict(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "n_events": self.n_events,
            "n_dropped": self.n_dropped,
            "dropped_reasons": dict(self.dropped_reasons),
            "train_cutoff": self.train_cutoff.isoformat() if self.train_cutoff else None,
            "first_ts": self.first_ts.isoformat() if self.first_ts else None,
            "last_ts": self.last_ts.isoformat() if self.last_ts else None,
            "skill": self.skill,
            "beats_market": self.beats_market,
            "is_out_of_sample": self.is_out_of_sample,
            "model": self.model.as_dict(),
            "market": self.market.as_dict(),
            "model_reliability": self.model_reliability.as_dict(),
            "market_reliability": self.market_reliability.as_dict(),
            "by_minutes_to_close": [b.as_dict() for b in self.by_minutes_to_close],
            "by_price_bucket": [b.as_dict() for b in self.by_price_bucket],
        }


# --------------------------------------------------------------------------------------
# The Calibrator
# --------------------------------------------------------------------------------------
@dataclass
class Calibrator:
    """Scores model probabilities against market mids using realised settlements.

    Usage:
        obs = build_observations(records, my_pricer)
        result = Calibrator().score(obs, settlements, train_cutoff=last_fit_ts)

    `settlements` maps event_ticker -> realised `expiration_value`. The outcome for a
    strike is `settlement > strike`, matching the contract's strike_type="greater".
    """

    bins: int = 10
    minute_edges: tuple[float, ...] = DEFAULT_MINUTE_EDGES
    price_edges: tuple[float, ...] = DEFAULT_PRICE_EDGES
    # Below this many observations a bucket's Brier is noise; we still emit the bucket so
    # the report can grey it out, but the skill number should not be acted on.
    min_bucket_n: int = 30

    dropped_reasons: dict[str, int] = field(default_factory=dict, init=False)

    def score(
        self,
        observations: Iterable[Observation],
        settlements: Mapping[str, float],
        *,
        train_cutoff: datetime | None = None,
    ) -> CalibrationResult:
        """Compute the full result.

        Args:
            observations: from `build_observations`, or constructed directly.
            settlements: event_ticker -> realised settlement value.
            train_cutoff: drop every observation at or before this timestamp. Pass the
                last timestamp used to FIT the model to guarantee an out-of-sample score.
        """
        self.dropped_reasons = {}
        cutoff = as_utc(train_cutoff) if train_cutoff is not None else None

        kept: list[Observation] = []
        outcomes: list[float] = []
        n_seen = 0
        for obs in observations:
            n_seen += 1
            outcome = self._screen(obs, settlements, cutoff)
            if outcome is None:
                continue
            kept.append(obs)
            outcomes.append(outcome)

        if not kept:
            raise ValueError(
                f"no scorable observations out of {n_seen} "
                f"(dropped: {self.dropped_reasons or 'none seen'})"
            )

        y = np.asarray(outcomes, dtype=float)
        pm = np.asarray([o.model_prob for o in kept], dtype=float)
        pk = np.asarray([o.market_prob for o in kept], dtype=float)
        mtc = np.asarray([o.minutes_to_close for o in kept], dtype=float)

        model = self._scoreset(pm, y)
        market = self._scoreset(pk, y)
        stamps = sorted(o.ts for o in kept)

        return CalibrationResult(
            n_observations=len(kept),
            n_events=len({o.event_ticker for o in kept}),
            n_dropped=n_seen - len(kept),
            dropped_reasons=dict(self.dropped_reasons),
            train_cutoff=cutoff,
            first_ts=stamps[0],
            last_ts=stamps[-1],
            model=model,
            market=market,
            skill=skill_score(model.brier, market.brier),
            model_reliability=ReliabilityCurve.build(pm, y, self.bins),
            market_reliability=ReliabilityCurve.build(pk, y, self.bins),
            by_minutes_to_close=self._bucket(mtc, pm, pk, y, self.minute_edges, "min"),
            # Bucket by the MARKET mid: it is known at ts and it is the benchmark's own
            # view, so the buckets do not depend on anything we are trying to score.
            by_price_bucket=self._bucket(pk, pm, pk, y, self.price_edges, ""),
        )

    # -- internals ---------------------------------------------------------------------
    def _drop(self, reason: str) -> None:
        self.dropped_reasons[reason] = self.dropped_reasons.get(reason, 0) + 1

    def _screen(
        self,
        obs: Observation,
        settlements: Mapping[str, float],
        cutoff: datetime | None,
    ) -> float | None:
        """Return the binary outcome, or None (with a recorded reason) if unscorable.

        This is the second line of anti-lookahead defence. `build_observations` already
        prevents the model from seeing the future; this re-checks the invariant on
        observations that may have been constructed by other code paths.
        """
        ts = as_utc(obs.ts)
        if cutoff is not None and ts <= cutoff:
            self._drop("at_or_before_train_cutoff")
            return None
        if ts >= as_utc(obs.close_time):
            self._drop("timestamp_at_or_after_close")
            return None
        if obs.minutes_to_close <= 0.0:
            self._drop("non_positive_minutes_to_close")
            return None
        if not (math.isfinite(obs.model_prob) and 0.0 <= obs.model_prob <= 1.0):
            self._drop("model_prob_out_of_range")
            return None
        if not (math.isfinite(obs.market_prob) and 0.0 <= obs.market_prob <= 1.0):
            self._drop("market_prob_out_of_range")
            return None

        settlement = settlements.get(obs.event_ticker)
        if settlement is None:
            self._drop("no_settlement")
            return None
        if not math.isfinite(settlement) or settlement <= 0:
            self._drop("bad_settlement")
            return None
        # strike_type is "greater": YES pays iff the settlement average exceeds the strike.
        return 1.0 if settlement > obs.strike else 0.0

    @staticmethod
    def _scoreset(probs: np.ndarray, y: np.ndarray) -> ScoreSet:
        return ScoreSet(
            n=int(y.size),
            brier=brier_score(probs, y),
            log_loss=log_loss(probs, y),
            mean_prob=float(probs.mean()),
            base_rate=float(y.mean()),
        )

    def _bucket(
        self,
        key: np.ndarray,
        model_probs: np.ndarray,
        market_probs: np.ndarray,
        y: np.ndarray,
        edges: tuple[float, ...],
        unit: str,
    ) -> tuple[BucketScore, ...]:
        """Half-open buckets [lo, hi), with the final bucket closed at the top."""
        out: list[BucketScore] = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            last = i == len(edges) - 2
            mask = (key >= lo) & (key <= hi) if last else (key >= lo) & (key < hi)
            if not mask.any():
                continue
            m = self._scoreset(model_probs[mask], y[mask])
            k = self._scoreset(market_probs[mask], y[mask])
            hi_label = "inf" if math.isinf(hi) else f"{hi:g}"
            out.append(
                BucketScore(
                    label=f"{lo:g}-{hi_label}{unit}",
                    low=lo,
                    high=hi,
                    model=m,
                    market=k,
                    skill=skill_score(m.brier, k.brier),
                    reliable=m.n >= self.min_bucket_n,
                )
            )
        return tuple(out)


__all__ = [
    "DEFAULT_MINUTE_EDGES",
    "DEFAULT_PRICE_EDGES",
    "LOG_LOSS_EPSILON",
    "BucketScore",
    "CalibrationResult",
    "Calibrator",
    "LadderRecord",
    "Observation",
    "ProbabilityModel",
    "ReliabilityCurve",
    "ScoreSet",
    "brier_score",
    "build_observations",
    "log_loss",
    "reliability_curve",
    "skill_score",
]
