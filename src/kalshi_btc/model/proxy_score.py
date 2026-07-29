"""THE PHASE 1 GATE: does our public-data proxy actually track the settlement index?

THE QUESTION THIS FILE ANSWERS
------------------------------
KXBTCD settles on the mean of sixty one-second CF Benchmarks BRTI prints in the final
minute before close. Real-time BRTI is licensed; we do not have it. What we have is a
proxy built from three public exchange feeds (`kalshi_btc.feed.spot_ws`).

Every settled event publishes `expiration_value`: the realised BRTI 60-second average,
free and public. So for any hour we captured spot through, we can compute OUR sixty-tick
average over the identical window and compare it to the truth. That comparison is the
whole go/no-go decision for this project:

    IF our proxy tracks the settlement index to within a few dollars,
    THEN the settlement-window edge is reachable on free data.
    IF IT DOES NOT,
    THEN it is not, and no amount of modelling fixes it.

WHY THE THRESHOLD IS $5
-----------------------
Strikes are $100 apart. In the last minute the fair value of a strike is dominated by
the distance from the running average to that strike, so a $D error in our estimate of
the settlement moves our fair price by roughly D/100 of a strike interval. At $5 that is
5% of a strike - comparable to the 1-cent tick and to the taker fee - so it is noise we
can price around. At $25 it is a quarter of a strike, which is larger than the entire
edge we are hunting: we would be systematically wrong about which side of a strike the
index will land, exactly when the market is most confident. There is no clever middle
ground; either the tracking error is small relative to $100 or the strategy needs a real
BRTI feed.

The median is the headline rather than the mean because one venue outage during one hour
should not be able to condemn or rescue the verdict. p90/p95 are reported beside it
precisely so a good median with a fat tail is visible rather than hidden.

WHY AN ASOF JOIN
----------------
The true settlement samples the index once per second at close-59s .. close-0s. Our feed
is event-driven, so it prints irregularly - several times in a busy second and possibly
not at all in a quiet one. Averaging raw rows would weight busy seconds more heavily and
would silently weight venues by their update rate, which is not the settlement's
statistic. So we build the same sixty-second grid the exchange uses and, for each grid
second, take the last proxy value at or before it (an ASOF join) with a staleness bound.
That is a genuine 60x1s average of the same quantity, not an approximation of one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Strike spacing on hourly KXBTCD. Daily events use $250 and weekly $500, but the
# settlement-window trade only exists on the hourly ladder.
STRIKE_SPACING_DOLLARS = 100.0

# Median absolute error below this is a PASS. See the module docstring for the reasoning.
PASS_THRESHOLD_DOLLARS = 5.0

# A grid second whose most recent proxy print is older than this is not counted. Five
# seconds is generous: at normal BTC activity all three venues print several times a
# second, so a five-second hole means a venue outage, not a quiet market.
MAX_TICK_AGE_S = 5.0

# Events covering fewer than this many of the sixty grid seconds are excluded rather than
# scored on a partial window. Half the window is the minimum at which our average is
# measuring the same thing the exchange's average measures.
MIN_TICKS = 30


_SCORE_SQL = """
WITH s AS (
    SELECT close_time, event_ticker, expiration_value
    FROM settlements
    WHERE expiration_value IS NOT NULL AND expiration_value > 0
),
grid AS (
    -- The exact sixty seconds the exchange averages: close-59s through close-0s.
    SELECT s.event_ticker, s.close_time, s.expiration_value,
           s.close_time - to_seconds(g.i) AS tick_ts
    FROM s, generate_series(0, 59) AS g(i)
),
px AS (
    SELECT ts, proxy FROM spot WHERE proxy IS NOT NULL
)
SELECT grid.event_ticker,
       grid.close_time,
       grid.expiration_value,
       avg(px.proxy)   FILTER (WHERE px.ts >= grid.tick_ts - to_seconds(?)) AS proxy_avg,
       count(px.proxy) FILTER (WHERE px.ts >= grid.tick_ts - to_seconds(?)) AS n_ticks
FROM grid ASOF LEFT JOIN px ON grid.tick_ts >= px.ts
GROUP BY 1, 2, 3
ORDER BY 2
"""


@dataclass(frozen=True)
class ProxyScore:
    """How closely our public-data proxy reproduced the realised settlement index.

    Errors are `proxy_avg - expiration_value` in DOLLARS, so a positive `mean_error`
    means our proxy runs HIGH relative to the index.
    """

    n: int
    median_abs_error: float
    mean_error: float  # signed: the bias, not the size
    p90_abs_error: float
    p95_abs_error: float
    max_abs_error: float
    median_ticks: float  # of 60 - how complete our windows were
    strike_spacing: float
    threshold: float
    events_settled: int  # settlements on file
    events_scored: int  # of those, how many had usable spot coverage
    events_thin: int  # excluded for too few ticks in the window
    per_event: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def median_error_frac_spacing(self) -> float:
        """Median absolute error as a fraction of the $100 strike interval."""
        if self.strike_spacing <= 0:
            return float("nan")
        return self.median_abs_error / self.strike_spacing

    @property
    def p95_error_frac_spacing(self) -> float:
        if self.strike_spacing <= 0:
            return float("nan")
        return self.p95_abs_error / self.strike_spacing

    @property
    def passed(self) -> bool:
        """PASS only with a real sample. Zero events is 'unknown', not 'fine'."""
        return self.n > 0 and self.median_abs_error < self.threshold

    @property
    def verdict(self) -> str:
        if self.n == 0:
            return "NO DATA"
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "median_abs_error": self.median_abs_error,
            "mean_error": self.mean_error,
            "p90_abs_error": self.p90_abs_error,
            "p95_abs_error": self.p95_abs_error,
            "max_abs_error": self.max_abs_error,
            "median_ticks": self.median_ticks,
            "median_error_frac_spacing": self.median_error_frac_spacing,
            "p95_error_frac_spacing": self.p95_error_frac_spacing,
            "strike_spacing": self.strike_spacing,
            "threshold": self.threshold,
            "events_settled": self.events_settled,
            "events_scored": self.events_scored,
            "events_thin": self.events_thin,
            "verdict": self.verdict,
        }


def _connection(store_or_conn: Any) -> Any:
    """Accept a Store, a raw DuckDB connection, or anything exposing `.conn`."""
    conn = getattr(store_or_conn, "conn", store_or_conn)
    if not hasattr(conn, "execute"):
        raise TypeError(
            f"score_proxy() needs a Store or a DuckDB connection, got {type(store_or_conn).__name__}"
        )
    return conn


def score_proxy(
    store_or_conn: Any,
    *,
    strike_spacing: float = STRIKE_SPACING_DOLLARS,
    threshold_dollars: float = PASS_THRESHOLD_DOLLARS,
    min_ticks: int = MIN_TICKS,
    max_tick_age_s: float = MAX_TICK_AGE_S,
) -> ProxyScore:
    """Score the captured spot proxy against every settled event's `expiration_value`.

    Returns a ProxyScore with `n == 0` when nothing overlaps yet - that is the normal
    state on a fresh install and is reported as NO DATA, never as a pass.
    """
    conn = _connection(store_or_conn)

    settled = conn.execute(
        "SELECT count(*) FROM settlements WHERE expiration_value IS NOT NULL"
    ).fetchone()[0]

    df = conn.execute(_SCORE_SQL, [max_tick_age_s, max_tick_age_s]).df()
    if df.empty:
        return _empty(strike_spacing, threshold_dollars, settled, 0)

    df = df.dropna(subset=["proxy_avg"])
    n_any = len(df)
    df = df[df["n_ticks"] >= min_ticks].copy()
    thin = n_any - len(df)
    if df.empty:
        return _empty(strike_spacing, threshold_dollars, settled, thin)

    # expiration_value arrives as DECIMAL; force float once, here, so every statistic
    # below is plain numpy rather than a mix of Decimal and float.
    truth = df["expiration_value"].astype(float).to_numpy()
    ours = df["proxy_avg"].astype(float).to_numpy()
    err = ours - truth
    abs_err = np.abs(err)

    df["proxy_avg"] = ours
    df["expiration_value"] = truth
    df["error"] = err
    df["abs_error"] = abs_err
    df = df.sort_values("close_time")

    return ProxyScore(
        n=len(df),
        median_abs_error=float(np.median(abs_err)),
        mean_error=float(np.mean(err)),
        p90_abs_error=float(np.percentile(abs_err, 90)),
        p95_abs_error=float(np.percentile(abs_err, 95)),
        max_abs_error=float(np.max(abs_err)),
        median_ticks=float(np.median(df["n_ticks"].to_numpy())),
        strike_spacing=float(strike_spacing),
        threshold=float(threshold_dollars),
        events_settled=int(settled),
        events_scored=len(df),
        events_thin=int(thin),
        per_event=df.reset_index(drop=True),
    )


def _empty(spacing: float, threshold: float, settled: int, thin: int) -> ProxyScore:
    nan = float("nan")
    return ProxyScore(
        n=0,
        median_abs_error=nan,
        mean_error=nan,
        p90_abs_error=nan,
        p95_abs_error=nan,
        max_abs_error=nan,
        median_ticks=0.0,
        strike_spacing=float(spacing),
        threshold=float(threshold),
        events_settled=int(settled),
        events_scored=0,
        events_thin=int(thin),
        per_event=pd.DataFrame(),
    )


def explain(score: ProxyScore) -> str:
    """The verdict in plain sentences, with no spin in either direction.

    This text is what a human reads before deciding whether to keep building, so it says
    what a failure MEANS rather than merely that one occurred.
    """
    if score.n == 0:
        return (
            "NO DATA. No settled event yet has both a published expiration_value and "
            f"at least {MIN_TICKS} of its 60 settlement seconds covered by captured spot "
            f"({score.events_settled:,} settlement(s) on file, {score.events_thin} event(s) "
            "had partial spot coverage). Run `kbtc capture` across at least one full hourly "
            "close, then `kbtc settlements`, and score again. This is not a pass."
        )
    if score.passed:
        return (
            f"PASS. Over {score.n} settled event(s) the proxy's 60-second average missed the "
            f"realised BRTI settlement by a median of ${score.median_abs_error:,.2f} "
            f"({score.median_error_frac_spacing:.1%} of the ${score.strike_spacing:,.0f} strike "
            f"interval), with a bias of ${score.mean_error:+,.2f} and a p95 of "
            f"${score.p95_abs_error:,.2f}. That is small relative to a strike, so the "
            "settlement-window edge is reachable on free public data. The bias is systematic "
            "and should be corrected for; the p95 is the number that sizes the risk."
        )
    return (
        f"FAIL. Over {score.n} settled event(s) the proxy's 60-second average missed the "
        f"realised BRTI settlement by a median of ${score.median_abs_error:,.2f} "
        f"({score.median_error_frac_spacing:.1%} of the ${score.strike_spacing:,.0f} strike "
        f"interval), against a ${score.threshold:,.0f} threshold, with a p95 of "
        f"${score.p95_abs_error:,.2f}.\n\n"
        "WHAT THIS MEANS: at this tracking error we cannot tell which side of a strike the "
        "index will settle on, in the one minute where that is the entire trade. The "
        "settlement-window edge is NOT reachable from these public feeds. Options, honestly "
        f"ranked: (1) pay for a real BRTI feed - Kalshi proxies it on the cfbenchmarks_value "
        "channel with credentials, cost 50; (2) if the bias term dominates the spread "
        f"(mean error ${score.mean_error:+,.2f} vs median absolute ${score.median_abs_error:,.2f}), "
        "a constant correction may recover most of it - re-score after applying one; "
        "(3) abandon the settlement-window trade and keep only strategies that do not "
        "depend on knowing the index in real time."
    )


__all__ = [
    "MAX_TICK_AGE_S",
    "MIN_TICKS",
    "PASS_THRESHOLD_DOLLARS",
    "STRIKE_SPACING_DOLLARS",
    "ProxyScore",
    "explain",
    "score_proxy",
]
