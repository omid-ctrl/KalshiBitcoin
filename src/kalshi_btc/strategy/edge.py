"""Fair value for the KXBTCD strike ladder, and the edge it implies against the book.

WHAT THIS MODULE IS FOR
-----------------------
`model/pricing.py` prices ONE strike given a spot, a sigma and a horizon. That is the
physics. This module is the trading layer on top of it: it takes the whole 188-strike
ladder as the venue reports it, produces a fair probability for every strike, and turns
that into a signed edge against the actual bid and ask we could trade on right now.

THREE DECISIONS ARE MADE HERE, AND THEY ALL MATTER
--------------------------------------------------
1. FAT TAILS BY DEFAULT.
   Measured on 1,596 real hourly settlements (2026-05-22..2026-07-29): sigma is
   0.466%/hour and EXCESS KURTOSIS IS 12.88. A Gaussian has excess kurtosis 0. Pricing a
   digital off a Gaussian when the truth has tails that heavy systematically UNDERPRICES
   every out-of-the-money strike — which is precisely where a 1-cent tick means the
   market is quoting in 1-cent increments over a range where the true probability moves
   by tenths of a cent. So the default distribution is Student-t.

   The degrees of freedom are not a free parameter: a Student-t with df > 4 has excess
   kurtosis 6/(df - 4), so matching 12.88 pins df = 4 + 6/12.88 = 4.47. That is what
   `DEFAULT_DF` is, and `student_t_df_for_excess_kurtosis` is the inverse map so a
   recalibration can update it from data instead of taste.

   Caveat kept deliberately visible: the 12.88 was measured on hour-over-hour SETTLEMENT
   returns, so it describes the terminal variable we price, which is the one we need. We
   do NOT model horizon-dependent kurtosis (real returns get fatter-tailed as the horizon
   shortens). Holding df fixed across the hour is therefore conservative early and mildly
   optimistic in the last seconds — and the last seconds are handled by the in-window
   pricer, where the residual distribution is a sum of ~m ticks and closer to normal anyway.

2. THE IN-WINDOW PRICER IS USED WHENEVER IT CAN BE.
   Inside the final 60 seconds part of the settlement average is already a known constant.
   Ignoring that overstates residual uncertainty by up to ~21x with five seconds left. If
   a `SettlementWindow` is supplied (i.e. we actually have the running BRTI average), the
   engine switches to `price_above_in_window`. Without it we fall back to the standard
   path and say so in the reason string, because pretending to know the running average is
   how you lose money confidently.

3. EVERY TAKER ACTION IS GATED ON `fees.min_taker_edge`.
   The taker fee is ceil(0.07 * C * P * (1-P)) — ~1.75c at the money, and MAKER FEES ARE
   ZERO. `min_taker_edge(price)` returns the fee plus a half-spread buffer. We require the
   raw book edge to clear that whole hurdle, which is deliberately conservative: the fee
   part covers the fee, and the half-cent buffer covers the fact that `fair` is an
   ESTIMATE from an estimated sigma and an estimated spot. Trades that clear the fee but
   not the buffer are recorded with `takeable=False` and a reason, not silently dropped.

SPOT WITHOUT A LICENSED FEED
----------------------------
Real-time BRTI is a licensed product; Kalshi proxies it but only with credentials. When
there is no BRTI tick, `implied_spot_from_ladder` inverts the market's own mids through
the pricer and takes the median. That makes the level of our curve agree with the market
by construction and leaves only the SHAPE as our signal — which is the honest description
of what a credential-less bot can actually see, and it is stated in the reason strings so
nobody mistakes shape edge for directional edge.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Sequence

from scipy import stats

from kalshi_btc.core.fees import min_taker_edge, taker_fee_per_contract
from kalshi_btc.core.types import Book, MarketSnapshot, Side, dec
from kalshi_btc.model.pricing import (
    TICKS,
    Quote,
    price_above,
    price_above_in_window,
    settlement_std_dollars,
)

# Measured on 1,596 real KXBTCD hourly settlements, 2026-05-22..2026-07-29.
MEASURED_SIGMA_PER_HOUR = 0.00466
MEASURED_EXCESS_KURTOSIS = 12.88

# Prices are on a 1-cent grid, 1..99. A "price" of 0 or 1 is not a quote, it is a pin.
TICK = Decimal("0.01")
MIN_PRICE = Decimal("0.01")
MAX_PRICE = Decimal("0.99")

# KXBTCD spreads are locked at the 1-cent minimum tick. A strike showing a 14-cent spread
# is not a wide market, it is a BROKEN one — a half-refreshed REST cache or a book with one
# side pulled — and its mid is not a fair value. Measured live on 2026-07-29: healthy
# strikes quote exactly 1 cent wide, and the moments this bot found 25-cent "edges" were
# all moments when the ladder had collapsed to one or two strikes quoted 14 cents wide.
MAX_QUOTABLE_SPREAD = Decimal("0.03")

# Fewer quotable strikes than this and the ladder cannot identify a spot level: with two
# strikes the median of the implied spots is an average of two noisy numbers, and with one
# it is that strike's own mid restated, which by construction has zero edge and tells us
# nothing about its neighbours.
MIN_STRIKES_FOR_SPOT = 3

# How far the per-strike implied spots may disagree, in units of the residual settlement
# sigma, before we declare the ladder internally inconsistent.
#
# Calibrated, not guessed. 35 consecutive 2-second samples of the live KXBTCD hourly
# ladder on 2026-07-29 (3-4 quotable strikes, 8-10 minutes to close) gave a dispersion of
# median 0.17 sigma, p90 0.46, MAX 0.47 - every single sample comfortably inside 0.75.
# During the same session a separate stretch sat at 1.0-1.24 sigma for 45 seconds while
# the underlying was moving and the REST-cached ladder could not keep its strikes in sync
# with each other. 0.75 sits cleanly between the two populations.
MAX_SPOT_DISPERSION_STD = 0.75


def student_t_df_for_excess_kurtosis(excess_kurtosis: float) -> float:
    """Degrees of freedom whose excess kurtosis matches the measurement.

    Var-standardised Student-t has excess kurtosis 6/(df-4) for df > 4, so the inverse is
    df = 4 + 6/k. Clamped to df > 4.05 because the kurtosis of a t with df <= 4 is
    infinite and the pricer needs df > 2 merely for a finite variance — an infinite fourth
    moment would make the calibration meaningless even where the price is well defined.
    """
    if not math.isfinite(excess_kurtosis) or excess_kurtosis <= 0:
        return 1e6  # effectively Gaussian
    return max(4.05, 4.0 + 6.0 / excess_kurtosis)


DEFAULT_DIST = "t"
DEFAULT_DF = student_t_df_for_excess_kurtosis(MEASURED_EXCESS_KURTOSIS)  # ~4.466


# --------------------------------------------------------------------------------------
# Unit-variance innovation helpers (mirrors pricing._tail_prob, which has no public inverse)
# --------------------------------------------------------------------------------------
def _t_scale(df: float) -> float:
    """Factor that rescales a raw t_df to unit variance."""
    return math.sqrt(df / (df - 2.0))


def unit_tail_inverse(p: float, dist: str = DEFAULT_DIST, df: float = DEFAULT_DF) -> float:
    """z such that P(X > z) = p for a zero-mean UNIT-VARIANCE innovation."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    if dist == "normal":
        return float(stats.norm.isf(p))
    if dist == "t":
        if df <= 2:
            raise ValueError("Student-t needs df > 2 for finite variance")
        return float(stats.t.isf(p, df)) / _t_scale(df)
    raise ValueError(f"unknown dist {dist!r}")


def unit_density(z: float, dist: str = DEFAULT_DIST, df: float = DEFAULT_DF) -> float:
    """Density at z of the zero-mean UNIT-VARIANCE innovation.

    This is |d P(X > z) / dz|, i.e. how fast the fair probability moves when the
    standardised distance to the strike moves. The quoting module uses it to price
    adverse selection, so it must be the density of the SAME distribution we priced with.
    """
    if dist == "normal":
        return float(stats.norm.pdf(z))
    if dist == "t":
        s = _t_scale(df)
        return float(stats.t.pdf(z * s, df)) * s
    raise ValueError(f"unknown dist {dist!r}")


# --------------------------------------------------------------------------------------
# Market state
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LadderQuote:
    """One strike's tradeable state, normalised away from the two API shapes.

    `yes_bid`/`yes_ask` are None when that side is absent. The venue reports a missing bid
    as 0.00 and a missing ask as 1.00, which are legal-looking numbers that quietly turn
    "nobody is there" into "you can sell at zero". Normalising them to None at the
    boundary is the cheapest place to kill that class of bug.
    """

    ticker: str
    strike: Decimal
    yes_bid: Decimal | None
    yes_ask: Decimal | None
    yes_bid_size: Decimal = Decimal("0")
    yes_ask_size: Decimal = Decimal("0")

    @staticmethod
    def _clean_bid(p: Decimal | None) -> Decimal | None:
        return None if p is None or p < MIN_PRICE else p

    @staticmethod
    def _clean_ask(p: Decimal | None) -> Decimal | None:
        return None if p is None or p > MAX_PRICE else p

    @classmethod
    def from_snapshot(cls, m: MarketSnapshot) -> LadderQuote:
        return cls(
            ticker=m.ticker,
            strike=m.strike,
            yes_bid=cls._clean_bid(m.yes_bid),
            yes_ask=cls._clean_ask(m.yes_ask),
            yes_bid_size=m.yes_bid_size,
            yes_ask_size=m.yes_ask_size,
        )

    @classmethod
    def from_book(cls, book: Book, strike: Decimal) -> LadderQuote:
        """Build from an L2 book.

        The ask side is derived: Kalshi holds both sides as resting BIDS, so the size
        available to LIFT at yes_ask is the size of the NO bid at 1 - yes_ask.
        """
        bid, ask = book.best_yes_bid, book.best_yes_ask
        return cls(
            ticker=book.ticker,
            strike=strike,
            yes_bid=cls._clean_bid(bid),
            yes_ask=cls._clean_ask(ask),
            yes_bid_size=(
                book.size_at(Side.YES, bid) if bid is not None else Decimal("0")
            ),
            yes_ask_size=(
                book.size_at(Side.NO, Decimal("1") - ask) if ask is not None else Decimal("0")
            ),
        )

    @property
    def is_two_sided(self) -> bool:
        return self.yes_bid is not None and self.yes_ask is not None

    @property
    def mid(self) -> Decimal | None:
        if not self.is_two_sided:
            return None
        return (self.yes_bid + self.yes_ask) / 2  # type: ignore[operator]

    @property
    def spread(self) -> Decimal | None:
        if not self.is_two_sided:
            return None
        return self.yes_ask - self.yes_bid  # type: ignore[operator]

    @property
    def is_quotable(self) -> bool:
        """Two-sided, not pinned, and quoted at a credible width.

        All three conditions are load-bearing. Pinned strikes carry no information in
        their mid, and abnormally wide strikes carry WRONG information — see
        MAX_QUOTABLE_SPREAD.
        """
        m, s = self.mid, self.spread
        if m is None or s is None:
            return False
        return Decimal("0.02") <= m <= Decimal("0.98") and s <= MAX_QUOTABLE_SPREAD


@dataclass(frozen=True)
class SettlementWindow:
    """The running state of the 60-second BRTI averaging window.

    `known_ticks` is how many of the 60 settlement ticks have already printed, and
    `known_sum` is their sum. Both come from the `cfbenchmarks_value` feed
    (`tick_count` and `windowed_avg * tick_count`); neither can be guessed.
    """

    known_sum: float
    known_ticks: int
    spot_now: float

    @property
    def is_active(self) -> bool:
        return 0 < self.known_ticks < TICKS and self.spot_now > 0

    @classmethod
    def from_windowed_average(
        cls, windowed_avg: float, tick_count: int, spot_now: float
    ) -> SettlementWindow:
        return cls(known_sum=windowed_avg * tick_count, known_ticks=tick_count, spot_now=spot_now)


# --------------------------------------------------------------------------------------
# Edge
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StrikeEdge:
    """Fair value and both signed edges for a single strike.

    Sign convention, fixed once here so nothing downstream has to think about it:
      edge_buy_yes  = fair - yes_ask   (what we gain per contract by LIFTING the offer)
      edge_sell_yes = yes_bid - fair   (what we gain per contract by HITTING the bid)
    Both are in dollars per contract and both are positive when the trade is attractive.
    """

    ticker: str
    strike: Decimal
    fair_prob: float
    fair: Decimal
    yes_bid: Decimal | None
    yes_ask: Decimal | None
    yes_bid_size: Decimal
    yes_ask_size: Decimal
    mid: Decimal | None
    edge_buy_yes: Decimal | None
    edge_sell_yes: Decimal | None
    hurdle_buy: Decimal | None
    hurdle_sell: Decimal | None
    takeable_buy: bool
    takeable_sell: bool
    residual_std: float
    minutes_to_close: float
    z_score: float
    in_window: bool
    reason: str

    @property
    def best_edge(self) -> Decimal:
        """The larger of the two signed edges, or 0 when neither side is quoted."""
        vals = [e for e in (self.edge_buy_yes, self.edge_sell_yes) if e is not None]
        return max(vals) if vals else Decimal("0")

    @property
    def any_takeable(self) -> bool:
        return self.takeable_buy or self.takeable_sell

    def describe(self) -> str:
        b = "--" if self.yes_bid is None else f"{self.yes_bid:.2f}"
        a = "--" if self.yes_ask is None else f"{self.yes_ask:.2f}"
        return (
            f"{self.strike:,.0f} fair={self.fair_prob:.4f} mkt={b}/{a} "
            f"edge_buy={_c(self.edge_buy_yes)} edge_sell={_c(self.edge_sell_yes)} "
            f"[{self.reason}]"
        )


def _c(x: Decimal | None) -> str:
    return "--" if x is None else f"{x * 100:+.2f}c"


@dataclass(frozen=True)
class LadderEdges:
    """Every strike's edge for one instant, plus the inputs that produced them."""

    ts: datetime
    event_ticker: str
    spot: float
    spot_source: str
    sigma_per_minute: float
    minutes_to_close: float
    dist: str
    df: float
    in_window: bool
    edges: list[StrikeEdge] = field(default_factory=list)

    def takeable(self) -> list[StrikeEdge]:
        return [e for e in self.edges if e.any_takeable]

    def by_ticker(self) -> dict[str, StrikeEdge]:
        return {e.ticker: e for e in self.edges}

    @property
    def atm(self) -> StrikeEdge | None:
        """The strike our model puts closest to a coin flip."""
        if not self.edges:
            return None
        return min(self.edges, key=lambda e: abs(e.fair_prob - 0.5))


@dataclass(frozen=True)
class FairValueEngine:
    """Turns a ladder plus (spot, sigma, horizon) into per-strike fair values and edges.

    Frozen because it is configuration, not state: nothing about a quote should depend on
    what the engine saw last cycle. All the time-varying inputs are arguments.
    """

    dist: str = DEFAULT_DIST
    df: float = DEFAULT_DF
    # Buffer added on top of the taker fee before a cross is allowed. Defaults to the
    # `fees.min_taker_edge` half-spread, i.e. our fair value must beat the fee AND leave
    # half a tick of room for being wrong.
    taker_buffer: Decimal = Decimal("0.005")
    # Below this residual sigma the strike is decided; a "fair value" there is arithmetic,
    # not a forecast, and quoting off it is how you get picked off by the settlement print.
    min_residual_std_dollars: float = 0.5

    # How wrong our SPOT input is, in dollars, versus the index the contract actually
    # settles on. This is the single most important number in this file when running off
    # the public spot proxy rather than the licensed BRTI tape.
    #
    # WHY IT MUST GATE TRADING
    # ------------------------
    # Fair value is P(settlement > K), so an error in spot propagates to an error in
    # probability with derivative  dp/dS = density(z) / sigma_settlement.  That 1/sigma is
    # the whole problem: the Asian-settlement edge exists BECAUSE sigma collapses in the
    # final minute, and the proxy error is amplified by exactly the same 1/sigma. The edge
    # and the noise scale identically, so a proxy that is merely "close" is not close
    # enough precisely where the strategy makes its money.
    #
    # Measured with the free Coinbase/Kraken/Bitstamp composite (`kbtc proxy-score`),
    # the tracking error against the realised BRTI settlement was $9.04. At the money that
    # is worth ~1.2c of probability error an hour out, but ~16c when the settlement window
    # opens and ~124c forty-five seconds in - against a taker hurdle of 2.25c. Trading on
    # that is not trading on edge, it is trading on our own measurement error.
    #
    # Set to 0.0 only when spot IS the settlement index (the authenticated
    # `cfbenchmarks_value` feed). Anything else should carry its measured error.
    spot_uncertainty_dollars: float = 0.0
    # Multiple of the propagated probability error to require on top of the fee hurdle.
    # 1.0 means "the edge must exceed one standard error of our own spot input".
    spot_uncertainty_k: float = 1.0

    def probability_uncertainty(self, quote: Quote, z: float) -> Decimal:
        """Probability error implied by our spot uncertainty at this strike.

        dp/dS = density(z) / sigma_settlement, so the probability error is
        density(z) * spot_error / sigma. Returns 0 when spot is exact.
        """
        if self.spot_uncertainty_dollars <= 0 or quote.residual_std <= 0:
            return Decimal("0")
        dp = unit_density(z, self.dist, self.df) * self.spot_uncertainty_dollars / quote.residual_std
        return _to_dollars(min(dp, 1.0))

    # ---------------------------------------------------------------- fair value
    def fair_quote(
        self,
        strike: Decimal | float,
        spot: float,
        sigma_per_minute: float,
        minutes_to_close: float,
        *,
        window: SettlementWindow | None = None,
    ) -> Quote:
        """One strike's model quote, in-window pricer when we can, standard path otherwise."""
        k = float(strike)
        if window is not None and window.is_active:
            return price_above_in_window(
                k,
                window.known_sum,
                window.known_ticks,
                window.spot_now,
                sigma_per_minute,
                dist=self.dist,
                df=self.df,
            )
        return price_above(
            spot, k, sigma_per_minute, max(0.0, minutes_to_close), dist=self.dist, df=self.df
        )

    # ---------------------------------------------------------------- edges
    def evaluate(
        self,
        quotes: Sequence[LadderQuote],
        *,
        spot: float,
        sigma_per_minute: float,
        minutes_to_close: float,
        event_ticker: str = "",
        window: SettlementWindow | None = None,
        spot_source: str = "unknown",
        ts: datetime | None = None,
    ) -> LadderEdges:
        """Price the whole ladder and compute both signed edges against the live book."""
        ts = ts or datetime.now(UTC)
        in_window = window is not None and window.is_active
        out: list[StrikeEdge] = []

        for q in quotes:
            mq = self.fair_quote(
                q.strike, spot, sigma_per_minute, minutes_to_close, window=window
            )
            fair_prob = min(max(mq.prob_above, 0.0), 1.0)
            fair = _to_dollars(fair_prob)
            z = _z_of(mq, float(q.strike), spot, window)

            reasons: list[str] = []
            if in_window:
                reasons.append(f"in-window m={TICKS - window.known_ticks}")  # type: ignore[union-attr]
            if spot_source != "unknown":
                reasons.append(f"spot={spot_source}")
            decided = mq.residual_std < self.min_residual_std_dollars

            edge_buy = None if q.yes_ask is None else fair - q.yes_ask
            edge_sell = None if q.yes_bid is None else q.yes_bid - fair

            # Our own spot input is uncertain, so require the edge to clear that too.
            # Without this the bot happily "finds" 3-4c of edge in the settlement window
            # that is entirely proxy tracking error, takes it, and loses.
            spot_noise = self.probability_uncertainty(mq, z) * Decimal(
                str(self.spot_uncertainty_k)
            )
            if spot_noise > 0:
                reasons.append(f"spot_noise={float(spot_noise) * 100:.1f}c")

            hurdle_buy = None if q.yes_ask is None else self.hurdle(q.yes_ask) + spot_noise
            hurdle_sell = None if q.yes_bid is None else self.hurdle(q.yes_bid) + spot_noise

            takeable_buy = bool(
                edge_buy is not None and hurdle_buy is not None and edge_buy >= hurdle_buy
            )
            takeable_sell = bool(
                edge_sell is not None and hurdle_sell is not None and edge_sell >= hurdle_sell
            )
            if decided:
                # Residual uncertainty has collapsed: our "edge" is now a bet that the
                # already-locked-in ticks say something different from what the tape says.
                # They do not.
                takeable_buy = takeable_sell = False
                reasons.append(f"decided residual={mq.residual_std:.1f}")
            if not q.is_two_sided:
                reasons.append("one-sided")
            if takeable_buy:
                reasons.append("TAKE_BUY")
            if takeable_sell:
                reasons.append("TAKE_SELL")
            if not (takeable_buy or takeable_sell) and not decided:
                reasons.append(_shortfall(edge_buy, hurdle_buy, edge_sell, hurdle_sell))

            out.append(
                StrikeEdge(
                    ticker=q.ticker,
                    strike=q.strike,
                    fair_prob=fair_prob,
                    fair=fair,
                    yes_bid=q.yes_bid,
                    yes_ask=q.yes_ask,
                    yes_bid_size=q.yes_bid_size,
                    yes_ask_size=q.yes_ask_size,
                    mid=q.mid,
                    edge_buy_yes=edge_buy,
                    edge_sell_yes=edge_sell,
                    hurdle_buy=hurdle_buy,
                    hurdle_sell=hurdle_sell,
                    takeable_buy=takeable_buy,
                    takeable_sell=takeable_sell,
                    residual_std=mq.residual_std,
                    minutes_to_close=mq.minutes_to_close,
                    z_score=z,
                    in_window=in_window,
                    reason=" ".join(r for r in reasons if r),
                )
            )

        return LadderEdges(
            ts=ts,
            event_ticker=event_ticker,
            spot=spot,
            spot_source=spot_source,
            sigma_per_minute=sigma_per_minute,
            minutes_to_close=minutes_to_close,
            dist=self.dist,
            df=self.df,
            in_window=in_window,
            edges=out,
        )

    def hurdle(self, price: Decimal) -> Decimal:
        """Minimum book edge, in dollars per contract, before crossing is allowed.

        This is `fees.min_taker_edge` and nothing else. Every taker decision in the
        codebase goes through here so there is exactly one place to audit.
        """
        return min_taker_edge(price, half_spread=self.taker_buffer)

    def taker_cost(self, price: Decimal) -> Decimal:
        """Per-contract taker fee at `price` (unrounded — the right form for EV maths)."""
        return taker_fee_per_contract(price)


def _shortfall(
    edge_buy: Decimal | None,
    hurdle_buy: Decimal | None,
    edge_sell: Decimal | None,
    hurdle_sell: Decimal | None,
) -> str:
    """Human-readable 'how far off were we', for the decisions table."""
    gaps: list[tuple[Decimal, str]] = []
    if edge_buy is not None and hurdle_buy is not None:
        gaps.append((hurdle_buy - edge_buy, "buy"))
    if edge_sell is not None and hurdle_sell is not None:
        gaps.append((hurdle_sell - edge_sell, "sell"))
    if not gaps:
        return "no-quote"
    gap, which = min(gaps)
    return f"below hurdle by {gap * 100:.2f}c ({which})"


def _to_dollars(prob: float) -> Decimal:
    """Probability -> price in dollars, at sub-cent precision (money stays Decimal)."""
    return dec(f"{prob:.6f}")


def _z_of(q: Quote, strike: float, spot: float, window: SettlementWindow | None) -> float:
    """Standardised distance to the strike, recovered from the quote's own residual std."""
    if q.residual_std <= 0:
        return math.inf if strike > spot else -math.inf
    if window is not None and window.is_active:
        m = TICKS - window.known_ticks
        projected = (window.known_sum + m * window.spot_now) / TICKS
        return (strike - projected) / q.residual_std
    return (strike - spot) / q.residual_std


# --------------------------------------------------------------------------------------
# Spot inference from the ladder
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpotEstimate:
    """A ladder-implied spot together with the reasons to distrust it."""

    value: float | None
    n_strikes: int
    dispersion: float  # max - min of the per-strike implied spots, in dollars
    dispersion_std: float  # the same, in units of the residual settlement sigma
    usable: bool
    reason: str

    def describe(self) -> str:
        if self.value is None:
            return f"no spot: {self.reason}"
        return (
            f"spot={self.value:,.1f} from {self.n_strikes} strikes, dispersion "
            f"${self.dispersion:,.0f} ({self.dispersion_std:.2f} sigma): {self.reason}"
        )


def estimate_spot_from_ladder(
    quotes: Sequence[LadderQuote],
    sigma_per_minute: float,
    minutes_to_close: float,
    *,
    dist: str = DEFAULT_DIST,
    df: float = DEFAULT_DF,
    iterations: int = 4,
    min_strikes: int = MIN_STRIKES_FOR_SPOT,
    max_dispersion_std: float = MAX_SPOT_DISPERSION_STD,
) -> SpotEstimate:
    """Spot implied by the quotable strikes' mids, with a usability verdict.

    Each quotable strike inverts exactly: mid = P(S > k) gives z = F^-1(mid) and hence
    spot = k - z * std. The std itself depends on spot (sigma is a fraction), so we iterate
    from an anchor a few times — it converges immediately because the dependence is second
    order.

    The MEDIAN, not the mean or a least-squares fit: with only 4-9 quotable strikes and a
    1-cent tick, one strike sitting a tick away from its neighbours would drag a mean
    around by tens of dollars. The median ignores it.

    TWO GATES, BOTH LEARNED THE HARD WAY on a live 45-second paper session:

    * `min_strikes`. With one quotable strike the median IS that strike's own mid and the
      "edge" it implies on its neighbours is pure noise. The session that motivated this
      reported 25-cent edges, every one of them from a cycle where the ladder had
      collapsed to one or two strikes.

    * `max_dispersion_std`. Even with enough strikes, if they do not agree on a level then
      the ladder is internally inconsistent — a stale side, a half-updated cache, or a
      market whose shape genuinely disagrees with our distribution. In all three cases the
      right answer is to sit out the cycle, because a "20-cent edge" against a book that
      does not agree with itself is a model disagreement, not an opportunity.
    """
    live = [q for q in quotes if q.is_quotable]
    if minutes_to_close <= 0:
        return SpotEstimate(None, len(live), 0.0, 0.0, False, "event has closed")
    if len(live) < min_strikes:
        return SpotEstimate(
            None, len(live), 0.0, 0.0, False,
            f"only {len(live)} quotable strike(s), need {min_strikes} to identify a level",
        )

    anchor = float(min(live, key=lambda q: abs((q.mid or Decimal("0.5")) - Decimal("0.5"))).strike)
    spot = anchor
    implied: list[float] = []
    std = 0.0
    for _ in range(max(1, iterations)):
        std = settlement_std_dollars(sigma_per_minute, spot, minutes_to_close)
        if std <= 0:
            return SpotEstimate(spot, len(live), 0.0, 0.0, False, "residual sigma has collapsed")
        implied = [
            float(q.strike) - unit_tail_inverse(float(q.mid), dist, df) * std  # type: ignore[arg-type]
            for q in live
        ]
        spot = float(statistics.median(implied))

    dispersion = max(implied) - min(implied)
    dispersion_std = dispersion / std if std > 0 else float("inf")
    if dispersion_std > max_dispersion_std:
        return SpotEstimate(
            spot, len(live), dispersion, dispersion_std, False,
            f"ladder disagrees with itself by {dispersion_std:.2f} sigma "
            f"(limit {max_dispersion_std:.2f}) - stale book or a shape we do not share",
        )
    return SpotEstimate(spot, len(live), dispersion, dispersion_std, True, "ok")


def implied_spot_from_ladder(
    quotes: Sequence[LadderQuote],
    sigma_per_minute: float,
    minutes_to_close: float,
    *,
    dist: str = DEFAULT_DIST,
    df: float = DEFAULT_DF,
    iterations: int = 4,
) -> float | None:
    """The usable ladder-implied spot, or None. Thin wrapper over `estimate_spot_from_ladder`."""
    est = estimate_spot_from_ladder(
        quotes, sigma_per_minute, minutes_to_close, dist=dist, df=df, iterations=iterations
    )
    return est.value if est.usable else None


__all__ = [
    "DEFAULT_DF",
    "DEFAULT_DIST",
    "MAX_QUOTABLE_SPREAD",
    "MEASURED_EXCESS_KURTOSIS",
    "MEASURED_SIGMA_PER_HOUR",
    "MIN_STRIKES_FOR_SPOT",
    "FairValueEngine",
    "LadderEdges",
    "LadderQuote",
    "SettlementWindow",
    "SpotEstimate",
    "StrikeEdge",
    "estimate_spot_from_ladder",
    "implied_spot_from_ladder",
    "student_t_df_for_excess_kurtosis",
    "unit_density",
    "unit_tail_inverse",
]
