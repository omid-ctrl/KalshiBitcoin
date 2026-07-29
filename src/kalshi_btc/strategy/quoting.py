"""Maker-first quoting, and the rule for when to stop quoting.

THE ECONOMIC FACT THIS WHOLE MODULE EXISTS FOR
----------------------------------------------
On KXBTCD the maker fee multiplier is 0. A resting order that fills costs NOTHING. A taker
fill pays ceil(0.07 * C * P * (1-P)), which is 1.75c at the money and still ~1.1c at a
dime. The spread is locked at the 1-cent minimum tick, so:

    resting and getting filled   ~ +0.5c of spread capture, 0.0c of fee
    crossing to get filled       ~ -0.5c of spread paid,   -1.75c of fee

That is a swing of about 2.75 cents per contract at the money on a market whose entire
tick is 1 cent. No plausible forecasting edge on an hourly BTC digital is worth 2.75c;
several are worth 0.5c. So the default posture is passive, and taking liquidity is the
exception that has to clear `fees.min_taker_edge` before it is even considered.

WHERE WE CAN ACTUALLY QUOTE
---------------------------
With a locked 1-cent spread there is no room to quote "inside" — any improvement crosses,
which a post_only order would have rejected anyway. So a maker quote here means JOINING
the best bid or the best ask, and the real question is not price but whether we will ever
reach the front of the queue. That is why every intent carries `queue_ahead`: joining a
2,000-contract queue on a strike that trades 40 contracts an hour is not a quote, it is a
decoration, and the paper runner's fill simulator will (correctly) never fill it.

INVENTORY SKEW
--------------
Position is a forecast error we are already carrying. Long inventory means we want to be
likelier to sell than to buy. Because the spread is locked at one tick there is only one
price per side available to a maker — the touch — so the skew cannot move our price. It
moves the EDGE WE DEMAND instead: long inventory raises the capture required before we
will buy more and lowers the capture required to sell, which withdraws the side that would
deepen the position. The size of the adjustment is proportional to
position / max_position_per_strike and capped at `max_skew_ticks`.

THE ADVERSE-SELECTION GUARD — THE PART THAT IS EASY TO GET WRONG
----------------------------------------------------------------
A resting quote is an option we wrote for free. We get filled preferentially when the fair
value has just moved THROUGH our price, which is exactly when we did not want the trade.
The expected cost of that is the size of the fair-value move that triggers the fill, and
near expiry it explodes.

Quantify it properly. The fair value P_t = P(S_settle > k) is a martingale, and
P = G(z) with z = (k - projected_settlement)/residual_std. Over a refresh interval the
residual std falls from s_now to s_next, and the information released is exactly the
variance difference, so the standardised distance moves by

    Var(dz) = 1 - (s_next / s_now)^2

and the fair value therefore moves by, to first order,

    fair_move_std = g(z) * sqrt(1 - (s_next/s_now)^2)                     [ADVERSE MOVE]

with g the density of the SAME unit-variance innovation we priced with (Student-t by
default — using a normal density here while pricing with a t would understate the wings).

This has the right behaviour at both ends. Early in the hour s_next/s_now is ~1 over a
two-second refresh and the move is fractions of a cent, comfortably smaller than the 0.5c
we capture. Inside the settlement window the residual std collapses like (m/60)^1.5, so
the ratio drops fast and the bound approaches g(0) ~ 0.4 = FORTY cents of adverse move
per fill. Half a cent of spread capture never pays for that.

    PULL RULE: quote only while  captured_edge >= adverse_multiple * fair_move_std,
               and never inside `hard_pull_seconds` of the close.

The hard time floor is not redundant with the quantitative test. It covers the failure
mode where the quantitative test is fed a stale sigma or a stale spot and therefore does
not know it should be alarmed. Two independent reasons to pull, either sufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from kalshi_btc.config import RiskLimits
from kalshi_btc.core.types import Action, Liquidity, Side
from kalshi_btc.model.pricing import (
    SECONDS_PER_MINUTE,
    residual_std_in_window,
    settlement_std_dollars,
)
from kalshi_btc.strategy.edge import (
    DEFAULT_DF,
    DEFAULT_DIST,
    MAX_PRICE,
    MIN_PRICE,
    TICK,
    LadderEdges,
    StrikeEdge,
    unit_density,
)

# The spread is locked at one tick, so a passive fill captures about half of it.
SPREAD_CAPTURE = TICK / 2


@dataclass(frozen=True)
class QuoteIntent:
    """One passive order we want resting. Prices are on the 1-cent grid, always."""

    ticker: str
    strike: Decimal
    side: Side
    action: Action
    price: Decimal
    contracts: int
    fair: Decimal
    edge: Decimal  # expected capture per contract at this price, fee-free (maker)
    queue_ahead: Decimal
    reason: str
    liquidity: Liquidity = Liquidity.MAKER

    def describe(self) -> str:
        return (
            f"{self.action.value} {self.side.value} {self.contracts}x {self.ticker} "
            f"@{self.price:.2f} (fair {self.fair:.4f}, +{self.edge * 100:.2f}c, "
            f"queue {self.queue_ahead:.0f}) {self.reason}"
        )


@dataclass(frozen=True)
class PullNotice:
    """A strike we deliberately are NOT quoting, and the number that decided it."""

    ticker: str
    strike: Decimal
    reason: str
    fair_move_std: float = 0.0
    seconds_to_close: float = 0.0

    def describe(self) -> str:
        return f"{self.strike:,.0f} pulled: {self.reason}"


@dataclass(frozen=True)
class QuotePlan:
    """Everything the quoter decided this cycle, including what it refused to do."""

    intents: list[QuoteIntent] = field(default_factory=list)
    pulls: list[PullNotice] = field(default_factory=list)
    seconds_to_close: float = 0.0

    @property
    def n_quotes(self) -> int:
        return len(self.intents)

    def by_ticker(self) -> dict[str, list[QuoteIntent]]:
        out: dict[str, list[QuoteIntent]] = {}
        for i in self.intents:
            out.setdefault(i.ticker, []).append(i)
        return out


def adverse_move_std(
    z: float,
    std_now: float,
    std_next: float,
    *,
    dist: str = DEFAULT_DIST,
    df: float = DEFAULT_DF,
) -> float:
    """Std of the fair-value move over one refresh, in probability units.

    See [ADVERSE MOVE] in the module docstring. Returns the worst case (the density at the
    money) when `std_now` has already collapsed to zero, because at that point a resting
    quote is a free option on a decided outcome.
    """
    if std_now <= 0.0:
        return unit_density(0.0, dist, df)
    ratio = max(0.0, min(1.0, std_next / std_now))
    return unit_density(z, dist, df) * math.sqrt(max(0.0, 1.0 - ratio * ratio))


def residual_std_after(
    sigma_per_minute: float,
    spot: float,
    minutes_to_close: float,
    refresh_seconds: float,
) -> tuple[float, float]:
    """(std now, std after one refresh) of the outstanding settlement, in dollars.

    Uses the in-window formula once we are inside the final minute, because that is where
    the ratio between the two collapses and the whole guard earns its keep.
    """
    if minutes_to_close <= 1.0:
        elapsed = max(0.0, (1.0 - minutes_to_close) * SECONDS_PER_MINUTE)
        now = residual_std_in_window(sigma_per_minute, spot, elapsed)
        nxt = residual_std_in_window(sigma_per_minute, spot, elapsed + refresh_seconds)
        return now, nxt
    now = settlement_std_dollars(sigma_per_minute, spot, minutes_to_close)
    nxt = settlement_std_dollars(
        sigma_per_minute, spot, max(0.0, minutes_to_close - refresh_seconds / SECONDS_PER_MINUTE)
    )
    return now, nxt


def round_to_tick(price: Decimal, *, mode: str = ROUND_HALF_UP) -> Decimal:
    """Snap to the 1-cent grid and keep it inside 0.01..0.99."""
    p = price.quantize(TICK, rounding=mode)
    return min(max(p, MIN_PRICE), MAX_PRICE)


@dataclass(frozen=True)
class MakerQuoter:
    """Turns per-strike fair values into passive quotes, or into an explicit refusal.

    Frozen for the same reason `FairValueEngine` is: this is policy, and policy that
    remembers last cycle is policy nobody can reason about.
    """

    risk: RiskLimits
    # Minimum expected capture (dollars/contract) to bother resting an order. Slightly
    # under a half-tick so joining the touch at exactly fair still qualifies.
    min_quote_edge: Decimal = Decimal("0.004")
    # How much of our own spread capture we insist on keeping after adverse selection.
    # 2.0 means: only quote while we expect to capture at least twice what informed flow
    # is expected to take off us per fill.
    adverse_multiple: float = 2.0
    # Assumed reprice cadence. The adverse-selection integral is over exactly this window,
    # so quoting slower than you claim here is not conservative, it is wrong.
    refresh_seconds: float = 2.0
    # Never quote inside this many seconds of close, whatever the numbers say.
    hard_pull_seconds: float = 20.0
    # Maximum inventory skew, in ticks.
    max_skew_ticks: int = 2
    # Do not join a queue deeper than this many contracts; we would never reach the front.
    max_queue_ahead: Decimal = Decimal("500")
    # Strikes this far into the wings have no fills to win and plenty of gap risk.
    max_abs_z: float = 3.0
    dist: str = DEFAULT_DIST
    df: float = DEFAULT_DF

    # ------------------------------------------------------------------ public
    def plan(
        self,
        ladder: LadderEdges,
        *,
        positions: dict[str, int] | None = None,
        sizes: dict[str, int] | None = None,
    ) -> QuotePlan:
        """Quote (or refuse to quote) every strike in `ladder`.

        `sizes` optionally overrides the contract count per ticker — that is where the
        Kelly/ladder sizing result gets injected. Absent an entry we quote the per-order
        maximum, which is the venue-facing clip size, not a view.
        """
        positions = positions or {}
        sizes = sizes or {}
        seconds_to_close = ladder.minutes_to_close * SECONDS_PER_MINUTE

        intents: list[QuoteIntent] = []
        pulls: list[PullNotice] = []

        if seconds_to_close <= self.hard_pull_seconds:
            # One notice for the whole ladder: a per-strike list here would be 188 lines
            # saying the same thing.
            pulls.append(
                PullNotice(
                    ticker=ladder.event_ticker,
                    strike=Decimal("0"),
                    reason=(
                        f"inside the hard pull window ({seconds_to_close:.0f}s to close "
                        f"<= {self.hard_pull_seconds:.0f}s); resting quotes near the "
                        "settlement print are pure adverse selection"
                    ),
                    seconds_to_close=seconds_to_close,
                )
            )
            return QuotePlan(intents=[], pulls=pulls, seconds_to_close=seconds_to_close)

        std_now, std_next = residual_std_after(
            ladder.sigma_per_minute, ladder.spot, ladder.minutes_to_close, self.refresh_seconds
        )

        for e in ladder.edges:
            pull = self._pull_reason(e, std_now, std_next, seconds_to_close)
            if pull is not None:
                pulls.append(pull)
                continue
            n = sizes.get(e.ticker, self.risk.max_contracts_per_order)
            intents.extend(self._quotes_for(e, positions.get(e.ticker, 0), n))

        return QuotePlan(intents=intents, pulls=pulls, seconds_to_close=seconds_to_close)

    # ------------------------------------------------------------------ internals
    def _pull_reason(
        self, e: StrikeEdge, std_now: float, std_next: float, seconds_to_close: float
    ) -> PullNotice | None:
        """None when the strike is quotable; a PullNotice with numbers when it is not."""
        if e.yes_bid is None and e.yes_ask is None:
            return PullNotice(
                e.ticker, e.strike, "no two-sided market to join", 0.0, seconds_to_close
            )
        if not math.isfinite(e.z_score) or abs(e.z_score) > self.max_abs_z:
            return PullNotice(
                e.ticker, e.strike, f"|z|={abs(e.z_score):.1f} beyond {self.max_abs_z:.1f}",
                0.0, seconds_to_close,
            )

        move = adverse_move_std(e.z_score, std_now, std_next, dist=self.dist, df=self.df)
        required = self.adverse_multiple * move
        capture = float(SPREAD_CAPTURE)
        if capture < required:
            return PullNotice(
                e.ticker,
                e.strike,
                (
                    f"adverse selection: fair moves {move * 100:.2f}c per "
                    f"{self.refresh_seconds:.0f}s refresh, need "
                    f"{required * 100:.2f}c but capture is only {capture * 100:.2f}c"
                ),
                move,
                seconds_to_close,
            )
        return None

    def _quotes_for(self, e: StrikeEdge, position: int, contracts: int) -> list[QuoteIntent]:
        """Both sides of one strike, after inventory skew and queue-depth screening.

        The spread is locked at one tick, so there is exactly one price per side a maker
        can use: the touch. Improving crosses (post_only would reject it) and stepping back
        puts us behind a queue that has to clear before ours even starts — on this market
        that is not a quote, it is a decoration.

        So the skew moves the REQUIREMENT, not the price. Long inventory raises the edge we
        demand before buying more and lowers the edge we demand to sell, which withdraws
        the side that would deepen the position. Implementing it as a price step instead
        would be strictly worse and, worse than worse, it can CREATE a bid: step a
        marginal 0.03 bid back to 0.01 and its apparent capture jumps, so the quoter would
        answer "I am too long" by adding another buy order.
        """
        if contracts <= 0:
            return []
        skew = self._skew(position)
        out: list[QuoteIntent] = []

        # BID: join the best bid. Being long makes `required` bigger, i.e. harder to buy.
        if e.yes_bid is not None:
            bid = round_to_tick(e.yes_bid, mode=ROUND_DOWN)
            capture = e.fair - bid
            required = self.min_quote_edge - skew
            if (
                capture >= required
                and bid >= MIN_PRICE
                and e.yes_bid_size <= self.max_queue_ahead
            ):
                out.append(
                    QuoteIntent(
                        ticker=e.ticker,
                        strike=e.strike,
                        side=Side.YES,
                        action=Action.BUY,
                        price=bid,
                        contracts=contracts,
                        fair=e.fair,
                        edge=capture,
                        queue_ahead=e.yes_bid_size,
                        reason=(
                            f"join bid, need {required * 100:.1f}c "
                            f"(skew {skew * 100:+.0f}c, pos {position:+d}), z={e.z_score:+.2f}"
                        ),
                    )
                )

        # ASK: join the best ask. Selling YES is buying NO at 1 - price; the venue lets us
        # express it as a YES sell and the fee (zero, we are the maker) is symmetric.
        if e.yes_ask is not None:
            ask = round_to_tick(e.yes_ask, mode=ROUND_HALF_UP)
            capture = ask - e.fair
            required = self.min_quote_edge + skew
            if (
                capture >= required
                and ask <= MAX_PRICE
                and e.yes_ask_size <= self.max_queue_ahead
            ):
                out.append(
                    QuoteIntent(
                        ticker=e.ticker,
                        strike=e.strike,
                        side=Side.YES,
                        action=Action.SELL,
                        price=ask,
                        contracts=contracts,
                        fair=e.fair,
                        edge=capture,
                        queue_ahead=e.yes_ask_size,
                        reason=(
                            f"join ask, need {required * 100:.1f}c "
                            f"(skew {skew * 100:+.0f}c, pos {position:+d}), z={e.z_score:+.2f}"
                        ),
                    )
                )
        return out

    def _skew(self, position: int) -> Decimal:
        """Inventory skew in dollars, negative when we are long.

        Being long YES means we want to sell, and the way to become likelier to sell than
        to buy on a locked spread is to withdraw the bid — so the skew is negative and it
        bites on the bid. Getting this sign backwards builds inventory instead of clearing
        it, which is the most expensive sign error available to a market maker.

        At the position cap the adjustment exceeds `min_quote_edge`, so the flattening side
        is quoted even at a NEGATIVE model edge: we are willing to pay up to
        `max_skew_ticks` to get flat. That is deliberate and it is bounded — carrying an
        unwanted position into the settlement print is worth more than two ticks.
        """
        cap = max(1, self.risk.max_position_per_strike)
        frac = max(-1.0, min(1.0, -position / cap))
        ticks = round(frac * self.max_skew_ticks)
        return Decimal(ticks) * TICK


__all__ = [
    "SPREAD_CAPTURE",
    "MakerQuoter",
    "PullNotice",
    "QuoteIntent",
    "QuotePlan",
    "adverse_move_std",
    "residual_std_after",
    "round_to_tick",
]
