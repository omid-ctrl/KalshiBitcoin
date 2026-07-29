"""Position book and hard risk limits.

WHAT MAKES BINARY RISK DIFFERENT
--------------------------------
On a normal instrument "max loss" needs an assumption about how far the price can move.
On a ladder of nested binaries it needs none: the payoff of a whole event position is a
STEP FUNCTION of one number, the settlement value S, and it only changes at the strikes we
hold. So the exact worst case is a scan over at most (number of strikes + 1) regions:

    P&L(S) = cash + sum_{k_i < S} v_i

where v_i is signed contracts (positive long YES), and `cash` is what the trades put in
or took out of the account net of fees. `worst_case_pnl` evaluates that at every region
and returns the minimum. No sigma, no confidence interval, no tail assumption — an exact
bound, which is the only kind of bound a limit should be built on.

That matters because the naive alternative (sum the premium paid on every leg) badly
overstates the risk of a hedged ladder: long YES at 117,800 and short YES at 118,000 can
lose at most the net premium in the middle region, not the sum of both legs. Overstating
risk is not "safe" — it makes the limit bind on positions that are not actually risky and
pushes the operator to raise it, which is how limits stop meaning anything.

THREE LIMITS, THREE DIFFERENT JOBS
----------------------------------
* max_position_per_strike — a concentration limit. Stops one strike from becoming the
  whole book because its model edge happened to be largest.
* max_loss_per_event — a per-hour budget. Checked against the exact worst case ABOVE,
  evaluated on the position we would hold AFTER the proposed order.
* max_loss_per_day — a solvency limit, and the only one wired to the kill switch. It is
  the one loss that means "stop", not "size down": a day past its budget is a day whose
  model assumptions are already suspect.

Every rejection returns a sentence with the numbers in it. `kbtc report` shows those
verbatim, and "blocked by per-event loss: worst case -$16.20 exceeds -$15.00" is a debug
session that lasts ten seconds instead of an hour.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Iterable, Mapping

from kalshi_btc.config import RiskLimits
from kalshi_btc.core.fees import maker_fee, taker_fee
from kalshi_btc.core.types import Action, Liquidity, MarketTicker
from kalshi_btc.risk.killswitch import KillSwitch

log = logging.getLogger(__name__)


@dataclass
class Position:
    """Net position in one market. Signed: positive is long YES, negative is short YES."""

    ticker: str
    strike: Decimal
    contracts: int = 0
    cash: Decimal = Decimal("0")  # net cash from trading this market, fees included
    fees: Decimal = Decimal("0")
    trades: int = 0

    def apply(self, action: Action, contracts: int, price: Decimal, fee: Decimal) -> None:
        signed = contracts if action is Action.BUY else -contracts
        # Buying pays out cash; selling (i.e. buying the NO side) takes it in. Fees always
        # leave the account regardless of direction.
        self.cash -= Decimal(signed) * price
        self.cash -= fee
        self.fees += fee
        self.contracts += signed
        self.trades += 1

    def pnl_if_yes(self) -> Decimal:
        """P&L if this market settles YES (S > strike)."""
        return self.cash + Decimal(self.contracts)

    def pnl_if_no(self) -> Decimal:
        return self.cash


@dataclass(frozen=True)
class RiskDecision:
    """Allowed or not, and exactly why. `contracts` may be a reduced allowance."""

    allowed: bool
    reason: str
    contracts: int = 0

    def __bool__(self) -> bool:  # so `if decision:` reads naturally
        return self.allowed


def worst_case_pnl(positions: Iterable[Position]) -> Decimal:
    """Exact minimum P&L over all possible settlement values. See the module docstring.

    Returns 0 for an empty book. The scan includes the region below every strike and above
    every strike, so no settlement outcome is missed.
    """
    pos = [p for p in positions if p.contracts != 0 or p.cash != 0]
    if not pos:
        return Decimal("0")

    cash = sum((p.cash for p in pos), Decimal("0"))
    ordered = sorted(pos, key=lambda p: p.strike)

    # Region 0: S below every strike -> no strike pays.
    worst = cash
    running = Decimal("0")
    for p in ordered:
        # Region above p.strike: every strike at or below p now pays out.
        running += Decimal(p.contracts)
        worst = min(worst, cash + running)
    return worst


def mark_to_market(positions: Iterable[Position], marks: Mapping[str, Decimal]) -> Decimal:
    """P&L marking each open contract at `marks[ticker]` (a YES price in dollars).

    Missing marks are treated as the position being worth its own cost basis, i.e. zero
    unrealised — a deliberately neutral assumption rather than an optimistic one.
    """
    total = Decimal("0")
    for p in positions:
        mark = marks.get(p.ticker)
        if mark is None:
            total += Decimal("0") if p.contracts else p.cash
            continue
        total += p.cash + Decimal(p.contracts) * mark
    return total


@dataclass
class RiskManager:
    """The position book plus the three hard limits, sharing one kill switch.

    It is intentionally the only object that mutates positions: sizing proposes,
    RiskManager disposes, and nothing else is allowed to believe it knows the position.
    """

    risk: RiskLimits
    killswitch: KillSwitch = field(default_factory=KillSwitch)
    positions: dict[str, Position] = field(default_factory=dict)
    # Latest YES mid per ticker, refreshed by the runner. A position book that does not
    # know the current mark cannot answer "how much am I down", which is the one question
    # the daily loss limit is built on.
    marks: dict[str, Decimal] = field(default_factory=dict)
    realised_by_day: dict[date, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    fees_paid: Decimal = Decimal("0")
    rejections: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fills_applied: int = 0

    # ------------------------------------------------------------------ book
    def position(self, ticker: str) -> Position:
        p = self.positions.get(ticker)
        if p is None:
            try:
                strike = MarketTicker.parse(ticker).strike
            except ValueError:
                strike = Decimal("0")
            p = Position(ticker=ticker, strike=strike)
            self.positions[ticker] = p
        return p

    def contracts(self, ticker: str) -> int:
        p = self.positions.get(ticker)
        return p.contracts if p else 0

    def position_map(self) -> dict[str, int]:
        """{ticker: signed contracts} — the shape `KillSwitch.check_reconciliation` wants."""
        return {t: p.contracts for t, p in self.positions.items() if p.contracts}

    def event_positions(self, event_ticker: str) -> list[Position]:
        return [p for t, p in self.positions.items() if t.startswith(f"{event_ticker}-")]

    # ------------------------------------------------------------------ accounting
    def fee_for(self, liquidity: Liquidity, price: Decimal, contracts: int) -> Decimal:
        """Fee for a fill. Maker is ZERO on KXBTCD; this is the only place that decides."""
        n = Decimal(contracts)
        if liquidity is Liquidity.MAKER:
            return maker_fee(price, n)
        return taker_fee(price, n)

    def record_fill(
        self,
        *,
        ticker: str,
        action: Action,
        contracts: int,
        price: Decimal,
        liquidity: Liquidity,
        fee: Decimal | None = None,
        when: datetime | None = None,
    ) -> Decimal:
        """Apply a fill to the book. Returns the fee charged."""
        when = when or datetime.now(UTC)
        f = self.fee_for(liquidity, price, contracts) if fee is None else fee
        self.position(ticker).apply(action, contracts, price, f)
        self.fees_paid += f
        self.fills_applied += 1
        return f

    def settle_event(self, event_ticker: str, settlement_value: Decimal) -> Decimal:
        """Realise every position under `event_ticker` against the settled BRTI average.

        Positions are removed afterwards: an event that has settled has no position, and
        leaving zeroed rows around is how a stale strike ends up inside a worst-case scan.
        """
        realised = Decimal("0")
        for p in self.event_positions(event_ticker):
            realised += p.pnl_if_yes() if settlement_value > p.strike else p.pnl_if_no()
            del self.positions[p.ticker]
        self.realised_by_day[datetime.now(UTC).date()] += realised
        log.info("settled %s at %s -> realised $%.2f", event_ticker, settlement_value, realised)
        return realised

    def abandon_event(
        self, event_ticker: str, marks: Mapping[str, Decimal] | None = None
    ) -> Decimal:
        """Realise an event we are dropping without a settlement value (rollover, restart).

        Marked at the supplied prices, or at cost when we have none. This exists so the
        daily P&L does not silently forget an hour just because we never saw it settle.
        """
        marks = marks or {}
        pos = self.event_positions(event_ticker)
        realised = mark_to_market(pos, marks)
        for p in pos:
            del self.positions[p.ticker]
        self.realised_by_day[datetime.now(UTC).date()] += realised
        return realised

    def realised_today(self, when: datetime | None = None) -> Decimal:
        return self.realised_by_day.get((when or datetime.now(UTC)).date(), Decimal("0"))

    def update_marks(self, marks: Mapping[str, Decimal]) -> None:
        self.marks.update(marks)

    def daily_pnl(self, marks: Mapping[str, Decimal] | None = None) -> Decimal:
        """Realised today plus the mark-to-market of everything still open."""
        return self.realised_today() + mark_to_market(
            self.positions.values(), self.marks if marks is None else marks
        )

    def event_worst_case(self, event_ticker: str) -> Decimal:
        return worst_case_pnl(self.event_positions(event_ticker))

    # ------------------------------------------------------------------ limits
    def check_order(
        self,
        *,
        ticker: str,
        action: Action,
        contracts: int,
        price: Decimal,
        liquidity: Liquidity,
        event_ticker: str | None = None,
    ) -> RiskDecision:
        """Would this order be allowed, and for how many contracts?

        Returns the largest allowed size rather than a bare no, so the caller can trade a
        smaller clip instead of skipping the opportunity entirely. A rejection is only
        returned when the answer is genuinely zero.
        """
        if self.killswitch.halted:
            return self._reject("killswitch", f"trading halted: {self.killswitch.reason}")
        if contracts <= 0:
            return self._reject("zero_size", "order size is zero")
        if not (Decimal("0") < price < Decimal("1")):
            return self._reject("bad_price", f"price {price} is outside (0, 1)")

        n = min(contracts, self.risk.max_contracts_per_order)
        note = [] if n == contracts else [f"clipped to per-order max {n}"]

        # --- concentration -------------------------------------------------------------
        held = self.contracts(ticker)
        signed = 1 if action is Action.BUY else -1
        target = held + signed * n
        cap = self.risk.max_position_per_strike
        if abs(target) > cap and abs(target) > abs(held):
            room = cap - abs(held)
            if room <= 0:
                return self._reject(
                    "per_strike",
                    f"{ticker} already at {held} contracts, per-strike cap is {cap}",
                )
            n = min(n, room)
            note.append(f"clipped to per-strike room {room}")

        # --- per-event worst case ------------------------------------------------------
        event_ticker = event_ticker or ticker.rsplit("-", 1)[0]
        fee = self.fee_for(liquidity, price, n)
        hypothetical = self._hypothetical(event_ticker, ticker, action, n, price, fee)
        worst = worst_case_pnl(hypothetical)
        if worst < -self.risk.max_loss_per_event:
            # Try to find a size that fits rather than refusing outright — the limit is on
            # dollars, and a smaller clip is a smaller number of dollars.
            fitted = self._largest_fitting_size(event_ticker, ticker, action, n, price, liquidity)
            if fitted <= 0:
                return self._reject(
                    "per_event",
                    f"per-event loss limit: worst case ${worst:.2f} would exceed "
                    f"${-self.risk.max_loss_per_event:.2f} on {event_ticker}",
                )
            note.append(f"clipped to {fitted} by per-event loss limit")
            n = fitted

        # --- daily solvency ------------------------------------------------------------
        pnl = self.daily_pnl()
        if pnl <= -self.risk.max_loss_per_day:
            self.killswitch.check_daily_loss(pnl, self.risk.max_loss_per_day)
            return self._reject(
                "per_day",
                f"daily loss limit: P&L ${pnl:.2f} at or beyond ${-self.risk.max_loss_per_day:.2f}",
            )

        reason = "ok" if not note else "ok (" + "; ".join(note) + ")"
        return RiskDecision(allowed=True, reason=reason, contracts=n)

    def check_daily_loss(self, marks: Mapping[str, Decimal] | None = None) -> bool:
        """Evaluate the daily limit and trip the kill switch if it is breached."""
        return self.killswitch.check_daily_loss(self.daily_pnl(marks), self.risk.max_loss_per_day)

    # ------------------------------------------------------------------ internals
    def _reject(self, key: str, message: str) -> RiskDecision:
        self.rejections[key] += 1
        return RiskDecision(allowed=False, reason=message, contracts=0)

    def _hypothetical(
        self,
        event_ticker: str,
        ticker: str,
        action: Action,
        contracts: int,
        price: Decimal,
        fee: Decimal,
    ) -> list[Position]:
        """Copy of the event's positions with the proposed order applied."""
        out: list[Position] = []
        found = False
        for p in self.event_positions(event_ticker):
            clone = Position(p.ticker, p.strike, p.contracts, p.cash, p.fees, p.trades)
            if clone.ticker == ticker:
                clone.apply(action, contracts, price, fee)
                found = True
            out.append(clone)
        if not found:
            try:
                strike = MarketTicker.parse(ticker).strike
            except ValueError:
                strike = Decimal("0")
            clone = Position(ticker, strike)
            clone.apply(action, contracts, price, fee)
            out.append(clone)
        return out

    def _largest_fitting_size(
        self,
        event_ticker: str,
        ticker: str,
        action: Action,
        upper: int,
        price: Decimal,
        liquidity: Liquidity,
    ) -> int:
        """Biggest clip that keeps the event worst case inside the limit.

        Linear scan downward: `upper` is already bounded by max_contracts_per_order, which
        is a handful of contracts, so a binary search would be more code than it saves.
        """
        for n in range(upper - 1, 0, -1):
            fee = self.fee_for(liquidity, price, n)
            worst = worst_case_pnl(self._hypothetical(event_ticker, ticker, action, n, price, fee))
            if worst >= -self.risk.max_loss_per_event:
                return n
        return 0

    # ------------------------------------------------------------------ reporting
    def describe(self) -> str:
        open_n = sum(1 for p in self.positions.values() if p.contracts)
        return (
            f"positions={open_n} fills={self.fills_applied} fees=${self.fees_paid:.4f} "
            f"realised=${self.realised_today():.2f} mtm=${self.daily_pnl():.2f} "
            f"rejects={dict(self.rejections)}"
        )


__all__ = [
    "Position",
    "RiskDecision",
    "RiskManager",
    "mark_to_market",
    "worst_case_pnl",
]
