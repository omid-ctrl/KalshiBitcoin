"""The kill switch: one place that can stop the bot, and it does not un-stop itself.

WHY STICKY
----------
Every condition here is a statement about our own reliability, not about the market. A
stale price feed does not become trustworthy the moment a tick arrives — it becomes a
feed that just dropped ticks for N seconds and we have no idea what happened during the
gap. A self-clearing halt turns "we lost the feed for 30 seconds" into a silent 30-second
hole in the risk model, and the failure mode is that the bot resumes quoting into a market
that moved while it was blind. So `trip()` latches, and only an explicit `reset(operator)`
call clears it. That reset is a human decision by construction.

WHY EVERY HALT CARRIES A REASON STRING
--------------------------------------
The question after any halt is always "what exactly did it see". A boolean cannot answer
it and neither can a log line that has already scrolled. Each `Halt` records the reason
enum, a human-readable detail with the actual numbers, and a timestamp; the whole history
is kept because the second halt is usually the interesting one.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Position and loss LIMITS live in `risk/limits.py`. This module only holds the conditions
that mean "stop trading entirely", never "size down". Mixing the two produces a system
that reduces size when it should be flat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

log = logging.getLogger(__name__)

# Defaults chosen from measured cadences: BRTI prints once per second and the ladder is
# polled every 2s, so 10s without a price is roughly ten missed prints — well past noise.
DEFAULT_MAX_FEED_AGE_S = 10.0
# Two independent views of BTC that disagree by more than this are not both right. $50 on
# a ~$118k index is ~4bp, comfortably wider than genuine venue dispersion inside BRTI.
DEFAULT_VENUE_TOLERANCE_DOLLARS = Decimal("50")


class HaltReason(StrEnum):
    """Why trading stopped. The enum exists so halts can be counted, not just read."""

    STALE_FEED = "stale_feed"
    VENUE_DISAGREEMENT = "venue_disagreement"
    WS_DISCONNECT = "ws_disconnect"
    RECONCILE_MISMATCH = "reconcile_mismatch"
    DAILY_LOSS = "daily_loss"
    EVENT_LOSS = "event_loss"
    MANUAL = "manual"


@dataclass(frozen=True)
class Halt:
    """One latched halt. Immutable so a post-mortem reads what actually happened."""

    reason: HaltReason
    detail: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def describe(self) -> str:
        return f"[{self.ts:%H:%M:%S}] {self.reason.value}: {self.detail}"


class TradingHalted(RuntimeError):
    """Raised by `require_live()` when something asks to trade while halted."""


@dataclass
class KillSwitch:
    """Latching halt state plus the checks that trip it.

    Usage is deliberately blunt: every check returns True when it is HAPPY, and trips
    internally when it is not, so a caller that forgets to look at the return value still
    ends up halted rather than still trading.
    """

    max_feed_age_s: float = DEFAULT_MAX_FEED_AGE_S
    venue_tolerance: Decimal = DEFAULT_VENUE_TOLERANCE_DOLLARS
    halts: list[Halt] = field(default_factory=list)
    resets: list[tuple[datetime, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ state
    @property
    def halted(self) -> bool:
        return bool(self.halts)

    @property
    def reason(self) -> str:
        """Human-readable summary of why we are stopped. Empty when running."""
        if not self.halts:
            return ""
        return "; ".join(h.describe() for h in self.halts)

    @property
    def first_halt(self) -> Halt | None:
        return self.halts[0] if self.halts else None

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for h in self.halts:
            out[h.reason.value] = out.get(h.reason.value, 0) + 1
        return out

    # ------------------------------------------------------------------ control
    def trip(self, reason: HaltReason, detail: str) -> Halt:
        """Latch a halt. Repeated trips are recorded, not deduplicated.

        Duplicates are kept on purpose: "the feed went stale eleven times" is a different
        situation from "the feed went stale once", and collapsing them hides it.
        """
        halt = Halt(reason=reason, detail=detail)
        self.halts.append(halt)
        log.error("KILL SWITCH: %s", halt.describe())
        return halt

    def reset(self, operator: str = "operator") -> int:
        """Clear every latched halt. Explicit by design — nothing calls this on a timer."""
        n = len(self.halts)
        if n:
            log.warning("kill switch reset by %s, clearing %d halt(s)", operator, n)
        self.halts.clear()
        self.resets.append((datetime.now(UTC), operator))
        return n

    def require_live(self, what: str = "trade") -> None:
        """Raise unless we are running. The last gate before anything touches the venue."""
        if self.halted:
            raise TradingHalted(f"refusing to {what}: {self.reason}")

    # ------------------------------------------------------------------ checks
    def check_feed_age(
        self, last_tick: datetime | None, now: datetime | None = None, *, label: str = "price feed"
    ) -> bool:
        """Trip when the price feed has gone quiet.

        `last_tick is None` counts as stale: never having had a price is strictly worse
        than having had an old one, and the code path that treats "no data" as "fine" is
        the one that quotes off a default.
        """
        now = now or datetime.now(UTC)
        if last_tick is None:
            self.trip(HaltReason.STALE_FEED, f"{label}: no tick has ever arrived")
            return False
        age = (now - _aware(last_tick)).total_seconds()
        if age > self.max_feed_age_s:
            self.trip(
                HaltReason.STALE_FEED,
                f"{label} last updated {age:.1f}s ago (limit {self.max_feed_age_s:.0f}s)",
            )
            return False
        return not self.halted

    def check_venue_agreement(
        self, a: Decimal | float, b: Decimal | float, *, labels: tuple[str, str] = ("A", "B")
    ) -> bool:
        """Trip when two independent price sources disagree beyond tolerance.

        One of them is wrong and we cannot tell which. Quoting off the average of a good
        price and a broken one is worse than not quoting.
        """
        da, db = Decimal(str(a)), Decimal(str(b))
        gap = abs(da - db)
        if gap > self.venue_tolerance:
            self.trip(
                HaltReason.VENUE_DISAGREEMENT,
                f"{labels[0]}={da:,.2f} vs {labels[1]}={db:,.2f} differ by {gap:,.2f} "
                f"(tolerance {self.venue_tolerance:,.2f})",
            )
            return False
        return not self.halted

    def check_websocket(self, connected: bool, *, detail: str = "") -> bool:
        """Trip on a WS disconnect.

        The socket carries the book deltas and the BRTI window. REST polling can keep the
        recorder alive but it cannot keep a QUOTER alive: a 1-cent spread sampled every
        two seconds is a picture of a market we are no longer trading in.
        """
        if not connected:
            self.trip(HaltReason.WS_DISCONNECT, detail or "websocket is not connected")
            return False
        return not self.halted

    def check_reconciliation(
        self, local: dict[str, int], exchange: dict[str, int], *, tolerance: int = 0
    ) -> bool:
        """Trip when our position book disagrees with the venue's.

        If we do not know what we own, every downstream number — exposure, P&L, the risk
        limits themselves — is fiction. This is the halt that protects the other halts.
        """
        mismatches = []
        for ticker in sorted(set(local) | set(exchange)):
            ours, theirs = local.get(ticker, 0), exchange.get(ticker, 0)
            if abs(ours - theirs) > tolerance:
                mismatches.append(f"{ticker}: local={ours} exchange={theirs}")
        if mismatches:
            self.trip(
                HaltReason.RECONCILE_MISMATCH,
                f"{len(mismatches)} position mismatch(es): " + "; ".join(mismatches[:5]),
            )
            return False
        return not self.halted

    def check_daily_loss(self, pnl: Decimal, limit: Decimal) -> bool:
        """Trip when realised+unrealised P&L for the day breaches the limit.

        `pnl` is signed; `limit` is a positive maximum loss. This is a kill switch rather
        than a size reduction on purpose: a day that has lost the daily budget is a day
        whose model assumptions are already in question.
        """
        if limit > 0 and pnl <= -limit:
            self.trip(
                HaltReason.DAILY_LOSS,
                f"daily P&L ${pnl:.2f} breached the ${-limit:.2f} limit",
            )
            return False
        return not self.halted

    def describe(self) -> str:
        if not self.halted:
            return "kill switch: armed and clear"
        return f"kill switch: HALTED ({len(self.halts)}) | {self.reason}"


def _aware(ts: datetime) -> datetime:
    """Naive timestamps are UTC by convention across this codebase."""
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def stale_after(last: datetime | None, seconds: float, now: datetime | None = None) -> bool:
    """Pure predicate version of the staleness test, for callers that only want a bool."""
    if last is None:
        return True
    now = now or datetime.now(UTC)
    return (now - _aware(last)) > timedelta(seconds=seconds)


__all__ = [
    "DEFAULT_MAX_FEED_AGE_S",
    "DEFAULT_VENUE_TOLERANCE_DOLLARS",
    "Halt",
    "HaltReason",
    "KillSwitch",
    "TradingHalted",
    "stale_after",
]
