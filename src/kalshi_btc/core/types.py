"""Domain types.

Money is Decimal everywhere. Kalshi's API returns prices AND sizes as decimal strings
(sizes are genuinely fractional, e.g. "1286.06"), so float is a correctness bug here,
not a style preference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

# KXBTCD-26JUL2819-T63999.99  ->  series, YYMMMDD, hour, strike
TICKER_RE = re.compile(
    r"^(?P<series>[A-Z]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hour>\d{2})-T(?P<strike>[\d.]+)$"
)


class Side(StrEnum):
    YES = "yes"
    NO = "no"


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Liquidity(StrEnum):
    """Which side of the book we were on - determines the fee (maker is free)."""

    MAKER = "maker"
    TAKER = "taker"


def dec(value: str | float | int | Decimal | None, default: str = "0") -> Decimal:
    """Parse Kalshi's decimal strings safely."""
    if value is None or value == "":
        return Decimal(default)
    return Decimal(str(value))


@dataclass(frozen=True)
class MarketTicker:
    """Parsed KXBTCD market ticker."""

    raw: str
    series: str
    hour: int
    strike: Decimal

    @classmethod
    def parse(cls, raw: str) -> MarketTicker:
        m = TICKER_RE.match(raw)
        if not m:
            raise ValueError(f"unparseable market ticker: {raw!r}")
        return cls(
            raw=raw,
            series=m["series"],
            hour=int(m["hour"]),
            strike=Decimal(m["strike"]),
        )

    @property
    def event_ticker(self) -> str:
        return self.raw.rsplit("-", 1)[0]


@dataclass(frozen=True)
class BookLevel:
    price: Decimal  # dollars, 0.01..0.99
    size: Decimal  # contracts (fractional)


@dataclass
class Book:
    """One market's order book.

    Kalshi expresses both sides as resting BIDS: `yes` levels are bids to buy YES,
    `no` levels are bids to buy NO. A NO bid at p is economically a YES ask at 1-p,
    which is how best_yes_ask is derived.
    """

    ticker: str
    yes: list[BookLevel] = field(default_factory=list)
    no: list[BookLevel] = field(default_factory=list)
    seq: int | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def best_yes_bid(self) -> Decimal | None:
        return max((lv.price for lv in self.yes), default=None)

    @property
    def best_no_bid(self) -> Decimal | None:
        return max((lv.price for lv in self.no), default=None)

    @property
    def best_yes_ask(self) -> Decimal | None:
        nb = self.best_no_bid
        return None if nb is None else Decimal("1") - nb

    @property
    def mid(self) -> Decimal | None:
        b, a = self.best_yes_bid, self.best_yes_ask
        if b is None or a is None:
            return None
        return (b + a) / 2

    @property
    def spread(self) -> Decimal | None:
        b, a = self.best_yes_bid, self.best_yes_ask
        if b is None or a is None:
            return None
        return a - b

    def size_at(self, side: Side, price: Decimal) -> Decimal:
        levels = self.yes if side is Side.YES else self.no
        return sum((lv.size for lv in levels if lv.price == price), Decimal("0"))

    def is_degenerate(self) -> bool:
        """True when the strike is pinned at 0 or 1 - not worth quoting."""
        m = self.mid
        return m is None or m <= Decimal("0.02") or m >= Decimal("0.98")


@dataclass(frozen=True)
class MarketSnapshot:
    """A market as reported by GET /markets - using the CURRENT dollars/fp schema.

    The legacy integer-cent fields (yes_bid, volume, open_interest) still appear in
    responses but are ALWAYS ZERO. Reading them is the single most common way a bot
    built from an old tutorial silently sees an empty market.
    """

    ticker: str
    strike: Decimal
    yes_bid: Decimal
    yes_ask: Decimal
    yes_bid_size: Decimal
    yes_ask_size: Decimal
    volume: Decimal
    open_interest: Decimal
    open_time: datetime
    close_time: datetime
    status: str
    expiration_value: Decimal | None = None

    @classmethod
    def from_api(cls, m: dict) -> MarketSnapshot:
        return cls(
            ticker=m["ticker"],
            strike=dec(m.get("floor_strike")),
            yes_bid=dec(m.get("yes_bid_dollars")),
            yes_ask=dec(m.get("yes_ask_dollars")),
            yes_bid_size=dec(m.get("yes_bid_size_fp")),
            yes_ask_size=dec(m.get("yes_ask_size_fp")),
            volume=dec(m.get("volume_fp")),
            open_interest=dec(m.get("open_interest_fp")),
            open_time=_ts(m.get("open_time")),
            close_time=_ts(m.get("close_time")),
            status=m.get("status", ""),
            expiration_value=(
                dec(m["expiration_value"]) if m.get("expiration_value") else None
            ),
        )

    @property
    def mid(self) -> Decimal:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def is_live(self) -> bool:
        """Two-sided and not pinned - i.e. actually worth modelling."""
        return self.yes_bid > 0 and self.yes_ask < 1 and Decimal("0.02") <= self.mid <= Decimal("0.98")


@dataclass(frozen=True)
class Fill:
    ticker: str
    side: Side
    action: Action
    price: Decimal
    count: Decimal
    liquidity: Liquidity
    fee: Decimal
    ts: datetime


def _ts(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
