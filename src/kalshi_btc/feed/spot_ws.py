"""Public spot feeds and the BRTI *proxy*.

WHY THIS FILE EXISTS
--------------------
Settlement of a KXBTCD hour is the simple mean of sixty one-second CF Benchmarks BRTI
prints in the final minute. Real-time BRTI is a licensed product: Kalshi will proxy it
over `cfbenchmarks_value`, but that channel - like every other channel on their
gateway - requires API credentials (measured: HTTP 401 on an anonymous upgrade).

That left `kbtc capture` recording a Kalshi ladder with no price series beside it, which
is close to useless for research. You cannot calibrate a settlement model against a
ladder alone; you need to know where BTC actually was while those quotes were showing.

So we build the price series from PUBLIC exchange WebSockets that need no key at all.
Three of BRTI's eight constituent venues publish free, unauthenticated top-of-book:

    Coinbase Exchange   wss://ws-feed.exchange.coinbase.com   ticker channel
    Kraken              wss://ws.kraken.com/v2                ticker channel, bbo trigger
    Bitstamp            wss://ws.bitstamp.net                 order_book_btcusd

All three subscribe payloads below were verified by actually connecting on 2026-07-29,
not recalled from memory.

THIS IS A PROXY. IT IS NOT BRTI.
--------------------------------
Say it plainly, because the whole strategy rests on it:

- BRTI aggregates EIGHT venues (Bitstamp, Coinbase, itBit, Kraken, Gemini, LMAX Digital,
  Bullish, Crypto.com). We see three. A move that starts on the five we cannot see
  reaches us late or not at all.
- BRTI is depth-weighted off the full order book with CF Benchmarks' own outlier and
  volume rules. We take the top-of-book midpoint. Those are different statistics even on
  identical input.
- Our clock, their clock and the venues' clocks are not the same clock.

The size of that tracking error is an empirical question, not an assumption, and
`kalshi_btc.model.proxy_score` answers it against realised `expiration_value`. Nothing
downstream should treat `proxy()` as the index until that score says it is close enough.

AGGREGATION CHOICE
------------------
The composite is the MEDIAN of fresh venue mids, not the mean. With three venues the
median is immune to one venue printing garbage (a stuck book, a fat-finger cross, a
feed that reconnects into a stale snapshot), and one bad venue is far more likely than
two. With two venues the median degenerates to their mean, which is the best available
answer anyway.

`mid()` is best effort and returns a number whenever any venue is fresh.
`proxy()` is the index-quality number and returns None unless at least
`min_venues_for_proxy` venues are fresh - because a one-venue "cross-exchange index" is
not an index, and a caller that gets a number back should be able to trust its shape.

BANDWIDTH, MEASURED
-------------------
Coinbase ~3 msg/s (1.2 KiB/s), Kraken ~1.5 msg/s (0.2 KiB/s), Bitstamp ~10 msg/s
(51 KiB/s - it pushes the whole 100-level book on any change, and there is no
top-of-book-only channel). Bitstamp therefore costs ~4 GB/day. We keep it because it is
a BRTI constituent and because two venues cannot outvote a liar, but the emitted-quote
stream is deduplicated on unchanged top-of-book, so the DATABASE cost is far lower than
the wire cost.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from websockets.asyncio.client import connect

from kalshi_btc.core.types import dec

log = logging.getLogger(__name__)

VENUE_COINBASE = "coinbase"
VENUE_KRAKEN = "kraken"
VENUE_BITSTAMP = "bitstamp"

DEFAULT_VENUES: tuple[str, ...] = (VENUE_COINBASE, VENUE_KRAKEN, VENUE_BITSTAMP)

# Six decimals matches the DECIMAL(18,6) storage columns, so nothing is silently
# re-rounded between here and disk.
_Q = Decimal("0.000001")

# A top-of-book wider than this is not a quote, it is a venue in trouble. BTC/USD on any
# of these three runs a spread of a cent or two; 50 bps is four orders of magnitude wider
# than normal and only shows up during an outage or a bad partial book.
MAX_SPREAD_BPS = Decimal("50")


# --------------------------------------------------------------------------- events
@dataclass(frozen=True)
class SpotQuote:
    """One venue's top of book at a point in time."""

    venue: str
    bid: Decimal
    ask: Decimal
    mid: Decimal
    ts: datetime
    # Local monotonic receipt time. Kept separately from `ts` because `ts` is the venue's
    # clock and staleness must be measured on a clock that cannot jump backwards.
    received: float = field(default_factory=time.monotonic, compare=False)

    @property
    def spread_bps(self) -> Decimal:
        if self.mid <= 0:
            return Decimal("0")
        return (self.ask - self.bid) / self.mid * 10_000


@dataclass(frozen=True)
class SpotTick:
    """What goes on the queue and, one-for-one, into the `spot` table.

    `proxy` is the composite computed at the instant this venue's quote landed, so a row
    records both what one venue said and what the aggregate believed at that moment.
    """

    venue: str
    bid: Decimal
    ask: Decimal
    mid: Decimal
    proxy: Decimal | None
    ts: datetime


@dataclass
class VenueStats:
    messages: int = 0
    quotes: int = 0
    connects: int = 0
    errors: int = 0
    rejected: int = 0  # frames that parsed but failed the sanity filter
    connected: bool = False
    last_message_at: float = 0.0


# --------------------------------------------------------------------------- parsing
def _price(raw: Any) -> Decimal | None:
    """Decimal or None. Venues send strings (Coinbase, Bitstamp) or floats (Kraken)."""
    if raw is None or raw == "":
        return None
    try:
        return dec(raw)
    except (InvalidOperation, ValueError, TypeError):
        return None


def _quote(venue: str, bid: Any, ask: Any, ts: datetime) -> SpotQuote | None:
    """Build a quote, rejecting anything that is not a usable two-sided market.

    Rejection is silent-but-counted rather than raising: a malformed frame is a normal
    event on a public feed and must never propagate into the capture loop.
    """
    b, a = _price(bid), _price(ask)
    if b is None or a is None or b <= 0 or a <= 0 or a < b:
        return None
    mid = ((b + a) / 2).quantize(_Q)
    if mid <= 0 or (a - b) / mid * 10_000 > MAX_SPREAD_BPS:
        return None
    return SpotQuote(venue=venue, bid=b, ask=a, mid=mid, ts=ts)


def _iso_ts(raw: Any) -> datetime:
    """Venue timestamp, falling back to local now(). Always tz-aware UTC."""
    if raw is None:
        return datetime.now(UTC)
    try:
        t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.now(UTC)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def parse_coinbase(msg: dict) -> SpotQuote | None:
    """Coinbase Exchange `ticker` frame.

    Verified shape::

        {"type":"ticker","product_id":"BTC-USD","price":"63944.26",
         "best_bid":"63944.26","best_bid_size":"0.108",
         "best_ask":"63944.27","best_ask_size":"0.037",
         "time":"2026-07-29T00:14:37.919203Z", ...}

    `price` is the last trade; we deliberately use best_bid/best_ask because BRTI is an
    order-book index and a last-trade print lags the book.
    """
    if msg.get("type") != "ticker" or msg.get("product_id") != "BTC-USD":
        return None
    return _quote(VENUE_COINBASE, msg.get("best_bid"), msg.get("best_ask"), _iso_ts(msg.get("time")))


def parse_kraken(msg: dict) -> SpotQuote | None:
    """Kraken v2 `ticker` frame (snapshot and update share one shape).

    Verified shape::

        {"channel":"ticker","type":"update","data":[{"symbol":"BTC/USD",
          "bid":63964.9,"bid_qty":10.35,"ask":63965.0,"ask_qty":0.36,
          "last":63964.9, ..., "timestamp":"2026-07-29T00:16:49.9Z"}]}

    Numbers arrive as JSON floats here, so they go through `dec(str(...))` rather than
    float arithmetic. Note that `timestamp` is absent on `event_trigger:"bbo"` updates,
    which is why `_iso_ts(None)` falls back to local time.
    """
    if msg.get("channel") != "ticker":
        return None
    rows = msg.get("data") or []
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") not in ("BTC/USD", "XBT/USD"):
            continue
        return _quote(VENUE_KRAKEN, row.get("bid"), row.get("ask"), _iso_ts(row.get("timestamp")))
    return None


def parse_bitstamp(msg: dict) -> SpotQuote | None:
    """Bitstamp `order_book_btcusd` frame - the full 100-level book, of which we keep [0].

    Verified shape::

        {"event":"data","channel":"order_book_btcusd",
         "data":{"timestamp":"1785284078","microtimestamp":"1785284078207406",
                 "bids":[["63947.07","0.195"], ...], "asks":[[...]]}}

    Bids arrive best-first and asks best-first, so index 0 is the top of book on both
    sides. Microseconds are the authoritative timestamp; the second-resolution
    `timestamp` field would quantise away exactly the resolution we care about.
    """
    if msg.get("event") != "data" or not str(msg.get("channel", "")).startswith("order_book"):
        return None
    data = msg.get("data") or {}
    bids, asks = data.get("bids") or [], data.get("asks") or []
    if not bids or not asks:
        return None
    micro = data.get("microtimestamp")
    if micro is not None:
        try:
            ts = datetime.fromtimestamp(int(micro) / 1e6, UTC)
        except (ValueError, TypeError, OSError, OverflowError):
            ts = datetime.now(UTC)
    else:
        ts = datetime.now(UTC)
    return _quote(VENUE_BITSTAMP, bids[0][0], asks[0][0], ts)


def _bitstamp_wants_reconnect(msg: dict) -> bool:
    """Bitstamp asks clients to cycle the socket before it drops them server-side."""
    return msg.get("event") == "bts:request_reconnect"


# --------------------------------------------------------------------------- venues
@dataclass(frozen=True)
class VenueSpec:
    """Everything venue-specific, in one place, so the client loop stays generic."""

    name: str
    url: str
    subscribe: tuple[dict, ...]
    parse: Callable[[dict], SpotQuote | None]
    wants_reconnect: Callable[[dict], bool] = lambda _msg: False
    # Coinbase and Bitstamp answer WebSocket pings; Kraken sends its own heartbeat frames.
    ping_interval_s: float | None = 20.0


VENUE_SPECS: dict[str, VenueSpec] = {
    VENUE_COINBASE: VenueSpec(
        name=VENUE_COINBASE,
        url="wss://ws-feed.exchange.coinbase.com",
        subscribe=({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]},),
        parse=parse_coinbase,
    ),
    VENUE_KRAKEN: VenueSpec(
        name=VENUE_KRAKEN,
        url="wss://ws.kraken.com/v2",
        # event_trigger "bbo" (not the default "trades") makes the ticker fire on every
        # best-bid/offer change. The default only updates on trades, which would leave us
        # blind through a quiet minute - precisely the minute we settle on.
        subscribe=(
            {
                "method": "subscribe",
                "params": {"channel": "ticker", "symbol": ["BTC/USD"], "event_trigger": "bbo"},
            },
        ),
        parse=parse_kraken,
    ),
    VENUE_BITSTAMP: VenueSpec(
        name=VENUE_BITSTAMP,
        url="wss://ws.bitstamp.net",
        subscribe=({"event": "bts:subscribe", "data": {"channel": "order_book_btcusd"}},),
        parse=parse_bitstamp,
        wants_reconnect=_bitstamp_wants_reconnect,
    ),
}


# --------------------------------------------------------------------------- the feed
class SpotFeed:
    """Aggregates public BTC/USD top-of-book into a single BRTI proxy.

    Usage::

        async with SpotFeed() as feed:
            ...
            for tick in feed.drain():
                store.add_spot(...)

    Every accessor is total: it returns None / inf / False rather than raising, because
    the caller is a capture loop that must survive anything a public venue does.
    """

    def __init__(
        self,
        venues: Iterable[str] = DEFAULT_VENUES,
        *,
        max_age_s: float = 5.0,
        min_venues_for_proxy: int = 2,
        # 5 bps is ~$32 on a $63k BTC. Healthy venues sit well under 1 bp apart, and our
        # proxy has to land within a few dollars of the index to be worth anything, so a
        # 5 bp spread already means the composite is not fit to trade on.
        agreement_bps: float = 5.0,
        queue_size: int = 50_000,
        max_backoff_s: float = 30.0,
    ) -> None:
        self.specs: list[VenueSpec] = []
        for name in venues:
            spec = VENUE_SPECS.get(name)
            if spec is None:
                log.warning("unknown spot venue %r ignored; known: %s", name, sorted(VENUE_SPECS))
                continue
            self.specs.append(spec)

        self.max_age_s = max_age_s
        self.min_venues_for_proxy = min_venues_for_proxy
        self.agreement_bps = agreement_bps
        self.max_backoff_s = max_backoff_s

        self.quotes: dict[str, SpotQuote] = {}
        self.stats: dict[str, VenueStats] = {s.name: VenueStats() for s in self.specs}

        self._queue: asyncio.Queue[SpotTick] = asyncio.Queue(maxsize=queue_size)
        self._tasks: list[asyncio.Task] = []
        self._stopping = asyncio.Event()
        self.dropped: int = 0

    # ------------------------------------------------------------------ lifecycle
    @property
    def venue_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    def start(self) -> None:
        """Spawn one reader task per venue. Idempotent."""
        if self._tasks or not self.specs:
            return
        for spec in self.specs:
            self._tasks.append(
                asyncio.create_task(self._run_venue(spec), name=f"spot-{spec.name}")
            )

    async def stop(self) -> None:
        self._stopping.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks = []

    async def __aenter__(self) -> SpotFeed:
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def wait_ready(self, timeout: float = 10.0, venues: int = 1) -> bool:
        """Block until `venues` venues have printed a quote, or `timeout` elapses.

        Convenience for callers that want their first heartbeat to carry a price instead
        of a dash. Returns whether the bar was met; never raises on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.fresh_quotes()) >= venues:
                return True
            await asyncio.sleep(0.1)
        return len(self.fresh_quotes()) >= venues

    # ------------------------------------------------------------------ consumer API
    def drain(self, limit: int = 100_000) -> list[SpotTick]:
        """Non-blocking pull of everything queued. Matches KalshiWebSocket.drain()."""
        out: list[SpotTick] = []
        for _ in range(limit):
            try:
                out.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    def fresh_quotes(self, max_age_s: float | None = None) -> list[SpotQuote]:
        """Quotes from venues whose last MESSAGE is recent enough to trust.

        Freshness keys off the last message rather than the last price CHANGE: a venue
        whose top of book has not moved for four seconds is telling us the price, not
        failing to. Only silence means we have stopped knowing.
        """
        age = self.max_age_s if max_age_s is None else max_age_s
        now = time.monotonic()
        out: list[SpotQuote] = []
        for name, q in self.quotes.items():
            st = self.stats.get(name)
            last = st.last_message_at if st else q.received
            if now - max(last, q.received) <= age:
                out.append(q)
        return out

    def mid(self, venue: str | None = None) -> Decimal | None:
        """Best-effort BTC mid. One venue's, or the composite across fresh venues.

        Unlike `proxy()` this never gates on venue count: it is the number you show a
        human, not the number you price a contract with.
        """
        if venue is not None:
            q = self.quotes.get(venue)
            if q is None:
                return None
            return q.mid if (time.monotonic() - q.received) <= self.max_age_s else None
        fresh = self.fresh_quotes()
        return _median([q.mid for q in fresh]) if fresh else None

    def proxy(self) -> Decimal | None:
        """The BRTI PROXY: median of fresh venue mids, or None if too few venues.

        NOT the BRTI index - see the module docstring. The gate exists so that a caller
        can never mistake a single surviving venue for a cross-exchange composite.
        """
        fresh = self.fresh_quotes()
        if len(fresh) < self.min_venues_for_proxy:
            return None
        return _median([q.mid for q in fresh])

    def staleness_s(self, venue: str | None = None) -> float:
        """Seconds since we last heard anything. `inf` if we never have."""
        now = time.monotonic()
        if venue is not None:
            st = self.stats.get(venue)
            if st is None or not st.last_message_at:
                return math.inf
            return now - st.last_message_at
        stamps = [s.last_message_at for s in self.stats.values() if s.last_message_at]
        return now - max(stamps) if stamps else math.inf

    def spread_bps(self) -> float | None:
        """Cross-venue disagreement in basis points: (max mid - min mid) / median mid."""
        mids = [q.mid for q in self.fresh_quotes()]
        if len(mids) < 2:
            return None
        med = _median(mids)
        if med is None or med <= 0:
            return None
        return float((max(mids) - min(mids)) / med * 10_000)

    def venues_agree(self, tolerance_bps: float | None = None) -> bool:
        """Cross-feed sanity check. Disagreement is a KILL-SWITCH condition.

        Two independent venues quoting BTC more than a few basis points apart means one
        of them is wrong, and we have no way to tell which. Fewer than two fresh venues
        returns False as well: an unverifiable price is not an agreeing price, and the
        safe reading of "cannot check" is "do not trade".
        """
        tol = self.agreement_bps if tolerance_bps is None else tolerance_bps
        bps = self.spread_bps()
        return bps is not None and bps <= tol

    def describe(self) -> str:
        """One-line human summary for logs and the capture banner."""
        fresh = self.fresh_quotes()
        px = self.proxy()
        bits = [f"{len(fresh)}/{len(self.specs)} venues"]
        bits.append(f"proxy ${px:,.2f}" if px is not None else "proxy --")
        bps = self.spread_bps()
        if bps is not None:
            bits.append(f"spread {bps:.1f}bps")
        return " | ".join(bits)

    # ------------------------------------------------------------------ internals
    def _publish(self, quote: SpotQuote) -> None:
        """Record a venue quote and queue the tick with the composite at that instant."""
        self.quotes[quote.venue] = quote
        tick = SpotTick(
            venue=quote.venue,
            bid=quote.bid,
            ask=quote.ask,
            mid=quote.mid,
            proxy=self.proxy(),
            ts=quote.ts,
        )
        try:
            self._queue.put_nowait(tick)
            return
        except asyncio.QueueFull:
            pass
        # Drop the OLDEST. A stalled consumer wants the newest price, and blocking the
        # socket here would only desync us further.
        with contextlib.suppress(asyncio.QueueEmpty):
            self._queue.get_nowait()
        self.dropped += 1
        if self.dropped == 1 or self.dropped % 1_000 == 0:
            log.warning("spot queue full: %d ticks dropped - consumer too slow", self.dropped)
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(tick)

    async def _run_venue(self, spec: VenueSpec) -> None:
        """Connect, subscribe, read, reconnect. Never exits except on stop()."""
        st = self.stats[spec.name]
        attempt = 0
        while not self._stopping.is_set():
            try:
                async with connect(
                    spec.url,
                    open_timeout=15,
                    ping_interval=spec.ping_interval_s,
                    ping_timeout=20,
                    close_timeout=5,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    attempt = 0
                    st.connects += 1
                    st.connected = True
                    log.info("spot %s connected (%s)", spec.name, spec.url)
                    for payload in spec.subscribe:
                        await ws.send(json.dumps(payload))
                    await self._read(spec, ws, st)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the feed must outlive any venue
                st.errors += 1
                log.warning("spot %s connection error: %s: %s", spec.name, type(e).__name__, e)
            finally:
                st.connected = False

            if self._stopping.is_set():
                return
            # Exponential backoff with jitter. The jitter matters more than usual here:
            # three venues that all blip on the same upstream network event would
            # otherwise retry in lockstep forever.
            delay = min(self.max_backoff_s, 1.0 * (2**attempt))
            delay += random.uniform(0, delay * 0.3)
            attempt += 1
            log.info("spot %s reconnecting in %.1fs", spec.name, delay)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)

    async def _read(self, spec: VenueSpec, ws: Any, st: VenueStats) -> None:
        last: tuple[Decimal, Decimal] | None = None
        async for raw in ws:
            if self._stopping.is_set():
                return
            st.messages += 1
            st.last_message_at = time.monotonic()
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                st.errors += 1
                continue
            if not isinstance(msg, dict):
                continue
            if spec.wants_reconnect(msg):
                log.info("spot %s asked us to reconnect", spec.name)
                return
            try:
                quote = spec.parse(msg)
            except Exception as e:  # noqa: BLE001 - one bad frame is not a bad feed
                st.errors += 1
                log.warning("spot %s parse error: %s: %s", spec.name, type(e).__name__, e)
                continue
            if quote is None:
                # Control frames (acks, heartbeats) are not rejections; only a frame that
                # LOOKED like a quote and failed the sanity filter is worth counting, and
                # we cannot distinguish those here, so count nothing and stay quiet.
                continue
            # Bitstamp republishes the whole book when a level 40 deep changes, so most of
            # its frames carry an identical top of book. Deduping here is the difference
            # between ~10 and ~1 stored rows per second from that venue, and drops exactly
            # zero information.
            key = (quote.bid, quote.ask)
            if key == last:
                continue
            last = key
            st.quotes += 1
            self._publish(quote)


def _median(values: list[Decimal]) -> Decimal | None:
    """Exact Decimal median. Even counts average the two middle values."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return ((s[n // 2 - 1] + s[n // 2]) / 2).quantize(_Q)


__all__ = [
    "DEFAULT_VENUES",
    "MAX_SPREAD_BPS",
    "VENUE_BITSTAMP",
    "VENUE_COINBASE",
    "VENUE_KRAKEN",
    "VENUE_SPECS",
    "SpotFeed",
    "SpotQuote",
    "SpotTick",
    "VenueSpec",
    "VenueStats",
    "parse_bitstamp",
    "parse_coinbase",
    "parse_kraken",
]
