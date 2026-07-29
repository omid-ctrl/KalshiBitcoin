"""Kalshi WebSocket feed.

WHY THIS FILE EXISTS
--------------------
REST responses are cached ~1s and two endpoints can disagree by a cent. On a market
whose entire spread is 1 cent, a stale cent is the whole edge. Every live decision must
come off the socket; REST is for discovery, backfill and reconciliation only.

AUTH (verified 2026-07-28 against the live gateway)
---------------------------------------------------
Same RSA-PSS scheme as REST, but the signed path is the WS route itself:

    msg = f"{timestamp_ms}GET/trade-api/ws/v2"

sent as HTTP headers on the upgrade request (KALSHI-ACCESS-KEY / -TIMESTAMP /
-SIGNATURE). There is no in-band login frame.

MEASURED, AND CONTRARY TO THE USUAL ASSUMPTION: the gateway rejects the *handshake*
itself with HTTP 401 `token_authentication_failure` when credentials are absent. The
"public" market-data channels are public over REST but NOT over this socket - there is
no anonymous WS access at all. So `available` is False without credentials and callers
must fall back to REST polling. We still filter to PUBLIC_CHANNELS in that case and say
so loudly rather than silently retrying a connection that can never succeed.

SEQUENCE INTEGRITY - THE PART THAT ACTUALLY LOSES MONEY
-------------------------------------------------------
`orderbook_delta` carries a monotonic `seq` per subscription id (`sid`), starting at the
`orderbook_snapshot`. If you apply deltas across a gap you get a book that looks
perfectly well-formed and is quietly wrong - phantom size at a price nobody is showing.
That is invisible in logs and expensive in fills, so this module refuses to guess:

    seq == last + 1   -> apply
    seq <= last       -> DUPLICATE, drop (counted)
    seq  > last + 1   -> GAP,   book marked stale + resubscribe
    seq == 1, last>1  -> RESET, book marked stale + resubscribe

A stale book is never served as if it were good: `book()` returns None for it, a
`BookStale` event is emitted, and the ticker stays in `stale_tickers` until a fresh
snapshot lands.

SCHEMA DEFENSIVENESS
--------------------
Kalshi is mid-migration from integer-cent fields to decimal-string `*_dollars` / `*_fp`
fields, and the socket has not moved in lockstep with REST. Every numeric parse here
accepts both spellings and infers the unit (a price of `53` is cents, `"0.5300"` is
dollars). This costs a few lines and removes a whole class of silent zero-data bugs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection, connect

from kalshi_btc.config import Settings
from kalshi_btc.core.types import Book, BookLevel, Side, dec
from kalshi_btc.exec.client import load_private_key, sign_request

log = logging.getLogger(__name__)

# The path that gets signed for the upgrade request. NOT the full wss:// URL.
WS_SIGN_PATH = "/trade-api/ws/v2"

CH_ORDERBOOK = "orderbook_delta"
CH_TICKER = "ticker_v2"
CH_TRADE = "trade"
CH_FILL = "fill"
CH_BRTI = "cfbenchmarks_value"

# Channels that key off a market_tickers list vs. account/index-wide channels.
MARKET_CHANNELS = frozenset({CH_ORDERBOOK, CH_TICKER, CH_TRADE})
# "Public" in the REST sense. Kept as a concept because it is what we would subscribe to
# unauthenticated if the gateway ever allows it - see the module docstring.
PUBLIC_CHANNELS = frozenset({CH_ORDERBOOK, CH_TICKER, CH_TRADE})
AUTHED_CHANNELS = frozenset({CH_FILL, CH_BRTI})

DEFAULT_CHANNELS = (CH_ORDERBOOK, CH_TRADE, CH_TICKER)


# --------------------------------------------------------------------------- events
@dataclass(frozen=True)
class BookUpdate:
    """A book that is known-good as of `seq`."""

    ticker: str
    book: Book
    seq: int | None
    is_snapshot: bool
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class BookDelta:
    """The raw delta, kept separately because the research value is in the flow."""

    ticker: str
    seq: int | None
    side: Side
    price: Decimal
    delta_size: Decimal
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class BookStale:
    """Emitted the moment we stop trusting a book. Never swallowed."""

    ticker: str
    reason: str  # "gap" | "reset" | "no_snapshot" | "disconnect"
    expected_seq: int | None
    got_seq: int | None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class TradeTick:
    ticker: str
    price: Decimal  # YES price in dollars
    size: Decimal
    taker_side: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class TickerUpdate:
    ticker: str
    yes_bid: Decimal | None
    yes_ask: Decimal | None
    volume: Decimal | None
    open_interest: Decimal | None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class FillTick:
    ticker: str
    side: str
    action: str
    price: Decimal
    count: Decimal
    is_taker: bool
    order_id: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class BrtiTick:
    """A CF Benchmarks index update.

    `avg_60s` (avg_60s_data) is a ROLLING trailing average and is always present. It is
    NOT the settlement figure and must never be used as one.

    `windowed_avg` (last_60s_windowed_average_15min) is the settlement-relevant number,
    and only exists inside the final minute before a quarter-hour close. `tick_count`
    counts 1..60 through that window, so tick_count tells you exactly how much of the
    settlement average is already locked in.
    """

    index_id: str
    value: Decimal | None
    avg_60s: Decimal | None
    windowed_avg: Decimal | None
    tick_count: int | None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def in_settlement_window(self) -> bool:
        return self.windowed_avg is not None


@dataclass(frozen=True)
class FeedStatus:
    """Connection lifecycle, surfaced to the consumer instead of hidden in logs."""

    state: str  # "connected" | "disconnected" | "auth_failed" | "subscribed" | "error"
    detail: str = ""
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


FeedEvent = (
    BookUpdate | BookDelta | BookStale | TradeTick | TickerUpdate | FillTick | BrtiTick | FeedStatus
)


@dataclass
class FeedStats:
    messages: int = 0
    connects: int = 0
    gaps: int = 0
    duplicates: int = 0
    resets: int = 0
    dropped: int = 0  # events discarded because the consumer could not keep up
    errors: int = 0
    last_message_at: float = 0.0

    def summary(self) -> str:
        return (
            f"msgs={self.messages} conn={self.connects} gap={self.gaps} "
            f"dup={self.duplicates} reset={self.resets} drop={self.dropped} err={self.errors}"
        )


@dataclass
class _SubState:
    """One live subscription.

    `seq` is per-SID, not per-ticker: a single subscription covering N markets shares one
    counter. That is why a gap invalidates every book under the sid, not just the book
    whose frame happened to expose the gap - the frames we missed could have belonged to
    any of them.
    """

    sid: int
    channel: str
    tickers: set[str] = field(default_factory=set)
    last_seq: int | None = None
    stale: bool = False


# --------------------------------------------------------------------------- parsing
def _num(msg: dict, *keys: str) -> Decimal | None:
    """First present key, parsed as Decimal. Returns None if none are present."""
    for k in keys:
        if k in msg and msg[k] not in (None, ""):
            return dec(msg[k])
    return None


def _price(msg: dict, *keys: str) -> Decimal | None:
    """Price in DOLLARS, tolerating the legacy integer-cent spelling.

    Kalshi prices live in [0.01, 0.99]. Anything > 1 arriving on a price field is the old
    integer-cent encoding, so scale it. `1` is ambiguous but means $1.00 either way.
    """
    v = _num(msg, *keys)
    if v is None:
        return None
    return v / 100 if v > 1 else v


def _levels(msg: dict, side: str) -> list[BookLevel]:
    """Snapshot levels for one side, from `<side>_dollars` or the legacy `<side>`."""
    rows = msg.get(f"{side}_dollars") or msg.get(side) or []
    out: list[BookLevel] = []
    for row in rows:
        if not row or len(row) < 2:
            continue
        price = dec(row[0])
        if price > 1:  # legacy integer cents
            price = price / 100
        out.append(BookLevel(price=price, size=dec(row[1])))
    return out


def _apply_delta(book: Book, side: Side, price: Decimal, delta: Decimal) -> None:
    """Add `delta` contracts at `price`, dropping the level when it empties.

    Sizes are genuinely fractional on this venue, so this is Decimal arithmetic and a
    level is only removed at <= 0, never at "close enough to zero".
    """
    levels = book.yes if side is Side.YES else book.no
    updated: list[BookLevel] = []
    found = False
    for lv in levels:
        if lv.price == price:
            found = True
            new_size = lv.size + delta
            if new_size > 0:
                updated.append(BookLevel(price=price, size=new_size))
        else:
            updated.append(lv)
    if not found and delta > 0:
        updated.append(BookLevel(price=price, size=delta))
    updated.sort(key=lambda lv: lv.price)
    if side is Side.YES:
        book.yes = updated
    else:
        book.no = updated


def _event_ts(msg: dict) -> datetime:
    """Prefer the venue's timestamp over local time when it is offered."""
    raw = msg.get("ts") or msg.get("timestamp")
    if raw is None:
        return datetime.now(UTC)
    try:
        if isinstance(raw, (int, float)):
            # Seconds vs milliseconds: anything past year ~2286 in seconds is really ms.
            secs = float(raw) / 1000 if float(raw) > 1e11 else float(raw)
            return datetime.fromtimestamp(secs, UTC)
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, OSError, OverflowError):
        return datetime.now(UTC)


# --------------------------------------------------------------------------- the feed
class KalshiWebSocket:
    """Self-healing Kalshi WS client with explicit book-integrity accounting.

    Usage::

        async with KalshiWebSocket(settings, tickers=[...]) as feed:
            async for ev in feed:
                ...

    The receive loop and the consumer are decoupled by a bounded queue. If the consumer
    stalls we drop the OLDEST events and count it, because for order books the newest
    state is the only one worth having and blocking the socket would desync us anyway.
    """

    def __init__(
        self,
        settings: Settings,
        tickers: list[str] | None = None,
        channels: tuple[str, ...] = DEFAULT_CHANNELS,
        *,
        queue_size: int = 20_000,
        on_brti: Callable[[BrtiTick], None] | None = None,
        max_backoff_s: float = 30.0,
    ) -> None:
        self.settings = settings
        self.tickers: list[str] = list(tickers or [])
        self.on_brti = on_brti
        self.max_backoff_s = max_backoff_s

        self._key = load_private_key(settings.private_key_path) if settings.has_credentials else None

        wanted = set(channels)
        if self._key is None:
            dropped = wanted - PUBLIC_CHANNELS
            if dropped:
                log.warning(
                    "no Kalshi credentials: dropping authenticated channels %s",
                    ", ".join(sorted(dropped)),
                )
            wanted &= PUBLIC_CHANNELS
        self.channels: tuple[str, ...] = tuple(sorted(wanted))

        self.books: dict[str, Book] = {}
        self.stale_tickers: set[str] = set()
        self.stats = FeedStats()
        self.auth_failed = False

        self._queue: asyncio.Queue[FeedEvent | None] = asyncio.Queue(maxsize=queue_size)
        self._subs: dict[int, _SubState] = {}
        self._pending: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        self._cmd_id = 0
        self._ws: ClientConnection | None = None
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._resub_requested = False

    # ------------------------------------------------------------------ lifecycle
    @property
    def available(self) -> bool:
        """False when this feed cannot possibly connect (no credentials).

        Callers use this to decide whether to fall back to REST polling. It is a hard
        capability check, not a hint - the gateway 401s the handshake without creds.
        """
        return self._key is not None and bool(self.channels)

    async def __aenter__(self) -> KalshiWebSocket:
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def start(self) -> None:
        if self._task is not None:
            return
        if not self.available:
            log.warning(
                "Kalshi WS not started: the gateway requires credentials for EVERY channel "
                "(verified: HTTP 401 token_authentication_failure on anonymous upgrade). "
                "Falling back to REST-only capture."
            )
            self._emit(FeedStatus("auth_failed", "no credentials; WS unavailable"))
            return
        self._task = asyncio.create_task(self._run(), name="kalshi-ws")

    async def stop(self) -> None:
        self._stopping.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # Wake any consumer parked on __anext__.
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)

    # ------------------------------------------------------------------ consumer API
    def __aiter__(self) -> AsyncIterator[FeedEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[FeedEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    def drain(self, limit: int = 10_000) -> list[FeedEvent]:
        """Non-blocking pull of everything queued right now.

        Poll-shaped consumers (the capture runner's fixed-cadence loop) want this rather
        than an async-for, so they stay in control of their own timing.
        """
        out: list[FeedEvent] = []
        for _ in range(limit):
            try:
                ev = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if ev is not None:
                out.append(ev)
        return out

    def book(self, ticker: str) -> Book | None:
        """The current book, or None if it is stale/unknown.

        Returning None for a stale book is deliberate: callers that forget to check a
        flag get no data rather than wrong data.
        """
        if ticker in self.stale_tickers:
            return None
        return self.books.get(ticker)

    async def set_tickers(self, tickers: list[str]) -> None:
        """Swap the subscribed market set (used at the hourly rollover)."""
        new = list(dict.fromkeys(tickers))
        if new == self.tickers:
            return
        self.tickers = new
        if self._ws is not None:
            await self._resubscribe_markets("ticker set changed")

    # ------------------------------------------------------------------ queue plumbing
    def _emit(self, ev: FeedEvent) -> None:
        try:
            self._queue.put_nowait(ev)
            return
        except asyncio.QueueFull:
            pass
        # Drop-oldest. Never block the recv loop for a slow consumer.
        with contextlib.suppress(asyncio.QueueEmpty):
            self._queue.get_nowait()
        self.stats.dropped += 1
        if self.stats.dropped == 1 or self.stats.dropped % 500 == 0:
            log.warning(
                "feed queue full (maxsize=%d): %d events dropped - consumer is too slow",
                self._queue.maxsize,
                self.stats.dropped,
            )
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(ev)

    # ------------------------------------------------------------------ connection
    def _auth_headers(self) -> dict[str, str]:
        assert self._key is not None
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.settings.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": sign_request(self._key, ts, "GET", WS_SIGN_PATH),
        }

    async def _run(self) -> None:
        attempt = 0
        while not self._stopping.is_set():
            try:
                async with connect(
                    self.settings.ws_base,
                    additional_headers=self._auth_headers(),
                    open_timeout=15,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    self.stats.connects += 1
                    log.info("WS connected to %s", self.settings.ws_base)
                    self._emit(FeedStatus("connected", self.settings.ws_base))
                    await self._subscribe_all()
                    await self._recv_loop(ws)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.InvalidStatus as e:
                status = e.response.status_code
                if status in (401, 403):
                    # Retrying cannot help. Say exactly what is wrong and stop.
                    self.auth_failed = True
                    log.error(
                        "WS handshake rejected with HTTP %d. Kalshi requires valid API "
                        "credentials for ALL WebSocket channels, including market data. "
                        "Check KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH and that the key "
                        "matches KALSHI_ENV=%s.",
                        status,
                        self.settings.env,
                    )
                    self._emit(FeedStatus("auth_failed", f"HTTP {status}"))
                    self._mark_all_stale("disconnect")
                    return
                self.stats.errors += 1
                log.warning("WS handshake failed: HTTP %d", status)
            except Exception as e:  # noqa: BLE001 - the feed must never die on us
                self.stats.errors += 1
                log.warning("WS connection error: %s: %s", type(e).__name__, e)
            finally:
                self._ws = None
                self._subs.clear()
                self._pending.clear()

            if self._stopping.is_set():
                return
            self._mark_all_stale("disconnect")
            self._emit(FeedStatus("disconnected", "reconnecting"))
            # Exponential backoff with jitter so a venue-side blip does not turn into a
            # synchronised thundering herd of reconnects.
            delay = min(self.max_backoff_s, 1.0 * (2**attempt))
            delay += random.uniform(0, delay * 0.3)
            attempt += 1
            log.info("WS reconnecting in %.1fs", delay)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)

    def _mark_all_stale(self, reason: str) -> None:
        for ticker in list(self.books):
            if ticker not in self.stale_tickers:
                self.stale_tickers.add(ticker)
                self._emit(BookStale(ticker=ticker, reason=reason, expected_seq=None, got_seq=None))

    # ------------------------------------------------------------------ subscriptions
    async def _send(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        await ws.send(json.dumps(payload))

    def _next_id(self) -> int:
        self._cmd_id += 1
        return self._cmd_id

    async def _subscribe_all(self) -> None:
        market_channels = tuple(c for c in self.channels if c in MARKET_CHANNELS)
        if market_channels and self.tickers:
            await self._subscribe(market_channels, tuple(self.tickers))
        for ch in self.channels:
            if ch not in MARKET_CHANNELS:
                await self._subscribe((ch,), ())

    async def _subscribe(self, channels: tuple[str, ...], tickers: tuple[str, ...]) -> None:
        cmd_id = self._next_id()
        params: dict[str, Any] = {"channels": list(channels)}
        if tickers:
            params["market_tickers"] = list(tickers)
        self._pending[cmd_id] = (channels, tickers)
        await self._send({"id": cmd_id, "cmd": "subscribe", "params": params})
        log.info("WS subscribe id=%d channels=%s tickers=%d", cmd_id, ",".join(channels), len(tickers))

    async def _resubscribe_markets(self, reason: str) -> None:
        """Tear down and rebuild the market subscriptions to force a fresh snapshot.

        This is the ONLY sanctioned repair for a gap. We never try to patch a book back
        into consistency from deltas we did not see.
        """
        if self._resub_requested:
            return  # one repair in flight is enough; extra churn only widens the outage
        self._resub_requested = True
        try:
            sids = [s.sid for s in self._subs.values() if s.channel in MARKET_CHANNELS]
            if sids:
                await self._send(
                    {"id": self._next_id(), "cmd": "unsubscribe", "params": {"sids": sids}}
                )
                for sid in sids:
                    self._subs.pop(sid, None)
            market_channels = tuple(c for c in self.channels if c in MARKET_CHANNELS)
            if market_channels and self.tickers:
                log.warning("resubscribing market channels (%s)", reason)
                await self._subscribe(market_channels, tuple(self.tickers))
        except Exception as e:  # noqa: BLE001 - a failed repair must not kill the loop
            log.warning("resubscribe failed (%s); connection will be recycled: %s", reason, e)
            if self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._ws.close()
        finally:
            self._resub_requested = False

    # ------------------------------------------------------------------ receive
    async def _recv_loop(self, ws: ClientConnection) -> None:
        async for raw in ws:
            if self._stopping.is_set():
                return
            self.stats.messages += 1
            self.stats.last_message_at = time.monotonic()
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                self.stats.errors += 1
                log.warning("undecodable WS frame: %r", str(raw)[:200])
                continue
            try:
                await self._handle(data)
            except Exception as e:  # noqa: BLE001 - one bad frame must not kill the feed
                self.stats.errors += 1
                log.warning("error handling %s frame: %s: %s", data.get("type"), type(e).__name__, e)

    async def _handle(self, data: dict[str, Any]) -> None:
        mtype = data.get("type")
        msg = data.get("msg") or {}

        if mtype == "subscribed":
            sid = msg.get("sid", data.get("sid"))
            channel = msg.get("channel", "")
            channels, tickers = self._pending.pop(data.get("id"), ((channel,), ()))
            existing = self._subs.get(sid)
            known = existing.tickers if existing else set()
            self._subs[sid] = _SubState(
                sid=sid,
                channel=channel or channels[0],
                tickers=set(tickers) | known,
                last_seq=existing.last_seq if existing else None,
            )
            log.info("WS subscribed sid=%s channel=%s", sid, channel)
            self._emit(FeedStatus("subscribed", f"{channel} sid={sid}"))
            return

        if mtype in ("error", "unsubscribed"):
            if mtype == "error":
                self.stats.errors += 1
                log.warning("WS error frame: %s", json.dumps(msg)[:300])
                self._emit(FeedStatus("error", json.dumps(msg)[:200]))
            return

        if mtype == "orderbook_snapshot":
            self._on_snapshot(data, msg)
        elif mtype == "orderbook_delta":
            await self._on_delta(data, msg)
        elif mtype == "trade":
            self._on_trade(msg)
        elif mtype in ("ticker_v2", "ticker"):
            self._on_ticker(msg)
        elif mtype == "fill":
            self._on_fill(msg)
        elif mtype == "cfbenchmarks_value":
            self._on_brti(msg)

    # ------------------------------------------------------------------ handlers
    def _sub_for(self, data: dict) -> _SubState | None:
        sid = data.get("sid")
        sub = self._subs.get(sid)
        if sub is None and sid is not None:
            # Snapshot can beat the `subscribed` ack on the wire; adopt the sid.
            sub = _SubState(sid=sid, channel=CH_ORDERBOOK)
            self._subs[sid] = sub
        return sub

    def _on_snapshot(self, data: dict, msg: dict) -> None:
        ticker = msg.get("market_ticker") or msg.get("ticker") or ""
        if not ticker:
            return
        seq = data.get("seq")
        sub = self._sub_for(data)
        if sub is not None:
            sub.last_seq = seq
            sub.stale = False
            sub.tickers.add(ticker)

        book = Book(
            ticker=ticker,
            yes=sorted(_levels(msg, "yes"), key=lambda lv: lv.price),
            no=sorted(_levels(msg, "no"), key=lambda lv: lv.price),
            seq=seq,
            ts=_event_ts(msg),
        )
        self.books[ticker] = book
        self.stale_tickers.discard(ticker)
        self._emit(BookUpdate(ticker=ticker, book=book, seq=seq, is_snapshot=True, ts=book.ts))

    async def _on_delta(self, data: dict, msg: dict) -> None:
        ticker = msg.get("market_ticker") or msg.get("ticker") or ""
        if not ticker:
            return
        seq = data.get("seq")
        sub = self._sub_for(data)

        if sub is not None and seq is not None:
            sub.tickers.add(ticker)
            last = sub.last_seq
            if last is None:
                # Deltas before any snapshot: nothing to apply them to. Once we are
                # already waiting on a snapshot, further deltas are expected - stay quiet
                # rather than re-firing a repair that is already in flight.
                if ticker not in self.stale_tickers:
                    self._go_stale(sub, ticker, "no_snapshot", None, seq)
                    await self._resubscribe_markets("delta before snapshot")
                return
            if seq <= last:
                # A sequence that restarts at 1 is the venue resetting the stream, not a
                # replay - it must be distinguished, because a replay is harmless to drop
                # and a reset means our whole book state is now fiction.
                if seq == 1 and last > 1:
                    self.stats.resets += 1
                    log.warning("orderbook RESET on sid=%s: seq went %d -> 1", sub.sid, last)
                    self._go_stale(sub, ticker, "reset", last + 1, seq)
                    await self._resubscribe_markets("sequence reset")
                    return
                # Duplicate/replayed frame. Applying it would double-count size.
                self.stats.duplicates += 1
                return
            if seq > last + 1:
                self.stats.gaps += 1
                log.warning(
                    "orderbook GAP on sid=%s: expected seq %d, got %d - %d book(s) stale",
                    sub.sid, last + 1, seq, len(sub.tickers) or 1,
                )
                self._go_stale(sub, ticker, "gap", last + 1, seq)
                await self._resubscribe_markets("sequence gap")
                return
            sub.last_seq = seq

        if ticker in self.stale_tickers:
            return  # waiting on a fresh snapshot; refuse to patch a book we do not trust

        book = self.books.get(ticker)
        if book is None:
            self._go_stale(sub, ticker, "no_snapshot", None, seq)
            return

        side = Side.NO if str(msg.get("side", "yes")).lower() == "no" else Side.YES
        price = _price(msg, "price_dollars", "price")
        delta = _num(msg, "delta_fp", "delta")
        if price is None or delta is None:
            return

        _apply_delta(book, side, price, delta)
        book.seq = seq
        book.ts = _event_ts(msg)
        self._emit(
            BookDelta(ticker=ticker, seq=seq, side=side, price=price, delta_size=delta, ts=book.ts)
        )
        self._emit(BookUpdate(ticker=ticker, book=book, seq=seq, is_snapshot=False, ts=book.ts))

    def _go_stale(
        self,
        sub: _SubState | None,
        ticker: str,
        reason: str,
        expected: int | None,
        got: int | None,
    ) -> None:
        """Invalidate every book sharing the broken sequence, not just the loud one.

        The books are DELETED rather than flagged-and-kept, so there is no path by which
        a caller can read a corrupted book back out. BookStale is edge-triggered: it
        fires on the transition into staleness, so a long outage produces one event per
        book rather than one per dropped frame.
        """
        affected = set(sub.tickers) if sub and sub.tickers else set()
        affected.add(ticker)
        if sub is not None:
            sub.stale = True
            sub.last_seq = None
        for t in affected:
            self.books.pop(t, None)
            if t in self.stale_tickers:
                continue
            self.stale_tickers.add(t)
            self._emit(BookStale(ticker=t, reason=reason, expected_seq=expected, got_seq=got))

    def _on_trade(self, msg: dict) -> None:
        ticker = msg.get("market_ticker") or msg.get("ticker") or ""
        price = _price(msg, "yes_price_dollars", "yes_price", "price_dollars", "price")
        size = _num(msg, "count_fp", "count", "size_fp", "size")
        if not ticker or price is None or size is None:
            return
        self._emit(
            TradeTick(
                ticker=ticker,
                price=price,
                size=size,
                taker_side=str(msg.get("taker_side", "")),
                ts=_event_ts(msg),
            )
        )

    def _on_ticker(self, msg: dict) -> None:
        ticker = msg.get("market_ticker") or msg.get("ticker") or ""
        if not ticker:
            return
        self._emit(
            TickerUpdate(
                ticker=ticker,
                yes_bid=_price(msg, "yes_bid_dollars", "yes_bid"),
                yes_ask=_price(msg, "yes_ask_dollars", "yes_ask"),
                volume=_num(msg, "volume_fp", "volume"),
                open_interest=_num(msg, "open_interest_fp", "open_interest"),
                ts=_event_ts(msg),
            )
        )

    def _on_fill(self, msg: dict) -> None:
        ticker = msg.get("market_ticker") or msg.get("ticker") or ""
        price = _price(msg, "yes_price_dollars", "yes_price", "price_dollars", "price")
        count = _num(msg, "count_fp", "count")
        if not ticker or price is None or count is None:
            return
        self._emit(
            FillTick(
                ticker=ticker,
                side=str(msg.get("side", "")),
                action=str(msg.get("action", "")),
                price=price,
                count=count,
                is_taker=bool(msg.get("is_taker", False)),
                order_id=str(msg.get("order_id", "")),
                ts=_event_ts(msg),
            )
        )

    def _on_brti(self, msg: dict) -> None:
        """Parse a cfbenchmarks_value frame.

        Both average fields are parsed defensively because only one of them is ever
        guaranteed: `avg_60s_data` is always there, and the windowed average exists ONLY
        inside the final minute before a quarter-hour close. Either may arrive as a bare
        scalar or as an object carrying its own tick count.
        """
        avg_raw = msg.get("avg_60s_data")
        win_raw = msg.get("last_60s_windowed_average_15min")

        def unpack(raw: Any) -> tuple[Decimal | None, int | None]:
            if raw in (None, ""):
                return None, None
            if isinstance(raw, dict):
                val = _num(raw, "value_dollars", "value", "average", "avg", "mean")
                cnt = raw.get("count", raw.get("tick_count", raw.get("num_values")))
                return val, int(cnt) if cnt is not None else None
            return dec(raw), None

        avg_60s, avg_count = unpack(avg_raw)
        windowed, win_count = unpack(win_raw)

        tick = BrtiTick(
            index_id=str(msg.get("index_id") or msg.get("index") or "BRTI"),
            value=_num(msg, "value_dollars", "value", "price_dollars", "price"),
            avg_60s=avg_60s,
            windowed_avg=windowed,
            tick_count=win_count if win_count is not None else avg_count,
            ts=_event_ts(msg),
        )
        if self.on_brti is not None:
            try:
                self.on_brti(tick)
            except Exception as e:  # noqa: BLE001 - a bad callback must not kill the feed
                log.warning("on_brti callback raised: %s: %s", type(e).__name__, e)
        self._emit(tick)
