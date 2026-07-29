"""Phase 0 recorder.

WHY THIS IS THE CRITICAL PATH
-----------------------------
Kalshi sells no historical L2 data. There is no vendor, no backfill endpoint, no way to
buy back a missed hour at any price. Every hour this process is not running is research
data that is gone permanently. That single fact drives every design choice below:

- It runs with NO credentials. Waiting on an API key would cost real hours.
- It never exits on a transient API error. A 500 from the venue is logged and the next
  cycle proceeds; only an explicit Ctrl-C stops the loop.
- It flushes on the way out, so a clean shutdown never drops the buffered tail.
- It records the FULL 188-strike ladder by default, not just the ~4-9 live strikes.
  The pinned strikes are nearly free to store (columnar, hugely repetitive) and they are
  the only record of when a strike woke up. `strike_window` trims this if disk ever
  matters more than completeness, but the default errs toward keeping everything.

REST *AND* WEBSOCKET, DELIBERATELY BOTH
---------------------------------------
The ladder poll is REST because it is the only way to see all 188 strikes at once, and a
2s cadence on a cached-1s endpoint is honest sampling. The per-strike book flow is
WebSocket because REST caching would smear the 1-cent spread we actually trade on. The
two are complementary, not redundant: the poll gives breadth, the socket gives fidelity.

MEASURED CAVEAT: Kalshi's WS gateway rejects the handshake with HTTP 401 when
credentials are absent - even for market-data channels that are public over REST. So
without an API key this degrades to a REST-only recorder and says so, rather than
retrying a connection that cannot succeed. Ladder capture is unaffected.

SPOT IS RECORDED ALWAYS
-----------------------
That caveat used to mean a credential-less capture recorded a ladder and no price at
all, which cannot be calibrated against anything. So we additionally run
`kalshi_btc.feed.spot_ws`, which reads PUBLIC Coinbase/Kraken/Bitstamp top-of-book with
no authentication whatsoever, and writes it to the `spot` table on every cycle. That
turns a keyless Phase 0 capture into something `kbtc proxy-score` can actually grade.
It is a PROXY for BRTI, with tracking error - see that module for exactly how much.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from rich.console import Console

from kalshi_btc.config import DECOY_SERIES, SERIES_TICKER, Settings
from kalshi_btc.core.types import Liquidity, MarketSnapshot, dec_or_none
from kalshi_btc.exec.client import KalshiClient, event_is_hourly
from kalshi_btc.feed.kalshi_ws import (
    CH_BRTI,
    CH_ORDERBOOK,
    CH_TICKER,
    CH_TRADE,
    BookStale,
    BookDelta,
    BrtiTick,
    FillTick,
    KalshiWebSocket,
    TradeTick,
)
from kalshi_btc.feed.spot_ws import DEFAULT_VENUES, SpotFeed
from kalshi_btc.store.db import Store

log = logging.getLogger(__name__)

# How long after close to keep retrying the settlement backfill. Kalshi's
# expected_expiration_time is close + 5 minutes; we allow generous slack because a late
# settlement is still free ground truth and a missed one is not recoverable cheaply.
SETTLEMENT_ATTEMPTS = 20
SETTLEMENT_RETRY_S = 45.0


@dataclass
class CaptureCounters:
    """Everything the heartbeat reports. Also the return value of run_capture()."""

    cycles: int = 0
    ladder_rows: int = 0
    deltas: int = 0
    trades: int = 0
    brti: int = 0
    spot_rows: int = 0
    fills: int = 0
    settlements: int = 0
    stale_events: int = 0
    disagreements: int = 0  # cycles where the public venues did not agree on BTC
    api_errors: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, int]:
        return {
            "cycles": self.cycles,
            "ladder_rows": self.ladder_rows,
            "deltas": self.deltas,
            "trades": self.trades,
            "brti": self.brti,
            "spot_rows": self.spot_rows,
            "fills": self.fills,
            "settlements": self.settlements,
            "stale_events": self.stale_events,
            "disagreements": self.disagreements,
            "api_errors": self.api_errors,
        }


def _human(n: int) -> str:
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def _countdown(seconds: float) -> str:
    """mm:ss to close, or a +mm:ss overrun once the event is past its close."""
    sign = "-" if seconds >= 0 else "+"
    s = int(abs(seconds))
    return f"T{sign}{s // 60:02d}:{s % 60:02d}"


def _is_target_series(event_ticker: str) -> bool:
    """KXBTC is a different, far less liquid series whose tickers are NOT a prefix match.

    `KXBTCD-...` does not start with `KXBTC-`, so the prefix test already separates them,
    but the decoy is named explicitly here because confusing the two is the single most
    expensive naming mistake available in this codebase.
    """
    if event_ticker.startswith(f"{DECOY_SERIES}-"):
        return False
    return event_ticker.startswith(f"{SERIES_TICKER}-")


async def discover_events(client: KalshiClient) -> tuple[dict | None, dict | None]:
    """Return (current, next) open KXBTCD events, ordered by close time.

    Kalshi lists several open events at once, but the far-future ones are half-built
    ladders that will not open for trading until an hour before their close. "Current" is
    simply the soonest-closing event whose close is still ahead of us.
    """
    events = await client.get_events(SERIES_TICKER, status="open", limit=20)
    live: list[dict] = []
    for e in events:
        ticker = e.get("event_ticker", "")
        if not _is_target_series(ticker):
            log.warning("ignoring non-%s event %s", SERIES_TICKER, ticker)
            continue
        # KXBTCD hosts DAILY ($250 spacing) and WEEKLY ($500) events under the very same
        # series ticker as the hourly one. Without this filter the hourly rollover lands
        # on whichever event closes soonest, which after a top-of-hour close is often the
        # daily - a different instrument with a completely different variance profile.
        # Observed live: 26JUL2821 rolled straight into the daily 26JUL2917.
        if not event_is_hourly(e):
            log.debug("skipping non-hourly %s event %s", SERIES_TICKER, ticker)
            continue
        live.append(e)

    now = datetime.now(UTC)

    def close_of(e: dict) -> datetime:
        mkts = e.get("markets") or []
        return MarketSnapshot.from_api(mkts[0]).close_time if mkts else now

    live.sort(key=close_of)
    upcoming = [e for e in live if close_of(e) > now]
    current = upcoming[0] if upcoming else None
    nxt = upcoming[1] if len(upcoming) > 1 else None
    return current, nxt


def _select_strikes(
    markets: list[MarketSnapshot], strike_window: int | None
) -> list[MarketSnapshot]:
    """Full ladder by default; optionally only the band around the money.

    The band is centred on the strike whose mid is closest to 0.50 rather than on spot,
    because the market's own opinion of the money is the thing we are trying to record.
    """
    if strike_window is None:
        return markets
    live = [m for m in markets if m.is_live] or markets
    atm = min(live, key=lambda m: abs(m.mid - Decimal("0.5")))
    ordered = sorted(markets, key=lambda m: m.strike)
    idx = min(range(len(ordered)), key=lambda i: abs(ordered[i].strike - atm.strike))
    lo = max(0, idx - strike_window)
    hi = min(len(ordered), idx + strike_window + 1)
    return ordered[lo:hi]


async def _record_ladder(
    client: KalshiClient,
    store: Store,
    event_ticker: str,
    strike_window: int | None,
) -> tuple[int, list[MarketSnapshot], MarketSnapshot | None, float]:
    """One ladder snapshot. Returns (rows, live_markets, atm, minutes_to_close).

    A single /events call with nested markets returns all 188 strikes for 10 rate-limit
    tokens. Fetching them per-market would cost 1880 tokens and take longer than the
    cadence it is trying to sample.
    """
    event = await client.get_event(event_ticker)
    raw = event.get("markets") or []
    if not raw:
        return 0, [], None, 0.0

    ts = datetime.now(UTC)
    markets = [MarketSnapshot.from_api(m) for m in raw]
    close_time = markets[0].close_time
    minutes_to_close = (close_time - ts).total_seconds() / 60.0

    rows = 0
    for m in _select_strikes(markets, strike_window):
        store.add_ladder_snapshot(
            ts=ts,
            event_ticker=event_ticker,
            ticker=m.ticker,
            strike=m.strike,
            yes_bid=m.yes_bid,
            yes_ask=m.yes_ask,
            yes_bid_size=m.yes_bid_size,
            yes_ask_size=m.yes_ask_size,
            volume=m.volume,
            open_interest=m.open_interest,
            minutes_to_close=minutes_to_close,
        )
        rows += 1

    live = [m for m in markets if m.is_live]
    atm = min(live, key=lambda m: abs(m.mid - Decimal("0.5"))) if live else None
    return rows, live, atm, minutes_to_close


async def backfill_settlement(
    client: KalshiClient,
    store: Store,
    event_ticker: str,
    probe_ticker: str,
    counters: CaptureCounters,
) -> None:
    """Poll one market of a just-closed event until its expiration_value appears.

    Only one market is polled because `expiration_value` is a property of the event -
    all 188 markets under it carry the identical realised BRTI average (verified). This
    is our free ground truth for scoring the settlement model, so it is worth waiting
    several minutes for.
    """
    for attempt in range(SETTLEMENT_ATTEMPTS):
        try:
            data = await client.request("GET", f"/markets/{probe_ticker}")
            m = data.get("market", {})
            raw = m.get("expiration_value")
            # Not every populated expiration_value is a number: the venue also serves
            # outcome strings ("a", "Yes", "No") in this field. Those carry no index
            # level, so keep polling rather than storing a bogus settlement.
            value = dec_or_none(raw)
            if value is not None:
                store.upsert_settlement(
                    close_time=MarketSnapshot.from_api(m).close_time,
                    event_ticker=event_ticker,
                    expiration_value=value,
                )
                counters.settlements += 1
                log.info("settlement %s -> BRTI 60s avg %s", event_ticker, raw)
                return
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - backfill is best-effort, never fatal
            counters.api_errors += 1
            log.warning("settlement backfill for %s failed: %s", event_ticker, e)
        await asyncio.sleep(SETTLEMENT_RETRY_S if attempt else 5.0)
    log.warning("gave up backfilling settlement for %s", event_ticker)


async def backfill_latest_settlement(
    client: KalshiClient, store: Store, counters: CaptureCounters
) -> None:
    """Grab the most recently settled event at startup.

    One page, no pagination: /markets?status=settled is returned newest-first, so a
    limit of 1 is the whole cost. This makes even a short capture run produce a
    settlements row, and proves the ground-truth path works before we depend on it at a
    rollover an hour from now.
    """
    try:
        data = await client.request(
            "GET",
            "/markets",
            params={"series_ticker": SERIES_TICKER, "status": "settled", "limit": 1},
        )
        markets = data.get("markets") or []
        if not markets:
            return
        m = markets[0]
        snap = MarketSnapshot.from_api(m)
        # None covers both "absent" and "present but not a number" (see dec_or_none).
        if snap.expiration_value is None:
            return
        store.upsert_settlement(
            close_time=snap.close_time,
            event_ticker=m.get("event_ticker", snap.ticker.rsplit("-", 1)[0]),
            expiration_value=snap.expiration_value,
        )
        counters.settlements += 1
        log.info(
            "seeded settlement %s -> %s", m.get("event_ticker"), m.get("expiration_value")
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        counters.api_errors += 1
        log.warning("could not seed latest settlement: %s", e)


def _drain_feed(feed: KalshiWebSocket, store: Store, counters: CaptureCounters) -> BrtiTick | None:
    """Move everything the socket produced since last cycle into the store."""
    latest_brti: BrtiTick | None = None
    for ev in feed.drain():
        if isinstance(ev, BookDelta):
            store.add_book_delta(
                ts=ev.ts,
                ticker=ev.ticker,
                seq=ev.seq,
                side=str(ev.side),
                price=ev.price,
                delta_size=ev.delta_size,
            )
            counters.deltas += 1
        elif isinstance(ev, TradeTick):
            store.add_trade(
                ts=ev.ts, ticker=ev.ticker, price=ev.price, size=ev.size, taker_side=ev.taker_side
            )
            counters.trades += 1
        elif isinstance(ev, BrtiTick):
            store.add_brti(
                ts=ev.ts,
                index_id=ev.index_id,
                value=ev.value,
                avg_60s=ev.avg_60s,
                windowed_avg=ev.windowed_avg,
                tick_count=ev.tick_count,
            )
            counters.brti += 1
            latest_brti = ev
        elif isinstance(ev, FillTick):
            store.add_fill(
                ts=ev.ts,
                ticker=ev.ticker,
                side=ev.side,
                action=ev.action,
                price=ev.price,
                count=ev.count,
                liquidity=str(Liquidity.TAKER if ev.is_taker else Liquidity.MAKER),
                fee=Decimal("0"),
                order_id=ev.order_id,
            )
            counters.fills += 1
        elif isinstance(ev, BookStale):
            counters.stale_events += 1
            log.warning(
                "book stale: %s (%s, expected seq %s got %s)",
                ev.ticker, ev.reason, ev.expected_seq, ev.got_seq,
            )
    return latest_brti


def _drain_spot(spot: SpotFeed | None, store: Store, counters: CaptureCounters) -> int:
    """Move every public spot tick since last cycle into the store.

    Separate from `_drain_feed` because this path has no credential dependency and must
    keep running even when the Kalshi socket is dead - it is the only price series a
    keyless capture produces.
    """
    if spot is None:
        return 0
    n = 0
    for tick in spot.drain():
        store.add_spot(
            ts=tick.ts,
            venue=tick.venue,
            bid=tick.bid,
            ask=tick.ask,
            mid=tick.mid,
            proxy=tick.proxy,
        )
        n += 1
    counters.spot_rows += n
    # Cross-venue disagreement is a kill-switch condition for anything that trades off
    # this price, so it is counted here (once per cycle) rather than only rendered. A
    # session that ends with a non-zero count had minutes it should not have traded.
    if len(spot.fresh_quotes()) >= 2 and not spot.venues_agree():
        counters.disagreements += 1
        log.warning(
            "spot venues disagree by %.1f bps - the composite is not trade-quality",
            spot.spread_bps() or 0.0,
        )
    return n


def _spot_text(spot: SpotFeed | None) -> str:
    """Live proxy price plus the cross-venue agreement state, in ~20 characters.

    Agreement is rendered as the venue count coloured by the kill-switch condition, so
    the one glance that reads the price also reads whether the price is trustworthy:
    green = venues agree, red = they do not, yellow = too few venues to check.
    """
    if spot is None:
        return "[dim]spot off[/]"
    fresh = len(spot.fresh_quotes())
    total = len(spot.specs)
    px = spot.proxy()
    if px is None:
        px = spot.mid()
        label = f"[dim]~{px:,.0f}[/]" if px is not None else "[dim]spot --[/]"
    else:
        label = f"[bold]${px:,.0f}[/]"

    if fresh < 2:
        agree = f"[yellow]{fresh}/{total}v[/]"
    elif spot.venues_agree():
        agree = f"[green]{fresh}/{total}v[/]"
    else:
        agree = f"[bold red]{fresh}/{total}v {spot.spread_bps() or 0.0:.0f}bps[/]"
    return f"{label} {agree}"


def _heartbeat(
    console: Console,
    event_ticker: str,
    minutes_to_close: float,
    live: list[MarketSnapshot],
    atm: MarketSnapshot | None,
    brti: BrtiTick | None,
    counters: CaptureCounters,
    feed: KalshiWebSocket | None,
    spot: SpotFeed | None = None,
) -> None:
    """One compact, colour-coded line per cycle. Meant to be watched for hours."""
    now = datetime.now(UTC).strftime("%H:%M:%S")
    left = _countdown(minutes_to_close * 60)
    # The final minute is the settlement window - make it impossible to miss.
    left_style = "bold red" if 0 <= minutes_to_close <= 1 else "cyan"

    if atm is not None:
        atm_txt = (
            f"{atm.strike:,.0f} [green]{atm.yes_bid:.2f}[/]/[red]{atm.yes_ask:.2f}[/]"
        )
    else:
        atm_txt = "[dim]atm --[/]"

    val = None if brti is None else (brti.windowed_avg or brti.avg_60s or brti.value)
    if val is not None:
        # "W" = the settlement-relevant windowed average, "r" = the rolling trailing one.
        # Confusing the two is the difference between knowing the settlement and guessing.
        brti_txt = f"[{'bold yellow' if brti.windowed_avg is not None else 'dim'}]"
        brti_txt += f"{'W' if brti.windowed_avg is not None else 'r'}{val:,.0f}[/]"
    else:
        brti_txt = "[dim]brti --[/]"

    if feed is None or not feed.available:
        ws_txt = "[yellow]rest-only[/]"
    elif feed.stale_tickers:
        ws_txt = f"[bold red]STALE:{len(feed.stale_tickers)}[/]"
    else:
        ws_txt = f"[green]ws {_human(feed.stats.messages)}[/]"

    # BRTI is the real index and only exists with credentials; the spot proxy is always
    # there. Showing BRTI only when we actually have it keeps the line short and stops
    # the proxy from ever being mistaken for the index.
    brti_txt = f"{brti_txt} " if brti is not None else ""

    extra = counters.deltas + counters.trades + counters.brti
    console.print(
        f"[dim]{now}[/] [{left_style}]{left}[/] {event_ticker[-9:]} "
        f"k[bold]{len(live)}[/] {atm_txt} {brti_txt}{_spot_text(spot)} "
        f"[bold]{_human(counters.ladder_rows)}[/]"
        + (f"[dim]+{_human(extra)}[/]" if extra else "")
        + (f"[dim]/{_human(counters.spot_rows)}s[/]" if counters.spot_rows else "")
        + f" {ws_txt}"
        + (f" [red]err{counters.api_errors}[/]" if counters.api_errors else ""),
        highlight=False,
    )


async def run_capture(
    settings: Settings,
    duration_s: float | None = None,
    *,
    ladder_interval_s: float = 2.0,
    strike_window: int | None = 25,
    export_interval_s: float = 300.0,
    spot_venues: tuple[str, ...] | None = DEFAULT_VENUES,
    store: Store | None = None,
    console: Console | None = None,
) -> dict[str, int]:
    """Record KXBTCD market data until `duration_s` elapses or Ctrl-C.

    Returns the session counters. `duration_s=None` means run forever, which is the
    intended production mode. `spot_venues=None` disables the public price feed, which
    you should only do if you already have a real BRTI feed.
    """
    console = console or Console()
    counters = CaptureCounters()
    owns_store = store is None
    store = store or Store(settings)
    await store.start()

    feed: KalshiWebSocket | None = None
    spot: SpotFeed | None = None
    background: list[asyncio.Task] = []
    deadline = None if duration_s is None else time.monotonic() + duration_s
    parquet_dir = Path(settings.data_dir).expanduser() / "parquet"
    next_export = time.monotonic() + export_interval_s

    console.print(f"[bold]kbtc capture[/] | {settings.describe()}")
    console.print(
        f"[dim]db {store.path} | cadence {ladder_interval_s}s | "
        f"strike window +/-{strike_window if strike_window else 'all'} | "
        f"parquet export every {export_interval_s:.0f}s[/]"
    )

    # Started before the Kalshi client so the first heartbeat already carries a price:
    # the public venues need no handshake of ours and print within a second or two.
    if spot_venues:
        spot = SpotFeed(spot_venues)
        spot.start()
        console.print(f"[dim]spot proxy: {', '.join(spot.venue_names)} (public, no key)[/]")

    try:
        async with KalshiClient(settings) as client:
            current, _nxt = await discover_events(client)
            if current is None:
                console.print(f"[red]no open {SERIES_TICKER} event found - nothing to record[/]")
                return counters.as_dict()

            event_ticker = current["event_ticker"]
            probe_ticker = (current.get("markets") or [{}])[0].get("ticker", "")
            console.print(f"[bold green]recording[/] {event_ticker}")

            await backfill_latest_settlement(client, store, counters)

            feed = KalshiWebSocket(
                settings,
                tickers=[],
                channels=(CH_ORDERBOOK, CH_TRADE, CH_TICKER, CH_BRTI),
            )
            feed.start()
            if not feed.available:
                console.print(
                    "[yellow]Kalshi WebSocket disabled (no credentials): recording the REST "
                    "ladder only. Book deltas, trades and the real BRTI need an API key — "
                    "the public spot proxy above does not.[/]"
                )
            if spot is not None:
                await spot.wait_ready(timeout=6.0, venues=2)
                console.print(f"[dim]spot ready: {spot.describe()}[/]")

            subscribed: list[str] = []
            latest_brti: BrtiTick | None = None
            next_tick = time.monotonic()

            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break

                try:
                    rows, live, atm, mtc = await _record_ladder(
                        client, store, event_ticker, strike_window
                    )
                    counters.ladder_rows += rows
                    counters.cycles += 1

                    # Rollover: the event we were recording has closed.
                    if mtc <= 0:
                        nxt_current, _ = await discover_events(client)
                        if nxt_current and nxt_current["event_ticker"] != event_ticker:
                            old_event, old_probe = event_ticker, probe_ticker
                            event_ticker = nxt_current["event_ticker"]
                            probe_ticker = (nxt_current.get("markets") or [{}])[0].get("ticker", "")
                            console.print(
                                f"[bold magenta]rollover[/] {old_event} -> {event_ticker}"
                            )
                            if old_probe:
                                background.append(
                                    asyncio.create_task(
                                        backfill_settlement(
                                            client, store, old_event, old_probe, counters
                                        )
                                    )
                                )
                            subscribed = []  # force a resubscribe on the new ladder

                    # Keep the socket pointed at the strikes that actually quote. The
                    # pinned strikes never move, so subscribing to all 188 would be pure
                    # bandwidth for zero information.
                    if feed.available and live:
                        want = sorted(m.ticker for m in live)
                        if want != subscribed:
                            await feed.set_tickers(want)
                            subscribed = want

                    brti = _drain_feed(feed, store, counters)
                    if brti is not None:
                        latest_brti = brti
                    _drain_spot(spot, store, counters)

                    _heartbeat(
                        console, event_ticker, mtc, live, atm, latest_brti, counters, feed, spot
                    )

                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - a bad cycle is not a bad session
                    counters.api_errors += 1
                    log.warning("capture cycle failed (continuing): %s: %s", type(e).__name__, e)

                # Periodically publish Parquet so other processes (`kbtc report`,
                # `kbtc calibrate`) can read captured data while we hold the DB lock.
                if time.monotonic() >= next_export:
                    next_export = time.monotonic() + export_interval_s
                    await store.flush()
                    with contextlib.suppress(Exception):
                        store.export_parquet(parquet_dir)

                # Fixed-cadence scheduling: sleep to the next slot rather than for a
                # fixed interval, so a slow cycle does not permanently shift the phase.
                next_tick += ladder_interval_s
                sleep_for = next_tick - time.monotonic()
                if sleep_for < 0:
                    next_tick = time.monotonic()
                    sleep_for = 0
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None:
                    sleep_for = min(sleep_for, max(0.0, remaining))
                await asyncio.sleep(sleep_for)

    except (asyncio.CancelledError, KeyboardInterrupt):
        console.print("[yellow]interrupted - flushing buffers[/]")
    finally:
        for t in background:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        if feed is not None:
            await feed.stop()
            _drain_feed(feed, store, counters)
        if spot is not None:
            await spot.stop()
            _drain_spot(spot, store, counters)  # the tail is real data; do not lose it
        pending = store.pending
        await store.flush()
        totals = store.counts()
        # Export before releasing the DB so `kbtc report` has something to read even
        # while a long-running capture holds the writer lock.
        with contextlib.suppress(Exception):
            store.export_parquet(Path(settings.data_dir).expanduser() / "parquet")
        if owns_store:
            await store.close()
        console.print(
            f"[bold]capture done[/] cycles={counters.cycles} spot={counters.spot_rows} "
            f"tail_flushed={pending} rows_written={store.rows_written}"
        )
        if spot is not None:
            for name, st in spot.stats.items():
                console.print(
                    f"[dim]  spot {name:<9} msgs={st.messages:<7} quotes={st.quotes:<7} "
                    f"connects={st.connects} errors={st.errors}[/]"
                )
        console.print(f"[dim]table totals: {totals}[/]")

    return counters.as_dict()
