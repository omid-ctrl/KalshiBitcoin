"""Backfill realised settlements — the project's free ground truth.

Every settled Kalshi market carries `expiration_value`: the actual CF Benchmarks BRTI
60-second average that the contract resolved against. Kalshi publishes it on a public,
unauthenticated endpoint.

That single field is worth a great deal:

- It is the *exact* quantity our pricer is trying to predict, so we can score the model
  against reality without paying CF Benchmarks for an index licence.
- It gives the correct volatility basis. Vol measured settlement-to-settlement is the
  right input for pricing these contracts; vol measured on Coinbase spot bars is not.
- It lets `kbtc calibrate` answer the only question that matters before risking money:
  does our probability beat the market maker's mid?

All 188 strikes of one event share a single `expiration_value` (verified against the
API), so we collapse them to one row per event. Upserts are keyed on event_ticker,
making re-runs free and idempotent.

WHY THIS WALKS THE PAGES ITSELF INSTEAD OF CALLING get_settled_markets()
------------------------------------------------------------------------
`client.get_settled_markets()` follows the cursor until the series is exhausted, with no
upper bound. On an hourly series that is not a large fetch, it is an unbounded one.

Measured against the live prod API: a page of 1000 markets contains ~5.3 events and
reaches back about five hours, because 188 of every 1000 rows are the same event. So
each page buys five hours of history, and exhausting even six months of KXBTCD would be
~800 requests of which 99.5% are duplicates of the one field we want. A previous run of
the unbounded helper earned a 429 that lasted over two minutes.

The walk here is therefore bounded three ways:

  1. `max_events` counts DISTINCT EVENTS, not markets. Events are the unit of ground
     truth, and it is the only budget an operator can reason about ("give me 20 days").
  2. It stops early after `stop_after_known` consecutive events that are already stored,
     BUT only once the store already holds `max_events` events. The feed is newest-first,
     so the known events are a prefix: hitting them means "caught up" only if the history
     behind them is already as deep as the operator asked for. Without that guard,
     `--events 300` run after an `--events 40` would stop on the first page and silently
     store nothing, because the 40 newest events are exactly the ones already on file.
     With it, a routine refresh exits on page one while a request to go deeper still
     extends the history backwards.
  3. It pauses between pages. The client's token bucket is sized for the authenticated
     read tier; the unauthenticated tier is much tighter, and this is the only command
     that issues back-to-back requests in a tight loop.

Paging markets rather than `/events` is deliberate: `/events?status=settled` is a smaller
payload but carries no `expiration_value`, so it would cost one extra request per event
to resolve — several times more requests than de-duplicating 188-market pages.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from kalshi_btc.config import SERIES_TICKER, Settings
from kalshi_btc.core.types import MarketSnapshot
from kalshi_btc.exec.client import KalshiClient
from kalshi_btc.store.db import Store

log = logging.getLogger(__name__)

# ~5.3 events per 1000-market page (measured), so 480 events is ~91 requests and covers
# roughly twenty days of hourly settlements. Enough to fit a volatility model on, and
# far short of the unbounded walk that earned a 429.
DEFAULT_MAX_EVENTS = 480
DEFAULT_STOP_AFTER_KNOWN = 3
PAGE_PAUSE_S = 0.25


@dataclass
class BackfillResult:
    events: int = 0
    markets_scanned: int = 0
    pages: int = 0
    skipped_no_value: int = 0

    def __int__(self) -> int:
        return self.events


def _known_event_tickers(store: Store) -> set[str]:
    """Events already carrying a settlement value, so a re-run can stop early."""
    rows = store.conn.execute(
        "SELECT event_ticker FROM settlements WHERE expiration_value IS NOT NULL"
    ).fetchall()
    return {r[0] for r in rows}


async def backfill_settlements(
    settings: Settings,
    limit: int = 1000,
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
    stop_after_known: int = DEFAULT_STOP_AFTER_KNOWN,
    store: Store | None = None,
) -> int:
    """Pull settled KXBTCD markets and store one settlement row per event.

    Returns the number of distinct events stored. Needs no credentials.
    """
    owns_store = store is None
    store = store or Store(settings)
    await store.start()

    result = BackfillResult()
    known = _known_event_tickers(store)
    seen: set[str] = set()
    consecutive_known = 0
    # Only treat a run of known events as "caught up" if the history already goes as deep
    # as the caller asked for. Otherwise the gap is behind them, not in front of them.
    may_stop_early = len(known) >= max_events

    try:
        async with KalshiClient(settings) as client:
            cursor: str | None = None
            while len(seen) < max_events:
                params: dict[str, Any] = {
                    "series_ticker": SERIES_TICKER,
                    # Valid filter values are open/closed/settled/unopened. A freshly
                    # closed event spends a couple of minutes as status="determined"
                    # before it turns up here as "finalized"; that is why the rollover
                    # backfill in capture.py retries rather than assuming one pass is
                    # enough. Measured: an event closing at 01:00Z appeared under this
                    # filter at ~01:04Z with expiration_value=63880.45.
                    "status": "settled",
                    "limit": limit,
                }
                if cursor:
                    params["cursor"] = cursor

                data = await client.request("GET", "/markets", params=params)
                result.pages += 1
                markets = data.get("markets") or []
                if not markets:
                    break
                result.markets_scanned += len(markets)

                exhausted_budget = False
                for raw in markets:
                    event = raw.get("event_ticker") or ""
                    # KXBTC is a different, far less liquid series. Its tickers are not a
                    # prefix match for KXBTCD-, but the check is explicit because fitting
                    # vol on the wrong series is a silent, expensive error.
                    if not event.startswith(f"{SERIES_TICKER}-") or event in seen:
                        continue
                    seen.add(event)

                    if event in known:
                        consecutive_known += 1
                        if may_stop_early and consecutive_known >= stop_after_known:
                            break
                        continue
                    consecutive_known = 0

                    try:
                        snap = MarketSnapshot.from_api(raw)
                    except Exception:  # noqa: BLE001 - a malformed row must not kill a backfill
                        continue
                    if snap.expiration_value is None:
                        # Settled but not yet valued. Capture's rollover backfill retries
                        # these; it is not an error.
                        result.skipped_no_value += 1
                        continue

                    store.upsert_settlement(
                        close_time=snap.close_time,
                        event_ticker=event,
                        expiration_value=snap.expiration_value,
                    )
                    result.events += 1
                    if len(seen) >= max_events:
                        exhausted_budget = True
                        break

                if may_stop_early and consecutive_known >= stop_after_known:
                    log.info(
                        "stopping early: %d consecutive events already stored and the "
                        "store already covers the requested %d event(s)",
                        consecutive_known, max_events,
                    )
                    break
                if exhausted_budget:
                    break
                cursor = data.get("cursor")
                if not cursor:
                    break
                await asyncio.sleep(PAGE_PAUSE_S)

        await store.flush()
        log.info(
            "backfilled %d settlements from %d settled markets over %d page(s)",
            result.events,
            result.markets_scanned,
            result.pages,
        )
    finally:
        if owns_store:
            await store.close()

    return result.events


# The CLI probes for several conventional entry-point names; alias the obvious ones.
run_settlements = backfill_settlements
run = backfill_settlements
