"""`kbtc calibrate` must work on a no-credentials install.

Real-time BRTI is licensed and Kalshi only serves it on the authenticated
`cfbenchmarks_value` channel, so an install with no API key records ZERO rows in `brti`.
The calibration query used to hard-join to that table, which meant calibration was
unreachable for exactly the operators the credential-free capture path exists to serve
(verified live: `kbtc calibrate` on a real capture reported "No usable ladder snapshots"
while the database held 50k ladder rows and 3k spot rows).

`kbtc capture` does record a free public composite in `spot`, so calibration falls back to
it and REPORTS which source it used. These tests pin both halves: the fallback fires, and
it never silently pretends to be BRTI.

NO NETWORK.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from kalshi_btc.runner.calibrate import _load_records
from kalshi_btc.store.db import SCHEMA

CLOSE = datetime(2026, 7, 28, 12, 0, 0)
SPOT_PX = 63_800.0
STRIKE = 63_799.99


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    for ddl in SCHEMA.values():
        c.execute(ddl)
    return c


def _add_ladder(c, *, n: int = 40) -> None:
    """n ladder snapshots at 2s cadence, walking in from 20 minutes before the close."""
    for i in range(n):
        ts = CLOSE - timedelta(minutes=20) + timedelta(seconds=2 * i)
        mtc = (CLOSE - ts).total_seconds() / 60.0
        c.execute(
            "INSERT INTO ladder_snapshots (ts, event_ticker, ticker, strike, yes_bid, "
            "yes_ask, yes_bid_size, yes_ask_size, volume, open_interest, minutes_to_close) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [ts, "KXBTCD-26JUL2812", f"KXBTCD-26JUL2812-T{STRIKE}", STRIKE,
             0.49, 0.50, 100.0, 100.0, 1000.0, 500.0, mtc],
        )


def _add_spot(c, *, n: int = 60, offset: float = 0.0) -> None:
    for i in range(n):
        ts = CLOSE - timedelta(minutes=21) + timedelta(seconds=2 * i)
        px = SPOT_PX + offset
        c.execute("INSERT INTO spot VALUES (?,?,?,?,?,?)",
                  [ts, "coinbase", px - 1, px + 1, px, px])


def _add_brti(c, *, n: int = 60) -> None:
    for i in range(n):
        ts = CLOSE - timedelta(minutes=21) + timedelta(seconds=2 * i)
        c.execute(
            "INSERT INTO brti (ts, index_id, value) VALUES (?,?,?)",
            [ts, "BRTI", SPOT_PX],
        )


# `days` must span from "now" back to the 2026-07-28 fixture window.
DAYS = max(2, (datetime.now(UTC) - CLOSE.replace(tzinfo=UTC)).days + 2)


def test_falls_back_to_the_spot_proxy_when_brti_is_empty(con):
    """The no-credentials case: brti has zero rows, spot has the composite."""
    _add_ladder(con)
    _add_spot(con)
    records, source = _load_records(con, DAYS)
    assert source == "spot-proxy"
    assert records, "the fallback must actually produce observations"
    assert all(r.spot == pytest.approx(SPOT_PX) for r in records)


def test_prefers_the_real_brti_tape_when_it_is_available(con):
    """With credentials the licensed tape wins; the proxy is a substitute, not a default."""
    _add_ladder(con)
    _add_spot(con, offset=500.0)  # deliberately wrong, so a mix-up is visible
    _add_brti(con)
    records, source = _load_records(con, DAYS)
    assert source == "brti"
    assert records
    assert all(r.spot == pytest.approx(SPOT_PX) for r in records), "must not read `spot`"


def test_no_spot_of_any_kind_yields_no_records(con):
    """A ladder alone cannot be scored - we refuse rather than invent a spot."""
    _add_ladder(con)
    records, source = _load_records(con, DAYS)
    assert records == []
    assert source == "spot-proxy"


def test_stale_proxy_rows_are_dropped_not_carried_forward(con):
    """ASOF joins carry the last value forward forever; a model priced off a stale spot
    is not the model we mean to score, so the staleness bound must discard it."""
    _add_ladder(con)
    # One print a full day before the ladder window.
    con.execute("INSERT INTO spot VALUES (?,?,?,?,?,?)",
                [CLOSE - timedelta(days=1), "coinbase",
                 SPOT_PX - 1, SPOT_PX + 1, SPOT_PX, SPOT_PX])
    records, _ = _load_records(con, DAYS)
    assert records == [], "a day-old proxy print is not a spot for this snapshot"


def test_rows_without_a_composite_proxy_are_ignored(con):
    """The feed writes proxy=NULL when too few venues are fresh; that is not a spot."""
    _add_ladder(con)
    for i in range(60):
        ts = CLOSE - timedelta(minutes=21) + timedelta(seconds=2 * i)
        con.execute("INSERT INTO spot VALUES (?,?,?,?,?,?)",
                    [ts, "coinbase", SPOT_PX - 1, SPOT_PX + 1, SPOT_PX, None])
    records, _ = _load_records(con, DAYS)
    assert records == []
