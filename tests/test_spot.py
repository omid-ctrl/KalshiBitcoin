"""Tests for the public spot feed, the `spot` table and the proxy score.

NO NETWORK. Every message here is a synthetic copy of a frame that was captured from the
live venue on 2026-07-29, so the parsers are tested against the real wire format without
making the suite depend on Coinbase being up.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import duckdb
import pytest

from kalshi_btc.config import Settings
from kalshi_btc.feed.spot_ws import (
    VENUE_BITSTAMP,
    VENUE_COINBASE,
    VENUE_KRAKEN,
    SpotFeed,
    SpotQuote,
    parse_bitstamp,
    parse_coinbase,
    parse_kraken,
)
from kalshi_btc.model.proxy_score import score_proxy
from kalshi_btc.store.db import SCHEMA, Store

# --------------------------------------------------------------------------- fixtures
# Verbatim shapes from the live feeds, trimmed of fields we never read.
COINBASE_TICKER = {
    "type": "ticker",
    "sequence": 133370566659,
    "product_id": "BTC-USD",
    "price": "63944.26",
    "best_bid": "63944.26",
    "best_bid_size": "0.10880515",
    "best_ask": "63944.27",
    "best_ask_size": "0.03781955",
    "side": "sell",
    "time": "2026-07-29T00:14:37.919203Z",
}

KRAKEN_TICKER = {
    "channel": "ticker",
    "type": "update",
    "data": [
        {
            "symbol": "BTC/USD",
            "bid": 63964.9,
            "bid_qty": 10.35208221,
            "ask": 63965.0,
            "ask_qty": 0.36131763,
            "last": 63964.9,
            "timestamp": "2026-07-29T00:16:50.100000Z",
        }
    ],
}

BITSTAMP_BOOK = {
    "event": "data",
    "channel": "order_book_btcusd",
    "data": {
        "timestamp": "1785284078",
        "microtimestamp": "1785284078207406",
        "bids": [["63947.07", "0.19547472"], ["63946.88", "0.27365881"]],
        "asks": [["63947.09", "0.57534297"], ["63947.50", "0.10000000"]],
    },
}


def _quote(venue: str, bid: str, ask: str) -> SpotQuote:
    b, a = Decimal(bid), Decimal(ask)
    return SpotQuote(
        venue=venue, bid=b, ask=a, mid=(b + a) / 2, ts=datetime.now(UTC), received=0.0
    )


# --------------------------------------------------------------------------- parsers
def test_parse_coinbase_uses_best_bid_ask_not_last_trade():
    q = parse_coinbase(COINBASE_TICKER)
    assert q is not None
    assert q.venue == VENUE_COINBASE
    assert q.bid == Decimal("63944.26")
    assert q.ask == Decimal("63944.27")
    assert q.mid == Decimal("63944.265000")
    assert q.ts == datetime(2026, 7, 29, 0, 14, 37, 919203, tzinfo=UTC)


def test_parse_coinbase_ignores_subscription_and_other_products():
    assert parse_coinbase({"type": "subscriptions", "channels": []}) is None
    assert parse_coinbase({**COINBASE_TICKER, "product_id": "ETH-USD"}) is None


def test_parse_kraken_handles_json_floats_exactly():
    q = parse_kraken(KRAKEN_TICKER)
    assert q is not None
    assert q.venue == VENUE_KRAKEN
    # Floats must round-trip through str(), not through binary float arithmetic.
    assert q.bid == Decimal("63964.9")
    assert q.ask == Decimal("63965.0")
    assert q.mid == Decimal("63964.950000")


def test_parse_kraken_ignores_status_and_heartbeat():
    assert parse_kraken({"channel": "heartbeat"}) is None
    assert parse_kraken({"channel": "status", "type": "update", "data": [{}]}) is None
    assert parse_kraken({"method": "subscribe", "success": True}) is None


def test_parse_kraken_snapshot_shape_is_the_same_as_update():
    snap = {**KRAKEN_TICKER, "type": "snapshot"}
    assert parse_kraken(snap) == parse_kraken(KRAKEN_TICKER)


def test_parse_bitstamp_takes_top_of_book_and_microsecond_time():
    q = parse_bitstamp(BITSTAMP_BOOK)
    assert q is not None
    assert q.venue == VENUE_BITSTAMP
    assert q.bid == Decimal("63947.07")
    assert q.ask == Decimal("63947.09")
    assert q.ts == datetime.fromtimestamp(1785284078.207406, UTC)


def test_parse_bitstamp_ignores_subscription_ack_and_empty_books():
    assert parse_bitstamp({"event": "bts:subscription_succeeded", "channel": "order_book_btcusd",
                           "data": {}}) is None
    assert parse_bitstamp({**BITSTAMP_BOOK, "data": {"bids": [], "asks": []}}) is None


@pytest.mark.parametrize(
    "bid,ask",
    [
        ("0", "63944.27"),        # zero bid
        ("-1", "63944.27"),       # negative
        ("63944.27", "63944.26"),  # crossed book
        ("63000", "64000"),        # ~157 bps: far beyond any real BTC/USD spread
    ],
)
def test_sanity_filter_rejects_impossible_quotes(bid, ask):
    assert parse_coinbase({**COINBASE_TICKER, "best_bid": bid, "best_ask": ask}) is None


def test_missing_fields_return_none_rather_than_raising():
    assert parse_coinbase({"type": "ticker", "product_id": "BTC-USD"}) is None
    assert parse_kraken({"channel": "ticker", "data": [{"symbol": "BTC/USD"}]}) is None


# --------------------------------------------------------------------------- aggregation
def _feed_with(*quotes: SpotQuote, **kwargs) -> SpotFeed:
    """A feed with hand-placed state and no sockets, for pure aggregation tests."""
    feed = SpotFeed((VENUE_COINBASE, VENUE_KRAKEN, VENUE_BITSTAMP), **kwargs)
    now_ish = __import__("time").monotonic()
    for q in quotes:
        feed.quotes[q.venue] = SpotQuote(
            venue=q.venue, bid=q.bid, ask=q.ask, mid=q.mid, ts=q.ts, received=now_ish
        )
        feed.stats[q.venue].last_message_at = now_ish
    return feed


def test_proxy_is_the_median_and_ignores_one_lying_venue():
    feed = _feed_with(
        _quote(VENUE_COINBASE, "63000", "63000.02"),
        _quote(VENUE_KRAKEN, "63000.10", "63000.12"),
        _quote(VENUE_BITSTAMP, "62000", "62000.02"),  # the liar
    )
    # Median of {63000.01, 63000.11, 62000.01} is the middle one, unmoved by the outlier.
    assert feed.proxy() == Decimal("63000.010000")
    assert feed.mid() == Decimal("63000.010000")


def test_proxy_of_two_venues_is_their_mean():
    feed = _feed_with(
        _quote(VENUE_COINBASE, "63000", "63000.02"),
        _quote(VENUE_KRAKEN, "63100", "63100.02"),
    )
    assert feed.proxy() == Decimal("63050.010000")


def test_proxy_refuses_a_one_venue_composite_but_mid_still_answers():
    feed = _feed_with(_quote(VENUE_COINBASE, "63000", "63000.02"))
    assert feed.proxy() is None, "a single venue is not a cross-exchange index"
    assert feed.mid() == Decimal("63000.010000")
    assert feed.mid(VENUE_COINBASE) == Decimal("63000.010000")
    assert feed.mid(VENUE_KRAKEN) is None


def test_stale_venues_are_excluded_from_the_composite():
    feed = _feed_with(
        _quote(VENUE_COINBASE, "63000", "63000.02"),
        _quote(VENUE_KRAKEN, "63100", "63100.02"),
        max_age_s=5.0,
    )
    # Push Kraken's last message an hour into the past.
    feed.stats[VENUE_KRAKEN].last_message_at -= 3600
    feed.quotes[VENUE_KRAKEN] = SpotQuote(
        venue=VENUE_KRAKEN,
        bid=Decimal("63100"),
        ask=Decimal("63100.02"),
        mid=Decimal("63100.01"),
        ts=datetime.now(UTC),
        received=feed.quotes[VENUE_KRAKEN].received - 3600,
    )
    assert [q.venue for q in feed.fresh_quotes()] == [VENUE_COINBASE]
    assert feed.proxy() is None  # only one fresh venue left
    assert feed.mid() == Decimal("63000.010000")


def test_venues_agree_when_close_and_not_when_far():
    close = _feed_with(
        _quote(VENUE_COINBASE, "63000", "63000.02"),
        _quote(VENUE_KRAKEN, "63001", "63001.02"),
    )
    # $1 apart on a $63k price is 0.159 bps - a completely normal cross-venue gap.
    assert close.spread_bps() == pytest.approx(0.1587, abs=0.001)
    assert close.venues_agree(tolerance_bps=5.0) is True

    # $500 apart is 79 bps. One of these two venues is broken and we cannot tell which.
    far = _feed_with(
        _quote(VENUE_COINBASE, "63000", "63000.02"),
        _quote(VENUE_KRAKEN, "63500", "63500.02"),
    )
    assert far.spread_bps() == pytest.approx(79.05, abs=0.1)
    assert far.venues_agree(tolerance_bps=5.0) is False


def test_venues_agree_is_false_when_it_cannot_be_checked():
    """Fewer than two fresh venues means 'unverifiable', which must read as 'do not trade'."""
    assert _feed_with().venues_agree() is False
    assert _feed_with(_quote(VENUE_COINBASE, "63000", "63000.02")).venues_agree() is False
    assert _feed_with().spread_bps() is None


def test_staleness_is_infinite_before_the_first_message():
    feed = SpotFeed()
    assert feed.staleness_s() == float("inf")
    assert feed.staleness_s(VENUE_COINBASE) == float("inf")
    fed = _feed_with(_quote(VENUE_COINBASE, "63000", "63000.02"))
    assert fed.staleness_s() < 1.0
    assert fed.staleness_s(VENUE_KRAKEN) == float("inf")


def test_unknown_venue_names_are_ignored_not_fatal():
    feed = SpotFeed(("coinbase", "not-a-venue"))
    assert feed.venue_names == (VENUE_COINBASE,)


def test_describe_never_raises_on_an_empty_feed():
    assert "0/3 venues" in SpotFeed().describe()


# --------------------------------------------------------------------------- store
@pytest.fixture
def store(tmp_path) -> Store:
    s = Settings(data_dir=tmp_path)
    st = Store(s, path=tmp_path / "t.duckdb")
    st.open()
    return st


def test_spot_table_exists_and_round_trips(store: Store):
    assert "spot" in SCHEMA
    ts = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
    store.add_spot(
        ts=ts,
        venue=VENUE_COINBASE,
        bid=Decimal("63944.26"),
        ask=Decimal("63944.27"),
        mid=Decimal("63944.265"),
        proxy=Decimal("63944.30"),
    )
    store.add_spot(ts=ts, venue=VENUE_KRAKEN, bid=None, ask=None, mid=None, proxy=None)
    import asyncio

    assert asyncio.run(store.flush()) == 2
    assert store.counts()["spot"] == 2

    df = store.spot_history()
    assert list(df.columns) == ["ts", "venue", "bid", "ask", "mid", "proxy"]
    assert float(df.iloc[0]["mid"]) == pytest.approx(63944.265)
    assert df.iloc[1]["bid"] is None or df.iloc[1]["bid"] != df.iloc[1]["bid"]  # NULL/NaN


def test_spot_history_bounds_are_inclusive(store: Store):
    import asyncio

    base = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
    for i in range(5):
        store.add_spot(
            ts=base + timedelta(seconds=i),
            venue=VENUE_COINBASE,
            bid=Decimal("1"),
            ask=Decimal("1.01"),
            mid=Decimal("1.005"),
            proxy=Decimal("1.005"),
        )
    asyncio.run(store.flush())
    win = store.spot_history(base + timedelta(seconds=1), base + timedelta(seconds=3))
    assert len(win) == 3, "both bounds inclusive: seconds 1, 2 and 3"


def test_spot_exports_to_parquet(store: Store, tmp_path):
    import asyncio

    store.add_spot(
        ts=datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC),
        venue=VENUE_BITSTAMP,
        bid=Decimal("63947.07"),
        ask=Decimal("63947.09"),
        mid=Decimal("63947.08"),
        proxy=Decimal("63947.10"),
    )
    asyncio.run(store.flush())
    written = store.export_parquet(tmp_path / "pq")
    assert written.get("spot") == 1
    assert list((tmp_path / "pq" / "spot").rglob("*.parquet"))


# --------------------------------------------------------------------------- proxy score
def _seeded_conn(offset_dollars: float, *, seconds: range = range(60)) -> duckdb.DuckDBPyConnection:
    """An in-memory store with one settled event and a proxy series offset from truth."""
    con = duckdb.connect(":memory:")
    for ddl in SCHEMA.values():
        con.execute(ddl)
    close = datetime(2026, 7, 28, 12, 0, 0)
    truth = 63000.0
    con.execute("INSERT INTO settlements VALUES (?, ?, ?)", [close, "KXBTCD-26JUL2812", truth])
    rows = []
    for i in seconds:
        # One print per settlement second, 200 ms before the grid tick, so the ASOF join
        # has something at-or-before every second it asks about.
        ts = close - timedelta(seconds=i) - timedelta(milliseconds=200)
        px = truth + offset_dollars
        rows.append((ts, "coinbase", px - 0.01, px + 0.01, px, px))
    con.executemany("INSERT INTO spot VALUES (?,?,?,?,?,?)", rows)
    return con


def test_score_proxy_recovers_a_known_offset():
    con = _seeded_conn(offset_dollars=3.0)
    score = score_proxy(con)
    assert score.n == 1
    assert score.median_abs_error == pytest.approx(3.0, abs=1e-6)
    assert score.mean_error == pytest.approx(3.0, abs=1e-6), "bias is signed"
    assert score.median_ticks == 60.0
    assert score.passed is True
    assert score.verdict == "PASS"
    assert score.median_error_frac_spacing == pytest.approx(0.03)


def test_score_proxy_fails_a_large_offset():
    score = score_proxy(_seeded_conn(offset_dollars=-40.0))
    assert score.n == 1
    assert score.median_abs_error == pytest.approx(40.0, abs=1e-6)
    assert score.mean_error == pytest.approx(-40.0, abs=1e-6)
    assert score.passed is False
    assert score.verdict == "FAIL"


def test_score_proxy_excludes_events_with_thin_coverage():
    # Only 10 of the 60 settlement seconds covered - below the default min_ticks of 30.
    score = score_proxy(_seeded_conn(offset_dollars=1.0, seconds=range(10)))
    assert score.n == 0
    assert score.events_thin == 1
    assert score.verdict == "NO DATA"
    assert score.passed is False, "unknown must never report as a pass"


def test_score_proxy_on_an_empty_database_is_no_data_not_pass():
    con = duckdb.connect(":memory:")
    for ddl in SCHEMA.values():
        con.execute(ddl)
    score = score_proxy(con)
    assert score.n == 0
    assert score.events_settled == 0
    assert score.verdict == "NO DATA"
    assert score.passed is False


def test_score_proxy_ignores_ticks_older_than_the_staleness_bound():
    """A grid second whose newest proxy print is minutes old must not be counted."""
    con = duckdb.connect(":memory:")
    for ddl in SCHEMA.values():
        con.execute(ddl)
    close = datetime(2026, 7, 28, 12, 0, 0)
    con.execute("INSERT INTO settlements VALUES (?, ?, ?)", [close, "E", 63000.0])
    # A single print ten minutes before close. Without the staleness bound the ASOF join
    # would happily carry it forward across all sixty seconds and score a perfect zero.
    con.execute(
        "INSERT INTO spot VALUES (?,?,?,?,?,?)",
        [close - timedelta(minutes=10), "coinbase", 62999.0, 63001.0, 63000.0, 63000.0],
    )
    score = score_proxy(con)
    assert score.n == 0, "stale carry-forward must not be mistaken for coverage"


def test_score_proxy_accepts_a_store_as_well_as_a_connection(store: Store):
    import asyncio

    close = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
    store.upsert_settlement(
        close_time=close, event_ticker="KXBTCD-26JUL2812", expiration_value=Decimal("63000")
    )
    for i in range(60):
        px = Decimal("63002")
        store.add_spot(
            ts=close - timedelta(seconds=i) - timedelta(milliseconds=200),
            venue=VENUE_COINBASE,
            bid=px - 1,
            ask=px + 1,
            mid=px,
            proxy=px,
        )
    asyncio.run(store.flush())
    score = score_proxy(store)
    assert score.n == 1
    assert score.median_abs_error == pytest.approx(2.0, abs=1e-6)


def test_score_proxy_rejects_a_nonsense_argument():
    with pytest.raises(TypeError):
        score_proxy("not a connection")
