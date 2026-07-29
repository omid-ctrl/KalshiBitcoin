"""DuckDB store for captured market data.

WHY DUCKDB
----------
Phase 0 writes a few hundred rows a second and Phase 2 reads them back as whole-column
scans for calibration. That is an OLAP workload with an OLTP-ish ingest rate, on one
machine, with zero ops budget. DuckDB is an embedded columnar engine that does exactly
that and hands us a pandas DataFrame for free at the other end.

WHY BUFFERED WRITES
-------------------
A per-row INSERT costs a transaction; at 2s cadence over a 188-strike ladder that is
~94 transactions/second of pure overhead, and the write would sit on the same event loop
as the socket. So every writer appends to an in-memory list and a background task flushes
them in batches with `executemany`. The flush is deliberately synchronous inside the loop
rather than punted to a thread: a few thousand rows lands in single-digit milliseconds,
and a single-threaded DuckDB connection is far easier to reason about than a shared one.

TIME AND MONEY CONVENTIONS
--------------------------
- Timestamps are stored as naive TIMESTAMP that are ALWAYS UTC. DuckDB's TIMESTAMPTZ ->
  pandas conversion drags in a pytz dependency we do not otherwise need, and every
  timestamp in this system is UTC anyway. `_utc()` strips tzinfo on the way in.
- Money and sizes are DECIMAL(18,6). Prices are exact cents, but sizes are genuinely
  fractional on this venue ("1286.06") and volumes run into the millions, so 18/6 covers
  both without the float rounding that would quietly corrupt a fee calculation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from kalshi_btc.config import Settings
from kalshi_btc.core.types import dec

log = logging.getLogger(__name__)

# DECIMAL(18,6): six fractional digits is enough for fractional contract sizes and
# leaves twelve integer digits, comfortably more than a BRTI print will ever need.
_SCALE = Decimal("0.000001")

SCHEMA: dict[str, str] = {
    "ladder_snapshots": """
        CREATE TABLE IF NOT EXISTS ladder_snapshots (
            ts                TIMESTAMP      NOT NULL,
            event_ticker      VARCHAR        NOT NULL,
            ticker            VARCHAR        NOT NULL,
            strike            DECIMAL(18,6),
            yes_bid           DECIMAL(18,6),
            yes_ask           DECIMAL(18,6),
            yes_bid_size      DECIMAL(18,6),
            yes_ask_size      DECIMAL(18,6),
            volume            DECIMAL(18,6),
            open_interest     DECIMAL(18,6),
            minutes_to_close  DOUBLE
        )
    """,
    "book_deltas": """
        CREATE TABLE IF NOT EXISTS book_deltas (
            ts          TIMESTAMP     NOT NULL,
            ticker      VARCHAR       NOT NULL,
            seq         BIGINT,
            side        VARCHAR,
            price       DECIMAL(18,6),
            delta_size  DECIMAL(18,6)
        )
    """,
    "trades": """
        CREATE TABLE IF NOT EXISTS trades (
            ts          TIMESTAMP     NOT NULL,
            ticker      VARCHAR       NOT NULL,
            price       DECIMAL(18,6),
            size        DECIMAL(18,6),
            taker_side  VARCHAR
        )
    """,
    "brti": """
        CREATE TABLE IF NOT EXISTS brti (
            ts            TIMESTAMP    NOT NULL,
            index_id      VARCHAR,
            value         DECIMAL(18,6),
            avg_60s       DECIMAL(18,6),
            windowed_avg  DECIMAL(18,6),
            tick_count    INTEGER
        )
    """,
    # Public exchange top-of-book plus the composite BRTI proxy at that instant.
    #
    # One row PER VENUE UPDATE, not one row per composite: keeping the venue's own
    # bid/ask alongside the aggregate is what makes it possible to re-derive the proxy
    # later under a different aggregation rule, or to prove after the fact that a bad
    # print came from one venue rather than from our arithmetic. `proxy` is nullable
    # because the feed refuses to publish a composite when too few venues are fresh.
    "spot": """
        CREATE TABLE IF NOT EXISTS spot (
            ts     TIMESTAMP     NOT NULL,
            venue  VARCHAR       NOT NULL,
            bid    DECIMAL(18,6),
            ask    DECIMAL(18,6),
            mid    DECIMAL(18,6),
            proxy  DECIMAL(18,6)
        )
    """,
    # event_ticker is the PK because settlement is a property of the EVENT, not of the
    # 188 markets under it - they all carry the identical expiration_value.
    "settlements": """
        CREATE TABLE IF NOT EXISTS settlements (
            close_time        TIMESTAMP     NOT NULL,
            event_ticker      VARCHAR       NOT NULL PRIMARY KEY,
            expiration_value  DECIMAL(18,6)
        )
    """,
    "fills": """
        CREATE TABLE IF NOT EXISTS fills (
            ts         TIMESTAMP     NOT NULL,
            ticker     VARCHAR       NOT NULL,
            side       VARCHAR,
            action     VARCHAR,
            price      DECIMAL(18,6),
            count      DECIMAL(18,6),
            liquidity  VARCHAR,
            fee        DECIMAL(18,6),
            order_id   VARCHAR
        )
    """,
    "decisions": """
        CREATE TABLE IF NOT EXISTS decisions (
            ts          TIMESTAMP     NOT NULL,
            ticker      VARCHAR       NOT NULL,
            fair_prob   DOUBLE,
            market_mid  DOUBLE,
            edge        DOUBLE,
            action      VARCHAR,
            reason      VARCHAR,
            armed       BOOLEAN
        )
    """,
}

# Which column each table is partitioned by on Parquet export.
_DATE_COLUMN: dict[str, str] = {t: "ts" for t in SCHEMA}
_DATE_COLUMN["settlements"] = "close_time"

_INSERTS: dict[str, str] = {
    "ladder_snapshots": "INSERT INTO ladder_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    "book_deltas": "INSERT INTO book_deltas VALUES (?,?,?,?,?,?)",
    "trades": "INSERT INTO trades VALUES (?,?,?,?,?)",
    "brti": "INSERT INTO brti VALUES (?,?,?,?,?,?)",
    "spot": "INSERT INTO spot VALUES (?,?,?,?,?,?)",
    "fills": "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?)",
    "decisions": "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?)",
    # Re-running a backfill must be a no-op, not a duplicate row.
    "settlements": (
        "INSERT INTO settlements VALUES (?,?,?) "
        "ON CONFLICT (event_ticker) DO UPDATE SET "
        "close_time = excluded.close_time, expiration_value = excluded.expiration_value"
    ),
}


def _utc(ts: datetime | None) -> datetime:
    """Naive-UTC for storage. Naive input is assumed to already be UTC."""
    if ts is None:
        return datetime.now(UTC).replace(tzinfo=None)
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(UTC).replace(tzinfo=None)


def _q(value: Decimal | float | int | str | None) -> Decimal | None:
    """Quantize to the DECIMAL(18,6) column scale.

    Done explicitly rather than relying on DuckDB's implicit rounding so that the
    truncation point is visible in this file instead of buried in the engine.
    """
    if value is None or value == "":
        return None
    return dec(value).quantize(_SCALE)


class Store:
    """Buffered DuckDB writer plus the read helpers calibration and reporting need.

    Writers (`add_*`) are plain synchronous appends to a list - safe to call from
    anywhere on the event loop and cheap enough to sit in a hot path. Only `flush()` and
    the lifecycle methods are async, and they serialise on a lock so a manual flush
    cannot interleave with the periodic one.
    """

    def __init__(
        self,
        settings: Settings,
        path: Path | None = None,
        *,
        flush_interval_s: float = 2.0,
        max_buffer: int = 5_000,
    ) -> None:
        self.settings = settings
        self.path = path or (settings.data_dir / "kbtc.duckdb")
        self.flush_interval_s = flush_interval_s
        self.max_buffer = max_buffer

        self._buffers: dict[str, list[tuple]] = {t: [] for t in SCHEMA}
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._flusher: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self.rows_written: int = 0

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> duckdb.DuckDBPyConnection:
        """Open the database and create any missing tables. Safe to call repeatedly."""
        if self._conn is not None:
            return self._conn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        for ddl in SCHEMA.values():
            self._conn.execute(ddl)
        log.info("store open at %s", self.path)
        return self._conn

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self.open()

    async def start(self) -> Store:
        self.open()
        if self._flusher is None:
            self._flusher = asyncio.create_task(self._flush_loop(), name="store-flush")
        return self

    async def close(self) -> None:
        """Flush everything buffered, then close. Never loses a buffered row."""
        self._stopping.set()
        self._wake.set()
        if self._flusher is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(self._flusher, timeout=10)
            self._flusher = None
        await self.flush()
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Store:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _flush_loop(self) -> None:
        while not self._stopping.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self.flush_interval_s)
            self._wake.clear()
            try:
                await self.flush()
            except Exception as e:  # noqa: BLE001 - a failed flush must not kill capture
                log.error("flush failed: %s: %s", type(e).__name__, e)

    def _buffer(self, table: str, row: tuple) -> None:
        buf = self._buffers[table]
        buf.append(row)
        if len(buf) >= self.max_buffer:
            self._wake.set()

    @property
    def pending(self) -> int:
        return sum(len(b) for b in self._buffers.values())

    async def flush(self) -> int:
        """Write every buffered row. Returns the number of rows written."""
        async with self._lock:
            # Swap the buffers out first so writers can keep appending during the write.
            batches = {t: rows for t, rows in self._buffers.items() if rows}
            if not batches:
                return 0
            for table in batches:
                self._buffers[table] = []

            conn = self.conn
            written = 0
            for table, rows in batches.items():
                try:
                    conn.executemany(_INSERTS[table], rows)
                    written += len(rows)
                except Exception as e:  # noqa: BLE001
                    # Dropping the batch beats spinning forever on a poison row, but it
                    # must be loud - this is irreplaceable data.
                    log.error("insert into %s failed (%d rows dropped): %s", table, len(rows), e)
            self.rows_written += written
            return written

    # ------------------------------------------------------------------ writers
    def add_ladder_snapshot(
        self,
        *,
        ts: datetime,
        event_ticker: str,
        ticker: str,
        strike: Decimal | None,
        yes_bid: Decimal | None,
        yes_ask: Decimal | None,
        yes_bid_size: Decimal | None,
        yes_ask_size: Decimal | None,
        volume: Decimal | None,
        open_interest: Decimal | None,
        minutes_to_close: float,
    ) -> None:
        self._buffer(
            "ladder_snapshots",
            (
                _utc(ts), event_ticker, ticker, _q(strike), _q(yes_bid), _q(yes_ask),
                _q(yes_bid_size), _q(yes_ask_size), _q(volume), _q(open_interest),
                float(minutes_to_close),
            ),
        )

    def add_book_delta(
        self,
        *,
        ts: datetime,
        ticker: str,
        seq: int | None,
        side: str,
        price: Decimal,
        delta_size: Decimal,
    ) -> None:
        self._buffer(
            "book_deltas",
            (_utc(ts), ticker, seq, str(side), _q(price), _q(delta_size)),
        )

    def add_trade(
        self, *, ts: datetime, ticker: str, price: Decimal, size: Decimal, taker_side: str
    ) -> None:
        self._buffer("trades", (_utc(ts), ticker, _q(price), _q(size), str(taker_side)))

    def add_brti(
        self,
        *,
        ts: datetime,
        index_id: str,
        value: Decimal | None,
        avg_60s: Decimal | None,
        windowed_avg: Decimal | None,
        tick_count: int | None,
    ) -> None:
        self._buffer(
            "brti",
            (_utc(ts), index_id, _q(value), _q(avg_60s), _q(windowed_avg), tick_count),
        )

    def add_spot(
        self,
        *,
        ts: datetime,
        venue: str,
        bid: Decimal | None,
        ask: Decimal | None,
        mid: Decimal | None,
        proxy: Decimal | None,
    ) -> None:
        """One public-exchange top-of-book update plus the composite proxy at that time.

        This is the only price series `kbtc capture` can record without credentials, so
        it is written unconditionally - see kalshi_btc.feed.spot_ws for why it is a PROXY
        for BRTI rather than BRTI itself.
        """
        self._buffer("spot", (_utc(ts), str(venue), _q(bid), _q(ask), _q(mid), _q(proxy)))

    def add_fill(
        self,
        *,
        ts: datetime,
        ticker: str,
        side: str,
        action: str,
        price: Decimal,
        count: Decimal,
        liquidity: str,
        fee: Decimal,
        order_id: str = "",
    ) -> None:
        self._buffer(
            "fills",
            (
                _utc(ts), ticker, str(side), str(action), _q(price), _q(count),
                str(liquidity), _q(fee), order_id,
            ),
        )

    def add_decision(
        self,
        *,
        ts: datetime,
        ticker: str,
        fair_prob: float,
        market_mid: float | None,
        edge: float | None,
        action: str,
        reason: str,
        armed: bool,
    ) -> None:
        self._buffer(
            "decisions",
            (
                _utc(ts), ticker, float(fair_prob),
                None if market_mid is None else float(market_mid),
                None if edge is None else float(edge),
                action, reason, bool(armed),
            ),
        )

    def upsert_settlement(
        self, *, close_time: datetime, event_ticker: str, expiration_value: Decimal | None
    ) -> None:
        """Idempotent by event_ticker - re-running a backfill never duplicates."""
        self._buffer("settlements", (_utc(close_time), event_ticker, _q(expiration_value)))

    # ------------------------------------------------------------------ readers
    def counts(self) -> dict[str, int]:
        """Row count per table. The verification workhorse."""
        return {t: self.conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in SCHEMA}

    def settlement_series(self) -> pd.DataFrame:
        """Realised BRTI 60s averages, oldest first, with the hour-over-hour return.

        `log_return` is what the volatility estimate is fitted to, so it is computed here
        once rather than in every consumer.
        """
        df = self.conn.execute(
            "SELECT close_time, event_ticker, expiration_value "
            "FROM settlements WHERE expiration_value IS NOT NULL ORDER BY close_time"
        ).df()
        if not df.empty:
            import numpy as np

            df["log_return"] = np.log(df["expiration_value"]).diff()
        return df

    def ladder_history(self, event_ticker: str) -> pd.DataFrame:
        return self.conn.execute(
            "SELECT * FROM ladder_snapshots WHERE event_ticker = ? ORDER BY ts, strike",
            [event_ticker],
        ).df()

    def decisions(self, since: datetime | None = None) -> pd.DataFrame:
        if since is None:
            return self.conn.execute("SELECT * FROM decisions ORDER BY ts").df()
        return self.conn.execute(
            "SELECT * FROM decisions WHERE ts >= ? ORDER BY ts", [_utc(since)]
        ).df()

    def fills(self, since: datetime | None = None) -> pd.DataFrame:
        if since is None:
            return self.conn.execute("SELECT * FROM fills ORDER BY ts").df()
        return self.conn.execute(
            "SELECT * FROM fills WHERE ts >= ? ORDER BY ts", [_utc(since)]
        ).df()

    def spot_history(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> pd.DataFrame:
        """Public spot ticks in [start, end], oldest first. Both bounds are optional.

        Bounds are inclusive because the settlement window is defined inclusively at both
        ends (close-59s through close), and an off-by-one second there is an off-by-one
        tick in a sixty-tick average.
        """
        where: list[str] = []
        params: list[datetime] = []
        if start is not None:
            where.append("ts >= ?")
            params.append(_utc(start))
        if end is not None:
            where.append("ts <= ?")
            params.append(_utc(end))
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        return self.conn.execute(f"SELECT * FROM spot{clause} ORDER BY ts", params).df()

    def brti_window(self, event_ticker: str, lookback_minutes: float = 60.0) -> pd.DataFrame:
        """BRTI ticks over the hour leading into that event's close.

        The close time is taken from `settlements` when the event has settled; otherwise
        it is reconstructed from the ladder (`ts + minutes_to_close`), which is why that
        column is recorded on every snapshot.
        """
        close = self._close_time_for(event_ticker)
        if close is None:
            return pd.DataFrame(columns=["ts", "index_id", "value", "avg_60s", "windowed_avg", "tick_count"])
        return self.conn.execute(
            "SELECT * FROM brti WHERE ts > ? AND ts <= ? ORDER BY ts",
            [close - pd.Timedelta(minutes=lookback_minutes), close],
        ).df()

    def _close_time_for(self, event_ticker: str) -> datetime | None:
        row = self.conn.execute(
            "SELECT close_time FROM settlements WHERE event_ticker = ?", [event_ticker]
        ).fetchone()
        if row and row[0] is not None:
            return row[0]
        row = self.conn.execute(
            "SELECT ts + INTERVAL (CAST(minutes_to_close * 60 AS BIGINT)) SECOND "
            "FROM ladder_snapshots WHERE event_ticker = ? ORDER BY ts DESC LIMIT 1",
            [event_ticker],
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------ export
    def export_parquet(self, outdir: Path | str) -> dict[str, int]:
        """Write each non-empty table to `outdir/<table>/` partitioned by date.

        Parquet is the handoff format for anything outside this process (notebooks, a
        different machine, cold archive) and Hive-style date partitions mean a backtest
        over one week never touches the other 51.
        """
        out = Path(outdir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, int] = {}
        for table in SCHEMA:
            n = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            if not n:
                continue
            date_col = _DATE_COLUMN[table]
            target = out / table
            self.conn.execute(
                f"COPY (SELECT *, CAST({date_col} AS DATE) AS dt FROM {table}) "
                f"TO '{target.as_posix()}' "
                "(FORMAT PARQUET, PARTITION_BY (dt), OVERWRITE_OR_IGNORE 1)"
            )
            written[table] = n
            log.info("exported %d rows of %s -> %s", n, table, target)
        return written


def open_store(settings: Settings, **kwargs: Any) -> Store:
    """Convenience constructor mirroring get_settings()."""
    return Store(settings, **kwargs)


class ReaderUnavailable(RuntimeError):
    """No readable source of captured data exists yet."""


def open_reader(settings: Settings) -> tuple[duckdb.DuckDBPyConnection, str]:
    """Open a READ connection that works even while `kbtc capture` is running.

    DuckDB is single-writer and takes an exclusive file lock, so a second process
    cannot open the same .duckdb file - not even read-only. That matters here because
    the whole point of this bot is that capture runs 24/7 while you separately run
    `kbtc report` and `kbtc calibrate`. Naively connecting would fail exactly when the
    system is working as intended.

    So: try the database first (fast, always current), and if it is locked by the
    capture process, fall back to an in-memory connection with views over the Parquet
    exports that capture writes periodically. The Parquet path is slightly stale -
    bounded by the export interval - which the caller is told via the returned source
    label so it can be surfaced in reports rather than silently misleading anyone.

    Returns (connection, source) where source is "duckdb" or "parquet".
    """
    path = Path(settings.data_dir).expanduser() / "kbtc.duckdb"
    if path.exists():
        try:
            con = duckdb.connect(str(path), read_only=True)
            return con, "duckdb"
        except (duckdb.IOException, duckdb.ConnectionException):
            # IOException: another PROCESS holds the file lock (the usual case - capture
            # is running). ConnectionException: this same process already holds a
            # read-write handle, so a second handle with different config is refused.
            # Both mean "the database is busy", and both fall back to Parquet.
            log.info("database is busy; reading Parquet exports instead")

    parquet_dir = Path(settings.data_dir).expanduser() / "parquet"
    if not parquet_dir.exists():
        raise ReaderUnavailable(
            f"No captured data available. The database at {path} is locked by a running "
            f"capture and no Parquet export exists yet at {parquet_dir}. Either wait for "
            f"capture's next export, or stop capture and re-run."
        )

    con = duckdb.connect(":memory:")
    found = False
    for table in SCHEMA:
        target = parquet_dir / table
        if not target.exists():
            # Create an empty view with the right shape so downstream queries still work.
            con.execute(SCHEMA[table])
            continue
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{(target / '**' / '*.parquet').as_posix()}', "
            "hive_partitioning=1)"
        )
        found = True
    if not found:
        raise ReaderUnavailable(f"Parquet export at {parquet_dir} contains no tables yet.")
    return con, "parquet"
