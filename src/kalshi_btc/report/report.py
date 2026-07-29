"""Static HTML performance and calibration report.

WHY THIS EXISTS
---------------
A trading bot that you cannot audit is a random number generator with a bank
account attached. This module answers, in one file you can open offline, the only
three questions that matter:

    1. Did we make money, and where did it come from?
    2. Is the pricing model actually better than just reading the market mid?
    3. What did execution cost us (fees, spread, adverse fills)?

Question 2 is the important one and it is deliberately given the most space. Our
whole thesis is that an Asian-settled digital is mispriced by anyone treating it
as point-in-time. If the reliability diagram shows our model tracking the market
mid with no skill, there is no edge and the honest move is to stop.

DESIGN CONSTRAINTS
------------------
* Self-contained: no CDN, no external CSS/JS/fonts, no image files. Charts are
  hand-rolled inline SVG. You can email this file or open it on a plane.
* Never crashes. A fresh install with zero rows must still produce a readable page
  that says "no data yet" and shows whatever capture data does exist. Every
  section is gathered inside its own try/except and degrades to a note.
* Schema-tolerant. Sibling modules (store, runner, calibration) are still moving.
  Rather than hard-coding table names we probe the DuckDB catalogue and match
  tables by column signature. If the schema drifts, the report says so instead of
  raising.

Entry point: `build_report(settings)` -> Path to reports/out/latest.html.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Layout constants. Chart geometry lives here so every chart shares a visual language.
# --------------------------------------------------------------------------------------
CHART_W = 860
CHART_H = 300
PAD_L = 62
PAD_R = 18
PAD_T = 18
PAD_B = 42

# Reliability bins. Ten equal-width bins on [0,1] is the convention in the forecast
# verification literature (Murphy 1973) and is what makes the diagram comparable to
# published ones. Bins with fewer than MIN_BIN_N observations are drawn hollow because
# their observed frequency is mostly sampling noise.
N_BINS = 10
MIN_BIN_N = 20

# Price buckets for P&L attribution. Fees are quadratic in price (0.07*P*(1-P)), so
# attribution by price bucket is really attribution by fee burden.
PRICE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("1-10c", 0.00, 0.10),
    ("10-25c", 0.10, 0.25),
    ("25-40c", 0.25, 0.40),
    ("40-60c", 0.40, 0.60),
    ("60-75c", 0.60, 0.75),
    ("75-90c", 0.75, 0.90),
    ("90-99c", 0.90, 1.01),
)

# Fallback vol when we have to price captured snapshots ourselves. Measured on the
# realised KXBTCD settlement series: 0.466%/hour == 43.6% annualised, over 1,596 hourly
# settlements spanning 2026-05-22 to 2026-07-29 (~2 months, one broad regime).
DEFAULT_ANNUAL_VOL = 0.436

_STRIKE_RE = re.compile(r"-T([\d.]+)$")


# ======================================================================================
# Data model
# ======================================================================================
@dataclass
class Bin:
    """One reliability-diagram bucket."""

    lo: float
    hi: float
    n: int
    mean_pred: float
    obs_freq: float

    @property
    def reliable(self) -> bool:
        return self.n >= MIN_BIN_N


@dataclass
class Scores:
    """Proper scoring rule results for one forecaster."""

    label: str
    n: int
    brier: float
    log_loss: float
    bins: list[Bin] = field(default_factory=list)


@dataclass
class EventPnl:
    event: str
    close_time: datetime | None
    contracts: float
    gross: float
    fees: float
    net: float


@dataclass
class BucketPnl:
    label: str
    contracts: float
    gross: float
    fees: float
    net: float


@dataclass
class FillQuality:
    maker_contracts: float = 0.0
    taker_contracts: float = 0.0
    maker_fees: float = 0.0
    taker_fees: float = 0.0
    # Signed slippage in cents vs the mid at decision time. Positive = we paid up.
    slippage_samples: int = 0
    mean_slippage_cents: float | None = None
    median_slippage_cents: float | None = None

    @property
    def total_contracts(self) -> float:
        return self.maker_contracts + self.taker_contracts

    @property
    def maker_share(self) -> float | None:
        t = self.total_contracts
        return None if t <= 0 else self.maker_contracts / t


@dataclass
class CaptureStats:
    book_rows: int = 0
    brti_rows: int = 0
    # Public Coinbase/Kraken/Bitstamp composite. Reported separately from brti_rows
    # because on a no-credentials install brti_rows is ALWAYS 0 - real-time BRTI is
    # licensed - and without this tile a perfectly healthy capture looks idle.
    spot_rows: int = 0
    settled_events: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    distinct_events: int = 0


@dataclass
class ReportData:
    """Everything the renderer needs. Fully populated even when empty."""

    generated_at: datetime
    env: str
    armed: bool
    bankroll: float
    data_dir: str
    db_path: str | None = None

    period_start: datetime | None = None
    period_end: datetime | None = None

    # Headline
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    events_traded: int = 0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0

    equity: list[tuple[datetime | None, float]] = field(default_factory=list)
    per_event: list[EventPnl] = field(default_factory=list)
    per_bucket: list[BucketPnl] = field(default_factory=list)
    fill_quality: FillQuality = field(default_factory=FillQuality)

    model_scores: Scores | None = None
    market_scores: Scores | None = None
    calibration_source: str = ""
    # Straight from CalibrationResult.as_dict() when `kbtc calibrate` has run.
    cal_meta: dict[str, Any] = field(default_factory=dict)
    cal_by_minutes: list[dict[str, Any]] = field(default_factory=list)
    cal_by_price: list[dict[str, Any]] = field(default_factory=list)

    capture: CaptureStats = field(default_factory=CaptureStats)
    notes: list[str] = field(default_factory=list)

    @property
    def has_trades(self) -> bool:
        return self.events_traded > 0 or bool(self.per_event)

    @property
    def has_calibration(self) -> bool:
        return self.model_scores is not None or self.market_scores is not None

    @property
    def win_rate(self) -> float | None:
        total = self.wins + self.losses
        return None if total == 0 else self.wins / total

    @property
    def skill_score(self) -> float | None:
        """Brier skill score of our model against the market mid as the reference.

        1.0 = perfect, 0.0 = no better than the market, negative = worse than the
        market. This is THE number. Everything else on the page is supporting detail.
        """
        if "skill" in self.cal_meta:
            return _as_float(self.cal_meta["skill"], default=float("nan")) or 0.0
        if not (self.model_scores and self.market_scores):
            return None
        if self.market_scores.brier <= 0:
            return None
        return 1.0 - (self.model_scores.brier / self.market_scores.brier)

    @property
    def out_of_sample(self) -> bool | None:
        """Whether the scored window excludes the hours the vol model was fitted on."""
        if "is_out_of_sample" not in self.cal_meta:
            return None
        return bool(self.cal_meta["is_out_of_sample"])


# ======================================================================================
# Gathering
# ======================================================================================
def _find_db(data_dir: Path) -> Path | None:
    """Locate the capture database without depending on the store module's naming."""
    if not data_dir.exists():
        return None
    candidates = sorted(data_dir.glob("*.duckdb")) + sorted(data_dir.glob("*.db"))
    if not candidates:
        return None
    # Largest file wins: a half-initialised sidecar shouldn't shadow the real one.
    return max(candidates, key=lambda p: p.stat().st_size)


def open_read_only(path: Path) -> tuple[Any, str | None]:
    """Open a DuckDB file for reading, even while `kbtc capture` holds the write lock.

    DuckDB permits a single writer process per file, so a naive read-only connect fails
    exactly when you most want a report: mid-session, with the recorder running. Falling
    back to a snapshot copy costs a file copy and gives up nothing — the report is a
    point-in-time view anyway. Returns (connection, warning-or-None).
    """
    import duckdb

    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="kbtc-report-")) / path.name
        try:
            shutil.copy2(path, tmp)
            for suffix in (".wal", ".tmp"):
                side = path.with_name(path.name + suffix)
                if side.exists():
                    shutil.copy2(side, tmp.with_name(tmp.name + suffix))
            con = duckdb.connect(str(tmp), read_only=True)
        except Exception as copy_exc:
            raise RuntimeError(f"could not open {path}: {exc}; snapshot also failed: {copy_exc}")
        return con, (
            f"{path.name} is locked by another process (probably `kbtc capture`), so this "
            "report was built from a snapshot copy taken just now."
        )

    # Timestamps are stored naive-UTC; pin the session so any implicit cast agrees.
    try:
        con.execute("SET TimeZone='UTC'")
    except Exception:
        pass
    return con, None


def _catalogue(con: Any) -> dict[str, list[str]]:
    """table name -> lowercase column names, for every table and view we can see."""
    out: dict[str, list[str]] = {}
    try:
        rows = con.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema NOT IN ('information_schema','pg_catalog') "
            "ORDER BY table_name, ordinal_position"
        ).fetchall()
    except Exception:
        return out
    for table, column in rows:
        out.setdefault(str(table), []).append(str(column).lower())
    return out


def _match_table(
    catalogue: dict[str, list[str]],
    required: Sequence[Iterable[str]],
    name_hints: Sequence[str] = (),
) -> str | None:
    """Find the table that best satisfies a column signature.

    `required` is a list of alias-groups: each group must be satisfied by at least one
    of its aliases. This is what lets the report survive a sibling module renaming
    `count` to `contracts` without anyone updating this file.
    """
    best: tuple[int, str] | None = None
    for table, cols in catalogue.items():
        colset = set(cols)
        if not all(colset & set(group) for group in required):
            continue
        score = len(colset)
        lname = table.lower()
        for i, hint in enumerate(name_hints):
            if hint in lname:
                score += 1000 - i  # earlier hints win
        if best is None or score > best[0]:
            best = (score, table)
    return None if best is None else best[1]


def _ts_expr(col: str) -> str:
    """Read timestamps as strings.

    DuckDB's Python conversion of TIMESTAMP WITH TIME ZONE pulls in `pytz`, which is not
    a dependency of this project. Casting server-side keeps the report working on a
    stock install regardless of how the store declared its timestamp columns.
    """
    return f'CAST("{col}" AS VARCHAR)'


def _col(cols: Iterable[str], *aliases: str) -> str | None:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _strike_from_ticker(ticker: str) -> float | None:
    m = _STRIKE_RE.search(str(ticker))
    return float(m.group(1)) if m else None


def _event_from_ticker(ticker: str) -> str:
    """Market ticker -> event ticker, and event ticker -> itself.

    `KXBTCD-26JUL2819-T63999.99` -> `KXBTCD-26JUL2819`, but `KXBTCD-26JUL2819` must come
    back unchanged: the settlements table is keyed by event, so blindly stripping the last
    segment would collapse every event to the bare series name.
    """
    head, sep, tail = str(ticker).rpartition("-")
    if sep and _STRIKE_RE.search(f"-{tail}"):
        return head
    return str(ticker)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, Decimal):
            return float(x)
        f = float(x)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _as_dt(x: Any) -> datetime | None:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None


def gather(settings: Any) -> ReportData:
    """Build a ReportData from whatever exists on disk. Never raises."""
    data = ReportData(
        generated_at=datetime.now(UTC),
        env=getattr(settings, "env", "demo"),
        armed=bool(getattr(settings, "armed", False)),
        bankroll=_as_float(getattr(getattr(settings, "risk", None), "bankroll", 0)),
        data_dir=str(getattr(settings, "data_dir", "./data")),
    )

    # The calibration result is a standalone JSON file, so it is loaded before (and
    # independently of) the database: a report built on a machine that only has the
    # scoring output should still show the section that matters most.
    have_calibration = _load_calibration_json(data)

    db = _find_db(Path(data.data_dir).expanduser())
    if db is None:
        data.notes.append(
            f"No capture database found under {data.data_dir}. "
            "Run `kbtc capture` to start recording — Kalshi order book history "
            "cannot be bought or backfilled, so every hour not captured is gone."
        )
        return data
    data.db_path = str(db)

    con = None
    try:
        con, warning = open_read_only(db)
        if warning:
            data.notes.append(warning)
    except Exception as exc:
        data.notes.append(str(exc))
        return data

    try:
        cat = _catalogue(con)
        if not cat:
            data.notes.append("Capture database exists but has no tables yet.")
            return data

        _gather_capture_stats(con, cat, data)
        fills = _gather_fills(con, cat, data)
        settle = _gather_settlements(con, cat, data)
        if fills:
            _compute_pnl(fills, settle, data)
            _gather_fill_quality(con, cat, fills, data)
        else:
            data.notes.append(
                "No fills recorded yet. Everything below the calibration section is "
                "empty until `kbtc paper` or `kbtc live` has traded."
            )
        _gather_calibration(con, cat, settle, data, already_loaded=have_calibration)
    except Exception as exc:  # last-ditch: a broken schema must not kill the report
        data.notes.append(f"Report gathering stopped early: {type(exc).__name__}: {exc}")
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

    return data


def _gather_capture_stats(con: Any, cat: dict[str, list[str]], data: ReportData) -> None:
    """Row counts and coverage window, so a fresh install still shows progress."""
    stats = data.capture

    book_tbl = _match_table(
        cat,
        [("ticker",), ("ts", "timestamp", "time", "recorded_at")],
        name_hints=("book", "orderbook", "snapshot", "quote"),
    )
    if book_tbl:
        ts = _col(cat[book_tbl], "ts", "timestamp", "time", "recorded_at")
        try:
            n, lo, hi, ev = con.execute(
                f"SELECT count(*), CAST(min(\"{ts}\") AS VARCHAR), CAST(max(\"{ts}\") AS VARCHAR), "
                f'count(DISTINCT regexp_replace(CAST("ticker" AS VARCHAR), \'-T[0-9.]+$\', \'\')) '
                f'FROM "{book_tbl}"'
            ).fetchone()
            stats.book_rows = int(n or 0)
            stats.first_seen = _as_dt(lo)
            stats.last_seen = _as_dt(hi)
            stats.distinct_events = int(ev or 0)
        except Exception:
            pass

    brti_tbl = _match_table(
        cat,
        [("value", "brti", "price", "spot", "avg_60s")],
        name_hints=("brti", "cfbench", "index", "benchmark"),
    )
    if brti_tbl and brti_tbl != book_tbl:
        try:
            stats.brti_rows = int(con.execute(f'SELECT count(*) FROM "{brti_tbl}"').fetchone()[0])
        except Exception:
            pass

    if "spot" in cat:
        try:
            stats.spot_rows = int(con.execute("SELECT count(*) FROM spot").fetchone()[0])
        except Exception:
            pass

    data.period_start = stats.first_seen
    data.period_end = stats.last_seen


def _gather_settlements(con: Any, cat: dict[str, list[str]], data: ReportData) -> dict[str, float]:
    """ticker -> realised BRTI 60s average (`expiration_value`).

    This is free ground truth: Kalshi publishes it on every settled market.
    """
    tbl = _match_table(
        cat,
        [
            ("event_ticker", "ticker"),
            ("expiration_value", "settlement_value", "settle_value", "result"),
        ],
        name_hints=("settle", "settlement", "market"),
    )
    if not tbl:
        data.notes.append(
            "No settlement history found. Run `kbtc settlements` to backfill "
            "expiration_value — without it nothing can be scored."
        )
        return {}
    cols = cat[tbl]
    val = _col(cols, "expiration_value", "settlement_value", "settle_value", "result")
    # Settlement is a property of the EVENT: all 188 strikes under one hour share the
    # identical expiration_value, so the store keys it by event_ticker. Accept either.
    key = _col(cols, "event_ticker", "ticker")
    out: dict[str, float] = {}
    try:
        for k, v in con.execute(
            f'SELECT "{key}", "{val}" FROM "{tbl}" WHERE "{val}" IS NOT NULL'
        ).fetchall():
            f = _as_float(v, default=float("nan"))
            if not math.isnan(f):
                out[str(k)] = f
    except Exception as exc:
        data.notes.append(f"Could not read settlements from {tbl}: {exc}")
    data.capture.settled_events = len({_event_from_ticker(t) for t in out})
    return out


def _settle_for(ticker: str, settle: dict[str, float]) -> float | None:
    """Settlement value for a market ticker, whether the map is keyed by market or event."""
    v = settle.get(ticker)
    return v if v is not None else settle.get(_event_from_ticker(ticker))


def _gather_fills(con: Any, cat: dict[str, list[str]], data: ReportData) -> list[dict[str, Any]]:
    tbl = _match_table(
        cat,
        [("ticker",), ("price", "yes_price", "fill_price"), ("count", "contracts", "size", "qty")],
        name_hints=("fill", "trade", "execution"),
    )
    if not tbl:
        return []
    cols = cat[tbl]
    c_price = _col(cols, "price", "fill_price", "yes_price")
    c_count = _col(cols, "count", "contracts", "size", "qty")
    c_side = _col(cols, "side")
    c_action = _col(cols, "action")
    c_liq = _col(cols, "liquidity", "taker", "is_taker", "maker_taker")
    c_fee = _col(cols, "fee", "fees", "fee_dollars")
    c_ts = _col(cols, "ts", "timestamp", "time", "created_time", "filled_at")

    select = [f'"{c_price}" AS price', f'"{c_count}" AS cnt', '"ticker" AS ticker']
    select.append(f'"{c_side}" AS side' if c_side else "'yes' AS side")
    select.append(f'"{c_action}" AS action' if c_action else "'buy' AS action")
    select.append(f'"{c_liq}" AS liq' if c_liq else "NULL AS liq")
    select.append(f'"{c_fee}" AS fee' if c_fee else "0 AS fee")
    select.append(f"{_ts_expr(c_ts)} AS ts" if c_ts else "NULL AS ts")

    order = f' ORDER BY "{c_ts}"' if c_ts else ""
    try:
        rows = con.execute(f'SELECT {", ".join(select)} FROM "{tbl}"{order}').fetchall()
    except Exception as exc:
        data.notes.append(f"Could not read fills from {tbl}: {exc}")
        return []

    out: list[dict[str, Any]] = []
    for price, cnt, ticker, side, action, liq, fee, ts in rows:
        n = _as_float(cnt)
        if n <= 0:
            continue
        p = _as_float(price)
        # Tolerate integer-cent storage: a "price" above 1.5 can only be cents.
        if p > 1.5:
            p /= 100.0
        liq_s = str(liq).lower() if liq is not None else ""
        is_maker = "maker" in liq_s or liq_s in {"false", "0", "m"}
        out.append(
            {
                "ticker": str(ticker),
                "event": _event_from_ticker(ticker),
                "strike": _strike_from_ticker(ticker),
                "price": p,
                "count": n,
                "side": "no" if str(side).lower().startswith("n") else "yes",
                "action": "sell" if str(action).lower().startswith("s") else "buy",
                "maker": is_maker,
                "fee": _as_float(fee),
                "ts": _as_dt(ts),
            }
        )
    return out


def _compute_pnl(
    fills: list[dict[str, Any]], settle: dict[str, float], data: ReportData
) -> None:
    """Realised P&L, netting each ticker's cash flows against its settlement payoff.

    Convention: buying either side costs cash and adds contracts; selling does the
    reverse. At expiry YES pays $1 if the settlement average exceeds the strike, NO
    pays the complement. Fees always leave the account. Only tickers whose settlement
    we actually know contribute — open positions are excluded rather than marked, so
    the headline P&L is never flattered by an optimistic mark.
    """
    per_ticker: dict[str, dict[str, float]] = {}
    for f in fills:
        t = per_ticker.setdefault(
            f["ticker"],
            {"cash": 0.0, "yes": 0.0, "no": 0.0, "fees": 0.0, "contracts": 0.0, "notional": 0.0},
        )
        signed = f["count"] if f["action"] == "buy" else -f["count"]
        t["cash"] += (-1.0 if f["action"] == "buy" else 1.0) * f["price"] * f["count"]
        t[f["side"]] += signed
        t["fees"] += f["fee"]
        t["contracts"] += f["count"]
        t["notional"] += f["price"] * f["count"]

    close_times: dict[str, datetime | None] = {}
    for f in fills:
        ev = f["event"]
        if f["ts"] and (close_times.get(ev) is None or f["ts"] > close_times[ev]):
            close_times[ev] = f["ts"]

    events: dict[str, dict[str, float]] = {}
    unsettled = 0
    for ticker, t in per_ticker.items():
        sv = _settle_for(ticker, settle)
        if sv is None:
            unsettled += 1
            continue
        strike = _strike_from_ticker(ticker)
        if strike is None:
            unsettled += 1
            continue
        yes_payoff = 1.0 if sv > strike else 0.0
        gross = t["cash"] + t["yes"] * yes_payoff + t["no"] * (1.0 - yes_payoff)
        ev = events.setdefault(
            _event_from_ticker(ticker),
            {"gross": 0.0, "fees": 0.0, "contracts": 0.0},
        )
        ev["gross"] += gross
        ev["fees"] += t["fees"]
        ev["contracts"] += t["contracts"]

    if unsettled:
        data.notes.append(
            f"{unsettled} traded market(s) have no settlement value yet and are excluded "
            "from P&L. Run `kbtc settlements` after their hour closes."
        )

    ordered = sorted(events.items(), key=lambda kv: (close_times.get(kv[0]) or datetime.min.replace(tzinfo=UTC)))
    running = 0.0
    peak = 0.0
    equity: list[tuple[datetime | None, float]] = []
    for name, ev in ordered:
        net = ev["gross"] - ev["fees"]
        running += net
        peak = max(peak, running)
        data.max_drawdown = max(data.max_drawdown, peak - running)
        equity.append((close_times.get(name), running))
        data.per_event.append(
            EventPnl(
                event=name,
                close_time=close_times.get(name),
                contracts=ev["contracts"],
                gross=ev["gross"],
                fees=ev["fees"],
                net=net,
            )
        )
        if net > 0:
            data.wins += 1
        elif net < 0:
            data.losses += 1

    data.equity = equity
    data.events_traded = len(ordered)
    data.gross_pnl = sum(e.gross for e in data.per_event)
    data.total_fees = sum(e.fees for e in data.per_event)
    data.net_pnl = data.gross_pnl - data.total_fees

    # Price-bucket attribution. Each fill is attributed at its OWN price rather than at
    # the ticker's average, because the point of the exercise is to find out which part
    # of the probability ladder is paying for itself once fees are charged.
    yes_payoffs: dict[str, float] = {}
    for ticker in per_ticker:
        sv = _settle_for(ticker, settle)
        strike = _strike_from_ticker(ticker)
        if sv is not None and strike is not None:
            yes_payoffs[ticker] = 1.0 if sv > strike else 0.0

    buckets: dict[str, dict[str, float]] = {}
    for f in fills:
        yp = yes_payoffs.get(f["ticker"])
        if yp is None:
            continue
        payoff = yp if f["side"] == "yes" else 1.0 - yp
        sign = 1.0 if f["action"] == "buy" else -1.0
        f["_pnl"] = sign * (payoff - f["price"]) * f["count"]
        label = next(
            (lbl for lbl, lo, hi in PRICE_BUCKETS if lo <= f["price"] < hi),
            PRICE_BUCKETS[-1][0],
        )
        b = buckets.setdefault(label, {"gross": 0.0, "fees": 0.0, "contracts": 0.0})
        b["gross"] += f["_pnl"]
        b["fees"] += f["fee"]
        b["contracts"] += f["count"]
    for label, _lo, _hi in PRICE_BUCKETS:
        if label in buckets:
            b = buckets[label]
            data.per_bucket.append(
                BucketPnl(
                    label=label,
                    contracts=b["contracts"],
                    gross=b["gross"],
                    fees=b["fees"],
                    net=b["gross"] - b["fees"],
                )
            )


def _gather_fill_quality(
    con: Any, cat: dict[str, list[str]], fills: list[dict[str, Any]], data: ReportData
) -> None:
    fq = data.fill_quality
    for f in fills:
        if f["maker"]:
            fq.maker_contracts += f["count"]
            fq.maker_fees += f["fee"]
        else:
            fq.taker_contracts += f["count"]
            fq.taker_fees += f["fee"]

    if fq.maker_fees > 1e-9:
        data.notes.append(
            f"Maker fills were charged ${fq.maker_fees:.4f} in fees. KXBTCD maker fees are "
            "ZERO — this means fills tagged 'maker' were actually taker fills, or the fee "
            "field is being populated from the wrong source. Investigate before scaling."
        )

    # Slippage vs the mid at decision time. Prefer an explicit column on the fills
    # table; otherwise reconstruct it from captured book snapshots (last book at or
    # before the fill for that ticker).
    slips: list[float] = []
    fill_tbl = _match_table(
        cat,
        [("ticker",), ("price", "fill_price"), ("mid_at_decision", "decision_mid", "mid")],
        name_hints=("fill", "trade"),
    )
    if fill_tbl:
        cols = cat[fill_tbl]
        c_mid = _col(cols, "mid_at_decision", "decision_mid", "mid")
        c_price = _col(cols, "price", "fill_price")
        try:
            for p, m in con.execute(
                f'SELECT "{c_price}", "{c_mid}" FROM "{fill_tbl}" WHERE "{c_mid}" IS NOT NULL'
            ).fetchall():
                pf, mf = _as_float(p), _as_float(m)
                if pf > 1.5:
                    pf /= 100.0
                if mf > 1.5:
                    mf /= 100.0
                slips.append((pf - mf) * 100.0)
        except Exception:
            slips = []

    if not slips:
        book_tbl = _match_table(
            cat,
            [
                ("ticker",),
                ("ts", "timestamp", "time", "recorded_at"),
                ("mid", "yes_bid", "yes_bid_dollars"),
            ],
            name_hints=("book", "orderbook", "snapshot", "quote"),
        )
        if book_tbl:
            cols = cat[book_tbl]
            c_ts = _col(cols, "ts", "timestamp", "time", "recorded_at")
            c_mid = _col(cols, "mid")
            c_bid = _col(cols, "yes_bid", "yes_bid_dollars", "best_yes_bid")
            c_ask = _col(cols, "yes_ask", "yes_ask_dollars", "best_yes_ask")
            expr = f'"{c_mid}"' if c_mid else (
                f'(("{c_bid}" + "{c_ask}") / 2)' if c_bid and c_ask else None
            )
            if expr:
                # One lookup per fill. Capped, because this is a diagnostic and a
                # multi-thousand-fill history does not need every sample to see the shape.
                for f in fills[-2000:]:
                    if f["ts"] is None:
                        continue
                    try:
                        row = con.execute(
                            f'SELECT {expr} FROM "{book_tbl}" '
                            f'WHERE "ticker" = ? AND "{c_ts}" <= CAST(? AS TIMESTAMPTZ) '
                            f'ORDER BY "{c_ts}" DESC LIMIT 1',
                            [f["ticker"], f["ts"].isoformat()],
                        ).fetchone()
                    except Exception:
                        break
                    if not row or row[0] is None:
                        continue
                    mid = _as_float(row[0])
                    if mid > 1.5:
                        mid /= 100.0
                    if mid <= 0:
                        continue
                    # A buy above the mid is adverse; a sell below the mid is adverse.
                    signed = (f["price"] - mid) if f["action"] == "buy" else (mid - f["price"])
                    slips.append(signed * 100.0)

    if slips:
        slips.sort()
        fq.slippage_samples = len(slips)
        fq.mean_slippage_cents = sum(slips) / len(slips)
        fq.median_slippage_cents = slips[len(slips) // 2]


def _score(label: str, probs: list[float], outcomes: list[int]) -> Scores:
    """Brier score, log loss and reliability bins for one forecaster."""
    n = len(probs)
    if n == 0:
        return Scores(label=label, n=0, brier=float("nan"), log_loss=float("nan"))
    eps = 1e-6
    brier = sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / n
    ll = -sum(
        math.log(max(eps, min(1 - eps, p))) if o == 1 else math.log(max(eps, min(1 - eps, 1 - p)))
        for p, o in zip(probs, outcomes)
    ) / n

    bins: list[Bin] = []
    for i in range(N_BINS):
        lo, hi = i / N_BINS, (i + 1) / N_BINS
        sel = [(p, o) for p, o in zip(probs, outcomes) if (lo <= p < hi or (i == N_BINS - 1 and p == 1.0))]
        if not sel:
            continue
        bins.append(
            Bin(
                lo=lo,
                hi=hi,
                n=len(sel),
                mean_pred=sum(p for p, _ in sel) / len(sel),
                obs_freq=sum(o for _, o in sel) / len(sel),
            )
        )
    return Scores(label=label, n=n, brier=brier, log_loss=ll, bins=bins)


CALIBRATION_JSON = "calibration.json"


def _scores_from_result(payload: dict, which: str, label: str) -> Scores | None:
    """Adapt one leg of `CalibrationResult.as_dict()` into the renderer's shapes."""
    block = payload.get(which)
    if not isinstance(block, dict):
        return None
    curve = payload.get(f"{which}_reliability") or {}
    centers = curve.get("centers") or []
    freqs = curve.get("frequencies") or []
    counts = curve.get("counts") or []
    bins: list[Bin] = []
    width = 1.0 / max(1, len(centers))
    for c, f, n in zip(centers, freqs, counts):
        n = int(n or 0)
        f = _as_float(f, default=float("nan"))
        if n <= 0 or math.isnan(f):
            continue  # empty bins carry no information; drawing them invents a point
        c = _as_float(c)
        bins.append(Bin(lo=max(0.0, c - width / 2), hi=min(1.0, c + width / 2),
                        n=n, mean_pred=c, obs_freq=f))
    return Scores(
        label=label,
        n=int(_as_float(block.get("n"))),
        brier=_as_float(block.get("brier"), default=float("nan")),
        log_loss=_as_float(block.get("log_loss"), default=float("nan")),
        bins=bins,
    )


def _load_calibration_json(data: ReportData) -> bool:
    """Load the persisted CalibrationResult written by `kbtc calibrate`, if any.

    This is the authoritative source: it is produced by the calibration module itself,
    with its own anti-lookahead screening and its own out-of-sample cutoff. The DB
    probing below is only a fallback for people who have not run `calibrate` yet.
    """
    import json

    path = Path(data.data_dir).expanduser() / CALIBRATION_JSON
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        data.notes.append(f"Ignoring unreadable {path}: {exc}")
        return False

    data.model_scores = _scores_from_result(payload, "model", "Our model")
    data.market_scores = _scores_from_result(payload, "market", "Market mid")
    if data.model_scores is None and data.market_scores is None:
        return False

    for key in (
        "skill", "n_observations", "n_events", "n_dropped", "dropped_reasons",
        "is_out_of_sample", "train_cutoff", "first_ts", "last_ts", "generated_at",
    ):
        if key in payload:
            data.cal_meta[key] = payload[key]
    data.cal_by_minutes = list(payload.get("by_minutes_to_close") or [])
    data.cal_by_price = list(payload.get("by_price_bucket") or [])
    data.calibration_source = f"`{path}` (written by `kbtc calibrate`)"

    dropped = data.cal_meta.get("dropped_reasons") or {}
    if isinstance(dropped, dict) and dropped:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        data.notes.append(f"Calibration dropped observations: {summary}.")
    if data.cal_meta.get("is_out_of_sample") is False:
        data.notes.append(
            "The calibration is IN-SAMPLE: the volatility model may have been fitted on "
            "the same hours it is being scored on. Treat the skill score as an upper "
            "bound, not a backtest."
        )
    return True


def _gather_calibration(
    con: Any,
    cat: dict[str, list[str]],
    settle: dict[str, float],
    data: ReportData,
    *,
    already_loaded: bool = False,
) -> None:
    """Score model vs market.

    Preference order: the persisted `CalibrationResult` from `kbtc calibrate`, then any
    table carrying model/market probabilities with a realised outcome, then a
    reconstruction from captured ladders joined to settlement values. The last of those
    is what makes the report useful on day one, before anything has been calibrated.
    """
    if already_loaded or _load_calibration_json(data):
        return

    tbl = _match_table(
        cat,
        [
            ("model_prob", "model_p", "fair", "fair_prob", "prob_above"),
            ("outcome", "realised", "realized", "label", "y"),
        ],
        name_hints=("calib", "prediction", "score"),
    )
    if tbl:
        cols = cat[tbl]
        c_model = _col(cols, "model_prob", "model_p", "fair", "fair_prob", "prob_above")
        c_market = _col(cols, "market_prob", "market_mid", "mid", "market_p")
        c_out = _col(cols, "outcome", "realised", "realized", "label", "y")
        sel = f'"{c_model}", "{c_out}"' + (f', "{c_market}"' if c_market else ", NULL")
        try:
            rows = con.execute(f'SELECT {sel} FROM "{tbl}" WHERE "{c_out}" IS NOT NULL').fetchall()
        except Exception as exc:
            data.notes.append(f"Could not read calibration table {tbl}: {exc}")
            rows = []
        mp, kp, out_m, out_k = [], [], [], []
        for model, outcome, market in rows:
            o = 1 if _as_float(outcome) > 0.5 else 0
            m = _as_float(model, default=float("nan"))
            if not math.isnan(m):
                mp.append(min(1.0, max(0.0, m if m <= 1.5 else m / 100.0)))
                out_m.append(o)
            if market is not None:
                k = _as_float(market, default=float("nan"))
                if not math.isnan(k):
                    kp.append(min(1.0, max(0.0, k if k <= 1.5 else k / 100.0)))
                    out_k.append(o)
        if mp:
            data.model_scores = _score("Our model", mp, out_m)
        if kp:
            data.market_scores = _score("Market mid", kp, out_k)
        if mp or kp:
            data.calibration_source = f"table `{tbl}`"
            return

    _derive_calibration_from_capture(con, cat, settle, data)


def _derive_calibration_from_capture(
    con: Any, cat: dict[str, list[str]], settle: dict[str, float], data: ReportData
) -> None:
    """Fallback: score the market mid, and our pricer where a spot value is available.

    This is what makes the report useful on day one — with nothing but `kbtc capture`
    and `kbtc settlements` you can already see how well the market mid is calibrated,
    which is the bar our model has to clear. It is deliberately cruder than
    `kbtc calibrate`: a single unconditional volatility, no seasonal adjustment, and no
    train/test split, so treat the model leg here as indicative only.
    """
    if not settle:
        return
    book_tbl = _match_table(
        cat,
        [
            ("ticker",),
            ("ts", "timestamp", "time", "recorded_at"),
            ("mid", "yes_bid", "yes_bid_dollars"),
        ],
        name_hints=("ladder", "book", "orderbook", "snapshot", "quote"),
    )
    if not book_tbl:
        data.notes.append("No calibration data and no captured ladders — nothing to score yet.")
        return

    cols = cat[book_tbl]
    c_ts = _col(cols, "ts", "timestamp", "time", "recorded_at")
    c_mid = _col(cols, "mid")
    c_bid = _col(cols, "yes_bid", "yes_bid_dollars", "best_yes_bid")
    c_ask = _col(cols, "yes_ask", "yes_ask_dollars", "best_yes_ask")
    c_mtc = _col(cols, "minutes_to_close")
    c_close = _col(cols, "close_time", "close_ts", "expires_at")
    c_spot = _col(cols, "spot", "brti", "index", "brti_value", "avg_60s_data")

    mid_expr = (
        f'"{c_mid}"' if c_mid else (f'(("{c_bid}" + "{c_ask}") / 2)' if c_bid and c_ask else None)
    )
    if not mid_expr:
        return

    # Spot: prefer a column on the ladder itself, else ASOF-join the BRTI tape. The
    # windowed average is the settlement-relevant figure once inside the final minute;
    # outside it the rolling 60s average is the best same-timestamp spot proxy we have.
    brti_tbl = _match_table(
        cat,
        [("ts", "timestamp", "time"), ("value", "avg_60s", "windowed_avg")],
        name_hints=("brti", "cfbench", "benchmark", "index"),
    )
    join_sql = ""
    if c_spot:
        spot_expr = f'avg(l."{c_spot}")'
    elif brti_tbl and brti_tbl != book_tbl:
        bcols = cat[brti_tbl]
        b_ts = _col(bcols, "ts", "timestamp", "time")
        pieces = [
            f'b."{c}"'
            for c in (
                _col(bcols, "windowed_avg"),
                _col(bcols, "avg_60s"),
                _col(bcols, "value"),
            )
            if c
        ]
        spot_expr = f"avg(COALESCE({', '.join(pieces)}))"
        join_sql = f' ASOF JOIN "{brti_tbl}" b ON l."{c_ts}" >= b."{b_ts}"'
    else:
        spot_expr = "NULL"

    # One observation per (ticker, minute): a fast-updating ladder would otherwise let a
    # single event contribute thousands of near-identical rows and dominate the score.
    aliased_mid = f'l."{c_mid}"' if c_mid else f'((l."{c_bid}" + l."{c_ask}") / 2)'
    sel = [
        'l."ticker" AS ticker',
        f"avg({aliased_mid}) AS mid",
        f'CAST(max(l."{c_ts}") AS VARCHAR) AS ts',
        f"{spot_expr} AS spot",
        (f'avg(l."{c_mtc}")' if c_mtc else "NULL") + " AS mtc",
        (f'CAST(max(l."{c_close}") AS VARCHAR)' if c_close else "NULL") + " AS close_time",
    ]
    try:
        rows = con.execute(
            f'SELECT {", ".join(sel)} FROM "{book_tbl}" l{join_sql} '
            f'GROUP BY l."ticker", date_trunc(\'minute\', l."{c_ts}")'
        ).fetchall()
    except Exception as exc:
        data.notes.append(f"Could not derive calibration from {book_tbl}: {exc}")
        return

    try:
        from kalshi_btc.model.pricing import annual_to_per_minute, price_above

        sigma_min: float | None = annual_to_per_minute(DEFAULT_ANNUAL_VOL)
    except Exception:
        sigma_min = None

    market_p: list[float] = []
    model_p: list[float] = []
    out_k: list[int] = []
    out_m: list[int] = []

    for ticker, mid_raw, ts_raw, spot_raw, mtc_raw, close_raw in rows:
        ticker = str(ticker)
        sv = _settle_for(ticker, settle)
        strike = _strike_from_ticker(ticker)
        if sv is None or strike is None:
            continue
        outcome = 1 if sv > strike else 0

        mid = _as_float(mid_raw, default=float("nan"))
        if math.isnan(mid):
            continue
        if mid > 1.5:
            mid /= 100.0
        if not (0.0 <= mid <= 1.0):
            continue
        market_p.append(mid)
        out_k.append(outcome)

        if sigma_min is None:
            continue
        spot = _as_float(spot_raw, default=float("nan"))
        if math.isnan(spot) or spot <= 0:
            continue

        # minutes_to_close is recorded on every snapshot; only reconstruct it if absent.
        minutes = _as_float(mtc_raw, default=float("nan"))
        if math.isnan(minutes):
            ts = _as_dt(ts_raw)
            close_time = _as_dt(close_raw) or _close_time_from_ticker(ticker)
            if ts is None or close_time is None:
                continue
            minutes = (close_time - ts).total_seconds() / 60.0
        if not (0.0 < minutes <= 65.0):
            continue
        try:
            q = price_above(spot, strike, sigma_min, minutes)
        except Exception:
            continue
        model_p.append(min(1.0, max(0.0, q.prob_above)))
        out_m.append(outcome)

    if market_p:
        data.market_scores = _score("Market mid", market_p, out_k)
    if model_p:
        data.model_scores = _score("Our model", model_p, out_m)
    if market_p or model_p:
        data.calibration_source = (
            f"derived from `{book_tbl}` + settlements (run `kbtc calibrate` for the real score)"
        )
        if not model_p:
            data.notes.append(
                "Only the market mid could be scored: no BRTI spot was available at the "
                "captured timestamps, so our model had nothing to price against. Run "
                "`kbtc calibrate` once the BRTI tape has some coverage."
            )


def _close_time_from_ticker(ticker: str) -> datetime | None:
    """KXBTCD-26JUL2819-T63999.99 -> 2026-07-28 19:00 UTC."""
    try:
        from kalshi_btc.core.types import MarketTicker

        parsed = MarketTicker.parse(ticker)
    except Exception:
        return None
    m = re.match(r"^[A-Z]+-(\d{2})([A-Z]{3})(\d{2})(\d{2})-", ticker)
    if not m:
        return None
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    mon = months.get(m.group(2))
    if mon is None:
        return None
    try:
        return datetime(2000 + int(m.group(1)), mon, int(m.group(3)), parsed.hour, tzinfo=UTC)
    except ValueError:
        return None


# ======================================================================================
# SVG chart primitives
# ======================================================================================
def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _fmt_money(x: float) -> str:
    sign = "-" if x < -1e-9 else ""
    return f"{sign}${abs(x):,.2f}"


def _fmt_pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Human-readable axis ticks spanning [lo, hi]."""
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo <= 0:
        return [lo]
    raw = (hi - lo) / max(1, count)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 0.5 and len(out) < 40:
        out.append(round(v, 10))
        v += step
    return out


def _empty_chart(message: str, height: int = 160) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {CHART_W} {height}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="{_esc(message)}">'
        f'<rect x="0" y="0" width="{CHART_W}" height="{height}" class="plot-bg" rx="8"/>'
        f'<text x="{CHART_W // 2}" y="{height // 2}" class="empty-label" '
        f'text-anchor="middle">{_esc(message)}</text></svg>'
    )


def _equity_chart(points: Sequence[tuple[datetime | None, float]]) -> str:
    if len(points) < 2:
        return _empty_chart("Not enough settled events to draw an equity curve yet.")

    ys = [0.0] + [p[1] for p in points]
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.12
    lo, hi = lo - pad, hi + pad

    n = len(points)
    iw = CHART_W - PAD_L - PAD_R
    ih = CHART_H - PAD_T - PAD_B

    def px(i: int) -> float:
        return PAD_L + (iw * i / max(1, n - 1))

    def py(v: float) -> float:
        return PAD_T + ih * (1 - (v - lo) / (hi - lo))

    parts = [
        (
            f'<svg class="chart" viewBox="0 0 {CHART_W} {CHART_H}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'aria-label="Cumulative net profit and loss by settled event">'
        ),
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{iw}" height="{ih}" class="plot-bg" rx="6"/>',
    ]
    for t in _nice_ticks(lo, hi):
        if not (lo <= t <= hi):
            continue
        y = py(t)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + iw}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" class="axis" text-anchor="end">'
            f"{_esc(_fmt_money(t))}</text>"
        )
    if lo < 0 < hi:
        y0 = py(0.0)
        parts.append(f'<line x1="{PAD_L}" y1="{y0:.1f}" x2="{PAD_L + iw}" y2="{y0:.1f}" class="zero"/>')

    coords = [(px(i), py(v)) for i, (_ts, v) in enumerate(points)]
    area = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    base = py(max(lo, min(hi, 0.0)))
    parts.append(
        f'<polygon class="equity-fill" points="{coords[0][0]:.1f},{base:.1f} {area} '
        f'{coords[-1][0]:.1f},{base:.1f}"/>'
    )
    parts.append(f'<polyline class="equity-line" points="{area}"/>')
    for x, y in coords[-1:]:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="equity-dot"/>')

    first = points[0][0]
    last = points[-1][0]
    if first:
        parts.append(
            f'<text x="{PAD_L}" y="{CHART_H - 14}" class="axis">'
            f'{_esc(first.strftime("%d %b %H:%M"))}</text>'
        )
    if last:
        parts.append(
            f'<text x="{PAD_L + iw}" y="{CHART_H - 14}" class="axis" text-anchor="end">'
            f'{_esc(last.strftime("%d %b %H:%M"))}</text>'
        )
    parts.append(
        f'<text x="{PAD_L + iw / 2}" y="{CHART_H - 14}" class="axis" text-anchor="middle">'
        f"{n} settled events</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _reliability_chart(model: Scores | None, market: Scores | None) -> str:
    if not (model and model.bins) and not (market and market.bins):
        return _empty_chart("No scored forecasts yet — run `kbtc calibrate`.", height=220)

    size = 360
    pad = 44
    inner = size - pad - 16

    def sx(p: float) -> float:
        return pad + inner * p

    def sy(p: float) -> float:
        return pad + inner * (1 - p)

    parts = [
        (
            f'<svg class="chart reliability" viewBox="0 0 {size + 300} {size + 20}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'aria-label="Reliability diagram comparing model and market forecasts">'
        ),
        f'<rect x="{pad}" y="{pad}" width="{inner}" height="{inner}" class="plot-bg" rx="6"/>',
    ]
    for k in range(0, 11, 2):
        p = k / 10
        parts.append(f'<line x1="{sx(p):.1f}" y1="{pad}" x2="{sx(p):.1f}" y2="{pad + inner}" class="grid"/>')
        parts.append(f'<line x1="{pad}" y1="{sy(p):.1f}" x2="{pad + inner}" y2="{sy(p):.1f}" class="grid"/>')
        parts.append(
            f'<text x="{sx(p):.1f}" y="{pad + inner + 18}" class="axis" text-anchor="middle">{p:.1f}</text>'
        )
        parts.append(
            f'<text x="{pad - 8}" y="{sy(p) + 4:.1f}" class="axis" text-anchor="end">{p:.1f}</text>'
        )
    parts.append(
        f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" class="perfect"/>'
    )

    def draw(s: Scores, cls: str) -> None:
        pts = [(sx(b.mean_pred), sy(b.obs_freq)) for b in s.bins]
        if len(pts) > 1:
            parts.append(
                f'<polyline class="rel-line {cls}" points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                + '"/>'
            )
        for b, (x, y) in zip(s.bins, pts):
            r = 3.0 + min(7.0, math.log10(max(1, b.n)) * 3.0)
            solid = "" if b.reliable else " hollow"
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" class="rel-dot {cls}{solid}">'
                f"<title>{_esc(s.label)}: predicted {b.mean_pred:.3f}, observed "
                f"{b.obs_freq:.3f}, n={b.n}</title></circle>"
            )

    if market and market.bins:
        draw(market, "market")
    if model and model.bins:
        draw(model, "model")

    parts.append(
        f'<text x="{pad + inner / 2}" y="{size + 12}" class="axis-title" text-anchor="middle">'
        "Forecast probability</text>"
    )
    parts.append(
        f'<text transform="translate(14,{pad + inner / 2}) rotate(-90)" class="axis-title" '
        'text-anchor="middle">Observed frequency</text>'
    )

    lx = size + 16
    ly = pad + 10
    parts.append(f'<text x="{lx}" y="{ly}" class="legend-title">How to read this</text>')
    lines = [
        "Dots on the diagonal = honest",
        "probabilities. Above it = the",
        "forecaster is too pessimistic;",
        "below = too confident.",
        "",
        "Dot size = number of forecasts.",
        f"Hollow dots have &lt; {MIN_BIN_N} samples",
        "and are mostly noise.",
    ]
    for i, line in enumerate(lines):
        parts.append(f'<text x="{lx}" y="{ly + 22 + i * 17}" class="legend-note">{line}</text>')
    ly2 = ly + 22 + len(lines) * 17 + 12
    if model:
        parts.append(f'<circle cx="{lx + 6}" cy="{ly2 - 4}" r="5" class="rel-dot model"/>')
        parts.append(f'<text x="{lx + 20}" y="{ly2}" class="legend-note">Our model</text>')
        ly2 += 20
    if market:
        parts.append(f'<circle cx="{lx + 6}" cy="{ly2 - 4}" r="5" class="rel-dot market"/>')
        parts.append(f'<text x="{lx + 20}" y="{ly2}" class="legend-note">Market mid</text>')
    parts.append("</svg>")
    return "".join(parts)


def _bucket_chart(buckets: Sequence[BucketPnl]) -> str:
    if not buckets:
        return _empty_chart("No fills to attribute by price bucket yet.")
    h = 44 * len(buckets) + 46
    label_w = 92
    iw = CHART_W - label_w - 130
    vals = [b.net for b in buckets] + [0.0]
    lo, hi = min(vals), max(vals)
    span = max(abs(lo), abs(hi)) or 1.0
    zero_x = label_w + iw / 2

    parts = [
        (
            f'<svg class="chart" viewBox="0 0 {CHART_W} {h}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'aria-label="Net profit and loss by fill price bucket">'
        )
    ]
    parts.append(f'<line x1="{zero_x:.1f}" y1="12" x2="{zero_x:.1f}" y2="{h - 34}" class="zero"/>')
    for i, b in enumerate(buckets):
        y = 18 + i * 44
        w = (abs(b.net) / span) * (iw / 2 - 6)
        x = zero_x if b.net >= 0 else zero_x - w
        cls = "bar-pos" if b.net >= 0 else "bar-neg"
        parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{max(1.0, w):.1f}" height="26" rx="4" class="{cls}">'
            f"<title>{_esc(b.label)}: net {_fmt_money(b.net)} on {b.contracts:g} contracts"
            f"</title></rect>"
        )
        parts.append(
            f'<text x="{label_w - 10}" y="{y + 18}" class="axis" text-anchor="end">'
            f"{_esc(b.label)}</text>"
        )
        # Place the value outside the bar, but flip it inside when the bar is long
        # enough that an outside label would collide with the row labels or the edge.
        if b.net >= 0:
            tx, anchor = x + w + 8, "start"
            if tx > CHART_W - 8:
                tx, anchor = x + w - 8, "end"
        else:
            tx, anchor = x - 8, "end"
            if tx < label_w + 8:
                tx, anchor = x + 8, "start"
        parts.append(
            f'<text x="{tx:.1f}" y="{y + 18}" class="bar-label" text-anchor="{anchor}">'
            f"{_esc(_fmt_money(b.net))}</text>"
        )
    parts.append(
        f'<text x="{zero_x:.1f}" y="{h - 12}" class="axis" text-anchor="middle">'
        "net P&amp;L after fees</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# ======================================================================================
# HTML rendering
# ======================================================================================
CSS = """
:root {
  color-scheme: light dark;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #14181f;
  --muted: #5d6874;
  --line: #dfe3e8;
  --grid: #eceef1;
  --accent: #2f6df6;
  --market: #8a93a0;
  --pos: #16855c;
  --neg: #c73a3a;
  --warn: #a8690b;
  --warn-bg: #fdf5e4;
  --plot: #fafbfc;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1216;
    --panel: #171b21;
    --ink: #e7ebf0;
    --muted: #97a1ae;
    --line: #262c35;
    --grid: #222831;
    --accent: #6a9bff;
    --market: #7e8896;
    --pos: #3fcf8e;
    --neg: #ff6b6b;
    --warn: #f0b54a;
    --warn-bg: #2a2313;
    --plot: #12161b;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 20px 72px;
  background: var(--bg); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.wrap { max-width: 980px; margin: 0 auto; }
header { padding: 40px 0 8px; }
h1 { font-size: 27px; margin: 0 0 6px; letter-spacing: -0.015em; }
h2 { font-size: 19px; margin: 0 0 4px; letter-spacing: -0.01em; }
h3 { font-size: 14px; margin: 22px 0 8px; text-transform: uppercase;
     letter-spacing: 0.08em; color: var(--muted); font-weight: 600; }
p { margin: 0 0 12px; }
.sub { color: var(--muted); font-size: 14px; }
.mono { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
        font-size: 13px; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
          padding: 22px 24px; margin: 18px 0; }
.section-head { margin-bottom: 14px; }
.section-head .sub { margin-top: 2px; }
.tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.tile .k { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); }
.tile .v { font-size: 25px; font-weight: 650; margin-top: 4px; letter-spacing: -0.02em; }
.tile .n { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.pos { color: var(--pos); } .neg { color: var(--neg); } .muted { color: var(--muted); }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: 11.5px;
     text-transform: uppercase; letter-spacing: 0.06em; }
tbody tr:last-child td { border-bottom: none; }
tfoot td { font-weight: 650; border-top: 2px solid var(--line); border-bottom: none; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.callout { border-left: 3px solid var(--accent); background: color-mix(in srgb, var(--accent) 7%, transparent);
           padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 0 0 4px; }
.warn { border-left: 3px solid var(--warn); background: var(--warn-bg); padding: 12px 16px;
        border-radius: 0 8px 8px 0; margin: 10px 0; font-size: 14px; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11.5px;
         font-weight: 650; letter-spacing: 0.04em; text-transform: uppercase;
         border: 1px solid var(--line); }
.badge.armed { background: var(--neg); color: #fff; border-color: transparent; }
.badge.paper { background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent); }
.headline-skill { font-size: 34px; font-weight: 700; letter-spacing: -0.025em; margin: 4px 0 2px; }
.chart { display: block; margin: 6px 0 4px; max-width: 100%; }
.plot-bg { fill: var(--plot); stroke: var(--line); }
.grid { stroke: var(--grid); stroke-width: 1; }
.zero { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 3; opacity: 0.8; }
.perfect { stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 5 4; }
.axis { fill: var(--muted); font-size: 11px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.axis-title { fill: var(--muted); font-size: 12px; }
.legend-title { fill: var(--ink); font-size: 12.5px; font-weight: 650; }
.legend-note { fill: var(--muted); font-size: 12px; }
.empty-label { fill: var(--muted); font-size: 13px; }
.equity-line { fill: none; stroke: var(--accent); stroke-width: 2.25;
               stroke-linejoin: round; stroke-linecap: round; }
.equity-fill { fill: var(--accent); opacity: 0.12; stroke: none; }
.equity-dot { fill: var(--accent); }
.rel-line { fill: none; stroke-width: 2; }
.rel-line.model { stroke: var(--accent); }
.rel-line.market { stroke: var(--market); stroke-dasharray: 6 4; }
.rel-dot.model { fill: var(--accent); stroke: var(--panel); stroke-width: 1.5; }
.rel-dot.market { fill: var(--market); }
.rel-dot.hollow { fill: var(--plot); stroke-width: 2; }
.rel-dot.model.hollow { stroke: var(--accent); }
.rel-dot.market.hollow { stroke: var(--market); }
.bar-pos { fill: var(--pos); } .bar-neg { fill: var(--neg); }
.bar-label { fill: var(--ink); font-size: 12px; font-weight: 600; }
footer { color: var(--muted); font-size: 12.5px; text-align: center; padding: 8px 0 0; }
ul { margin: 0 0 12px; padding-left: 20px; }
li { margin-bottom: 5px; }
"""


def _tile(key: str, value: str, note: str = "", cls: str = "") -> str:
    note_html = f'<div class="n">{note}</div>' if note else ""
    return (
        f'<div class="tile"><div class="k">{_esc(key)}</div>'
        f'<div class="v {cls}">{value}</div>{note_html}</div>'
    )


def _plain_english(d: ReportData) -> str:
    """The paragraph a non-quant reads first. Must be honest, including about failure."""
    bits: list[str] = []
    mode = "LIVE with real money" if d.armed else "paper (no real money at risk)"
    bits.append(f"This report covers a bot running in <strong>{mode}</strong> mode on the ")
    bits.append(f"<span class='mono'>{_esc(d.env)}</span> environment.")

    if d.has_trades:
        sign = "made" if d.net_pnl >= 0 else "lost"
        wr = _fmt_pct(d.win_rate)
        bits.append(
            f" Across <strong>{d.events_traded}</strong> settled hourly events it {sign} "
            f"<strong>{_fmt_money(abs(d.net_pnl))}</strong> after fees, winning {wr} of them. "
            f"Fees took {_fmt_money(d.total_fees)}"
        )
        if abs(d.gross_pnl) > 1e-9:
            bits.append(f" — that is {_fmt_pct(d.total_fees / abs(d.gross_pnl))} of gross P&amp;L")
        bits.append(
            f". The worst peak-to-trough loss along the way was {_fmt_money(d.max_drawdown)}."
        )
    else:
        bits.append(
            " <strong>No completed trades yet</strong>, so there is no P&amp;L to show. "
            "That is the expected state on a fresh install — the sections below still "
            "report what the data capture and the pricing model are doing."
        )

    ss = d.skill_score
    if ss is None and d.market_scores:
        bits.append(
            " Our model has not been scored against the market yet, but the market mid's own "
            "calibration is shown below — that is the bar we have to beat."
        )
    elif ss is None:
        bits.append(
            " There is not yet enough settled data to say whether the pricing model is any good. "
            "Leave <span class='mono'>kbtc capture</span> running and check back."
        )
    elif ss > 0.02:
        bits.append(
            f" <strong>The model currently beats the market mid</strong>: its Brier score is "
            f"{_fmt_pct(ss)} better. That is the number the whole strategy rests on. "
            "A positive skill score means our probabilities are genuinely more accurate "
            "than simply reading the price off the screen."
        )
    elif ss > -0.02:
        bits.append(
            " <strong>The model is currently no better than the market mid</strong> "
            f"(skill score {ss:+.3f}). Without an accuracy edge there is nothing to monetise "
            "beyond the maker rebate. Do not increase size."
        )
    else:
        bits.append(
            f" <strong>The model is currently WORSE than the market mid</strong> "
            f"(skill score {ss:+.3f}). Trading on it is expected to lose money. "
            "Stay in paper mode and fix the model before doing anything else."
        )
    return "".join(bits)


def _calibration_section(d: ReportData) -> str:
    parts = ['<section id="calibration">']
    parts.append(
        '<div class="section-head"><h2>Calibration — is the model any good?</h2>'
        '<div class="sub">The most important section on this page. Everything else is '
        "bookkeeping; this is the evidence.</div></div>"
    )

    ss = d.skill_score
    if ss is not None:
        cls = "pos" if ss > 0 else "neg"
        verdict = (
            "Our model is more accurate than the market."
            if ss > 0.02
            else "No measurable edge over the market."
            if ss > -0.02
            else "Our model is LESS accurate than the market. Do not trade it."
        )
        parts.append(
            f'<div class="callout"><div class="k sub">Brier skill score vs market mid</div>'
            f'<div class="headline-skill {cls}">{ss:+.4f}</div>'
            f'<div class="sub">{_esc(verdict)} '
            "Positive means our probabilities beat the price on the screen; "
            "zero means we are just re-reading the market; negative means we are adding noise."
            "</div></div>"
        )
    else:
        parts.append(
            '<div class="warn">No skill score yet. It needs both a model probability and a '
            "realised outcome for the same market. Run <span class='mono'>kbtc capture</span> "
            "for a few days, then <span class='mono'>kbtc settlements</span> and "
            "<span class='mono'>kbtc calibrate</span>.</div>"
        )

    parts.append(_reliability_chart(d.model_scores, d.market_scores))

    rows = [s for s in (d.model_scores, d.market_scores) if s is not None]
    if rows:
        parts.append("<h3>Scores</h3>")
        parts.append('<div class="scroll"><table><thead><tr>')
        parts.append(
            "<th>Forecaster</th><th>Forecasts</th><th>Brier score</th>"
            "<th>Log loss</th><th>vs market</th></tr></thead><tbody>"
        )
        base = d.market_scores.brier if d.market_scores else None
        for s in rows:
            if base and base > 0 and s is not d.market_scores:
                rel = 1.0 - s.brier / base
                rel_html = f'<span class="{"pos" if rel > 0 else "neg"}">{rel:+.4f}</span>'
            else:
                rel_html = '<span class="muted">reference</span>' if s is d.market_scores else "—"
            parts.append(
                f"<tr><td>{_esc(s.label)}</td><td>{s.n:,}</td>"
                f"<td>{s.brier:.5f}</td><td>{s.log_loss:.5f}</td><td>{rel_html}</td></tr>"
            )
        parts.append("</tbody></table></div>")
        parts.append(
            '<p class="sub">Brier score is mean squared error on probabilities — lower is '
            "better, 0.25 is what you get by always saying 50%. Log loss punishes confident "
            "mistakes much harder, so a good Brier with a bad log loss means the model is "
            "occasionally very wrong while sounding very sure.</p>"
        )

    parts.append(
        _bucket_skill_table(
            d.cal_by_minutes,
            "Skill by time to close",
            "Minutes left",
            "The Asian-settlement correction is largest in the final minutes, so that is "
            "where the model should beat the market by the most. If it does not, the edge "
            "is not coming from where the theory says it should.",
        )
    )
    parts.append(
        _bucket_skill_table(
            d.cal_by_price,
            "Skill by price bucket",
            "Price",
            "Taker fees peak at 50c, so skill in the middle of the ladder has to be much "
            "larger to be worth crossing the spread for.",
        )
    )

    meta = d.cal_meta
    if meta:
        facts = []
        if "n_observations" in meta:
            facts.append(f'{int(_as_float(meta["n_observations"])):,} observations')
        if "n_events" in meta:
            facts.append(f'{int(_as_float(meta["n_events"])):,} events')
        if "n_dropped" in meta:
            facts.append(f'{int(_as_float(meta["n_dropped"])):,} dropped by screening')
        oos = d.out_of_sample
        if oos is True:
            facts.append("out-of-sample")
        elif oos is False:
            facts.append("IN-SAMPLE (upper bound, not a backtest)")
        if facts:
            parts.append(f'<p class="sub">{_esc(" · ".join(facts))}</p>')
    if d.calibration_source:
        parts.append(f'<p class="sub mono">source: {_esc(d.calibration_source)}</p>')
    parts.append("</section>")
    return "".join(parts)


def _bucket_skill_table(
    buckets: Sequence[dict[str, Any]], title: str, first_col: str, blurb: str
) -> str:
    """Render one BucketScore breakdown. Thin buckets are greyed out, not hidden."""
    rows = [b for b in buckets if int(_as_float(b.get("n"))) > 0]
    if not rows:
        return ""
    out = [f"<h3>{_esc(title)}</h3>", f'<p class="sub">{blurb}</p>', '<div class="scroll"><table>']
    out.append(
        f"<thead><tr><th>{_esc(first_col)}</th><th>Observations</th>"
        "<th>Model Brier</th><th>Market Brier</th><th>Skill</th></tr></thead><tbody>"
    )
    for b in rows:
        n = int(_as_float(b.get("n")))
        skill = _as_float(b.get("skill"), default=float("nan"))
        reliable = bool(b.get("reliable", True))
        model_b = _as_float((b.get("model") or {}).get("brier"), default=float("nan"))
        market_b = _as_float((b.get("market") or {}).get("brier"), default=float("nan"))
        cls = "muted" if not reliable else ("pos" if skill > 0 else "neg")
        suffix = "" if reliable else ' <span class="muted">(too thin)</span>'
        out.append(
            f'<tr><td>{_esc(b.get("label", "?"))}</td><td>{n:,}</td>'
            f"<td>{model_b:.5f}</td><td>{market_b:.5f}</td>"
            f'<td class="{cls}">{skill:+.4f}{suffix}</td></tr>'
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _pnl_section(d: ReportData) -> str:
    parts = ['<section id="pnl">']
    parts.append(
        '<div class="section-head"><h2>P&amp;L attribution</h2>'
        '<div class="sub">Where the money actually came from, per event and per price '
        "bucket. Fees are quadratic in price, so the price bucket is also the fee "
        "bucket.</div></div>"
    )
    if not d.per_event:
        parts.append(_empty_chart("No settled trades yet."))
        parts.append("</section>")
        return "".join(parts)

    parts.append("<h3>By price bucket</h3>")
    parts.append(_bucket_chart(d.per_bucket))

    parts.append("<h3>By event</h3>")
    parts.append('<div class="scroll"><table><thead><tr>')
    parts.append(
        "<th>Event</th><th>Closed</th><th>Contracts</th><th>Gross</th>"
        "<th>Fees</th><th>Net</th></tr></thead><tbody>"
    )
    # Most recent first: that is what you actually want to look at after a session.
    for e in sorted(d.per_event, key=lambda x: (x.close_time or datetime.min.replace(tzinfo=UTC)), reverse=True)[:60]:
        cls = "pos" if e.net > 0 else ("neg" if e.net < 0 else "muted")
        when = e.close_time.strftime("%d %b %H:%M") if e.close_time else "—"
        parts.append(
            f'<tr><td class="mono">{_esc(e.event)}</td><td class="mono">{_esc(when)}</td>'
            f"<td>{e.contracts:g}</td><td>{_esc(_fmt_money(e.gross))}</td>"
            f"<td>{_esc(_fmt_money(e.fees))}</td>"
            f'<td class="{cls}">{_esc(_fmt_money(e.net))}</td></tr>'
        )
    parts.append("</tbody><tfoot><tr><td>Total</td><td></td>")
    parts.append(f"<td>{sum(e.contracts for e in d.per_event):g}</td>")
    parts.append(f"<td>{_esc(_fmt_money(d.gross_pnl))}</td>")
    parts.append(f"<td>{_esc(_fmt_money(sum(e.fees for e in d.per_event)))}</td>")
    cls = "pos" if d.net_pnl > 0 else ("neg" if d.net_pnl < 0 else "muted")
    parts.append(f'<td class="{cls}">{_esc(_fmt_money(d.net_pnl))}</td></tr></tfoot></table></div>')
    if len(d.per_event) > 60:
        parts.append(f'<p class="sub">Showing the 60 most recent of {len(d.per_event)} events.</p>')
    parts.append("</section>")
    return "".join(parts)


def _execution_section(d: ReportData) -> str:
    fq = d.fill_quality
    parts = ['<section id="execution">']
    parts.append(
        '<div class="section-head"><h2>Fill quality &amp; fee drag</h2>'
        '<div class="sub">Maker fills on KXBTCD are free. Taker fills cost '
        "0.07&thinsp;&times;&thinsp;P&thinsp;&times;&thinsp;(1&minus;P) per contract — about "
        "1.75&cent; at the money. Maker share is therefore the single biggest lever on net "
        "returns.</div></div>"
    )
    if fq.total_contracts <= 0:
        parts.append(_empty_chart("No fills recorded yet."))
        parts.append("</section>")
        return "".join(parts)

    parts.append('<div class="scroll"><table><thead><tr>')
    parts.append(
        "<th>Liquidity</th><th>Contracts</th><th>Share</th><th>Fees paid</th>"
        "<th>Fee per contract</th></tr></thead><tbody>"
    )
    for label, n, fees in (
        ("Maker (resting)", fq.maker_contracts, fq.maker_fees),
        ("Taker (crossing)", fq.taker_contracts, fq.taker_fees),
    ):
        share = n / fq.total_contracts if fq.total_contracts else 0.0
        per = fees / n if n else 0.0
        parts.append(
            f"<tr><td>{_esc(label)}</td><td>{n:g}</td><td>{_fmt_pct(share)}</td>"
            f"<td>{_esc(_fmt_money(fees))}</td><td>${per:.4f}</td></tr>"
        )
    parts.append("</tbody><tfoot><tr><td>Total</td>")
    parts.append(f"<td>{fq.total_contracts:g}</td><td>100.0%</td>")
    parts.append(f"<td>{_esc(_fmt_money(fq.maker_fees + fq.taker_fees))}</td><td></td>")
    parts.append("</tr></tfoot></table></div>")

    if fq.maker_fees > 1e-9:
        parts.append(
            '<div class="warn"><strong>Maker fills show a non-zero fee.</strong> '
            "KXBTCD maker fees are zero, so either the liquidity flag is wrong or the fee "
            "field is being filled from the wrong place. Fix this before trusting any P&amp;L "
            "number on this page.</div>"
        )

    parts.append("<h3>Fee drag</h3>")
    drag = d.total_fees / abs(d.gross_pnl) if abs(d.gross_pnl) > 1e-9 else None
    per_event_fee = d.total_fees / d.events_traded if d.events_traded else 0.0
    parts.append('<div class="tiles">')
    parts.append(_tile("Total fees", _esc(_fmt_money(d.total_fees)), "all taker; maker is free"))
    parts.append(_tile("Fees / gross P&L", _fmt_pct(drag) if drag is not None else "—",
                       "how much of the edge the exchange took"))
    parts.append(_tile("Fees per event", _esc(_fmt_money(per_event_fee))))
    parts.append("</div>")

    parts.append("<h3>Fill price vs mid at decision time</h3>")
    if fq.slippage_samples:
        cls = "neg" if (fq.mean_slippage_cents or 0) > 0 else "pos"
        parts.append('<div class="tiles">')
        parts.append(
            _tile("Mean slippage", f"{fq.mean_slippage_cents:+.3f}&cent;", "positive = we paid up", cls)
        )
        parts.append(_tile("Median slippage", f"{fq.median_slippage_cents:+.3f}&cent;"))
        parts.append(_tile("Samples", f"{fq.slippage_samples:,}"))
        parts.append("</div>")
        parts.append(
            '<p class="sub">A resting maker order that fills should show <em>negative</em> '
            "slippage: we got filled better than the mid because someone crossed to us. "
            "Persistently positive slippage on maker fills means adverse selection — we are "
            "being picked off just before the price moves.</p>"
        )
    else:
        parts.append(
            '<p class="sub">Not available. This needs either a mid-at-decision column on the '
            "fills table, or captured order books covering the moments we traded.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def _capture_section(d: ReportData) -> str:
    c = d.capture
    parts = ['<section id="capture">']
    parts.append(
        '<div class="section-head"><h2>Data capture</h2>'
        '<div class="sub">Kalshi order book history cannot be bought or backfilled. '
        "Every hour the recorder is not running is data that is gone forever.</div></div>"
    )
    parts.append('<div class="tiles">')
    parts.append(_tile("Book snapshots", f"{c.book_rows:,}"))
    parts.append(
        _tile(
            "BRTI ticks",
            f"{c.brti_rows:,}",
            "licensed feed — needs an API key" if not c.brti_rows else "the real index",
        )
    )
    parts.append(
        _tile("Spot proxy rows", f"{c.spot_rows:,}", "public Coinbase/Kraken/Bitstamp")
    )
    parts.append(_tile("Events seen", f"{c.distinct_events:,}"))
    parts.append(_tile("Settled events", f"{c.settled_events:,}", "with expiration_value"))
    parts.append("</div>")
    if c.first_seen and c.last_seen:
        hours = (c.last_seen - c.first_seen).total_seconds() / 3600.0
        parts.append(
            f'<p class="sub">Coverage: <span class="mono">'
            f'{_esc(c.first_seen.strftime("%Y-%m-%d %H:%M UTC"))}</span> to '
            f'<span class="mono">{_esc(c.last_seen.strftime("%Y-%m-%d %H:%M UTC"))}</span> '
            f"({hours:,.1f} hours wall clock).</p>"
        )
    elif c.book_rows == 0:
        parts.append(
            '<div class="warn">Nothing captured yet. Start the recorder now and leave it '
            "running: <span class='mono'>kbtc capture</span>. It needs no API key.</div>"
        )
    if d.db_path:
        parts.append(f'<p class="sub mono">db: {_esc(d.db_path)}</p>')
    parts.append("</section>")
    return "".join(parts)


def render_html(d: ReportData) -> str:
    """Render a complete, self-contained HTML document."""
    badge = (
        '<span class="badge armed">ARMED · real money</span>'
        if d.armed
        else '<span class="badge paper">paper · no money at risk</span>'
    )
    period = "no data yet"
    if d.period_start and d.period_end:
        period = (
            f'{d.period_start.strftime("%d %b %Y %H:%M")} — '
            f'{d.period_end.strftime("%d %b %Y %H:%M")} UTC'
        )

    net_cls = "pos" if d.net_pnl > 0 else ("neg" if d.net_pnl < 0 else "muted")
    drag = d.total_fees / abs(d.gross_pnl) if abs(d.gross_pnl) > 1e-9 else None

    parts: list[str] = []
    parts.append('<div class="wrap">')
    parts.append("<header>")
    parts.append("<h1>Kalshi KXBTCD — performance &amp; calibration</h1>")
    parts.append(
        f'<div class="sub">{badge} &nbsp;·&nbsp; env <span class="mono">{_esc(d.env)}</span>'
        f' &nbsp;·&nbsp; bankroll {_esc(_fmt_money(d.bankroll))}</div>'
    )
    parts.append(f'<div class="sub">Period covered: {_esc(period)}</div>')
    parts.append("</header>")

    parts.append('<section id="summary"><h2>What this means</h2>')
    parts.append(f"<p>{_plain_english(d)}</p>")
    if not d.has_trades:
        parts.append(
            '<div class="warn"><strong>Fresh install.</strong> The P&amp;L and execution '
            "sections below are empty by design. The useful next steps are: leave "
            "<span class='mono'>kbtc capture</span> running, run "
            "<span class='mono'>kbtc settlements</span> once a day, and check the calibration "
            "section as data accumulates.</div>"
        )
    parts.append("</section>")

    parts.append('<div class="tiles">')
    parts.append(
        _tile(
            "Net P&L",
            _esc(_fmt_money(d.net_pnl)),
            "after fees, settled events only",
            net_cls,
        )
    )
    parts.append(
        _tile("Win rate", _fmt_pct(d.win_rate), f"{d.wins}W / {d.losses}L")
    )
    parts.append(_tile("Events traded", f"{d.events_traded:,}", "settled hourly events"))
    parts.append(
        _tile(
            "Fee drag",
            _esc(_fmt_money(d.total_fees)),
            f"{_fmt_pct(drag)} of gross" if drag is not None else "no gross P&amp;L yet",
        )
    )
    parts.append(
        _tile(
            "Max drawdown",
            _esc(_fmt_money(d.max_drawdown)),
            "worst peak-to-trough",
            "neg" if d.max_drawdown > 0 else "",
        )
    )
    parts.append("</div>")

    parts.append('<section id="equity">')
    parts.append(
        '<div class="section-head"><h2>Equity curve</h2>'
        '<div class="sub">Cumulative net P&amp;L, one point per settled event. Open '
        "positions are excluded rather than marked, so this line never flatters "
        "itself.</div></div>"
    )
    parts.append(_equity_chart(d.equity))
    parts.append("</section>")

    parts.append(_calibration_section(d))
    parts.append(_pnl_section(d))
    parts.append(_execution_section(d))
    parts.append(_capture_section(d))

    if d.notes:
        parts.append('<section id="notes"><h2>Notes &amp; warnings</h2><ul>')
        for n in d.notes:
            parts.append(f"<li>{_esc(n)}</li>")
        parts.append("</ul></section>")

    parts.append(
        f'<footer>Generated {_esc(d.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"))} · '
        "self-contained, no network required · <span class='mono'>kbtc report</span></footer>"
    )
    parts.append("</div>")

    # A complete standalone document. The charset declaration is not optional: without
    # it a file:// open or a plain static server will mangle every em dash and cent sign.
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>KXBTCD report — "
        f'{_esc(d.generated_at.strftime("%Y-%m-%d %H:%M"))} UTC</title>\n'
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        + "".join(parts)
        + "\n</body>\n</html>\n"
    )


# ======================================================================================
# Entry point
# ======================================================================================
def build_report(
    settings: Any | None = None,
    out_dir: Path | str | None = None,
    *,
    data: ReportData | None = None,
) -> Path:
    """Write the report and return the path to `latest.html`.

    A timestamped copy is kept alongside it so you can diff two sessions and see what
    changed, which is the only honest way to tell whether a model tweak helped.
    """
    if settings is None:
        from kalshi_btc.config import get_settings

        settings = get_settings()
    if data is None:
        data = gather(settings)

    out = Path(out_dir) if out_dir else Path("reports") / "out"
    out.mkdir(parents=True, exist_ok=True)

    doc = render_html(data)
    stamp = data.generated_at.strftime("%Y%m%d-%H%M%S")
    stamped = out / f"report-{stamp}.html"
    stamped.write_text(doc, encoding="utf-8")
    latest = out / "latest.html"
    latest.write_text(doc, encoding="utf-8")
    return latest
