"""`kbtc calibrate` — score the pricing model against the market on captured data.

WHY THIS IS A SEPARATE RUNNER
-----------------------------
`model/calibration.py` deliberately exposes primitives rather than a pipeline: a
`ProbabilityModel` protocol whose signature makes lookahead impossible, a
`build_observations` choke point that never sees a settlement, and a `Calibrator` that
attaches outcomes afterwards. This module is the plumbing that connects those primitives
to what `kbtc capture` actually wrote, and nothing more. Keeping it out of the model
package is what stops store-schema details leaking into the scoring code.

THE ONE DESIGN DECISION THAT MATTERS
------------------------------------
The volatility model is fitted on realised settlements, and those same settlements decide
the outcomes we score against. Scoring the hours you fitted on is not a backtest, it is a
memory test. So we split: the vol model is fitted on the OLDER portion of the settlement
history, and only observations strictly after that cutoff are scored. `train_cutoff` is
handed to the Calibrator, which enforces the split itself.

If there is not enough history to split, we fall back to a constant unconditional sigma.
Nothing is fitted in that case, so nothing is contaminated — but the report says so
rather than quietly presenting it as a validated result.

Output: `CalibrationResult.as_dict()` written to `<DATA_DIR>/calibration.json`, which is
what `kbtc report` reads. The JSON is the contract; this module's internals are not.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kalshi_btc.config import Settings

log = logging.getLogger(__name__)

CALIBRATION_FILENAME = "calibration.json"

# Fraction of the settlement history reserved for fitting the vol model. The remainder is
# scored. Half is a compromise: enough hours to fit a HAR-RV, enough left to mean anything.
TRAIN_FRACTION = 0.5

# A BRTI tick older than this is not a usable spot for the snapshot it is joined to.
MAX_SPOT_STALENESS_SECONDS = 120.0


@dataclass(frozen=True)
class CalibrationSummary:
    """The handful of numbers `kbtc calibrate` prints. The JSON has everything else."""

    n_observations: int
    n_events: int
    n_dropped: int
    skill: float
    model_brier: float
    market_brier: float
    out_of_sample: bool
    output_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": self.n_observations,
            "events": self.n_events,
            "dropped": self.n_dropped,
            "skill_vs_market": self.skill,
            "model_brier": self.model_brier,
            "market_brier": self.market_brier,
            "out_of_sample": self.out_of_sample,
            "written_to": str(self.output_path),
        }


def _find_db(data_dir: Path) -> Path | None:
    if not data_dir.exists():
        return None
    candidates = sorted(data_dir.glob("*.duckdb"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def _rows(con: Any, sql: str, params: list | None = None) -> list[tuple]:
    return con.execute(sql, params or []).fetchall()


def run_calibration(settings: Settings, days: int = 30) -> dict[str, Any]:
    """Score the model over the last `days` of captured ladders. Returns a summary dict.

    Raises RuntimeError with an actionable message when there is not enough data — this
    is a batch job an operator runs by hand, so a clear sentence beats a stack trace.
    """
    from kalshi_btc.model.calibration import Calibrator, build_observations
    from kalshi_btc.model.pricing import price_above
    from kalshi_btc.report.report import open_read_only

    data_dir = Path(settings.data_dir).expanduser()
    db = _find_db(data_dir)
    if db is None:
        raise RuntimeError(
            f"No capture database under {data_dir}. Run `kbtc capture` first — there is "
            "nothing to calibrate against."
        )

    con, warning = open_read_only(db)
    if warning:
        log.info("%s", warning)
    try:
        settlements, close_times = _load_settlements(con)
        if len(settlements) < 3:
            raise RuntimeError(
                f"Only {len(settlements)} settled event(s) on file. Run `kbtc settlements` "
                "and let `kbtc capture` collect at least a few days of hours."
            )

        vol, cutoff, fitted = _fit_vol(close_times, settlements)
        records = _load_records(con, days)
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not records:
        raise RuntimeError(
            f"No usable ladder snapshots in the last {days} days. Each observation needs a "
            "quoted mid AND a BRTI value within "
            f"{MAX_SPOT_STALENESS_SECONDS:.0f}s — check that the cfbenchmarks_value feed is "
            "actually being recorded (`kbtc report` shows the BRTI tick count)."
        )

    # The protocol has no settlement parameter, so this closure cannot peek even by accident.
    def pricer(*, ts: datetime, spot: float, strike: float, minutes_to_close: float) -> float:
        return price_above(spot, strike, vol.sigma_per_minute(ts), minutes_to_close).prob_above

    observations = build_observations(records, pricer)
    if not observations:
        raise RuntimeError(
            "Every captured snapshot was already at or past its close time; nothing to "
            "forecast. This usually means only settlement-window rows were recorded."
        )

    result = Calibrator().score(observations, settlements, train_cutoff=cutoff)

    payload = result.as_dict()
    payload["generated_at"] = datetime.now(UTC).isoformat()
    payload["vol_fitted"] = fitted
    payload["vol_describe"] = vol.describe()
    payload["source_db"] = str(db)
    payload["days_requested"] = days

    out_path = data_dir / CALIBRATION_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    summary = CalibrationSummary(
        n_observations=result.n_observations,
        n_events=result.n_events,
        n_dropped=result.n_dropped,
        skill=result.skill,
        model_brier=result.model.brier,
        market_brier=result.market.brier,
        out_of_sample=result.is_out_of_sample,
        output_path=out_path,
    )
    log.info("%s", result.headline())
    return summary.as_dict()


def _load_settlements(con: Any) -> tuple[dict[str, float], list[tuple[datetime, float]]]:
    """event_ticker -> realised expiration_value, plus an ascending (close_time, value) list.

    Settlement is a property of the event: all 188 strikes under one hour share the same
    expiration_value, which is why the store keys it that way and why the Calibrator
    expects an event-keyed map.
    """
    settlements: dict[str, float] = {}
    series: list[tuple[datetime, float]] = []
    rows = _rows(
        con,
        "SELECT event_ticker, CAST(close_time AS VARCHAR), expiration_value "
        "FROM settlements WHERE expiration_value IS NOT NULL ORDER BY close_time",
    )
    for event, close_raw, value in rows:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        settlements[str(event)] = v
        when = _parse_ts(close_raw)
        if when is not None:
            series.append((when, v))
    return settlements, series


def _fit_vol(
    series: list[tuple[datetime, float]], settlements: dict[str, float]
) -> tuple[Any, datetime | None, bool]:
    """Fit the vol model on the older half of the settlement history.

    Returns (model, train_cutoff, fitted). `train_cutoff` is None when nothing was
    fitted, which is the honest signal that there is no train/test split to speak of.
    """
    from kalshi_btc.model.vol import VolModel

    if len(series) < 8:
        # Too short to split meaningfully. A constant sigma is not fitted to anything,
        # so it cannot be contaminated — but it is also not a validated model.
        return VolModel(), None, False

    split = max(4, int(len(series) * TRAIN_FRACTION))
    split = min(split, len(series) - 2)  # always leave something to score
    train = series[:split]
    cutoff = train[-1][0]

    model = VolModel()
    try:
        model.fit_settlements([t for t, _ in train], [v for _, v in train])
    except Exception as exc:
        log.warning("vol fit failed (%s); falling back to the unconditional sigma", exc)
        return VolModel(), None, False
    return model, cutoff, True


def _load_records(con: Any, days: int) -> list[Any]:
    """Build LadderRecords by ASOF-joining captured ladders to the BRTI tape.

    `minutes_to_close` is recorded on every snapshot, so the close time is reconstructed
    exactly rather than guessed from the ticker. Snapshots with no fresh BRTI value are
    dropped: a model priced off a stale spot is not the model we intend to score.
    """
    from kalshi_btc.model.calibration import LadderRecord

    since = datetime.now(UTC) - timedelta(days=max(1, days))
    sql = """
        SELECT
            CAST(l.ts AS VARCHAR)                          AS ts,
            l.event_ticker,
            l.ticker,
            l.strike,
            COALESCE(b.windowed_avg, b.avg_60s, b.value)   AS spot,
            (l.yes_bid + l.yes_ask) / 2                    AS mid,
            l.minutes_to_close,
            date_diff('second', b.ts, l.ts)                AS spot_age
        FROM ladder_snapshots l
        ASOF JOIN brti b ON l.ts >= b.ts
        WHERE l.ts >= CAST(? AS TIMESTAMP)
          AND l.yes_bid IS NOT NULL AND l.yes_ask IS NOT NULL
          AND l.minutes_to_close > 0
    """
    rows = _rows(con, sql, [since.replace(tzinfo=None).isoformat(sep=" ")])

    out: list[Any] = []
    for ts_raw, event, ticker, strike, spot, mid, mtc, spot_age in rows:
        when = _parse_ts(ts_raw)
        if when is None:
            continue
        try:
            spot_f = float(spot)
            mid_f = float(mid)
            strike_f = float(strike)
            mtc_f = float(mtc)
        except (TypeError, ValueError):
            continue
        if spot_age is None or abs(float(spot_age)) > MAX_SPOT_STALENESS_SECONDS:
            continue
        if not (math.isfinite(spot_f) and spot_f > 0):
            continue
        # Pinned strikes carry no information and would dominate the Brier score with
        # forecasts everyone agrees on. The Calibrator screens these too; dropping them
        # here keeps the observation count honest.
        if not (0.0 < mid_f < 1.0):
            continue
        out.append(
            LadderRecord(
                ts=when,
                event_ticker=str(event),
                ticker=str(ticker),
                strike=strike_f,
                spot=spot_f,
                market_mid=mid_f,
                close_time=when + timedelta(minutes=mtc_f),
            )
        )
    return out


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    try:
        t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)
