"""Regression tests for two ways the live venue breaks a naive KXBTCD parser.

NO NETWORK. Every payload below is a trimmed copy of a response captured from the
production Kalshi REST API on 2026-07-29. Both quirks were found by running against
prod, and both were silent-but-total failures rather than loud ones, which is exactly
why they are pinned here.

Quirk 1: `expiration_value` is not always a number.
Quirk 2: an hourly event's open->close span is not an hour, so cadence must be inferred
         from strike spacing instead.
"""

from __future__ import annotations

from decimal import Decimal

from kalshi_btc.core.types import MarketSnapshot, dec_or_none
from kalshi_btc.exec.client import event_is_hourly, strike_spacing
from kalshi_btc.runner.paper import is_hourly_event


def _market(strike: float, **over: object) -> dict:
    """One KXBTCD market row, shaped like the live /events payload."""
    m: dict = {
        "ticker": f"KXBTCD-26JUL2822-T{strike:.2f}",
        "event_ticker": "KXBTCD-26JUL2822",
        "floor_strike": f"{strike:.2f}",
        # Measured live: ladders are built days-to-years ahead of the close, so this span
        # is nowhere near 60 minutes even for a genuinely hourly event.
        "open_time": "2026-07-24T21:00:00Z",
        "close_time": "2026-07-29T02:00:00Z",
        "status": "active",
    }
    m.update(over)
    return m


def _event(spacing: float, n: int = 60, ticker: str = "KXBTCD-26JUL2822", **over) -> dict:
    base = 60000.0
    return {
        "event_ticker": ticker,
        "markets": [_market(base + i * spacing, **over) for i in range(n)],
    }


# ============================================================ quirk 1: expiration_value
# Observed on 9 markets across 62 open events on 2026-07-29. On a *settled* scalar market
# this field is the realised BRTI 60-second average, but on some finalized rows the venue
# reuses it for the settlement OUTCOME instead, as a bare string.
NON_NUMERIC_EXPIRATION_VALUES = ["a", "Yes", "No"]


def test_non_numeric_expiration_value_parses_to_none_not_a_crash():
    """`dec()` raises InvalidOperation on these; that used to kill the whole ladder parse."""
    for raw in NON_NUMERIC_EXPIRATION_VALUES:
        assert dec_or_none(raw) is None, f"{raw!r} should degrade to None"


def test_market_snapshot_survives_a_poisoned_expiration_value():
    for raw in NON_NUMERIC_EXPIRATION_VALUES:
        snap = MarketSnapshot.from_api(
            _market(63900, ticker="KXBTCD-28JAN0121-T99499.99", expiration_value=raw)
        )
        # The rest of the row must still be readable - that is the whole point.
        assert snap.expiration_value is None
        assert snap.strike == Decimal("63900.00")


def test_a_numeric_expiration_value_is_still_parsed():
    """The guard must not throw away the real settlements it exists to protect."""
    snap = MarketSnapshot.from_api(_market(63900, expiration_value="63421.87"))
    assert snap.expiration_value == Decimal("63421.87")


def test_absent_and_empty_expiration_value_are_none():
    assert MarketSnapshot.from_api(_market(63900)).expiration_value is None
    assert MarketSnapshot.from_api(_market(63900, expiration_value="")).expiration_value is None


def test_one_poisoned_market_does_not_break_a_whole_ladder():
    """The live failure mode: 1 bad row in a 188-strike event took down event discovery."""
    ev = _event(100.0, n=20)
    ev["markets"][7]["expiration_value"] = "a"
    snaps = [MarketSnapshot.from_api(m) for m in ev["markets"]]
    assert len(snaps) == 20
    assert snaps[7].expiration_value is None


# ================================================================ quirk 2: hourly cadence
def test_strike_spacing_separates_the_three_cadences():
    """Measured live: hourly $100 (60 events), daily $250 (1), weekly $500 (1)."""
    assert strike_spacing(_event(100.0)) == Decimal("100")
    assert strike_spacing(_event(250.0)) == Decimal("250")
    assert strike_spacing(_event(500.0)) == Decimal("500")


def test_hourly_event_is_detected_despite_a_multi_day_open_to_close_span():
    """The regression itself: this event's span is 4+ days, and it IS the hourly one.

    Before the fix, `event_is_hourly` tested span == 60min and so matched 0 of 62 live
    open events, which made capture and paper conclude there were no hourly markets.
    """
    ev = _event(100.0)
    span_hours = 4 * 24 + 5
    assert span_hours > 1, "fixture must not accidentally be a 60-minute span"
    assert event_is_hourly(ev) is True


def test_daily_and_weekly_events_are_rejected():
    """These trade under the SAME series ticker and have a different variance profile."""
    assert event_is_hourly(_event(250.0, n=80)) is False
    assert event_is_hourly(_event(500.0, n=50)) is False


def test_a_genuine_sixty_minute_span_is_still_accepted():
    """We did not delete the real definition, we just stopped depending on it."""
    ev = _event(
        250.0, n=5, open_time="2026-07-29T01:00:00Z", close_time="2026-07-29T02:00:00Z"
    )
    assert event_is_hourly(ev) is True


def test_a_two_strike_stub_is_not_enough_to_infer_hourly():
    """A single $100 gap proves nothing about a ladder that has not been built yet."""
    assert event_is_hourly(_event(100.0, n=2)) is False


def test_event_with_no_markets_is_not_hourly():
    assert event_is_hourly({"event_ticker": "KXBTCD-26JUL2822", "markets": []}) is False


# ------------------------------------------------- the paper runner's stricter variant
def test_paper_requires_a_full_ladder_not_just_the_right_cadence():
    """Discovery only needs to name the instrument; trading needs a ladder that prices."""
    thin = _event(100.0, n=10)
    assert event_is_hourly(thin) is True, "cadence is unambiguous"
    assert is_hourly_event(thin) is False, "but 10 strikes is not tradable"
    assert is_hourly_event(_event(100.0, n=60)) is True


def test_paper_refuses_the_decoy_series():
    """KXBTC is the illiquid range series - one character from the one we want."""
    decoy = _event(100.0, ticker="KXBTC-26JUL2822")
    for m in decoy["markets"]:
        m["event_ticker"] = "KXBTC-26JUL2822"
    assert is_hourly_event(decoy) is False
