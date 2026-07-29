"""Kalshi fee model for the KXBTCD hourly BTC series.

Verified against the official "Fee Schedule for July 2026 - 7.7.26 Update" PDF and
https://kalshi.com/fee-schedule on 2026-07-28, plus the live API
(GET /series/KXBTCD -> fee_type="quadratic", fee_multiplier=1) and
GET /series/fee_changes?series_ticker=KXBTCD -> empty (no override, scheduled or historical).

Verbatim from the schedule:

    fees = round up(M x 0.07 x C x P x (1-P))          <- taker
    fees = round up(M x 0.0175 x C x P x (1-P))        <- maker, M defaults to 0

    P = the price of a contract in dollars (50 cents is 0.5)
    C = the number of contracts being traded
    M = the multiplier for each contract (default is 1 unless otherwise indicated)
    round up = rounds up such that the fee + positionCost is rounded to a centicent

    "Settlement Fees - There is no settlement fee."

KXBTCD is absent from the Non-Standard Fees table, so it takes the plain standard
schedule: taker multiplier 1, maker multiplier 0 (i.e. MAKER FILLS ARE FREE), and no
settlement fee. That asymmetry is the single most important economic fact about this
market and it is why the strategy is maker-first.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

# A centicent is $0.0001 - the granularity fees round up to.
CENTICENT = Decimal("0.0001")

TAKER_RATE = Decimal("0.07")
MAKER_RATE = Decimal("0.0175")

# Multipliers for KXBTCD specifically (verified: no per-series override exists).
KXBTCD_TAKER_MULTIPLIER = Decimal("1")
KXBTCD_MAKER_MULTIPLIER = Decimal("0")


def _round_up_centicent(x: Decimal) -> Decimal:
    return x.quantize(CENTICENT, rounding=ROUND_CEILING)


def taker_fee(price: Decimal, contracts: Decimal, multiplier: Decimal = KXBTCD_TAKER_MULTIPLIER) -> Decimal:
    """Total taker fee in dollars for `contracts` at `price` (dollars, 0..1)."""
    if contracts <= 0:
        return Decimal("0")
    raw = multiplier * TAKER_RATE * contracts * price * (Decimal("1") - price)
    return _round_up_centicent(raw)


def maker_fee(price: Decimal, contracts: Decimal, multiplier: Decimal = KXBTCD_MAKER_MULTIPLIER) -> Decimal:
    """Total maker fee. Zero for KXBTCD - resting orders that fill cost nothing."""
    if contracts <= 0 or multiplier == 0:
        return Decimal("0")
    raw = multiplier * MAKER_RATE * contracts * price * (Decimal("1") - price)
    return _round_up_centicent(raw)


def settlement_fee(contracts: Decimal) -> Decimal:
    """There is no settlement fee on Kalshi."""
    return Decimal("0")


def taker_fee_per_contract(price: Decimal) -> Decimal:
    """Unrounded per-contract taker cost - the right quantity for edge thresholds."""
    return TAKER_RATE * price * (Decimal("1") - price)


def min_taker_edge(price: Decimal, half_spread: Decimal = Decimal("0.005")) -> Decimal:
    """Minimum model edge (in dollars per contract) for a taker trade to be +EV.

    Buy-and-hold-to-settlement pays the taker fee once and crosses half the spread.
    There is no settlement fee, so nothing else is charged.

    At P=0.50 this is ~2.25 cents; at P=0.10 it is ~1.13 cents. This asymmetry is why
    the bot trades away from the money far more often than at it.
    """
    return taker_fee_per_contract(price) + half_spread
