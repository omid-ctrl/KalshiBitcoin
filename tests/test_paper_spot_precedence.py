"""Which BTC level the paper runner trades off, and why that ordering.

Three possible sources, ranked BRTI > public spot proxy > ladder inference:

  brti        the licensed CF Benchmarks index. It IS the settlement value, so it wins.
  spot-proxy  the free Coinbase/Kraken/Bitstamp composite. Tracks BRTI to within
              single-digit dollars - immaterial against a $100 strike gap, decisive
              inside the settlement minute.
  ladder      spot implied by the KXBTCD quotes themselves. Last resort, because pricing
              off it is near-circular: it recovers the market's own view.

The proxy rung is not cosmetic. Before it existed, a no-credentials session had only the
dispersion-gated ladder estimate to fall back on, and a real 45-second run against prod
skipped 23 of 23 cycles for "no trustworthy spot" and never simulated a single trade.
With the proxy wired in, the same command on the same event skipped 0 of 22.

NO NETWORK.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kalshi_btc.feed.kalshi_ws import BrtiTick
from kalshi_btc.runner.paper import PROXY_MAX_AGE_S, SpotState
from kalshi_btc.strategy.edge import SpotEstimate

T0 = datetime(2026, 7, 29, 1, 30, 0, tzinfo=UTC)
BRTI_PX = 63_900.0
PROXY_PX = 63_850.0
LADDER_PX = 63_700.0


def _brti(value: float = BRTI_PX) -> BrtiTick:
    return BrtiTick(
        index_id="BRTI", ts=T0, value=value, avg_60s=None, windowed_avg=None, tick_count=None
    )


def _estimate(usable: bool, value: float | None = LADDER_PX) -> SpotEstimate:
    return SpotEstimate(
        value=value,
        n_strikes=6,
        dispersion=40.0,
        dispersion_std=0.2 if usable else 1.05,
        usable=usable,
        reason="ok" if usable else "ladder disagrees with itself by 1.05 sigma",
    )


# ------------------------------------------------------------------ the no-credentials path
def test_proxy_is_used_when_there_is_no_brti():
    """The whole point: a keyless install still gets a tradable spot."""
    s = SpotState()
    s.update_from_proxy(PROXY_PX, T0)
    assert s.value == PROXY_PX
    assert s.source == "spot-proxy"


def test_proxy_beats_an_unusable_ladder_estimate():
    """This is the exact case that produced 23/23 skipped cycles against prod."""
    s = SpotState()
    s.update_from_proxy(PROXY_PX, T0)
    s.update_from_ladder(_estimate(usable=False), T0 + timedelta(seconds=2))
    assert s.value == PROXY_PX, "an unusable ladder must not clear a fresh observed price"
    assert s.source == "spot-proxy"


def test_proxy_also_beats_a_usable_ladder_estimate():
    """Anti-circularity: an independent price outranks the market's own view of itself."""
    s = SpotState()
    s.update_from_proxy(PROXY_PX, T0)
    s.update_from_ladder(_estimate(usable=True), T0 + timedelta(seconds=2))
    assert s.value == PROXY_PX
    assert s.source == "spot-proxy"


def test_the_ladder_estimate_is_still_recorded_when_the_proxy_wins():
    """We keep the losing opinion so an operator can see how far apart the two were."""
    s = SpotState()
    s.update_from_proxy(PROXY_PX, T0)
    est = _estimate(usable=True)
    s.update_from_ladder(est, T0 + timedelta(seconds=2))
    assert s.estimate is est


# ------------------------------------------------------------------------- staleness
def test_a_stale_proxy_yields_to_the_ladder():
    """A feed that has gone quiet is not a price. Fall through rather than trade stale."""
    s = SpotState()
    s.update_from_proxy(PROXY_PX, T0)
    later = T0 + timedelta(seconds=PROXY_MAX_AGE_S + 1)
    s.update_from_ladder(_estimate(usable=True), later)
    assert s.value == LADDER_PX
    assert s.source == "ladder"


def test_a_stale_proxy_and_an_unusable_ladder_means_no_spot_at_all():
    """Refusing to trade is the correct outcome; a stale level is the failure mode."""
    s = SpotState()
    s.update_from_proxy(PROXY_PX, T0)
    later = T0 + timedelta(seconds=PROXY_MAX_AGE_S + 1)
    s.update_from_ladder(_estimate(usable=False), later)
    assert s.value is None, "must clear rather than leave a stale price in place"


# ------------------------------------------------------------------------------ BRTI
def test_brti_outranks_the_proxy():
    """BRTI is the settlement index; the proxy only tracks it."""
    s = SpotState()
    s.update_from_brti(_brti())
    s.update_from_proxy(PROXY_PX, T0 + timedelta(seconds=2))
    assert s.value == BRTI_PX
    assert s.source == "brti"


def test_a_stale_brti_yields_to_the_proxy():
    """If the licensed feed drops out we degrade to the public one rather than to nothing."""
    s = SpotState()
    s.update_from_brti(_brti())
    s.update_from_proxy(PROXY_PX, T0 + timedelta(seconds=20))
    assert s.value == PROXY_PX
    assert s.source == "spot-proxy"


def test_brti_outranks_the_ladder_too():
    s = SpotState()
    s.update_from_brti(_brti())
    s.update_from_ladder(_estimate(usable=True), T0 + timedelta(seconds=2))
    assert s.value == BRTI_PX
    assert s.source == "brti"
