"""Paper trading: the full engine against LIVE markets, with no orders, ever.

WHAT THIS IS FOR
----------------
Everything up to here is either measurable in isolation (the pricer against Monte Carlo,
the vol model against realised settlements) or free (data capture). Paper trading is the
first place the pieces have to work together against a market that does not care about our
model. It answers exactly one question the unit tests cannot: does the edge survive real
spreads, real queues and real timing?

NO ORDER PATH EXISTS IN THIS MODULE
-----------------------------------
`KalshiClient.create_order` is never imported here and never called. That is deliberate
and it is the strongest guarantee available: not "we check a flag before sending", but
"there is no code in this file that could send". The `armed` flag is still recorded on
every decision row so the report can prove nothing was live.

THE FILL SIMULATOR IS THE WHOLE POINT, SO IT IS PESSIMISTIC ON PURPOSE
----------------------------------------------------------------------
A paper runner that fills optimistically produces a beautiful equity curve and no
information. Two rules, both of which cost us fills:

* TAKER fills cross the spread and pay the taker fee. We fill at the ask (buying) or the
  bid (selling), capped by the size actually resting there, and the fee is
  `fees.taker_fee`, rounded up to a centicent exactly as the venue does it.

* MAKER fills require the market to TRADE THROUGH our resting price, and we sit at the
  BACK of the queue we joined. Being at the front of a 2,000-contract queue is not a
  modelling simplification, it is fiction: on a locked 1-cent spread the queue IS the
  competition. So each resting order carries `queue_ahead`, we consume it out of observed
  traded volume, and only the remainder can fill us. Repricing resets queue position,
  because it does at the venue too.

  Consequence, stated up front so it is not mistaken for a bug: on a short run the
  realistic number of simulated maker fills is ZERO. That is the finding, not a failure.

WHERE SPOT COMES FROM WITHOUT CREDENTIALS
------------------------------------------
Real-time BRTI is licensed. With credentials we take it from the `cfbenchmarks_value`
channel, including the running settlement average inside the final minute, which is the
high-value path. Without credentials there is no BRTI at any price, so spot is inferred
from the ladder's own mids (`edge.estimate_spot_from_ladder`). That anchors the LEVEL of
our curve to the market and leaves only its SHAPE as signal — genuinely useful (a live
5-strike ladder on 2026-07-29 fitted our Student-t to within 1.7c and a Gaussian only to
2.7c) but it is relative value, not a view on BTC, and the heartbeat labels which mode is
running so the two never get confused.

That inference is only as good as the ladder underneath it, which is why the estimate
carries a usability verdict and a cycle with no trustworthy level trades nothing at all.
The first live session recorded 25-cent "edges"; every one came from a moment when the
ladder had collapsed to one strike quoted 14 cents wide.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from rich.console import Console

from kalshi_btc.config import DECOY_SERIES, SERIES_TICKER, Settings
from kalshi_btc.core.fees import taker_fee
from kalshi_btc.core.types import Action, Liquidity, MarketSnapshot, Side, _ts
from kalshi_btc.exec.client import KalshiClient, event_is_hourly
from kalshi_btc.feed.kalshi_ws import CH_BRTI, CH_ORDERBOOK, CH_TRADE, BrtiTick, KalshiWebSocket
from kalshi_btc.feed.spot_ws import DEFAULT_VENUES, SpotFeed
from kalshi_btc.model.vol import VolModel
from kalshi_btc.risk.killswitch import KillSwitch
from kalshi_btc.risk.limits import RiskManager
from kalshi_btc.store.db import Store
from kalshi_btc.strategy.edge import (
    MEASURED_SIGMA_PER_HOUR,
    FairValueEngine,
    LadderEdges,
    LadderQuote,
    SettlementWindow,
    SpotEstimate,
    estimate_spot_from_ladder,
)
from kalshi_btc.strategy.quoting import MakerQuoter, QuoteIntent, QuotePlan
from kalshi_btc.strategy.sizing import SizingCandidate, size_ladder

log = logging.getLogger(__name__)

SECONDS_PER_MINUTE = 60.0

# Fallback sigma when there is not enough settlement history to fit the vol model. This is
# the MEASURED level (0.466%/hour over 1,596 hourly settlements), not the library default.
FALLBACK_SIGMA_PER_MINUTE = MEASURED_SIGMA_PER_HOUR / SECONDS_PER_MINUTE**0.5

# Minimum settled hours before we trust a fitted vol model over the measured constant.
MIN_SETTLEMENTS_TO_FIT = 40

# Of the volume that prints at a price level, how much do we assume hit OUR side. We
# cannot see trade direction in the REST ladder (only cumulative volume), so half is the
# neutral assumption. It is applied to the QUEUE, not to our fill, so being wrong here
# delays fills rather than inventing them.
QUEUE_FLOW_SHARE = Decimal("0.5")


# Hourly events use $100 strike spacing; daily uses $250 and weekly $500. Measured live
# on 2026-07-29 and consistent with the contract specs. The spacing constant now lives in
# `exec.client` alongside the shared `event_is_hourly`, so there is exactly one definition
# of what "hourly" means; we only keep our own, much stricter, minimum-strike bar.
#
# Discovery accepts a 3-strike ladder because it only needs to name the instrument. We
# need to *trade* it, and a half-built ladder prices nothing, so we demand a full one.
MIN_HOURLY_STRIKES = 50
# An hourly event closes every hour, so the soonest-closing one is never further away than
# that. Anything beyond means the discriminator picked the wrong instrument.
MAX_CURRENT_EVENT_MINUTES = 75.0

# How long a public spot-proxy print stays authoritative. The feed itself only publishes a
# composite when >=2 venues are fresh within 5s and agree to 5bps, so anything older than
# this is a feed that has gone quiet rather than a price - fall through to the ladder.
PROXY_MAX_AGE_S = 10.0


# ======================================================================================
# Event discovery
# ======================================================================================
def _event_close(event: dict) -> datetime | None:
    """Close time from the raw payload, without building a MarketSnapshot.

    Deliberately does NOT go through `MarketSnapshot.from_api`: on 2026-07-29 the live
    /events response contained an open KXBTCD event whose `expiration_value` was the
    string "a", which makes `dec()` raise InvalidOperation and takes the whole discovery
    call down. Discovery must survive one poisoned row in a list of twenty.
    """
    markets = event.get("markets") or []
    if not markets:
        return None
    raw = markets[0].get("close_time")
    return _ts(raw) if raw else None


def is_hourly_event(event: dict) -> bool:
    """True iff this is a KXBTCD hourly event with a ladder deep enough to trade.

    Cadence itself is decided by the shared `client.event_is_hourly`, which leads on
    strike spacing because the live venue opens ladders days-to-years ahead of the close
    and so the open->close span test matched 0 of 62 open events on 2026-07-29.

    On top of that we add the series guard (KXBTC, the illiquid range series, is a decoy
    whose name is one character away) and our own depth requirement.
    """
    ticker = event.get("event_ticker", "")
    if ticker.startswith(f"{DECOY_SERIES}-") or not ticker.startswith(f"{SERIES_TICKER}-"):
        return False
    if not event_is_hourly(event):
        return False
    return len(event.get("markets") or []) >= MIN_HOURLY_STRIKES


async def discover_hourly_event(client: KalshiClient) -> dict | None:
    """The KXBTCD hourly event we should be trading right now, or None.

    Kalshi lists many open events at once; the current one is simply the soonest-closing
    HOURLY event whose close is still ahead of us. We log loudly if that turns out to be
    more than ~an hour away, because it means the cadence filter let something through.
    """
    events = await client.get_events(SERIES_TICKER, status="open", limit=20)
    now = datetime.now(UTC)

    candidates: list[tuple[datetime, dict]] = []
    for e in events:
        if not is_hourly_event(e):
            continue
        close = _event_close(e)
        if close is None or close <= now:
            continue
        candidates.append((close, e))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    close, event = candidates[0]
    minutes = (close - now).total_seconds() / SECONDS_PER_MINUTE
    if minutes > MAX_CURRENT_EVENT_MINUTES:
        log.warning(
            "soonest hourly %s closes in %.0f minutes - that is not an hourly cadence; "
            "refusing to trade it",
            event.get("event_ticker"),
            minutes,
        )
        return None
    return event


# ======================================================================================
# Simulated execution
# ======================================================================================
@dataclass
class RestingOrder:
    """One simulated passive order and its position in the queue.

    `queue_ahead` starts at the full displayed size at our price: everyone already there
    is in front of a new order. It only ever decreases, and repricing throws it away.
    """

    ticker: str
    side: Side
    action: Action
    price: Decimal
    contracts: int
    queue_ahead: Decimal
    placed_at: datetime
    filled: int = 0

    @property
    def remaining(self) -> int:
        return self.contracts - self.filled

    @property
    def key(self) -> tuple[str, str, Decimal]:
        return (self.ticker, self.action.value, self.price)


@dataclass
class SimulatedFill:
    ticker: str
    side: Side
    action: Action
    price: Decimal
    contracts: int
    liquidity: Liquidity
    fee: Decimal
    ts: datetime
    reason: str


@dataclass
class FillSimulator:
    """Queue-aware fill model. See the module docstring for why it is this pessimistic."""

    resting: dict[tuple[str, str, Decimal], RestingOrder] = field(default_factory=dict)
    last_volume: dict[str, Decimal] = field(default_factory=dict)
    maker_fills: int = 0
    taker_fills: int = 0
    queue_consumed: Decimal = Decimal("0")
    cancels: int = 0

    # ---------------------------------------------------------------- passive orders
    def reconcile(self, plan: QuotePlan, now: datetime) -> int:
        """Make the resting book match the plan. Returns how many orders were cancelled.

        An order whose price is unchanged KEEPS its queue position; anything else is
        cancelled and replaced, losing priority. That asymmetry is why a quoter that
        rewrites its quotes every cycle never fills, and it must be visible in the sim.
        """
        wanted: dict[tuple[str, str, Decimal], QuoteIntent] = {
            (i.ticker, i.action.value, i.price): i for i in plan.intents
        }
        cancelled = 0
        for key in list(self.resting):
            if key not in wanted:
                del self.resting[key]
                cancelled += 1
        self.cancels += cancelled

        for key, intent in wanted.items():
            existing = self.resting.get(key)
            if existing is not None:
                # Same price: keep our place in line, only adjust the clip size.
                existing.contracts = max(existing.filled, intent.contracts)
                continue
            self.resting[key] = RestingOrder(
                ticker=intent.ticker,
                side=intent.side,
                action=intent.action,
                price=intent.price,
                contracts=intent.contracts,
                queue_ahead=intent.queue_ahead,
                placed_at=now,
            )
        return cancelled

    def on_market_update(
        self, markets: dict[str, MarketSnapshot], now: datetime
    ) -> list[SimulatedFill]:
        """Advance every resting order against observed volume and price moves."""
        fills: list[SimulatedFill] = []
        for key, order in list(self.resting.items()):
            m = markets.get(order.ticker)
            if m is None:
                continue

            traded = self._volume_delta(m)
            flow = traded * QUEUE_FLOW_SHARE

            # A trade THROUGH our price fills us regardless of queue: nobody trades past
            # resting size at a better price.
            through = self._traded_through(order, m)
            if through:
                flow = max(flow, Decimal(order.remaining) + order.queue_ahead)

            if flow <= 0:
                continue

            eaten = min(order.queue_ahead, flow)
            order.queue_ahead -= eaten
            self.queue_consumed += eaten
            flow -= eaten
            if flow <= 0 or order.queue_ahead > 0:
                continue

            n = int(min(Decimal(order.remaining), flow))
            if n <= 0:
                continue
            order.filled += n
            self.maker_fills += 1
            fills.append(
                SimulatedFill(
                    ticker=order.ticker,
                    side=order.side,
                    action=order.action,
                    price=order.price,
                    contracts=n,
                    liquidity=Liquidity.MAKER,
                    fee=Decimal("0"),  # maker multiplier is 0 on KXBTCD
                    ts=now,
                    reason="queue cleared" if not through else "traded through",
                )
            )
            if order.remaining <= 0:
                del self.resting[key]
        return fills

    def _volume_delta(self, m: MarketSnapshot) -> Decimal:
        """Contracts printed since the last poll. First sight of a market counts as zero."""
        prev = self.last_volume.get(m.ticker)
        self.last_volume[m.ticker] = m.volume
        if prev is None or m.volume < prev:
            return Decimal("0")
        return m.volume - prev

    @staticmethod
    def _traded_through(order: RestingOrder, m: MarketSnapshot) -> bool:
        """True when the touch has moved past our resting price.

        Buying at 0.42 when the best ASK has dropped to 0.42 or below means someone was
        willing to sell there, and on a FIFO book at a locked spread that flow reaches us.
        """
        if order.action is Action.BUY:
            return m.yes_ask <= order.price < Decimal("1")
        return m.yes_bid >= order.price > Decimal("0")

    # ---------------------------------------------------------------- aggressive orders
    def take(
        self,
        *,
        ticker: str,
        action: Action,
        price: Decimal,
        contracts: int,
        available: Decimal,
        now: datetime,
        reason: str,
    ) -> SimulatedFill | None:
        """Cross the spread. Capped by the size actually resting on the other side."""
        n = int(min(Decimal(contracts), available)) if available > 0 else 0
        if n <= 0:
            return None
        self.taker_fills += 1
        return SimulatedFill(
            ticker=ticker,
            side=Side.YES,
            action=action,
            price=price,
            contracts=n,
            liquidity=Liquidity.TAKER,
            fee=taker_fee(price, Decimal(n)),
            ts=now,
            reason=reason,
        )

    @property
    def n_resting(self) -> int:
        return len(self.resting)

    @property
    def total_queue_ahead(self) -> Decimal:
        return sum((o.queue_ahead for o in self.resting.values()), Decimal("0"))


# ======================================================================================
# Session state
# ======================================================================================
@dataclass
class PaperCounters:
    """Everything the heartbeat shows, and the return value of run_paper()."""

    cycles: int = 0
    ladder_rows: int = 0
    spot_rows: int = 0
    decisions: int = 0
    quotes_placed: int = 0
    quotes_cancelled: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
    taker_signals: int = 0
    blocked_by_risk: int = 0
    skipped_cycles: int = 0
    pulled: int = 0
    events: int = 0
    halts: int = 0
    api_errors: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, int]:
        return {
            "cycles": self.cycles,
            "ladder_rows": self.ladder_rows,
            "decisions": self.decisions,
            "quotes_placed": self.quotes_placed,
            "quotes_cancelled": self.quotes_cancelled,
            "maker_fills": self.maker_fills,
            "taker_fills": self.taker_fills,
            "taker_signals": self.taker_signals,
            "blocked_by_risk": self.blocked_by_risk,
            "skipped_cycles": self.skipped_cycles,
            "pulled": self.pulled,
            "events": self.events,
            "halts": self.halts,
            "api_errors": self.api_errors,
        }


def _countdown(seconds: float) -> str:
    sign = "-" if seconds >= 0 else "+"
    s = int(abs(seconds))
    return f"T{sign}{s // 60:02d}:{s % 60:02d}"


def build_vol_model(store: Store) -> tuple[VolModel, str]:
    """Fit the vol model on captured settlements, or fall back to the measured constant.

    The fallback is the number measured on 1,596 real settlements rather than
    `vol.DEFAULT_SIGMA_PER_MINUTE`, because a paper session run before any capture should
    still be quoting off the best available estimate of THIS market's volatility.
    """
    try:
        df = store.settlement_series()
    except Exception as exc:  # noqa: BLE001 - a missing table must not stop the session
        log.warning("could not read settlements: %s", exc)
        df = None

    if df is not None and len(df) >= MIN_SETTLEMENTS_TO_FIT:
        try:
            model = VolModel().fit_settlements(
                list(df["close_time"]), df["expiration_value"].astype(float).to_numpy()
            )
            return model, f"fitted on {len(df)} settlements"
        except (ValueError, RuntimeError) as exc:
            log.warning("vol fit failed (%s); using the measured constant", exc)

    n = 0 if df is None else len(df)
    model = VolModel.constant(FALLBACK_SIGMA_PER_MINUTE)
    return model, f"measured constant ({n} settlements on file, need {MIN_SETTLEMENTS_TO_FIT})"


@dataclass
class SpotState:
    """Best available BTC level, and an honest label for where it came from."""

    value: float | None = None
    source: str = "none"
    ts: datetime | None = None
    window: SettlementWindow | None = None
    estimate: SpotEstimate | None = None

    def update_from_brti(self, tick: BrtiTick) -> None:
        val = tick.value or tick.avg_60s
        if val is not None:
            self.value = float(val)
            self.source = "brti"
            self.ts = tick.ts
        if tick.windowed_avg is not None and tick.tick_count:
            self.window = SettlementWindow.from_windowed_average(
                float(tick.windowed_avg),
                int(tick.tick_count),
                self.value or float(tick.windowed_avg),
            )
        else:
            self.window = None

    def update_from_proxy(self, value: float, ts: datetime) -> None:
        """Adopt the public Coinbase/Kraken/Bitstamp composite.

        Ranks BELOW a real BRTI tick and ABOVE a ladder inference, and both halves of that
        ordering are deliberate.

        Below BRTI because BRTI *is* the settlement index while this only tracks it - to
        within single-digit dollars in our measurements, which is immaterial against a
        $100 strike gap and decisive inside the settlement minute. Grade it for yourself
        with `kbtc proxy-score` before trusting it near the close.

        Above the ladder because the ladder inference is derived from the very quotes we
        are trying to form an opinion about. Pricing off it is close to circular: it
        recovers the market's own view and finds, unsurprisingly, no edge. An independently
        observed price from other venues is what makes the opinion a differentiated one.

        Without this path a no-credentials install has NO usable spot at all: BRTI never
        arrives and the ladder estimate is dispersion-gated, so a live 45s session skipped
        23 of 23 cycles for "no trustworthy spot" and never placed a single simulated
        trade. That is the entire credential-free path failing silently.
        """
        if self.source == "brti" and self.ts is not None and (ts - self.ts).total_seconds() < 15:
            return
        self.value = float(value)
        self.source = "spot-proxy"
        self.ts = ts

    def update_from_ladder(self, estimate: SpotEstimate, ts: datetime) -> None:
        """Adopt a ladder-implied level, or drop to no-spot when it is not trustworthy.

        Three rules. A real BRTI tick always wins, because letting a ladder inference
        overwrite a licensed price would be a silent downgrade. A FRESH public spot proxy
        also wins, for the anti-circularity reason in `update_from_proxy`. And an UNUSABLE
        estimate clears the value rather than leaving the previous one in place: a stale
        spot from two seconds ago looks exactly like a good one to everything downstream,
        and that is the failure this whole guard exists to prevent.
        """
        if self.source == "brti" and self.ts is not None and (ts - self.ts).total_seconds() < 15:
            return
        if (
            self.source == "spot-proxy"
            and self.ts is not None
            and (ts - self.ts).total_seconds() < PROXY_MAX_AGE_S
        ):
            # Keep the observed price, but still record the estimate so the operator can
            # see what the ladder thought and how far apart the two were.
            self.estimate = estimate
            return
        self.estimate = estimate
        if estimate.usable and estimate.value is not None:
            self.value = estimate.value
            self.source = "ladder"
            self.ts = ts
        else:
            self.value = None
            self.source = "none"


# ======================================================================================
# The loop
# ======================================================================================
async def _poll_ladder(
    client: KalshiClient, store: Store, event_ticker: str, strike_window: int | None
) -> tuple[list[MarketSnapshot], float, int]:
    """Record one ladder snapshot and return (markets, minutes_to_close, rows written)."""
    event = await client.get_event(event_ticker)
    raw = event.get("markets") or []
    if not raw:
        return [], 0.0, 0

    ts = datetime.now(UTC)
    markets: list[MarketSnapshot] = []
    for m in raw:
        try:
            markets.append(MarketSnapshot.from_api(m))
        except Exception as exc:  # noqa: BLE001 - one malformed strike is not an outage
            # The venue does serve junk: an open event carried expiration_value="a" on
            # 2026-07-29. Drop the row, keep the other 187.
            log.warning("skipping unparseable market %s: %s", m.get("ticker"), exc)
    if not markets:
        return [], 0.0, 0
    minutes_to_close = (markets[0].close_time - ts).total_seconds() / SECONDS_PER_MINUTE

    selected = markets
    if strike_window is not None:
        live = [m for m in markets if m.is_live] or markets
        atm = min(live, key=lambda m: abs(m.mid - Decimal("0.5")))
        ordered = sorted(markets, key=lambda m: m.strike)
        idx = min(range(len(ordered)), key=lambda i: abs(ordered[i].strike - atm.strike))
        selected = ordered[max(0, idx - strike_window) : idx + strike_window + 1]

    for m in selected:
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
    return markets, minutes_to_close, len(selected)


def _record_fill(
    store: Store, risk: RiskManager, fill: SimulatedFill, counters: PaperCounters, armed: bool
) -> None:
    """Book a simulated fill into both the risk manager and the store."""
    risk.record_fill(
        ticker=fill.ticker,
        action=fill.action,
        contracts=fill.contracts,
        price=fill.price,
        liquidity=fill.liquidity,
        fee=fill.fee,
        when=fill.ts,
    )
    store.add_fill(
        ts=fill.ts,
        ticker=fill.ticker,
        side=str(fill.side),
        action=str(fill.action),
        price=fill.price,
        count=Decimal(fill.contracts),
        liquidity=str(fill.liquidity),
        fee=fill.fee,
        order_id=f"paper-{fill.liquidity.value}",
    )
    if fill.liquidity is Liquidity.MAKER:
        counters.maker_fills += 1
    else:
        counters.taker_fills += 1
    store.add_decision(
        ts=fill.ts,
        ticker=fill.ticker,
        fair_prob=float(fill.price),
        market_mid=float(fill.price),
        edge=None,
        action=f"fill_{fill.action.value}",
        reason=f"{fill.liquidity.value} {fill.contracts}x @ {fill.price} ({fill.reason})",
        armed=armed,
    )
    counters.decisions += 1


def _heartbeat(
    console: Console,
    ladder: LadderEdges,
    plan: QuotePlan,
    sim: FillSimulator,
    risk: RiskManager,
    counters: PaperCounters,
    vol_source: str,
) -> None:
    """One dense line per cycle, in the same visual language as `kbtc capture`."""
    now = datetime.now(UTC).strftime("%H:%M:%S")
    secs = ladder.minutes_to_close * SECONDS_PER_MINUTE
    left_style = "bold red" if 0 <= secs <= 60 else "cyan"

    atm = ladder.atm
    atm_txt = "[dim]atm --[/]"
    if atm is not None:
        b = "--" if atm.yes_bid is None else f"{atm.yes_bid:.2f}"
        a = "--" if atm.yes_ask is None else f"{atm.yes_ask:.2f}"
        atm_txt = f"{atm.strike:,.0f} [green]{b}[/]/[red]{a}[/] f{atm.fair_prob:.2f}"

    spot_style = "bold yellow" if ladder.spot_source == "brti" else "dim"
    spot_txt = f"[{spot_style}]{ladder.spot_source[0]}{ladder.spot:,.0f}[/]"
    sigma_txt = f"[dim]s{ladder.sigma_per_minute * (60**0.5) * 100:.2f}%/h[/]"

    q = f"[cyan]q{plan.n_quotes}[/]" if plan.n_quotes else "[dim]q0[/]"
    rest = f"[blue]r{sim.n_resting}[/]" if sim.n_resting else "[dim]r0[/]"
    fills = counters.maker_fills + counters.taker_fills
    fill_txt = f"[bold green]F{fills}[/]" if fills else "[dim]F0[/]"
    take_txt = f" [magenta]take{counters.taker_signals}[/]" if counters.taker_signals else ""
    pnl = risk.daily_pnl()
    pnl_style = "green" if pnl >= 0 else "red"
    halt = " [bold red]HALTED[/]" if risk.killswitch.halted else ""

    console.print(
        f"[dim]{now}[/] [{left_style}]{_countdown(secs)}[/] {ladder.event_ticker[-9:]} "
        f"k[bold]{len(ladder.edges)}[/] {atm_txt} {spot_txt} {sigma_txt} "
        f"{q} {rest} {fill_txt}{take_txt} "
        f"[{pnl_style}]${pnl:+.2f}[/]"
        + (f" [yellow]blk{counters.blocked_by_risk}[/]" if counters.blocked_by_risk else "")
        + (f" [red]err{counters.api_errors}[/]" if counters.api_errors else "")
        + halt,
        highlight=False,
    )
    if counters.cycles == 1:
        console.print(f"[dim]vol: {vol_source} | {ladder.dist} df={ladder.df:.2f}[/]")


async def run_paper(
    settings: Settings,
    duration_s: float | None = None,
    *,
    hours: int | None = None,
    ladder_interval_s: float = 2.0,
    strike_window: int | None = 25,
    store: Store | None = None,
    console: Console | None = None,
    engine: FairValueEngine | None = None,
    quoter: MakerQuoter | None = None,
) -> dict[str, int]:
    """Run the strategy against live KXBTCD markets without ever sending an order.

    Stops after `duration_s` seconds, after `hours` completed events, or on Ctrl-C —
    whichever comes first. Every decision lands in `decisions` and every simulated fill in
    `fills`, which is what `kbtc report` and `kbtc calibrate` read.
    """
    console = console or Console()
    counters = PaperCounters()
    owns_store = store is None
    store = store or Store(settings)
    try:
        await store.start()
    except Exception as exc:  # noqa: BLE001 - translated to an operator sentence below
        # DuckDB allows exactly ONE writer process, and paper is a writer: it records
        # ladder snapshots, decisions and simulated fills. `kbtc capture` is also a
        # writer and the README tells you to leave it running forever, so operators hit
        # this collision on their first Phase 2 run. A raw IOException traceback does not
        # tell anyone what to do about it, and the two readers (`report`, `calibrate`)
        # already degrade gracefully, so this one should explain itself too.
        if "lock" not in str(exc).lower():
            raise
        raise RuntimeError(
            f"The capture database {store.path} is locked by another process — almost "
            "certainly `kbtc capture`. DuckDB allows only one writer, and `kbtc paper` "
            "is a writer too (it records ladder snapshots, decisions and simulated "
            "fills).\n\n"
            "Stop `kbtc capture` for the duration of the paper session. Paper records "
            "the ladder itself, so you keep collecting ladder history while it runs — "
            "you only lose the public spot-proxy rows, which paper does not write. "
            "Restart capture when the session ends.\n\n"
            "`kbtc report` and `kbtc calibrate` are readers and DO work while capture "
            "holds the lock; only the writers collide."
        ) from exc

    engine = engine or FairValueEngine()
    quoter = quoter or MakerQuoter(risk=settings.risk, refresh_seconds=ladder_interval_s)
    killswitch = KillSwitch()
    risk = RiskManager(risk=settings.risk, killswitch=killswitch)
    sim = FillSimulator()
    spot_state = SpotState()

    feed: KalshiWebSocket | None = None
    spot_feed: SpotFeed | None = None
    deadline = None if duration_s is None else time.monotonic() + duration_s

    console.print(f"[bold]kbtc paper[/] | {settings.describe()}")
    console.print(
        f"[dim]db {store.path} | cadence {ladder_interval_s}s | bankroll "
        f"${settings.risk.bankroll} | kelly {settings.risk.kelly_fraction} | "
        f"max {settings.risk.max_contracts_per_order}/order, "
        f"{settings.risk.max_position_per_strike}/strike[/]"
    )
    console.print("[bold yellow]NO ORDERS ARE SENT FROM THIS RUNNER. Fills are simulated.[/]")

    vol_model, vol_source = build_vol_model(store)

    try:
        async with KalshiClient(settings) as client:
            current = await discover_hourly_event(client)
            if current is None:
                console.print(
                    f"[red]no open HOURLY {SERIES_TICKER} event found - nothing to trade.[/] "
                    "[dim](KXBTCD also hosts daily and weekly events; those are a different "
                    "instrument and this runner will not touch them.)[/]"
                )
                return counters.as_dict()

            event_ticker = current["event_ticker"]
            counters.events = 1
            console.print(f"[bold green]paper trading[/] {event_ticker}")

            feed = KalshiWebSocket(
                settings, tickers=[], channels=(CH_ORDERBOOK, CH_TRADE, CH_BRTI)
            )
            feed.start()

            # The public spot proxy runs ALWAYS, credentials or not. It is the only
            # independent price a keyless install ever sees, and without it the ladder
            # inference is the sole spot source - which the dispersion gate rejects often
            # enough that a session can trade nothing at all.
            spot_feed = SpotFeed(DEFAULT_VENUES)
            spot_feed.start()
            await spot_feed.wait_ready(timeout=8.0, venues=2)
            console.print(f"[dim]spot proxy: {spot_feed.describe()}[/]")

            if not feed.available:
                console.print(
                    "[yellow]No credentials: no BRTI feed.[/] Spot comes from the public "
                    "venue composite above, falling back to the ladder when it goes stale. "
                    "The proxy tracks BRTI to within a few dollars — fine against a $100 "
                    "strike gap, [bold]not[/] fine inside the settlement minute. Grade it "
                    "with [bold]kbtc proxy-score[/]."
                )

            next_tick = time.monotonic()
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if hours is not None and counters.events > hours:
                    break

                minutes_to_close = 1.0
                try:
                    minutes_to_close = await _cycle(
                        client=client,
                        store=store,
                        feed=feed,
                        spot_feed=spot_feed,
                        engine=engine,
                        quoter=quoter,
                        risk=risk,
                        sim=sim,
                        spot_state=spot_state,
                        vol_model=vol_model,
                        vol_source=vol_source,
                        counters=counters,
                        console=console,
                        settings=settings,
                        event_ticker=event_ticker,
                        strike_window=strike_window,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - a bad cycle is not a bad session
                    counters.api_errors += 1
                    log.warning("paper cycle failed (continuing): %s: %s", type(e).__name__, e)

                # Rollover: the event we were trading has closed.
                if minutes_to_close <= 0.0:
                    nxt_current = await discover_hourly_event(client)
                    if nxt_current and nxt_current["event_ticker"] != event_ticker:
                        realised = await _close_out(
                            client, store, risk, sim, event_ticker, settings.armed
                        )
                        console.print(
                            f"[bold magenta]rollover[/] {event_ticker} -> "
                            f"{nxt_current['event_ticker']} | realised ${realised:+.2f}"
                        )
                        event_ticker = nxt_current["event_ticker"]
                        counters.events += 1

                next_tick += ladder_interval_s
                sleep_for = max(0.0, next_tick - time.monotonic())
                if next_tick < time.monotonic():
                    next_tick = time.monotonic()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None:
                    sleep_for = min(sleep_for, max(0.0, remaining))
                await asyncio.sleep(sleep_for)

    except (asyncio.CancelledError, KeyboardInterrupt):
        console.print("[yellow]interrupted - flushing buffers[/]")
    finally:
        if feed is not None:
            await feed.stop()
        if spot_feed is not None:
            # Drain what the venues sent after the last cycle so a short session does not
            # throw away its final second of spot history.
            for tick in spot_feed.drain():
                store.add_spot(
                    ts=tick.ts, venue=tick.venue, bid=tick.bid, ask=tick.ask,
                    mid=tick.mid, proxy=tick.proxy,
                )
                counters.spot_rows += 1
            await spot_feed.stop()
        counters.halts = len(killswitch.halts)
        await store.flush()
        with contextlib.suppress(Exception):
            store.export_parquet(Path(settings.data_dir).expanduser() / "parquet")
        if owns_store:
            await store.close()
        _summary(console, counters, risk, sim, killswitch)

    return counters.as_dict()


async def _close_out(
    client: KalshiClient,
    store: Store,
    risk: RiskManager,
    sim: FillSimulator,
    event_ticker: str,
    armed: bool,
) -> Decimal:
    """Settle the finished event and drop every resting order.

    We ask the venue for the realised `expiration_value` — it is public and it is the only
    honest way to score a paper position. If it has not published yet (Kalshi settles up to
    ~5 minutes after close) we mark out at cost rather than guess, and say so in the
    decision row: a paper P&L that quietly invents a settlement is worse than no P&L.
    """
    sim.resting.clear()
    positions = risk.event_positions(event_ticker)
    if not positions:
        return Decimal("0")

    expiration: Decimal | None = None
    with contextlib.suppress(Exception):
        snap = await client.get_market(positions[0].ticker)
        expiration = snap.expiration_value
        if expiration is not None:
            store.upsert_settlement(
                close_time=snap.close_time,
                event_ticker=event_ticker,
                expiration_value=expiration,
            )

    if expiration is not None:
        realised = risk.settle_event(event_ticker, expiration)
        reason = f"settled at BRTI 60s avg {expiration}"
    else:
        realised = risk.abandon_event(event_ticker)
        reason = "no expiration_value published yet; marked out at cost"

    store.add_decision(
        ts=datetime.now(UTC),
        ticker=event_ticker,
        fair_prob=0.0,
        market_mid=None,
        edge=float(realised),
        action="settle",
        reason=reason,
        armed=armed,
    )
    return realised


async def _cycle(
    *,
    client: KalshiClient,
    store: Store,
    feed: KalshiWebSocket | None,
    spot_feed: SpotFeed | None,
    engine: FairValueEngine,
    quoter: MakerQuoter,
    risk: RiskManager,
    sim: FillSimulator,
    spot_state: SpotState,
    vol_model: VolModel,
    vol_source: str,
    counters: PaperCounters,
    console: Console,
    settings: Settings,
    event_ticker: str,
    strike_window: int | None,
) -> float:
    """One decision cycle: poll, price, size, risk-check, simulate, record.

    Returns minutes to close, which is how the outer loop detects a rollover.
    """
    now = datetime.now(UTC)

    # 1. Market data ------------------------------------------------------------------
    markets, minutes_to_close, rows = await _poll_ladder(
        client, store, event_ticker, strike_window
    )
    counters.ladder_rows += rows
    counters.cycles += 1
    if not markets:
        return minutes_to_close

    if feed is not None and feed.available:
        for ev in feed.drain():
            if isinstance(ev, BrtiTick):
                spot_state.update_from_brti(ev)
                store.add_brti(
                    ts=ev.ts,
                    index_id=ev.index_id,
                    value=ev.value,
                    avg_60s=ev.avg_60s,
                    windowed_avg=ev.windowed_avg,
                    tick_count=ev.tick_count,
                )

    # The public composite. Free, no credentials, and the only independent price a
    # keyless install ever sees - `spot_feed.proxy()` already enforces the venue-count and
    # cross-venue-agreement gates, returning None rather than a lone venue's opinion.
    if spot_feed is not None:
        for tick in spot_feed.drain():
            store.add_spot(
                ts=tick.ts, venue=tick.venue, bid=tick.bid, ask=tick.ask,
                mid=tick.mid, proxy=tick.proxy,
            )
            counters.spot_rows += 1
        px = spot_feed.proxy()
        if px is not None:
            spot_state.update_from_proxy(float(px), now)

    quotes = [LadderQuote.from_snapshot(m) for m in markets]
    sigma = vol_model.sigma_per_minute(now)
    estimate = estimate_spot_from_ladder(
        quotes, sigma, max(minutes_to_close, 1e-6), dist=engine.dist, df=engine.df
    )
    spot_state.update_from_ladder(estimate, now)
    if spot_state.value is None:
        # No trustworthy level. Common early in an event (everything pinned) and during
        # the momentary REST-cache blips where the ladder collapses to one wide strike.
        counters.skipped_cycles += 1
        store.add_decision(
            ts=now, ticker=event_ticker, fair_prob=0.0, market_mid=None, edge=None,
            action="skip", reason=estimate.describe(), armed=settings.armed,
        )
        counters.decisions += 1
        # Still print a line. A runner that goes silent when it declines to trade is
        # indistinguishable from a runner that has hung.
        console.print(
            f"[dim]{now:%H:%M:%S} {_countdown(minutes_to_close * SECONDS_PER_MINUTE)} "
            f"{event_ticker[-9:]} skip: {estimate.reason}[/]",
            highlight=False,
        )
        return minutes_to_close

    # 2. Fair value and edge ----------------------------------------------------------
    quotable = [q for q in quotes if q.is_quotable]
    ladder = engine.evaluate(
        quotable,
        spot=spot_state.value,
        sigma_per_minute=sigma,
        minutes_to_close=minutes_to_close,
        event_ticker=event_ticker,
        window=spot_state.window,
        spot_source=spot_state.source,
        ts=now,
    )

    # Marks first, so every P&L and every limit this cycle sees the current market.
    risk.update_marks({e.ticker: e.mid for e in ladder.edges if e.mid is not None})

    # 3. Simulated fills on what is already resting -----------------------------------
    by_ticker = {m.ticker: m for m in markets}
    for fill in sim.on_market_update(by_ticker, now):
        _record_fill(store, risk, fill, counters, settings.armed)

    # 4. Taker candidates: only those that cleared fees.min_taker_edge ----------------
    takers = [e for e in ladder.edges if e.any_takeable]
    counters.taker_signals += len(takers)
    if takers:
        candidates = [
            SizingCandidate(
                ticker=e.ticker,
                strike=e.strike,
                action=Action.BUY if e.takeable_buy else Action.SELL,
                price=(e.yes_ask if e.takeable_buy else e.yes_bid),  # type: ignore[arg-type]
                fair_prob=e.fair_prob,
                liquidity=Liquidity.TAKER,
                existing_position=risk.contracts(e.ticker),
                available_size=(e.yes_ask_size if e.takeable_buy else e.yes_bid_size),
            )
            for e in takers
        ]
        sizing = size_ladder(candidates, settings.risk)
        for order in sizing.orders:
            c = order.candidate
            edge_e = ladder.by_ticker()[c.ticker]
            edge_val = float(
                edge_e.edge_buy_yes if c.action is Action.BUY else edge_e.edge_sell_yes
            )
            if order.contracts <= 0:
                store.add_decision(
                    ts=now, ticker=c.ticker, fair_prob=c.fair_prob,
                    market_mid=float(edge_e.mid) if edge_e.mid is not None else None,
                    edge=edge_val, action="skip",
                    reason=f"sized to zero (ladder scale {sizing.ladder_scale:.3f})",
                    armed=settings.armed,
                )
                counters.decisions += 1
                continue

            decision = risk.check_order(
                ticker=c.ticker, action=c.action, contracts=order.contracts,
                price=c.price, liquidity=Liquidity.TAKER, event_ticker=event_ticker,
            )
            if not decision:
                counters.blocked_by_risk += 1
                store.add_decision(
                    ts=now, ticker=c.ticker, fair_prob=c.fair_prob,
                    market_mid=float(edge_e.mid) if edge_e.mid is not None else None,
                    edge=edge_val, action="blocked", reason=decision.reason,
                    armed=settings.armed,
                )
                counters.decisions += 1
                continue

            store.add_decision(
                ts=now, ticker=c.ticker, fair_prob=c.fair_prob,
                market_mid=float(edge_e.mid) if edge_e.mid is not None else None,
                edge=edge_val, action=f"take_{c.action.value}",
                reason=(
                    f"edge {edge_val * 100:.2f}c >= hurdle "
                    f"{float(engine.hurdle(c.price)) * 100:.2f}c; {order.describe()}"
                ),
                armed=settings.armed,
            )
            counters.decisions += 1

            fill = sim.take(
                ticker=c.ticker, action=c.action, price=c.price,
                contracts=decision.contracts,
                available=c.available_size or Decimal("0"), now=now,
                reason=f"crossed for {edge_val * 100:.2f}c",
            )
            if fill is not None:
                _record_fill(store, risk, fill, counters, settings.armed)

    # 5. Maker quotes ------------------------------------------------------------------
    plan = quoter.plan(ladder, positions=risk.position_map())
    counters.pulled += len(plan.pulls)
    allowed: list[QuoteIntent] = []
    for intent in plan.intents:
        decision = risk.check_order(
            ticker=intent.ticker, action=intent.action, contracts=intent.contracts,
            price=intent.price, liquidity=Liquidity.MAKER, event_ticker=event_ticker,
        )
        if not decision:
            counters.blocked_by_risk += 1
            continue
        allowed.append(intent)

    cancelled = sim.reconcile(QuotePlan(intents=allowed, pulls=plan.pulls), now)
    counters.quotes_cancelled += cancelled
    counters.quotes_placed += len(allowed)

    # One decision row per cycle for the strike our model calls the money. Writing all of
    # them would bury the interesting rows under thousands of "hold"s.
    atm = ladder.atm
    if atm is not None:
        resting_here = [i for i in allowed if i.ticker == atm.ticker]
        store.add_decision(
            ts=now,
            ticker=atm.ticker,
            fair_prob=atm.fair_prob,
            market_mid=float(atm.mid) if atm.mid is not None else None,
            edge=float(atm.best_edge),
            action="quote" if resting_here else "hold",
            reason=(
                "; ".join(i.describe() for i in resting_here)
                if resting_here
                else (plan.pulls[0].reason if plan.pulls else atm.reason)
            ),
            armed=settings.armed,
        )
        counters.decisions += 1

    # 6. Risk surveillance --------------------------------------------------------------
    risk.check_daily_loss()

    _heartbeat(console, ladder, plan, sim, risk, counters, vol_source)
    return minutes_to_close


def _summary(
    console: Console,
    counters: PaperCounters,
    risk: RiskManager,
    sim: FillSimulator,
    killswitch: KillSwitch,
) -> None:
    elapsed = time.monotonic() - counters.started_at
    console.print(
        f"[bold]paper done[/] {elapsed:.0f}s cycles={counters.cycles} "
        f"decisions={counters.decisions} quotes={counters.quotes_placed} "
        f"(cancelled {counters.quotes_cancelled}) fills: maker={counters.maker_fills} "
        f"taker={counters.taker_fills}"
    )
    console.print(f"[dim]risk: {risk.describe()}[/]")
    console.print(
        f"[dim]sim: resting={sim.n_resting} queue_ahead={sim.total_queue_ahead:.0f} "
        f"queue_consumed={sim.queue_consumed:.0f} taker_signals={counters.taker_signals} "
        f"pull_notices={counters.pulled} "
        f"skipped_cycles={counters.skipped_cycles}/{counters.cycles} "
        f"(no trustworthy spot)[/]"
    )
    if killswitch.halted:
        console.print(f"[bold red]{killswitch.describe()}[/]")
    else:
        console.print(f"[dim]{killswitch.describe()}[/]")
    if counters.maker_fills == 0 and counters.quotes_placed:
        console.print(
            "[dim]No maker fills: the simulator requires the market to trade through our "
            "resting price after clearing the queue we joined. On a short run that is the "
            "expected — and honest — result.[/]"
        )


__all__ = ["FillSimulator", "PaperCounters", "RestingOrder", "SimulatedFill", "run_paper"]
