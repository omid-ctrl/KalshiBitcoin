"""Kalshi REST client.

AUTH (verified 2026-07-28 against docs.kalshi.com/openapi.yaml and the official
kalshi_python_sync 3.25.0 SDK source):

    KALSHI-ACCESS-KEY        = API key id
    KALSHI-ACCESS-TIMESTAMP  = unix time in MILLISECONDS, decimal string
    KALSHI-ACCESS-SIGNATURE  = base64(RSA-PSS(msg))

    msg = f"{timestamp_ms}{METHOD}{path}"

where `path` INCLUDES the /trade-api/v2 prefix and EXCLUDES the query string, e.g.
    "1703123456789GET/trade-api/v2/portfolio/balance"
The request BODY is not signed. RSA-PSS uses MGF1-SHA256 with salt length = digest length.

There is no /login endpoint and no bearer-token scheme in the current API.

RATE LIMITS are token buckets, not request counts. On the Basic tier: 200 read
tokens/sec and 100 write tokens/sec, refilled continuously. Most endpoints cost 10
tokens, so that is 20 GET/s and 10 order-creates/s. Cancels cost only 2 (=> 50/s), and
the CF Benchmarks passthrough costs 50 (=> 4/s). Basic tier has NO write burst capacity.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_btc.config import SERIES_TICKER, Settings
from kalshi_btc.core.types import Book, BookLevel, MarketSnapshot, _ts, dec

log = logging.getLogger(__name__)

RATE_LIMIT_RETRIES = 8
RATE_LIMIT_MAX_BACKOFF_S = 20.0

DEFAULT_COST = 10
COST_CANCEL = 2
COST_CFBENCHMARKS = 50


class RateLimitError(RuntimeError):
    pass


class TokenBucket:
    """Continuous-refill token bucket matching Kalshi's documented model."""

    def __init__(self, rate_per_sec: float, capacity: float | None = None) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity if capacity is not None else rate_per_sec
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self, cost: float) -> None:
        if cost > self.capacity:
            raise RateLimitError(
                f"request costs {cost} tokens but bucket capacity is {self.capacity}"
            )
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                await asyncio.sleep((cost - self._tokens) / self.rate)


def load_private_key(path: str | Path) -> rsa.RSAPrivateKey:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"Kalshi private key not found at {p}. Create an API key in Kalshi -> Settings "
            f"-> API Keys, save the RSA private key to that path, and `chmod 600` it."
        )
    key = serialization.load_pem_private_key(p.read_bytes(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError(f"{p} is not an RSA private key")
    return key


def sign_request(key: rsa.RSAPrivateKey, timestamp_ms: int, method: str, path: str) -> str:
    """Produce the KALSHI-ACCESS-SIGNATURE value.

    `path` must include /trade-api/v2 and must NOT include the query string.
    """
    msg = f"{timestamp_ms}{method.upper()}{path}".encode()
    sig = key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


class KalshiClient:
    """Async Kalshi REST client.

    Market data endpoints are PUBLIC - this client works with no credentials at all,
    which is what lets Phase 0 data capture start before you have an API key.
    """

    def __init__(self, settings: Settings, session: aiohttp.ClientSession | None = None) -> None:
        self.settings = settings
        self.base = settings.rest_base.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self._key: rsa.RSAPrivateKey | None = None
        if settings.has_credentials:
            self._key = load_private_key(settings.private_key_path)
        self.read_bucket = TokenBucket(200.0, 200.0)
        self.write_bucket = TokenBucket(100.0, 100.0)

    async def __aenter__(self) -> KalshiClient:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20), raise_for_status=False
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("KalshiClient must be used as an async context manager")
        return self._session

    def _path(self, endpoint: str) -> str:
        from urllib.parse import urlparse

        return urlparse(self.base).path + endpoint

    async def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        cost: int = DEFAULT_COST,
        authed: bool = False,
        retries: int = 3,
    ) -> dict[str, Any]:
        write = method.upper() in {"POST", "PUT", "DELETE", "PATCH"}
        bucket = self.write_bucket if write else self.read_bucket

        headers: dict[str, str] = {"Accept": "application/json"}
        if authed:
            if self._key is None:
                raise PermissionError(
                    "This call needs Kalshi API credentials. Set KALSHI_API_KEY_ID and "
                    "KALSHI_PRIVATE_KEY_PATH in .env. (Market data does not need them.)"
                )
            ts = int(time.time() * 1000)
            path = self._path(endpoint)  # query string deliberately excluded
            headers |= {
                "KALSHI-ACCESS-KEY": self.settings.api_key_id,
                "KALSHI-ACCESS-TIMESTAMP": str(ts),
                "KALSHI-ACCESS-SIGNATURE": sign_request(self._key, ts, method, path),
            }

        url = f"{self.base}{endpoint}"
        last_exc: Exception | None = None
        # Rate limiting is a wait, not a failure, so it gets its own generous budget.
        # The token buckets here are sized for the authenticated Basic tier; the
        # UNAUTHENTICATED limit is materially tighter, so a public-data recorder will
        # meet 429s routinely. Counting those against the transport-error budget kills
        # a 24/7 capture within seconds of a burst - which is exactly what it did.
        rate_limit_budget = max(retries, RATE_LIMIT_RETRIES)
        rate_limited = 0
        attempt = 0
        while attempt < retries:
            await bucket.take(cost)
            try:
                async with self.session.request(
                    method, url, params=params, json=json_body, headers=headers
                ) as resp:
                    if resp.status == 429:
                        rate_limited += 1
                        if rate_limited > rate_limit_budget:
                            raise RateLimitError(
                                f"{method} {endpoint} rate limited {rate_limited} times in a "
                                f"row. Slow the polling cadence, or add API credentials - the "
                                f"unauthenticated read limit is much tighter than the "
                                f"authenticated one."
                            )
                        # Honour Retry-After when the venue tells us; otherwise back off
                        # exponentially on the RATE-LIMIT counter, capped so a recorder
                        # recovers rather than stalling for minutes.
                        retry_after = resp.headers.get("Retry-After")
                        wait = (
                            float(retry_after)
                            if retry_after and retry_after.replace(".", "", 1).isdigit()
                            else min(2**rate_limited, RATE_LIMIT_MAX_BACKOFF_S)
                        )
                        last_exc = RateLimitError(f"429 on {endpoint}")
                        log.warning(
                            "rate limited on %s (%d), waiting %.1fs", endpoint, rate_limited, wait
                        )
                        await asyncio.sleep(wait)
                        continue  # deliberately does NOT consume a transport attempt
                    text = await resp.text()
                    if resp.status >= 400:
                        # Do not retry client errors other than 429 - they will not fix themselves.
                        if resp.status < 500:
                            raise RuntimeError(f"{method} {endpoint} -> {resp.status}: {text[:300]}")
                        raise aiohttp.ClientError(f"{resp.status}: {text[:200]}")
                    return await resp.json() if text else {}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                attempt += 1
                # Only idempotent verbs may be retried; a retried POST could double-order.
                if write and method.upper() == "POST":
                    raise
                await asyncio.sleep(min(2**attempt, 8))
        raise RuntimeError(f"{method} {endpoint} failed after {retries} attempts: {last_exc}")

    # ---------------------------------------------------------------- public market data
    async def get_series(self, series: str = SERIES_TICKER) -> dict:
        return (await self.request("GET", f"/series/{series}")).get("series", {})

    async def get_events(
        self, series: str = SERIES_TICKER, status: str = "open", limit: int = 20
    ) -> list[dict]:
        data = await self.request(
            "GET",
            "/events",
            params={
                "series_ticker": series,
                "status": status,
                "limit": limit,
                "with_nested_markets": "true",
            },
        )
        return data.get("events", [])

    async def get_hourly_events(self, status: str = "open", limit: int = 20) -> list[dict]:
        """Open KXBTCD events that are actually HOURLY.

        This filter is mandatory, not cosmetic. `series_ticker` is NOT a cadence filter:
        KXBTCD simultaneously hosts hourly events ($100 strike spacing, 188 strikes,
        60-minute life), DAILY events (17:00 ET, $250 spacing, ~80 strikes, ~25h life)
        and WEEKLY events (Friday 17:00 ET, $500 spacing, ~50 strikes, 7-day life).

        Taking "the next open KXBTCD event" therefore lands you on a daily or weekly
        contract roughly whenever one is open - a completely different instrument with a
        completely different variance profile. Filter on the event's actual duration.
        """
        events = await self.get_events(series=SERIES_TICKER, status=status, limit=limit)
        return [e for e in events if event_is_hourly(e)]

    async def get_event(self, event_ticker: str) -> dict:
        data = await self.request(
            "GET", f"/events/{event_ticker}", params={"with_nested_markets": "true"}
        )
        return data.get("event", {})

    async def get_market(self, ticker: str) -> MarketSnapshot:
        data = await self.request("GET", f"/markets/{ticker}")
        return MarketSnapshot.from_api(data["market"])

    async def get_orderbook(self, ticker: str, depth: int = 10) -> Book:
        data = await self.request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})
        return parse_orderbook(ticker, data)

    async def get_settled_markets(self, series: str = SERIES_TICKER, limit: int = 1000) -> list[dict]:
        """Settled markets carry `expiration_value` - the realised BRTI 60s average.

        This is how we score our settlement model against ground truth for free.
        """
        out: list[dict] = []
        cursor: str | None = None
        while True:
            params = {"series_ticker": series, "status": "settled", "limit": limit}
            if cursor:
                params["cursor"] = cursor
            data = await self.request("GET", "/markets", params=params)
            batch = data.get("markets", [])
            out.extend(batch)
            cursor = data.get("cursor")
            if not cursor or not batch:
                return out

    # ------------------------------------------------------------------ authenticated
    async def get_balance(self) -> dict:
        return await self.request("GET", "/portfolio/balance", authed=True)

    async def get_positions(self) -> dict:
        return await self.request("GET", "/portfolio/positions", authed=True)

    async def get_fills(self, limit: int = 200) -> dict:
        return await self.request("GET", "/portfolio/fills", params={"limit": limit}, authed=True)

    async def create_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: int,
        client_order_id: str,
        post_only: bool = True,
        time_in_force: str | None = None,
    ) -> dict:
        """Place an order.

        Kalshi has NO market-order type - everything is a limit order. For immediate
        execution use time_in_force="immediate_or_cancel" with an aggressive price.
        `post_only=True` guarantees maker treatment (and therefore a zero fee).
        """
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": "limit",
            "client_order_id": client_order_id,
            f"{side}_price": price_cents,
        }
        if post_only:
            body["post_only"] = True
        if time_in_force:
            body["time_in_force"] = time_in_force
        return await self.request("POST", "/portfolio/orders", json_body=body, authed=True)

    async def cancel_order(self, order_id: str) -> dict:
        return await self.request(
            "DELETE", f"/portfolio/orders/{order_id}", cost=COST_CANCEL, authed=True
        )

    async def get_cfbenchmarks(self, endpoint: str = "") -> dict:
        """CF Benchmarks passthrough using Kalshi credentials (no CF account needed).

        Costs 50 tokens, so ~4 req/s on Basic. Prefer the `cfbenchmarks_value` WebSocket
        channel for anything real-time.
        """
        suffix = f"/{endpoint}" if endpoint else ""
        return await self.request(
            "GET", f"/cfbenchmarks{suffix}", cost=COST_CFBENCHMARKS, authed=True
        )


def parse_orderbook(ticker: str, payload: dict) -> Book:
    """Parse the CURRENT orderbook schema.

    Responses use `orderbook_fp` with `yes_dollars` / `no_dollars` arrays of
    [price, size] decimal STRINGS. Older code reading `orderbook.yes` gets nothing.
    """
    ob = payload.get("orderbook_fp") or payload.get("orderbook") or {}
    yes_raw = ob.get("yes_dollars") or ob.get("yes") or []
    no_raw = ob.get("no_dollars") or ob.get("no") or []

    def levels(rows: list) -> list[BookLevel]:
        out = []
        for row in rows or []:
            if not row or len(row) < 2:
                continue
            out.append(BookLevel(price=dec(row[0]), size=dec(row[1])))
        return out

    return Book(ticker=ticker, yes=levels(yes_raw), no=levels(no_raw))


HOURLY_SECONDS = 3600
HOURLY_TOLERANCE_S = 120
HOURLY_STRIKE_SPACING = Decimal("100")
MIN_HOURLY_STRIKES = 3


def strike_spacing(event: dict) -> Decimal | None:
    """Smallest gap between adjacent strikes, or None if the ladder is too thin to tell."""
    strikes = sorted(
        {dec(m.get("floor_strike")) for m in (event.get("markets") or []) if m.get("floor_strike")}
    )
    if len(strikes) < 2:
        return None
    return min(b - a for a, b in zip(strikes, strikes[1:]))


def event_is_hourly(event: dict) -> bool:
    """True iff this KXBTCD event is the 60-minute cadence.

    The series ticker cannot distinguish the cadences: KXBTCD hosts hourly, DAILY ($250
    spacing) and WEEKLY ($500) events all at once. So we need a discriminator.

    STRIKE SPACING IS THAT DISCRIMINATOR, and open_time is not. The obvious test -
    close_time - open_time == 60 minutes - is the correct *definition* but it does not
    survive contact with the live venue: Kalshi now builds the full ladder days to years
    ahead of the close, so an hourly event's open->close span is nothing like an hour.
    Measured on 62 open KXBTCD events on 2026-07-29, that span ranged up to 61,350 hours
    and matched 0 of 62 events, i.e. the span test concluded there were no hourly markets
    at all and silently blinded both `kbtc capture` and `kbtc paper`.

    Spacing separates the three cadences cleanly on that same live sample: 60 events at
    $100 (hourly), 1 at $250 (daily), 1 at $500 (weekly). We still accept a genuine
    60-minute span when the payload happens to carry one, because that is the real
    definition and it costs nothing to honour it.

    `MIN_HOURLY_STRIKES` guards against inferring a cadence from a two-strike stub, where
    a single $100 gap proves nothing about the ladder as a whole.
    """
    markets = event.get("markets") or []
    if not markets:
        return False

    m = markets[0]
    open_raw, close_raw = m.get("open_time"), m.get("close_time")
    if open_raw and close_raw:
        span = (_ts(close_raw) - _ts(open_raw)).total_seconds()
        if abs(span - HOURLY_SECONDS) <= HOURLY_TOLERANCE_S:
            return True

    return strike_spacing(event) == HOURLY_STRIKE_SPACING and len(markets) >= MIN_HOURLY_STRIKES


def market_is_kxbtcd(ticker: str) -> bool:
    """Guard against accidentally trading KXBTC (the illiquid range series)."""
    return ticker.startswith(f"{SERIES_TICKER}-")


def cents(price: Decimal) -> int:
    """Kalshi order prices are integer cents 1..99."""
    return int((price * 100).to_integral_value())
