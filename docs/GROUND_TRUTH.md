# Ground truth — verified market facts for KXBTCD

Every statement below was checked against a primary source on the date shown. They are
recorded here so that future work does not re-derive them, contradict them, or "fix" correct
code back into something that is wrong.

**If you are about to change behaviour that contradicts something on this page, do not.**
Re-verify against the cited source first, and if the market really has changed, update the
entry — including the date and the new source — in the same commit as the code change.

Verification date for everything below unless stated otherwise: **2026-07-28**.

---

## 1. The series

| Fact | Value |
|---|---|
| Series ticker | `KXBTCD` |
| Cadence | Hourly |
| Market type | Binary, above/below a floor strike (`strike_type: "greater"`) |
| Strikes per event | 188, spaced $100 apart |
| Strikes with a live two-sided quote | typically 4–9; the rest are pinned at 0 or 1 |
| Event lifetime | opens **exactly one hour** before it closes |
| Ticker format | `KXBTCD-<YY><MON><DD><HH>-T<strike>`, e.g. `KXBTCD-26JUL2819-T63999.99` |

*Source:* live `GET /events?series_ticker=KXBTCD&with_nested_markets=true` and
`GET /series/KXBTCD` on production. Confirmed by `kbtc status` output on 2026-07-28, which
showed 4 of 188 strikes carrying a two-sided quote.

### 1a. `KXBTC` is a different series — do not trade it by accident

`KXBTC` is a separate hourly BTC series of **range/bracket** markets. It carries roughly 80x
less volume and two-to-four cent spreads. It is not what this bot is designed for.

*Guard in code:* `kalshi_btc.exec.client.market_is_kxbtcd()` and
`kalshi_btc.config.DECOY_SERIES`.

---

## 2. Settlement — the central fact

> "a simple average of the CF Bitcoin Real-Time Index (BRTI) for the minute (60 seconds)
> prior to *\<time\>*"

and

> "The Last Trading Time will be *\<time\>*"

*Source:* Kalshi contract terms for KXBTCD; the same wording appears in the `rules_primary`
field of every KXBTCD market returned by the API, e.g.

> "If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time Index
> (BRTI) before 9 PM EST is above 99499.99 at 9 PM EST on Jan 1, 2028, then the market
> resolves to Yes."

Consequences, all of which the code depends on:

1. **Settlement is an arithmetic average, not a point-in-time price.** This is an Asian-
   settled digital. Pricing it as a coin flip on the closing tick is simply wrong.
2. **Trading stays OPEN through the entire averaging window.** The last trading time equals
   the close, so the final sixty seconds are tradeable while part of the settlement value is
   already locked in as a known constant.
3. **The 60 ticks land at 59, 58, ..., 1, 0 seconds before the close.** Per Kalshi's
   AsyncAPI the accumulation window is `(close_ts_ms - 60000, close_ts_ms]`, second-indexed
   `:01 -> 1` ... `close tick -> 60`.

### 2a. Variance consequences (derived, and Monte-Carlo verified)

- Outside the window: `Var(settlement) ≈ sigma_min² × (tau − 2/3)`. Averaging removes about
  forty seconds of variance from *every* quote, all hour long.
- Inside the window, with `s` seconds elapsed and `m = 60 − s` remaining:
  `residual_std ≈ sigma_min × (m/60)^1.5 / sqrt(3)`.
- A point-in-time model therefore overstates remaining uncertainty by **1.73x** at window
  open, **3.46x** at the halfway mark, and about **21x** with five seconds left.

The code implements the **exact discrete-tick** version, not the continuous approximation:
with only 60 ticks the difference is measurable (exact in-window factor 0.3417 vs the
continuous 1/3).

*Source:* `src/kalshi_btc/model/pricing.py` and the Monte Carlo checks in
`tests/test_pricing.py`.

---

## 3. The BRTI feed

Kalshi's WebSocket exposes a `cfbenchmarks_value` channel carrying the CF Benchmarks BRTI.

| Field | Availability | Meaning |
|---|---|---|
| `avg_60s_data` | **always** present | A rolling *trailing* 60-second average. **This is NOT the settlement figure.** |
| `last_60s_windowed_average_15min` | only in the final minute before a quarter-hour close (`:00`, `:15`, `:30`, `:45`) | The settlement-relevant windowed average over `(close_ts_ms − 60000, close_ts_ms]`, second-indexed `:01 -> 1` ... `close tick -> 60` |

For our hourly market — which always closes at `:00` — **the settlement-relevant field is
`last_60s_windowed_average_15min`**. Using `avg_60s_data` as the settlement estimate is a
correctness bug: it is a trailing window that includes ticks from before the settlement
window and excludes ticks inside it.

*Source:* Kalshi AsyncAPI specification for the `cfbenchmarks_value` channel.

There is also a REST passthrough, `GET /cfbenchmarks[/...]`, which uses your Kalshi
credentials (no CF Benchmarks account needed) and costs **50 rate-limit tokens** — about 4
requests/second on the Basic tier. Use the WebSocket for anything real-time.

---

## 4. Free ground truth: `expiration_value`

Settled markets expose `expiration_value`, the **realised BRTI 60-second average**. This is
how the pricing model is scored against reality at no cost.

- Fetch via `GET /markets?series_ticker=KXBTCD&status=settled` (paginated; see
  `KalshiClient.get_settled_markets`).
- Public endpoint — no credentials required.
- **Caveat:** the demo exchange returns junk here (observed literal `"a"` on
  `KXBTCD-28JAN0121` markets on 2026-07-28). Parse defensively; only trust production.

---

## 5. Fees

Verbatim from the official *"Fee Schedule for July 2026 — 7.7.26 Update"* PDF and
<https://kalshi.com/fee-schedule>:

```
fees = round up(M × 0.07   × C × P × (1−P))     <- taker
fees = round up(M × 0.0175 × C × P × (1−P))     <- maker, M defaults to 0

P = contract price in dollars (50 cents is 0.5)
C = number of contracts
M = per-contract multiplier (default 1 unless otherwise indicated)
round up = rounds up so that fee + positionCost lands on a centicent ($0.0001)
```

> "Settlement Fees — There is no settlement fee."

KXBTCD is **absent from the Non-Standard Fees table**, so it takes the plain standard
schedule:

| | Multiplier | Effect |
|---|---|---|
| Taker | 1 | `ceil(0.07 × C × P × (1−P))` to the centicent — about 1.75c/contract at the money |
| **Maker** | **0** | **Zero. Resting orders that fill cost nothing.** |
| Settlement | — | **Zero.** |

Cross-checked against the live API: `GET /series/KXBTCD` returns `fee_type: "quadratic"`,
`fee_multiplier: 1`, and `GET /series/fee_changes?series_ticker=KXBTCD` returns empty (no
override, scheduled or historical).

**This asymmetry is the single most important economic fact about the market** and is why
the strategy is maker-first. Minimum model edge for a taker trade to be +EV is
`0.07 × P × (1−P) + half_spread` — about 2.25c at P=0.50, about 1.13c at P=0.10. That
asymmetry is why the bot trades away from the money far more often than at it.

*Source:* `src/kalshi_btc/core/fees.py`.

---

## 6. API schema traps

### 6a. The legacy integer-cent fields are always ZERO

The current schema uses `*_dollars` and `*_fp` **string** fields. The legacy integer-cent
fields still appear in responses and are **always zero**.

| Use this | Not this (always 0) |
|---|---|
| `yes_bid_dollars`, `yes_ask_dollars` | `yes_bid`, `yes_ask` |
| `yes_bid_size_fp`, `yes_ask_size_fp` | `yes_bid_size`, `yes_ask_size` |
| `volume_fp` | `volume` |
| `open_interest_fp` | `open_interest` |
| `orderbook_fp.yes_dollars` / `.no_dollars` | `orderbook.yes` / `.no` |

This is the most common way a bot built from an older tutorial silently sees an empty
market. Sizes are genuinely **fractional** decimal strings (e.g. `"1286.06"`), which is why
money and sizes are `Decimal` throughout this codebase — float is a correctness bug here,
not a style preference.

### 6b. Order book sides are both expressed as bids

`yes` levels are bids to buy YES; `no` levels are bids to buy NO. A NO bid at `p` is
economically a YES ask at `1 − p`. See `Book.best_yes_ask`.

### 6c. Spreads

Spreads are **1 cent**, locked to the minimum tick (`price_ranges: [{start: 0.0000,
end: 1.0000, step: 0.0100}]`, `price_level_structure: "linear_cent"`).

### 6d. There is no market order type

Everything is a limit order. Immediate execution = limit price + `time_in_force:
"immediate_or_cancel"`. `post_only: true` guarantees maker treatment, and therefore a zero
fee.

### 6e. REST is cached

REST responses are cached for roughly one second, and two endpoints can disagree by a cent.
**Never poll REST for live trading decisions — use the WebSocket.** REST is for setup,
backfill and reconciliation.

---

## 7. Authentication

Verified against `docs.kalshi.com/openapi.yaml` and the official `kalshi_python_sync` 3.25.0
SDK source.

```
KALSHI-ACCESS-KEY        = API key id
KALSHI-ACCESS-TIMESTAMP  = unix time in MILLISECONDS, decimal string
KALSHI-ACCESS-SIGNATURE  = base64(RSA-PSS(msg))

msg = f"{timestamp_ms}{METHOD}{path}"
```

- `path` **includes** the `/trade-api/v2` prefix and **excludes** the query string, e.g.
  `"1703123456789GET/trade-api/v2/portfolio/balance"`.
- The request **body is not signed**.
- RSA-PSS uses MGF1-SHA256 with salt length = digest length.
- There is **no** `/login` endpoint and no bearer-token scheme in the current API.

Hosts (verified live 2026-07-28):

| | REST | WebSocket |
|---|---|---|
| Production | `https://external-api.kalshi.com/trade-api/v2` | `wss://external-api-ws.kalshi.com/trade-api/ws/v2` |
| Demo | `https://external-api.demo.kalshi.co/trade-api/v2` | `wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2` |

Demo credentials work only on demo hosts; production credentials only on production hosts.

---

## 8. Rate limits

Token buckets, **not** request counts, refilled continuously. Basic tier:

| | Rate | Burst |
|---|---|---|
| Read | 200 tokens/sec | none beyond the bucket |
| Write | 100 tokens/sec | **none** |

Costs: most endpoints **10** tokens (≈20 GET/s, ≈10 order-creates/s), cancels **2** (≈50/s),
CF Benchmarks passthrough **50** (≈4/s).

*Source:* `src/kalshi_btc/exec/client.py`, which implements matching buckets and backs off
on HTTP 429.

---

## 9. Measured statistics

Measured on the realised KXBTCD settlement series (`expiration_value`):

| Statistic | Value |
|---|---|
| Volatility | **0.393% per hour** = 36.8% annualised |
| Excess kurtosis | **3.66** |

The excess kurtosis is why `pricing.price_above(..., dist="t")` exists. The Gaussian default
is a benchmark, not a recommendation for live trading.

---

## 10. Operational facts

- **Order book history cannot be bought or backfilled.** Kalshi sells no historical
  microstructure data and offers no backfill endpoint. An hour not recorded is gone
  permanently. This is why `kbtc capture` should be started on day one and supervised.
- **Clock accuracy is a correctness requirement.** Settlement is a mean of sixty one-second
  ticks with the last landing at the close, so final-minute decisions are second-sensitive.
  `kbtc doctor` measures skew against Kalshi's `Date` header, which has one-second
  resolution — treat anything under ±0.5s as measurement noise.
- **The demo exchange serves synthetic junk**: events with close times in the past,
  `expiration_value: "a"`, markets created months before their listed open. Use it to test
  plumbing, never to draw conclusions about prices, spreads or fills.
