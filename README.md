# kbtc — a trading bot for Kalshi's hourly Bitcoin markets (KXBTCD)

Kalshi lists an hourly binary market on Bitcoin: *"will BTC be above $X at the top of the
hour?"* This bot prices those contracts, records the market so the pricing can be scored
against reality, and — only once you have told it to, twice — trades them.

## What edge this targets, honestly

KXBTCD does **not** settle on the Bitcoin price at the top of the hour. It settles on the
**simple average of sixty one-second CF Benchmarks BRTI ticks in the final minute**, and
trading stays open the whole way through that minute. Averaging destroys variance. Outside
the window that removes roughly two-thirds of a minute of variance from every quote; inside
it, the uncertainty still to be resolved collapses as `(minutes_left/60)^1.5`. Five seconds
before the close, a bot that prices this as a coin flip is overstating the remaining
uncertainty by about **21x** — at exactly the moment the mispricing is largest. On top of
that, Kalshi charges **zero maker fees** on this series and `0.07 × P × (1−P)` per contract
to takers, so resting quotes are free and crossing the spread is not.

That is the whole thesis: price the Asian settlement correctly, quote as a maker, and only
cross the spread in the final minute when the model's edge clearly exceeds the taker fee.

**What it realistically earns.** On a $1,000–$5,000 bankroll, expect **single-digit to
low-double-digit dollars per day**, with plenty of days at zero or slightly negative. Not a
salary. Not a percentage-per-day. The reasons are structural and will not go away: spreads
are locked at one cent, only four to nine of the 188 strikes carry a live two-sided quote at
any moment, resting size at the touch is often only a few contracts, and the market maker on
the other side is not stupid. If a strategy like this were worth thousands a day, the
spread would not be one cent. Treat any month that beats a savings account as a good month,
and treat this repository primarily as a well-instrumented way to find out whether the edge
is real *before* you fund it.

**This is not financial advice, and you can lose the money you put in.** The default
configuration will not place a single order.

---

## Requirements

- Python 3.12 or newer
- A Kalshi account (only needed for the *trading* phases — data capture needs nothing)
- A machine that can stay awake, with a correct clock (NTP on). We trade to the second.

---

## Quickstart — works today, no API key

Everything in this section uses only Kalshi's **public** market-data endpoints. You do not
need an account, an API key, or any credentials.

```bash
git clone <your-fork> KalshiBitcoin
cd KalshiBitcoin

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env      # optional at this stage; sensible defaults exist
```

### 1. Preflight

```bash
kbtc doctor
```

`doctor` is the command to run first, and again whenever anything behaves oddly. It checks
your config, that the data directory is writable, that Kalshi is reachable, **how far your
clock is off from Kalshi's**, whether the KXBTCD series has open events, whether your
credentials (if any) actually work, and whether the safety flag is armed. It prints
PASS/WARN/FAIL with the exact next step for anything that isn't PASS, and exits non-zero on
a FAIL so you can put it in a startup script.

A clean first run looks like two WARNs (no `.env`, no credentials) and no FAILs.

### 2. Look at the market

```bash
kbtc status
```

```
      KXBTCD-26JUL2820  ·  closes 00:00:00 UTC  ·  T-15:31

    Strike  Bid  Ask    Mid  Spread  Bid size  Ask size     Volume
$63,599.99  94c  95c  94.5c      1c  15614.55    262.14  366267.20
$63,699.99  75c  76c  75.5c      1c   3790.04   1098.00  410009.57
$63,799.99  24c  25c  24.5c      1c   3622.00    200.00  181820.07
$63,899.99   3c   4c   3.5c      1c   1236.00   1025.00   58025.78

4 of 188 strikes carry a live two-sided quote; the rest are pinned at 0 or 1.
```

That is the real shape of this market: 188 strikes, four of them tradeable, one-cent
spreads, fractional sizes.

### 3. Start recording — do this now

```bash
kbtc capture
```

**Start this immediately and leave it running.** This is the single most important thing in
the whole README. Kalshi does not sell historical order book data and there is no backfill
endpoint. Every hour the recorder is not running is an hour of market microstructure that
you can never get back, at any price, ever. The pricing model cannot be validated without
it, and validation is the only thing standing between you and donating money to a market
maker.

It needs no credentials, uses little bandwidth, and writes to `./data` (configurable via
`DATA_DIR`). Run it under `tmux`, `screen`, `systemd`, or `launchd` so it survives your
terminal closing:

```bash
tmux new -d -s kbtc-capture 'kbtc capture'
```

Once a day, pull down the settled outcomes — this is free ground truth:

```bash
kbtc settlements
```

### 4. Look at a report

```bash
kbtc report
```

This works with zero trades. It writes a self-contained HTML file to
`reports/out/latest.html` (plus a timestamped copy) and prints the path. Open it in any
browser; there is no server, no CDN, and no network access required.

---

## Getting an API key

You only need this once you want balances, positions, or actual orders. Public market data
and everything in the Quickstart works without it.

1. Log in to [kalshi.com](https://kalshi.com) and go to **Settings → API Keys**.
2. Click **Create API key**. Give it a name you will recognise later, e.g. `kbtc-laptop`.
3. Kalshi shows you two things:
   - an **API Key ID** (a UUID) — you can look this up again later;
   - an **RSA private key** — **shown exactly once**. If you close that dialog without
     saving it, the key is gone and your only option is to delete the key and create a new
     one.
4. Save the private key to a file. Copy the whole block including the
   `-----BEGIN ... PRIVATE KEY-----` and `-----END ... PRIVATE KEY-----` lines:

   ```bash
   mkdir -p ~/.kalshi
   # paste into the editor, save, quit
   nano ~/.kalshi/key.pem
   chmod 600 ~/.kalshi/key.pem
   ```

   The `chmod 600` matters. `doctor` will warn you if the file is readable by anyone else.

5. Put the key ID and the path in `.env`:

   ```ini
   KALSHI_API_KEY_ID=00000000-0000-0000-0000-000000000000
   KALSHI_PRIVATE_KEY_PATH=~/.kalshi/key.pem
   KALSHI_ENV=demo
   ```

6. Verify:

   ```bash
   kbtc doctor
   ```

   `credentials valid` should be PASS, showing your balance.

**Never commit the private key.** `.gitignore` already excludes `.env`, `*.pem` and `*.key`,
but the responsibility is yours. Nothing in this codebase ever logs the key. If you think it
has leaked, delete the API key in Kalshi's settings immediately — that revokes it.

### Demo vs production

`KALSHI_ENV=demo` points at Kalshi's demo exchange. **Demo credentials work only on demo
hosts and production credentials work only on production hosts** — they are separate
accounts with separate keys. Create a demo key from the demo site, not the live one. Demo is
also seeded with stale and occasionally malformed data, so treat it as a connectivity and
plumbing test, not as a market simulator.

---

## The phased path to going live

Do not skip a phase. Each one has a **go/no-go gate**, and the gates exist because the
cheapest place to discover that the model is wrong is the phase where it costs nothing.

### Phase 0 — Capture (day 1, forever)

```bash
kbtc capture        # leave running
kbtc settlements    # once a day
```

**Gate:** at least **two weeks** of continuous capture, and `kbtc report` showing a
non-trivial number of book snapshots and settled events with `expiration_value`. There is no
shortcut. If your capture has gaps, fix the supervision before continuing.

### Phase 1a — Score the spot proxy (no money, no orders)

```bash
kbtc proxy-score
```

Run this **before** `kbtc calibrate`, because it can kill the whole strategy on its own.

Settlement is the mean of 60 one-second CF Benchmarks BRTI ticks in the final minute.
Real-time BRTI is a licensed product and Kalshi only proxies it to accounts with
credentials, so `kbtc capture` builds a free substitute from public Coinbase / Kraken /
Bitstamp order books. This command asks the only question that matters about that
substitute: **does it actually track the index we settle against?** Every settled event
publishes `expiration_value` — the realised 60-second BRTI average — so we compute our
own average over the identical 60 seconds and measure the error. Free, public, no key.

It needs capture to have been running across at least one hourly close, plus
`kbtc settlements`. With no overlapping event it prints `NO DATA` and exits 0, because
unknown is not the same as failure.

**Gate:** median absolute error must be **< $5**, i.e. under 5% of the $100 strike
spacing. Strikes are $100 apart, so an error comparable to the spacing means we cannot
tell which side of a strike the index will land on during the one minute when that is the
entire trade. `--threshold` moves the bar; the command exits 1 on FAIL so it can gate a
pipeline.

If this FAILs, the settlement-window edge is not reachable from public data, and the
honest options are: pay for a real BRTI feed (Kalshi serves it on the `cfbenchmarks_value`
channel with credentials), apply a constant bias correction if the bias term dominates the
spread and re-score, or drop the settlement-window trade and keep only strategies that do
not need the index in real time. Do not proceed to Phase 2 on a failed proxy.

### Phase 1b — Calibrate (no money, no orders)

```bash
kbtc calibrate
kbtc report
```

**Gate:** the report's **Brier skill score vs market mid must be positive** and stay positive
as more data arrives. Zero means the model is just re-reading the price off the screen —
there is nothing to monetise. Negative means the model is actively adding noise and trading
it is expected to lose money. If you cannot clear this gate, stop. Everything downstream is
built on it.

### Phase 2 — Paper (live markets, simulated fills)

```bash
kbtc paper                  # runs until you stop it
kbtc paper --duration 45    # 45-second smoke test
kbtc paper --hours 3        # stop after 3 hourly events
```

No credentials, no orders, ever — there is no executable order path in this runner at all.
It reads the live ladder, prices it, sizes with Kelly, and simulates fills against the real
queue. `--duration` and `--hours` exist so you can prove the loop works end to end without
committing to a week-long session.

**Where its spot comes from** matters, and the runner prints which source it used every
cycle. It prefers, in order: the licensed **BRTI** feed (credentials only — this is the
actual settlement index); the free **public spot proxy** (Coinbase/Kraken/Bitstamp, the
same composite `kbtc capture` records); and finally spot implied by the **ladder** itself.
The ladder is last because pricing off it is close to circular — it recovers the market's
own view and finds no edge — and because its dispersion gate rejects it often enough that
it alone is not a reliable source. Run `kbtc proxy-score` to find out how much to trust
the middle rung.

**Gate:** at least a week, and the paper P&L in the report must be positive **after** the
modelled fees and after realistic queue assumptions. This is where most theoretical edges
die: the model can be right and still lose, because the fills you assumed you would get are
the ones that only happen when the price is about to move against you. Watch the
**fill quality** section — persistently *positive* slippage on maker fills is adverse
selection and it is fatal.

### Phase 3 — Demo (real API, real orders, fake money)

```bash
ARMED=true kbtc live --yes-i-understand --env demo
```

**Gate:** orders are accepted, cancels land, the reconciliation between what you think you
hold and what the exchange says you hold is exact, and nothing crashes for a full day. This
phase is about plumbing, not profit — demo prices are not realistic.

### Phase 4 — Live, minimum size

```bash
# in .env:  ARMED=true, KALSHI_ENV=prod, MAX_CONTRACTS_PER_ORDER=1
kbtc live --yes-i-understand --env prod
```

**Gate to scale up:** at least 200 settled events at one contract, net-positive after fees,
and a maximum drawdown you actually find boring. Only then raise
`MAX_CONTRACTS_PER_ORDER`, and raise it slowly. Size is the last thing you change, never the
first.

---

## How to read the report

`kbtc report` writes `reports/out/latest.html`. Open it in a browser. From the top:

**"What this means"** — a plain-English paragraph. If you read nothing else, read this. It
says whether the bot made or lost money, and whether the model is beating the market. It is
written to be blunt, including when the answer is bad.

**Headline tiles** — net P&L (after fees, settled events only; open positions are *excluded*
rather than optimistically marked), win rate, events traded, fee drag, and max drawdown.

**Equity curve** — cumulative net P&L, one point per settled hourly event.

**Calibration — is the model any good?** *This is the most important section on the page.*

- The **Brier skill score vs market mid** is the headline number. It compares our
  probabilities to simply reading the market's mid price:
  - **> 0** — our forecasts are genuinely more accurate than the market's. This is the only
    condition under which the strategy makes sense.
  - **≈ 0** — no edge. We are re-deriving the price we could have read for free.
  - **< 0** — we are worse than the market. Trading this model is expected to lose money.
    Stay in paper mode.
- The **reliability diagram** plots forecast probability against what actually happened.
  Dots on the diagonal mean honest probabilities. Above the diagonal means the forecaster is
  too pessimistic (things happen more often than it says); below means overconfident. Dot
  size is the number of forecasts in that bucket; hollow dots have too few samples to mean
  anything. Both our model and the market mid are drawn, so you can see *where* on the
  ladder we differ — usually the tails, which is exactly where the Asian-settlement
  correction bites.
- **Brier score** is mean squared error on probabilities: lower is better, and 0.25 is what
  you get by always saying 50%. **Log loss** punishes confident mistakes much harder, so a
  good Brier score alongside a bad log loss means the model is occasionally very wrong while
  sounding very sure — which is precisely the failure mode that blows up accounts.

**P&L attribution** — per event and per price bucket. Because taker fees are quadratic in
price (`0.07 × P × (1−P)`, maximised at 50c), the price bucket is also the fee bucket. If
your profit is concentrated at the money and your fees are too, you may be running very fast
to stand still.

**Fill quality & fee drag** — maker vs taker mix, and fill price versus the mid at decision
time. Maker fills on KXBTCD are free, so **the maker share is the single biggest lever on
net returns**. If maker fills ever show a non-zero fee, the report flags it loudly: either
the liquidity flag is wrong or the fee field is coming from the wrong place, and until it is
fixed no P&L number on the page can be trusted.

**Data capture** — row counts and the coverage window, so you can see at a glance whether
the recorder has been up.

---

## The safety model

Live trading is behind **two independent gates**. Neither implies the other, and neither
defaults to on.

**Gate 1 — `ARMED`.** An environment variable, normally set in `.env`:

```ini
ARMED=false   # default. The bot computes fair values and logs what it WOULD do.
ARMED=true    # the bot is permitted to send real orders.
```

Nothing in the codebase is allowed to default this to true. With `ARMED=false`, `kbtc live`
refuses to start and tells you exactly how to change it.

**Gate 2 — `--yes-i-understand`.** A flag you must type on the command line, every time:

```bash
kbtc live --yes-i-understand
```

`ARMED=true` alone does nothing. The flag alone does nothing. You need both, and before it
runs, the bot prints your current risk limits and makes you look at them.

### Risk limits

Set in `.env`, read at startup, enforced by the risk layer:

| Variable | Default | Meaning |
|---|---|---|
| `BANKROLL` | `2000` | Total capital the sizing model is allowed to consider |
| `MAX_CONTRACTS_PER_ORDER` | `1` | Hard cap on a single order. Leave at 1 until Phase 4 is done |
| `MAX_POSITION_PER_STRIKE` | `25` | Maximum net contracts held on any one strike |
| `MAX_LOSS_PER_EVENT` | `15` | Stop trading an hourly event after losing this much |
| `MAX_LOSS_PER_DAY` | `50` | Stop trading entirely for the day after losing this much |
| `KELLY_FRACTION` | `0.25` | Fraction of full Kelly used for sizing. Full Kelly is for people who enjoy drawdowns |

These are dollar amounts, not percentages. Lower them if in doubt; the cost of trading too
small is a smaller number in the report, and the cost of trading too big is not.

### Things the safety model does *not* do

- Ctrl-C stops the bot. It does **not** close open positions. Check them on kalshi.com or
  with `kbtc status`.
- Nothing here protects you from a Kalshi outage, an exchange halt, or a BRTI feed problem
  during the settlement window.

---

## Command reference

| Command | Credentials? | What it does |
|---|---|---|
| `kbtc doctor` | optional | Preflight: config, clock skew, connectivity, credentials, arming |
| `kbtc status` | no | Current event, time to close, live strike ladder |
| `kbtc capture` | **no** | Phase 0 recorder. Runs forever. Start this first |
| `kbtc settlements` | no | Backfill settled markets and their `expiration_value` |
| `kbtc proxy-score` | **no** | **Phase 1 gate.** Grade the public spot proxy against realised BRTI settlements. Exits 1 on FAIL |
| `kbtc calibrate` | no | Score the model against the market mid on captured data |
| `kbtc report` | no | Build the static HTML report and print its path |
| `kbtc paper` | no | Paper trade against live markets. Never sends orders |
| `kbtc live` | **yes** | Live trading. Refuses without `ARMED=true` *and* `--yes-i-understand` |

Global option: `--env demo|prod` overrides `KALSHI_ENV` for a single invocation.

Useful extras: `kbtc report --open` opens the report in your browser; `kbtc status --all`
shows all 188 strikes including the ones pinned at 0 and 1.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'kalshi_btc'` right after `pip install -e .` (macOS).**
Check the editable install's path file:

```bash
ls -lO .venv/lib/python3.12/site-packages/*.pth
```

If it says `hidden`, macOS has set the `UF_HIDDEN` flag on it and **Python 3.12+ silently
skips hidden `.pth` files** — the package is installed, but nothing puts it on the path.
Fix:

```bash
chflags nohidden .venv/lib/python3.12/site-packages/*.pth
```

Alternatively, skip the install entirely and run from the source tree:

```bash
PYTHONPATH=src python -m kalshi_btc.cli doctor
```

**`kbtc report` says the database is locked / built from a snapshot.**
That is expected and harmless. DuckDB allows one writer process, so while `kbtc capture`
is running the report copies the database and reads the copy. The note on the page tells
you it did. Nothing is lost — the report is a point-in-time view either way.

**`kbtc paper` says the capture database is locked.**
DuckDB allows exactly one *writer* process, and both `kbtc capture` and `kbtc paper` are
writers — paper records ladder snapshots, decisions and simulated fills. So the two cannot
run at the same time. Stop `kbtc capture` for the duration of the paper session; paper
records the ladder itself, so you keep collecting ladder history while it runs and only
lose the public spot-proxy rows (which paper does not write). Restart capture afterwards.

`kbtc report`, `kbtc calibrate` and `kbtc proxy-score` are *readers* and work fine while
capture holds the lock. Only the writers collide.

**`kbtc doctor` says my clock is off.**
Turn on NTP and leave it on. Settlement is a mean of sixty one-second ticks and the last of
them lands *at* the close, so decisions in the final minute are second-sensitive. The `Date`
header only has one-second resolution, so anything under ±0.5s is measurement noise; a WARN
above 1s is real and a FAIL above 5s means you should not trade until it is fixed.

**Everything looks empty — bids and volumes are all zero.**
You are reading the legacy integer-cent fields. The current API returns
`yes_bid_dollars`, `yes_ask_dollars`, `volume_fp`, `open_interest_fp`, `yes_bid_size_fp` as
decimal **strings**, and the old `yes_bid` / `volume` / `open_interest` integer fields are
still present in responses but are **always zero**. This is the single most common way a bot
built from an older tutorial silently sees a dead market. The same applies to order books:
the data is under `orderbook_fp.yes_dollars` / `no_dollars`, not `orderbook.yes`.

**I'm trading `KXBTC` and getting terrible fills.**
Wrong series. `KXBTCD` is the hourly above/below series this bot targets. `KXBTC` is a
different, far less liquid hourly range/bracket series with roughly 80x less volume and
two-to-four cent spreads. Every ticker this bot touches starts with `KXBTCD-`, and there is
an explicit guard for it, but if you are hand-testing endpoints, check the prefix.

**Rate limits / HTTP 429.**
Kalshi's limits are token buckets, not request counts. On the Basic tier you get 200 read
tokens/sec and 100 write tokens/sec, refilled continuously. Most endpoints cost 10 tokens
(so ~20 GET/s), cancels cost 2, and the CF Benchmarks passthrough costs 50. Basic has no
write burst capacity. The client already paces itself with matching token buckets and backs
off on 429; if you are still hitting limits, you are polling REST where you should be using
the WebSocket.

**The WebSocket keeps disconnecting.**
Expected on any long-lived connection. The right response is to reconnect, resubscribe, and
**re-snapshot the book** — do not resume from a stale sequence number. If you are near the
top of the hour when it happens, treat your position as unknown until you have re-read it
from `/portfolio/positions`.

**Two endpoints disagree by a cent.**
REST responses are cached for about a second and different endpoints can be a tick out of
sync. This is normal and it is why REST must never be used for live trading decisions. Use
the WebSocket for anything that moves money; use REST for setup, backfill and
reconciliation.

**`kbtc live` refuses to start.**
That is the design. Read the panel it printed: it names which of the two gates you are
missing and gives you the exact command.

**Demo shows nonsense (events closing in the past, `expiration_value: "a"`).**
The demo exchange is seeded with synthetic junk. The tooling is defensive about it, but do
not draw conclusions about prices, spreads or fills from demo.

---

## Repository layout

```
src/kalshi_btc/
  config.py          settings, risk limits, the ARMED flag
  core/types.py      Book, MarketSnapshot, Fill — Decimal everywhere
  core/fees.py       the fee schedule (maker is free; there is no settlement fee)
  model/pricing.py   the Asian-settlement digital pricer — the heart of the edge
  model/vol.py       volatility estimation
  model/calibration.py  scoring the model against the market
  exec/client.py     Kalshi REST client, RSA-PSS request signing, rate limiting
  feed/              WebSocket market data and the BRTI channel
  store/db.py        DuckDB capture store
  strategy/          quoting and edge logic
  risk/              position and loss limits
  runner/capture.py  the Phase 0 recorder
  runner/calibrate.py  wires the store + vol model into the Calibrator, writes calibration.json
  runner/            paper and live loops
  report/report.py   the self-contained HTML report
  cli.py             the kbtc command line
docs/GROUND_TRUTH.md verified market facts, with sources. Read before changing anything.
```

`docs/GROUND_TRUTH.md` is worth your time. It records the market facts that were verified
against primary sources, each with its citation, so that nobody has to re-derive them or
"fix" the code back to something that is wrong.

---

## Licence and disclaimer

For personal research use. Trading prediction markets involves risk of loss. Nothing in this
repository is financial advice, and the authors are not responsible for your P&L. The
default configuration will not place an order; making it place orders requires two
deliberate acts by you.
