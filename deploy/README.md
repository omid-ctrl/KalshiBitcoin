# Deploying kbtc

"Deploying" this bot means one thing: **keeping `kbtc capture` running 24/7**. Kalshi sells
no historical order-book data and has no backfill endpoint, so every hour you are not
recording is research data that cannot be bought back at any price.

Nothing in here places an order. `capture` has no order path at all, and `live` still
requires both `ARMED=true` and `--yes-i-understand`.

---

## macOS (your current setup)

```bash
cp deploy/com.kalshibtc.capture.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.kalshibtc.capture.plist
launchctl list | grep kalshibtc          # confirm it is running
tail -f data/capture.log                 # watch it
```

Stop it:

```bash
launchctl unload ~/Library/LaunchAgents/com.kalshibtc.capture.plist
```

**Two macOS gotchas that will silently cost you data:**

1. **Sleep.** A sleeping Mac records nothing. Either `caffeinate -s` alongside it, or set
   Energy Saver to prevent sleep on power. This is the single biggest source of gaps.
2. **The write lock.** DuckDB allows one writer. `kbtc paper` is also a writer, so unload
   the agent before a paper session. `kbtc report` and `kbtc proxy-score` read fine
   alongside a running capture.

---

## VPS / Linux (when you outgrow the laptop)

A small box is enough — this is a few hundred rows a second and one WebSocket. 1 vCPU and
1 GB is comfortable. Put it near `us-east-1` if you ever get to the latency-sensitive
strategies; for data capture the region is irrelevant.

```bash
sudo cp deploy/kbtc-capture.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kbtc-capture
journalctl -u kbtc-capture -f
```

Set `NTP` on (`timedatectl set-ntp true`). We settle on a 60-second average of one-second
index ticks, so a clock that drifts a second is a correctness problem, not a cosmetic one.
`kbtc doctor` measures your skew against Kalshi's own clock.

---

## What to run, and in what order

| Command | Cadence | Needs credentials |
|---|---|---|
| `kbtc capture` | always, forever | no |
| `kbtc settlements` | daily (or after any close you care about) | no |
| `kbtc proxy-score` | after a few closes | no |
| `kbtc calibrate` | weekly, once there is history | no |
| `kbtc report` | whenever you want to look | no |
| `kbtc paper` | when evaluating the strategy (stop capture first) | no |
| `kbtc live` | only after the gates pass | **yes** |

## Secrets

The RSA private key never belongs in the repo. Keep it at `~/.kalshi/key.pem` with `600`
permissions and point `KALSHI_PRIVATE_KEY_PATH` at it. `.env`, `*.pem` and `*.key` are all
gitignored; `git ls-files` should never show any of them.

On a VPS, prefer a dedicated user and `chmod 600` — a key readable by other accounts on a
shared box is a key you should rotate.
