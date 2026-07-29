"""`kbtc` command line.

WHY THIS FILE LOOKS THE WAY IT DOES
-----------------------------------
Two design decisions dominate:

1. **Every heavy import is lazy.** Nothing outside the standard library, typer and rich
   is imported at module scope. `kbtc --help` therefore starts instantly, and — more
   importantly — the CLI still loads when a sibling module (store, runner, model) is
   mid-edit or not written yet. A trading tool whose help screen breaks because an
   unrelated module has a syntax error is a tool you cannot debug at 3am.

2. **`doctor` is the flagship.** Most failures in this system are environmental, not
   algorithmic: a missing .env, a key file with the wrong permissions, demo credentials
   pointed at production, or a laptop clock two seconds slow. We settle on a 60-second
   average of one-second index ticks, so a clock that is off by more than a second is a
   correctness bug, not a nuisance. `doctor` finds all of that before you lose money to it.

Safety: `live` requires TWO independent gates — `ARMED=true` in the environment AND an
explicit `--yes-i-understand` flag on the command line. Neither implies the other, and
neither has a default that turns it on.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="kbtc",
    help="Kalshi hourly BTC (KXBTCD) trading bot. Start with `kbtc doctor`.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# doctor result levels, in escalating order of "you must fix this".
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_LEVEL_STYLE = {PASS: "bold green", WARN: "bold yellow", FAIL: "bold red"}

# We settle on second-resolution index ticks, so a clock offset of a second is material.
CLOCK_WARN_SECONDS = 1.0
CLOCK_FAIL_SECONDS = 5.0


# ======================================================================================
# Shared helpers
# ======================================================================================
def _settings() -> Any:
    from kalshi_btc.config import get_settings

    return get_settings()


def _fail(msg: str, hint: str = "") -> None:
    console.print(f"[bold red]✗[/] {msg}")
    if hint:
        console.print(f"  [dim]{hint}[/]")
    raise typer.Exit(code=1)


def _run_entrypoint(module_path: str, names: tuple[str, ...], **kwargs: Any) -> Any:
    """Import a sibling runner lazily and call whichever entry point it exposes.

    Sibling modules are being written in parallel, so we tolerate reasonable naming
    variation and pass only the keyword arguments the target actually accepts. If the
    module is missing we say exactly which one and stop, rather than emitting a
    traceback that looks like the bot crashed.
    """
    import importlib

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        _fail(
            f"`{module_path}` is not available yet ({exc}).",
            "This command depends on a module that has not been built. "
            "`kbtc doctor`, `kbtc status`, `kbtc capture` and `kbtc report` are the "
            "commands that work earliest.",
        )
        return None

    fn = next((getattr(mod, n) for n in names if hasattr(mod, n)), None)
    if fn is None:
        _fail(
            f"`{module_path}` exposes none of {names}.",
            "The module exists but its entry point is named something else.",
        )
        return None

    try:
        params = inspect.signature(fn).parameters
        accepts_all = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        call_kwargs = kwargs if accepts_all else {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        call_kwargs = kwargs

    # Never silently discard an argument the caller asked for. Dropping `duration`
    # because the target calls it `duration_s` turns `--duration 12` into "run forever",
    # which reads exactly like a hang. The same silent-drop would happily swallow a risk
    # limit, so this is a hard error rather than a warning.
    dropped = {k: v for k, v in kwargs.items() if k not in call_kwargs and v is not None}
    if dropped:
        raise TypeError(
            f"{module_path}.{fn.__name__}() does not accept {sorted(dropped)}; "
            f"it accepts {sorted(params)}. Refusing to run with silently ignored arguments."
        )

    result = fn(**call_kwargs)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def _fmt_td(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    seconds = abs(int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h:d}:{m:02d}:{s:02d}" if h else f"{sign}{m:d}:{s:02d}"


# ======================================================================================
# Global options
# ======================================================================================
@app.callback()
def main(
    env: Annotated[
        str | None,
        typer.Option(
            "--env",
            "-e",
            help="Override KALSHI_ENV for this invocation: [bold]demo[/] or [bold]prod[/].",
        ),
    ] = None,
) -> None:
    """Kalshi KXBTCD trading bot."""
    from kalshi_btc.config import load_dotenv

    load_dotenv()
    if env:
        low = env.strip().lower()
        if low not in {"demo", "prod"}:
            _fail(f"--env must be 'demo' or 'prod', got {env!r}")
        # Set AFTER load_dotenv: load_dotenv uses setdefault, so the flag wins.
        os.environ["KALSHI_ENV"] = low


# ======================================================================================
# doctor
# ======================================================================================
@app.command()
def doctor() -> None:
    """Preflight check: config, clock skew, connectivity, credentials, arming.

    Run this first, and run it again any time something behaves oddly. Exits non-zero
    if anything is a hard FAIL, so it is safe to use in a startup script.
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold]kbtc doctor[/] — preflight for the KXBTCD bot",
            border_style="cyan",
        )
    )
    rows: list[tuple[str, str, str, str]] = []

    def add(check: str, level: str, detail: str, action: str = "") -> None:
        rows.append((check, level, detail, action))

    # ---------------------------------------------------------------- config on disk
    env_file = Path(".env")
    if env_file.exists():
        add(".env file", PASS, f"found at {env_file.resolve()}", "")
    elif Path(".env.example").exists():
        add(
            ".env file",
            WARN,
            "missing — defaults will be used",
            "cp .env.example .env   (not needed for `kbtc capture`)",
        )
    else:
        add(".env file", WARN, "missing, and no .env.example either", "Create a .env")

    try:
        s = _settings()
    except Exception as exc:
        add("config parses", FAIL, f"{type(exc).__name__}: {exc}", "Fix the bad value in .env")
        _print_doctor(rows)
        raise typer.Exit(code=1)
    add("config parses", PASS, s.describe(), "")

    # ---------------------------------------------------------------- data directory
    try:
        s.data_dir.mkdir(parents=True, exist_ok=True)
        probe = s.data_dir / ".kbtc-write-test"
        probe.write_text("ok")
        probe.unlink()
        add("data dir writable", PASS, str(s.data_dir.resolve()), "")
    except Exception as exc:
        add("data dir writable", FAIL, f"{s.data_dir}: {exc}", "Fix permissions or set DATA_DIR")

    # ---------------------------------------------------------------- network + clock
    net = asyncio.run(_probe_network(s))
    if net["error"]:
        add(
            "REST reachable",
            FAIL,
            f'{s.rest_base} — {net["error"]}',
            "Check your internet connection, VPN, or whether Kalshi is up.",
        )
    else:
        add(
            "REST reachable",
            PASS,
            f'{s.rest_base} — HTTP {net["status"]} in {net["latency_ms"]:.0f} ms',
            "",
        )

        skew = net["skew"]
        if skew is None:
            add(
                "clock skew",
                WARN,
                "server sent no Date header; cannot measure",
                "Make sure NTP is enabled anyway.",
            )
        else:
            # The Date header has one-second resolution, so ±0.5s of the reading is
            # quantisation, not real drift. We report the raw number and say so.
            detail = f"local clock is {skew:+.2f}s vs Kalshi (±0.5s measurement floor)"
            if abs(skew) >= CLOCK_FAIL_SECONDS:
                add("clock skew", FAIL, detail, "Enable NTP: settlement is a 60x1s average.")
            elif abs(skew) >= CLOCK_WARN_SECONDS:
                add("clock skew", WARN, detail, "Enable NTP — we trade to the second.")
            else:
                add("clock skew", PASS, detail, "")

        if net["open_events"] is None:
            add("KXBTCD series live", WARN, "could not list open events", "")
        elif net["open_events"] == 0:
            add(
                "KXBTCD series live",
                WARN,
                "no open events right now",
                "Unusual for an hourly series — check kalshi.com.",
            )
        else:
            add("KXBTCD series live", PASS, f'{net["open_events"]} open event(s)', "")

    # ---------------------------------------------------------------- credentials
    if not s.has_credentials:
        add(
            "API credentials",
            WARN,
            "not configured — public market data only",
            "Fine for `capture`/`status`. Needed for balances, orders, CF Benchmarks. "
            "See README → 'Getting an API key'.",
        )
    else:
        key_path = Path(s.private_key_path).expanduser()
        if not key_path.exists():
            add(
                "private key file",
                FAIL,
                f"{key_path} does not exist",
                "Kalshi shows the RSA private key exactly ONCE. If you lost it, "
                "delete the key in Settings → API Keys and create a new one.",
            )
        else:
            mode = key_path.stat().st_mode & 0o777
            if mode & 0o077:
                add(
                    "private key perms",
                    WARN,
                    f"{key_path} is mode {mode:o} (group/other readable)",
                    f"chmod 600 {key_path}",
                )
            else:
                add("private key perms", PASS, f"{key_path} is mode {mode:o}", "")

            try:
                from kalshi_btc.exec.client import load_private_key

                load_private_key(key_path)
                add("private key loads", PASS, "valid RSA private key", "")
            except Exception as exc:
                add(
                    "private key loads",
                    FAIL,
                    f"{type(exc).__name__}: {exc}",
                    "The file must be the unencrypted PEM Kalshi gave you.",
                )
                key_path = None  # type: ignore[assignment]

            if key_path is not None:
                bal = asyncio.run(_probe_balance(s))
                if bal["error"]:
                    add(
                        "credentials valid",
                        FAIL,
                        f'/portfolio/balance → {bal["error"]}',
                        "Demo keys only work on demo hosts and prod keys only on prod. "
                        f"You are on env={s.env}.",
                    )
                else:
                    add("credentials valid", PASS, f'balance {bal["balance"]}', "")

    # ---------------------------------------------------------------- arming
    if s.armed:
        add(
            "ARMED",
            WARN,
            "TRUE — real orders are permitted",
            "`kbtc live` will still require --yes-i-understand. "
            "Set ARMED=false in .env to disarm.",
        )
    else:
        add("ARMED", PASS, "false — no orders will ever be sent", "")

    worst = _print_doctor(rows)
    _print_next_steps(s, rows)
    raise typer.Exit(code=1 if worst == FAIL else 0)


def _print_doctor(rows: list[tuple[str, str, str, str]]) -> str:
    table = Table(show_lines=False, header_style="bold", box=None, padding=(0, 2, 0, 0))
    table.add_column("Check", style="bold", no_wrap=True)
    table.add_column("", no_wrap=True)
    table.add_column("Detail")
    table.add_column("Next step", style="dim")
    for check, level, detail, action in rows:
        table.add_row(check, Text(level, style=_LEVEL_STYLE[level]), detail, action)
    console.print()
    console.print(table)
    console.print()

    n_fail = sum(1 for r in rows if r[1] == FAIL)
    n_warn = sum(1 for r in rows if r[1] == WARN)
    if n_fail:
        console.print(f"[bold red]{n_fail} FAIL[/], {n_warn} WARN — fix the failures above.")
        return FAIL
    if n_warn:
        console.print(f"[bold yellow]{n_warn} WARN[/], 0 FAIL — usable, read the warnings.")
        return WARN
    console.print("[bold green]All checks passed.[/]")
    return PASS


def _print_next_steps(s: Any, rows: list[tuple[str, str, str, str]]) -> None:
    levels = {r[0]: r[1] for r in rows}
    steps: list[str] = []
    if levels.get("REST reachable") == PASS:
        steps.append("[bold]kbtc capture[/]   start recording now — history cannot be backfilled")
        steps.append("[bold]kbtc status[/]    see the live strike ladder")
    if not s.has_credentials:
        steps.append("[dim]Add an API key when you want balances, fills or orders (README).[/]")
    steps.append("[bold]kbtc report[/]    build the HTML report (works with zero trades)")
    if steps:
        console.print()
        console.print(Panel("\n".join(steps), title="Next", border_style="dim", expand=False))


async def _probe_network(s: Any) -> dict[str, Any]:
    """One unauthenticated round trip: reachability, latency, clock skew, liveness."""
    import aiohttp

    from kalshi_btc.config import SERIES_TICKER

    out: dict[str, Any] = {
        "error": None,
        "status": None,
        "latency_ms": 0.0,
        "skew": None,
        "open_events": None,
    }
    url = f'{s.rest_base.rstrip("/")}/series/{SERIES_TICKER}'
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            t0 = time.time()
            async with sess.get(url) as resp:
                await resp.read()
                t1 = time.time()
                out["status"] = resp.status
                out["latency_ms"] = (t1 - t0) * 1000.0
                date_hdr = resp.headers.get("Date")
                if date_hdr:
                    from email.utils import parsedate_to_datetime

                    try:
                        server = parsedate_to_datetime(date_hdr)
                        if server.tzinfo is None:
                            server = server.replace(tzinfo=UTC)
                        # Compare against the midpoint of the round trip: that is the
                        # best unbiased estimate of "our time when the server stamped it".
                        local_mid = datetime.fromtimestamp((t0 + t1) / 2, tz=UTC)
                        out["skew"] = (local_mid - server).total_seconds()
                    except (TypeError, ValueError):
                        pass

            try:
                from kalshi_btc.exec.client import KalshiClient

                async with KalshiClient(s, session=sess) as client:
                    events = await client.get_events(status="open", limit=10)
                    out["open_events"] = len(events)
            except Exception:
                out["open_events"] = None
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


async def _probe_balance(s: Any) -> dict[str, Any]:
    from kalshi_btc.exec.client import KalshiClient

    try:
        async with KalshiClient(s) as client:
            data = await client.get_balance()
        raw = data.get("balance_dollars") or data.get("balance")
        return {"error": None, "balance": f"${raw}" if raw is not None else str(data)[:80]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "balance": None}


# ======================================================================================
# status
# ======================================================================================
@app.command()
def status(
    event: Annotated[
        str | None,
        typer.Option("--event", help="Show a specific event ticker instead of the nearest one."),
    ] = None,
    show_all: Annotated[
        bool,
        typer.Option("--all", help="Include degenerate strikes pinned at 0 or 1."),
    ] = False,
) -> None:
    """Show settings and the current live KXBTCD strike ladder. No credentials needed."""
    s = _settings()
    console.print()
    console.print(Panel.fit(s.describe(), title="settings", border_style="cyan"))
    try:
        asyncio.run(_show_status(s, event, show_all))
    except Exception as exc:
        _fail(f"could not fetch market data: {type(exc).__name__}: {exc}", "Try `kbtc doctor`.")


def _earliest_close(event: dict) -> datetime | None:
    """Earliest market close_time in an event, parsed from the raw payload only."""
    times: list[datetime] = []
    for m in event.get("markets") or []:
        raw = m.get("close_time")
        if not raw:
            continue
        try:
            t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        times.append(t if t.tzinfo else t.replace(tzinfo=UTC))
    return min(times) if times else None


async def _show_status(s: Any, event_ticker: str | None, show_all: bool) -> None:
    from kalshi_btc.core.types import MarketSnapshot
    from kalshi_btc.exec.client import KalshiClient

    async with KalshiClient(s) as client:
        if event_ticker:
            ev = await client.get_event(event_ticker)
        else:
            events = await client.get_events(status="open", limit=20)
            if not events:
                console.print("[yellow]No open KXBTCD events right now.[/]")
                return
            # Nearest close = the event actually being traded. Read close_time straight
            # off the raw payload: the demo exchange serves stale events with garbage in
            # other fields (e.g. expiration_value="a"), and a full parse would blow up.
            now = datetime.now(UTC)
            ranked = [(t, e) for e in events if (t := _earliest_close(e)) and t > now]
            if not ranked:
                console.print(
                    "[yellow]All open events have already closed (stale demo data?).[/]"
                )
                return
            ev = min(ranked, key=lambda pair: pair[0])[1]

        markets = ev.get("markets") or []
        if not markets:
            console.print(f"[yellow]Event {ev.get('event_ticker')} has no markets attached.[/]")
            return

        snaps = []
        for m in markets:
            try:
                snaps.append(MarketSnapshot.from_api(m))
            except Exception:
                continue
        if not snaps:
            console.print("[yellow]Could not parse any markets in this event.[/]")
            return

        close = _earliest_close(ev) or min(sn.close_time for sn in snaps)
        remaining = (close - datetime.now(UTC)).total_seconds()
        title = (
            f"{ev.get('event_ticker', '?')}  ·  closes "
            f"{close.strftime('%H:%M:%S UTC')}  ·  T-{_fmt_td(remaining)}"
        )
        if 0 < remaining <= 60:
            title += "  ·  [bold red]SETTLEMENT WINDOW OPEN[/]"

        live = [sn for sn in snaps if sn.is_live]
        shown = sorted(snaps if show_all else live, key=lambda sn: sn.strike)

        table = Table(title=title, header_style="bold", box=None, padding=(0, 2, 0, 0))
        table.add_column("Strike", justify="right")
        table.add_column("Bid", justify="right")
        table.add_column("Ask", justify="right")
        table.add_column("Mid", justify="right")
        table.add_column("Spread", justify="right")
        table.add_column("Bid size", justify="right")
        table.add_column("Ask size", justify="right")
        table.add_column("Volume", justify="right")
        for sn in shown:
            spread = sn.yes_ask - sn.yes_bid
            style = "" if sn.is_live else "dim"
            table.add_row(
                f"${sn.strike:,.2f}",
                f"{int(sn.yes_bid * 100)}c",
                f"{int(sn.yes_ask * 100)}c",
                f"{sn.mid * 100:.1f}c",
                f"{int(spread * 100)}c",
                f"{sn.yes_bid_size:g}",
                f"{sn.yes_ask_size:g}",
                f"{sn.volume:g}",
                style=style,
            )
        console.print()
        console.print(table)
        console.print()
        console.print(
            f"[dim]{len(live)} of {len(snaps)} strikes carry a live two-sided quote; "
            "the rest are pinned at 0 or 1 and are not worth modelling. "
            "Use --all to see them.[/]"
        )


# ======================================================================================
# capture
# ======================================================================================
@app.command()
def capture(
    duration: Annotated[
        int | None,
        typer.Option("--duration", help="Stop after N seconds (default: run forever)."),
    ] = None,
) -> None:
    """Record the order book and BRTI feed. Runs forever. NO credentials required.

    Start this now and leave it running. Kalshi order book history cannot be purchased
    or backfilled — an hour you did not record is gone permanently.
    """
    s = _settings()
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Recording KXBTCD[/]\n{s.describe()}\n\n"
            "[dim]Ctrl-C to stop. Nothing here places orders.[/]",
            border_style="cyan",
        )
    )
    try:
        _run_entrypoint(
            "kalshi_btc.runner.capture",
            ("run_capture", "main", "run"),
            settings=s,
            duration_s=duration,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Capture stopped.[/] Data written to "
                      f"[bold]{s.data_dir}[/].")


# ======================================================================================
# settlements
# ======================================================================================
@app.command()
def settlements(
    limit: Annotated[
        int, typer.Option("--limit", help="Max settled markets to pull per page.")
    ] = 1000,
    events: Annotated[
        int,
        typer.Option(
            "--events",
            help="Stop after this many distinct events (~24/day). Bounds the walk.",
        ),
    ] = 480,
) -> None:
    """Backfill settled markets and their `expiration_value` into the store.

    `expiration_value` is the realised BRTI 60-second average — free ground truth for
    scoring the model. This is a public endpoint; no credentials needed.

    A page of 1000 markets holds only ~5 events (188 strikes each share one settlement
    value), so the walk is bounded by --events rather than run to exhaustion. Re-runs
    stop as soon as they meet events already on file, which makes them nearly free.
    """
    s = _settings()
    console.print(
        f"[dim]Fetching settled KXBTCD markets (up to {events:,} events)…[/]"
    )
    result = _run_entrypoint(
        "kalshi_btc.runner.settlements",
        ("backfill_settlements", "run_settlements", "main", "run"),
        settings=s,
        limit=limit,
        max_events=events,
    )
    if isinstance(result, int):
        console.print(f"[green]Stored {result:,} newly settled event(s).[/]")
    else:
        console.print("[green]Settlement backfill complete.[/]")


# ======================================================================================
# calibrate
# ======================================================================================
@app.command()
def calibrate(
    days: Annotated[
        int, typer.Option("--days", help="Only score captured data from the last N days.")
    ] = 30,
) -> None:
    """Score the pricing model against the market mid on captured data.

    Writes Brier score, log loss and reliability bins so `kbtc report` can show them.
    A positive skill score means our probabilities beat the price on the screen; that
    is the only evidence that this bot has an edge at all.
    """
    s = _settings()
    try:
        result = _run_entrypoint(
            "kalshi_btc.runner.calibrate",
            ("run_calibration", "calibrate", "main", "run"),
            settings=s,
            days=days,
        )
    except RuntimeError as exc:
        # The runner raises RuntimeError with an operator-readable sentence when there is
        # not enough data yet. That is an expected state, not a crash.
        _fail(str(exc))
        return

    if isinstance(result, dict):
        table = Table(box=None, header_style="bold", padding=(0, 2, 0, 0))
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for k, v in result.items():
            if isinstance(v, bool):
                shown = "[green]yes[/]" if v else "[yellow]no[/]"
            elif isinstance(v, float):
                shown = f"{v:.6f}"
            else:
                shown = str(v)
            table.add_row(str(k), shown)
        console.print()
        console.print(table)

        skill = result.get("skill_vs_market")
        if isinstance(skill, (int, float)):
            cls = "green" if skill > 0 else "red"
            console.print(
                f"\nBrier skill score vs market mid: [{cls}]{skill:+.4f}[/] "
                + ("[dim](positive = our probabilities beat the screen)[/]" if skill > 0
                   else "[dim](not positive — there is no accuracy edge to trade)[/]")
            )
        if result.get("out_of_sample") is False:
            console.print(
                "[yellow]IN-SAMPLE:[/] not enough settlement history to hold out a test "
                "window, so this is an upper bound, not a backtest."
            )
    console.print("\n[dim]Now run [bold]kbtc report[/bold] to see the reliability diagram.[/]")


# ======================================================================================
# report
# ======================================================================================
@app.command()
def report(
    out: Annotated[
        Path | None, typer.Option("--out", help="Output directory (default reports/out).")
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the report in your browser when done.")
    ] = False,
) -> None:
    """Build the self-contained HTML report and print its path.

    Works with zero trades — on a fresh install it renders what capture and calibration
    data exist and says plainly that there is no P&L yet.
    """
    s = _settings()
    from kalshi_btc.report.report import build_report, gather

    data = gather(s)
    path = build_report(s, out, data=data)

    console.print()
    if data.has_trades:
        cls = "green" if data.net_pnl >= 0 else "red"
        console.print(
            f"Net P&L [{cls}]${data.net_pnl:,.2f}[/] across "
            f"{data.events_traded} settled events."
        )
    else:
        console.print("[yellow]No completed trades yet[/] — the report says so plainly.")
    ss = data.skill_score
    if ss is not None:
        cls = "green" if ss > 0 else "red"
        console.print(f"Brier skill score vs market mid: [{cls}]{ss:+.4f}[/]")
    for note in data.notes[:4]:
        console.print(f"  [dim]· {note}[/]")

    console.print(f"\n[bold]{path.resolve()}[/]")
    if open_browser:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())


# ======================================================================================
# paper
# ======================================================================================
@app.command()
def paper(
    hours: Annotated[
        int | None, typer.Option("--hours", help="Stop after N hourly events.")
    ] = None,
) -> None:
    """Paper trade against live markets. No orders are sent, ever.

    This is the phase where you find out whether the edge survives contact with real
    spreads and real queue position. Run it for at least a week before considering
    `kbtc live`.
    """
    s = _settings()
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Paper trading[/] — simulated fills only, no money at risk\n{s.describe()}",
            border_style="cyan",
        )
    )
    try:
        _run_entrypoint(
            "kalshi_btc.runner.paper",
            ("run_paper", "main", "run"),
            settings=s,
            hours=hours,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Paper session stopped.[/] Run `kbtc report`.")


# ======================================================================================
# live
# ======================================================================================
@app.command()
def live(
    yes_i_understand: Annotated[
        bool,
        typer.Option(
            "--yes-i-understand",
            help="Required acknowledgement that this risks REAL MONEY.",
        ),
    ] = False,
    hours: Annotated[
        int | None, typer.Option("--hours", help="Stop after N hourly events.")
    ] = None,
) -> None:
    """Live trading with real money. Requires ARMED=true AND --yes-i-understand.

    Two independent gates, deliberately. An environment variable alone cannot start
    live trading, and a command-line flag alone cannot either.
    """
    s = _settings()

    if not s.armed:
        console.print()
        console.print(
            Panel(
                "[bold red]Refusing to trade live: ARMED is not set.[/]\n\n"
                "This is the safety flag. To enable it, edit [bold].env[/] and set:\n\n"
                "    [bold]ARMED=true[/]\n\n"
                "or export it for one session:\n\n"
                "    [bold]ARMED=true kbtc live --yes-i-understand[/]\n\n"
                "[dim]ARMED alone is not enough — you also need --yes-i-understand.[/]",
                title="blocked · gate 1 of 2",
                border_style="red",
            )
        )
        raise typer.Exit(code=2)

    if not yes_i_understand:
        risk = s.risk
        console.print()
        console.print(
            Panel(
                "[bold red]REAL MONEY IS AT RISK.[/]\n\n"
                "ARMED=true is set, so the only thing standing between this command and "
                "live orders on your Kalshi account is the acknowledgement flag.\n\n"
                "Current risk limits:\n"
                f"  bankroll                 ${risk.bankroll}\n"
                f"  max contracts per order  {risk.max_contracts_per_order}\n"
                f"  max position per strike  {risk.max_position_per_strike}\n"
                f"  max loss per event       ${risk.max_loss_per_event}\n"
                f"  max loss per day         ${risk.max_loss_per_day}\n"
                f"  Kelly fraction           {risk.kelly_fraction}\n\n"
                f"Environment: [bold]{s.env}[/]\n\n"
                "If you accept this, re-run with:\n\n"
                "    [bold]kbtc live --yes-i-understand[/]",
                title="blocked · gate 2 of 2",
                border_style="red",
            )
        )
        raise typer.Exit(code=2)

    if not s.has_credentials:
        _fail(
            "ARMED and acknowledged, but there are no API credentials.",
            "Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env, then `kbtc doctor`.",
        )

    if s.env != "prod":
        console.print(
            f"[yellow]Note:[/] KALSHI_ENV={s.env}, so orders go to the demo exchange, "
            "not to real markets. Use --env prod when you actually mean it."
        )

    console.print()
    console.print(
        Panel.fit(
            f"[bold red]LIVE TRADING[/] — real orders, real money\n{s.describe()}\n"
            f"[dim]Max loss per day ${s.risk.max_loss_per_day}. Ctrl-C stops the bot; "
            "it does not close open positions.[/]",
            border_style="red",
        )
    )
    try:
        _run_entrypoint(
            "kalshi_btc.runner.live",
            ("run_live", "main", "run"),
            settings=s,
            hours=hours,
        )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Live session stopped.[/] Open positions are UNCHANGED — "
            "check them on kalshi.com or with `kbtc status`."
        )


def run() -> None:
    """setuptools console-script shim, and `python -m kalshi_btc.cli`."""
    app()


if __name__ == "__main__":
    run()
