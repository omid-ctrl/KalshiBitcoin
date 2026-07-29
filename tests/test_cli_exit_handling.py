"""`typer.Exit` is a `RuntimeError`, and that is a trap worth a test.

    >>> typer.Exit.__mro__
    (Exit, RuntimeError, Exception, BaseException, object)
    >>> str(typer.Exit(code=1))
    ''

Several commands catch `RuntimeError` to turn an expected operator state ("not enough
data yet", "the capture database is locked") into a clean message instead of a traceback.
That handler also catches `typer.Exit` — which `_fail` raises AFTER it has already printed
a good message — and re-reports it as a blank `✗` with no text. Observed live: `kbtc paper`
against a locked database printed the full lock explanation and then a bare `✗`.

Worse, it would convert a *successful* `typer.Exit(0)` into a failure exit.

So every `except RuntimeError` in the CLI must be preceded by `except typer.Exit: raise`.
These tests pin that for each command that has such a handler.

NO NETWORK: `_run_entrypoint` is monkeypatched, so no command reaches a runner.
"""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from kalshi_btc import cli

runner = CliRunner()

# Commands that wrap `_run_entrypoint` in an `except RuntimeError` handler.
COMMANDS = ["calibrate", "paper"]


def test_typer_exit_really_is_a_runtime_error():
    """The premise of this whole file. If typer ever changes this, these tests are moot."""
    assert issubclass(typer.Exit, RuntimeError)
    assert str(typer.Exit(code=1)) == ""


@pytest.mark.parametrize("command", COMMANDS)
def test_a_clean_zero_exit_is_not_turned_into_a_failure(monkeypatch, command):
    """`typer.Exit(0)` means success; the RuntimeError handler must not downgrade it."""
    monkeypatch.setattr(cli, "_run_entrypoint", lambda *a, **k: (_ for _ in ()).throw(typer.Exit(code=0)))
    result = runner.invoke(cli.app, ["--env", "prod", command])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_an_already_reported_failure_is_not_reported_twice(monkeypatch, command):
    """`_fail` prints then raises typer.Exit(1). We must not print a second, blank line."""
    def already_failed(*_a, **_k):
        cli.console.print("[bold red]✗[/] the real explanation")
        raise typer.Exit(code=1)

    monkeypatch.setattr(cli, "_run_entrypoint", already_failed)
    result = runner.invoke(cli.app, ["--env", "prod", command])
    assert result.exit_code == 1
    assert result.output.count("✗") == 1, f"duplicate failure marker:\n{result.output}"
    assert "the real explanation" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_a_genuine_runtime_error_is_still_rendered_as_a_message(monkeypatch, command):
    """The handler must keep doing its actual job: no traceback for an expected state."""
    def not_enough_data(*_a, **_k):
        raise RuntimeError("Only 2 settled event(s) on file.")

    monkeypatch.setattr(cli, "_run_entrypoint", not_enough_data)
    result = runner.invoke(cli.app, ["--env", "prod", command])
    assert result.exit_code == 1
    assert "Only 2 settled event(s) on file." in result.output
    assert "Traceback" not in result.output
