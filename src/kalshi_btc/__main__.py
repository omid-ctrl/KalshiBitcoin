"""Entry point for `python -m kalshi_btc`.

The console script `kbtc` is only present after an editable install, and on macOS
that install is fragile: setuptools can emit the `__editable__.*.pth` file with the
`UF_HIDDEN` flag set, and CPython 3.12+ *silently* skips hidden `.pth` files, so a
correctly-installed package still raises ModuleNotFoundError. `python -m kalshi_btc`
needs nothing but PYTHONPATH=src, so it is the one invocation that always works and
the one the runbooks should quote.
"""

from kalshi_btc.cli import app

if __name__ == "__main__":  # pragma: no cover - trivial dispatch
    app()
