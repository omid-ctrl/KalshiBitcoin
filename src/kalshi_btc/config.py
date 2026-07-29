"""Configuration, loaded from environment / .env.

The one thing to understand here: `ARMED` defaults to False and nothing else in the
codebase is allowed to default it to True. Live order placement is gated on it
explicitly at the router level.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

# Verified live 2026-07-28. See docs/GROUND_TRUTH.md.
PROD_REST = "https://external-api.kalshi.com/trade-api/v2"
PROD_WS = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
DEMO_REST = "https://external-api.demo.kalshi.co/trade-api/v2"
DEMO_WS = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"

# The series we trade. KXBTC is a DIFFERENT, far less liquid hourly BTC series
# (range/bracket markets, ~80x less volume, 2-4 cent spreads). Never trade it by accident.
SERIES_TICKER = "KXBTCD"
DECOY_SERIES = "KXBTC"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key, str(default)).lower()
    return raw in {"1", "true", "yes", "on"}


def _env_dec(key: str, default: str) -> Decimal:
    return Decimal(_env(key, default) or default)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader - avoids a hard dependency for the capture-only path."""
    p = path or Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


@dataclass(frozen=True)
class RiskLimits:
    bankroll: Decimal = field(default_factory=lambda: _env_dec("BANKROLL", "2000"))
    max_contracts_per_order: int = field(
        default_factory=lambda: int(_env("MAX_CONTRACTS_PER_ORDER", "1"))
    )
    max_position_per_strike: int = field(
        default_factory=lambda: int(_env("MAX_POSITION_PER_STRIKE", "25"))
    )
    max_loss_per_event: Decimal = field(default_factory=lambda: _env_dec("MAX_LOSS_PER_EVENT", "15"))
    max_loss_per_day: Decimal = field(default_factory=lambda: _env_dec("MAX_LOSS_PER_DAY", "50"))
    kelly_fraction: Decimal = field(default_factory=lambda: _env_dec("KELLY_FRACTION", "0.25"))


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _env("KALSHI_ENV", "demo"))
    api_key_id: str = field(default_factory=lambda: _env("KALSHI_API_KEY_ID"))
    private_key_path: str = field(default_factory=lambda: _env("KALSHI_PRIVATE_KEY_PATH"))
    armed: bool = field(default_factory=lambda: _env_bool("ARMED", False))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")).expanduser())
    telegram_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    telegram_chat: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    risk: RiskLimits = field(default_factory=RiskLimits)

    @property
    def rest_base(self) -> str:
        return PROD_REST if self.env == "prod" else DEMO_REST

    @property
    def ws_base(self) -> str:
        return PROD_WS if self.env == "prod" else DEMO_WS

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key_id and self.private_key_path)

    def describe(self) -> str:
        armed = "ARMED (REAL MONEY)" if self.armed else "disarmed (no orders will be sent)"
        creds = "credentials present" if self.has_credentials else "NO credentials (public data only)"
        return f"env={self.env} | {armed} | {creds} | data_dir={self.data_dir}"


def get_settings() -> Settings:
    load_dotenv()
    return Settings()
