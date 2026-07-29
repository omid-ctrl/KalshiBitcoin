"""Kelly sizing for binary contracts, with a correlated-ladder correction.

THE SINGLE-BET FORMULA (derived, not quoted)
--------------------------------------------
Buy one YES contract at price p (dollars, 0..1). It pays $1 if the event happens and $0
otherwise, so per contract we stake p to win (1 - p). Betting a fraction f of bankroll:

    G(f) = q * ln(1 + f * b) + (1 - q) * ln(1 - f),      b = (1 - p) / p

with q the TRUE probability. Setting G'(f) = 0:

    q*b / (1 + f*b) = (1 - q) / (1 - f)
    =>  f* = (q*b - (1 - q)) / b
    =>  f* = (q - p) / (1 - p)                                             [KELLY]

f* is the fraction of bankroll STAKED, so the contract count is f* * B / p. Two sanity
checks that the tests pin down: q = p gives f* = 0 (no edge, no bet) and q = 1 gives
f* = 1 (a certainty is worth the whole bankroll).

Selling YES at p is buying NO at 1 - p with win probability 1 - q, so the same formula
gives f* = (p - q) / p. Fees enter by moving the EFFECTIVE price: a buy costs p + c and a
sell nets p - c, where c is the per-contract taker fee (zero for maker fills — which is
the whole reason this strategy is maker-first).

WHY FRACTIONAL KELLY IS NOT OPTIONAL HERE
-----------------------------------------
Full Kelly is optimal only if q is KNOWN. Ours is a model output built on an estimated
sigma, an estimated spot and an assumed tail shape. Kelly's growth curve is flat to first
order at the optimum but its LOSS curve is not: betting at 2x the true optimum has
negative expected log growth. Since our q has error, the effective optimum is materially
below the nominal one, and a quarter-Kelly (settings.risk.kelly_fraction, default 0.25)
buys roughly 94% of the growth for 25% of the variance while surviving a q that is
systematically off by a few cents. This is a robustness decision, not timidity.

THE CORRELATED-LADDER CORRECTION — THE PART THAT ACTUALLY MATTERS
-----------------------------------------------------------------
All 188 strikes in one hourly event are indicators on the SAME settlement value S:
X_i = 1{S > k_i}. They are not merely correlated, they are NESTED — if S clears the
highest strike it has cleared every lower one. Sizing each strike with its own Kelly and
adding up is not "slightly aggressive", it is a category error: it treats one bet on S as
N independent bets and can over-bet by a factor of N.

The exact covariance is available in closed form and costs nothing to compute. For strikes
k_i < k_j we have q_i >= q_j and E[X_i X_j] = P(S > k_j) = q_j, hence

    Cov(X_i, X_j) = min(q_i, q_j) - q_i * q_j                              [EXACT]

which reduces to q_i(1 - q_i) on the diagonal. No estimation, no shrinkage, no assumption
beyond the fair probabilities we already computed.

Given a signed contract vector v (positive = long YES, negative = short YES), portfolio
P&L is sum_i v_i * (X_i - p_i), with

    mean  = v . m,      m_i = q_i - p_i        (per contract, fee-adjusted price)
    var   = v' C v,     C from the formula above

In the small-bet limit the log-growth objective is (v.m)/B - (v'Cv)/(2 B^2), so along a
fixed direction d the optimal scale is

    lambda* = B * (d . m) / (d' C d)                                       [JOINT SCALE]

We take d to be the naive FULL-Kelly per-strike vector and multiply it by lambda*. Two
properties make this the right correction rather than an arbitrary haircut:

  * One strike alone: lambda* = p(1-p) / (q(1-q)), which is 1 whenever q is near p — the
    quadratic approximation reproduces exact Kelly for small edges, so a single-strike
    position is unchanged.
  * N identical, perfectly correlated strikes: d'Cd grows like N^2 while d.m grows like N,
    so lambda* falls like 1/N and the TOTAL ladder position equals the single-strike
    position. Exactly the behaviour that naive per-strike Kelly destroys.

lambda* is clamped to (0, 1]: this correction is only ever allowed to shrink. The
quadratic approximation can suggest scaling UP when edges are large, and a model that
wants to double its bet because it is very confident is the one you least want to trust.

HARD LIMITS COME LAST
---------------------
Kelly answers "how much is optimal", not "how much am I allowed". After sizing we apply
settings.risk: max_contracts_per_order, max_position_per_strike (against the position we
already hold), and a bankroll cap on total capital at risk. Every clamp is recorded on the
order so `kbtc report` can show whether the limits or the model are binding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Sequence

import numpy as np

from kalshi_btc.config import RiskLimits
from kalshi_btc.core.fees import taker_fee_per_contract
from kalshi_btc.core.types import Action, Liquidity, Side

# Below this the "edge" is inside the rounding error of a 1-cent tick.
MIN_MEANINGFUL_EDGE = Decimal("0.0005")


def kelly_fraction_binary(price: float | Decimal, win_prob: float) -> float:
    """Full-Kelly bankroll fraction to STAKE on a binary paying $1 at `price`.

        f* = (q - p) / (1 - p)

    Returns 0 when there is no edge (never a negative bet — the caller decides direction
    by choosing which side to price, and a negative f here would silently invert it).
    """
    p = float(price)
    q = float(win_prob)
    if not (0.0 < p < 1.0) or not (0.0 <= q <= 1.0):
        return 0.0
    f = (q - p) / (1.0 - p)
    return max(0.0, min(1.0, f))


def kelly_fraction_for(
    action: Action, price: float | Decimal, fair_prob: float, fee_per_contract: float | Decimal = 0
) -> tuple[float, float]:
    """(full-Kelly fraction, effective price) for buying or selling YES.

    BUY YES  at p: we pay p + c, we win $1 with probability q      -> p_eff = p + c, win q
    SELL YES at p: we receive p - c and owe $1 with probability q, which is buying NO at
                   1 - (p - c) with win probability 1 - q          -> mapped accordingly
    """
    p = float(price)
    c = float(fee_per_contract)
    if action is Action.BUY:
        p_eff = p + c
        return kelly_fraction_binary(p_eff, fair_prob), p_eff
    p_eff = p - c
    # Buying NO at (1 - p_eff) with win probability (1 - q) is algebraically identical to
    # the expression below; we keep the YES frame so the caller's price stays comparable.
    return kelly_fraction_binary(1.0 - p_eff, 1.0 - fair_prob), p_eff


def ladder_covariance(fair_probs: Sequence[float]) -> np.ndarray:
    """Exact covariance of nested indicators on one settlement value.

        Cov(X_i, X_j) = min(q_i, q_j) - q_i q_j

    Valid for ANY ordering of the inputs because 1{S > k_i} and 1{S > k_j} are nested
    whichever way round the strikes are.
    """
    q = np.clip(np.asarray(fair_probs, dtype=float), 0.0, 1.0)
    return np.minimum.outer(q, q) - np.outer(q, q)


@dataclass(frozen=True)
class SizingCandidate:
    """One tradeable idea, already gated by the edge engine."""

    ticker: str
    strike: Decimal
    action: Action  # BUY or SELL of YES
    price: Decimal  # the price we would actually trade at (ask for a buy, bid for a sell)
    fair_prob: float
    liquidity: Liquidity  # MAKER fills are free; TAKER fills carry the quadratic fee
    side: Side = Side.YES
    existing_position: int = 0  # signed: + long YES, - short YES
    available_size: Decimal | None = None  # size resting on the other side, if known

    @property
    def fee_per_contract(self) -> Decimal:
        if self.liquidity is Liquidity.MAKER:
            return Decimal("0")
        return taker_fee_per_contract(self.price)

    @property
    def signed_direction(self) -> int:
        return 1 if self.action is Action.BUY else -1


@dataclass(frozen=True)
class SizedOrder:
    """A candidate with a contract count and a full audit trail of how it got there."""

    candidate: SizingCandidate
    contracts: int
    naive_contracts: int
    kelly_fraction_full: float
    effective_price: float
    capital_at_risk: Decimal
    clamps: tuple[str, ...] = ()

    @property
    def ticker(self) -> str:
        return self.candidate.ticker

    @property
    def action(self) -> Action:
        return self.candidate.action

    @property
    def price(self) -> Decimal:
        return self.candidate.price

    def describe(self) -> str:
        clamp = f" [{','.join(self.clamps)}]" if self.clamps else ""
        return (
            f"{self.candidate.action.value} {self.contracts}x {self.candidate.ticker} "
            f"@{self.candidate.price:.2f} (naive {self.naive_contracts}, "
            f"f*={self.kelly_fraction_full:.4f}){clamp}"
        )


@dataclass(frozen=True)
class LadderSizing:
    """The whole event's sizing decision, with the correction made visible."""

    orders: list[SizedOrder] = field(default_factory=list)
    naive_total: int = 0
    joint_total: int = 0
    ladder_scale: float = 1.0
    portfolio_edge: float = 0.0
    portfolio_std: float = 0.0
    capital_at_risk: Decimal = Decimal("0")

    @property
    def shrink(self) -> float:
        """joint / naive contract count. < 1 means the correlation correction bit."""
        return (self.joint_total / self.naive_total) if self.naive_total else 1.0

    def executable(self) -> list[SizedOrder]:
        return [o for o in self.orders if o.contracts > 0]

    def describe(self) -> str:
        return (
            f"naive={self.naive_total} joint={self.joint_total} "
            f"scale={self.ladder_scale:.3f} edge=${self.portfolio_edge:.3f} "
            f"sd=${self.portfolio_std:.3f} risk=${self.capital_at_risk:.2f}"
        )


def naive_contracts(
    candidate: SizingCandidate, bankroll: Decimal, kelly_fraction: float = 1.0
) -> tuple[float, float, float]:
    """(contracts, full-Kelly fraction, effective price) ignoring every other strike.

    Returned as a float so the joint step can scale it before rounding — rounding twice
    would quantise away exactly the shrinkage the correction is trying to apply.
    """
    f_full, p_eff = kelly_fraction_for(
        candidate.action, candidate.price, candidate.fair_prob, candidate.fee_per_contract
    )
    if f_full <= 0.0 or p_eff <= 0.0 or p_eff >= 1.0:
        return 0.0, f_full, p_eff
    # Stake per contract is what we actually put up: p_eff for a YES buy, (1 - p_eff) for
    # a YES sell (= a NO buy at that price).
    stake = p_eff if candidate.action is Action.BUY else (1.0 - p_eff)
    if stake <= 0.0:
        return 0.0, f_full, p_eff
    n = f_full * kelly_fraction * float(bankroll) / stake
    return n, f_full, p_eff


def size_ladder(
    candidates: Sequence[SizingCandidate],
    risk: RiskLimits,
    *,
    bankroll: Decimal | None = None,
    kelly_fraction: float | None = None,
    allow_scale_up: bool = False,
) -> LadderSizing:
    """Size a whole ladder jointly. See the module docstring for the derivation.

    `candidates` should all belong to ONE event: the covariance formula assumes a single
    underlying settlement value, which is exactly what an event is.
    """
    bankroll = Decimal(bankroll if bankroll is not None else risk.bankroll)
    kf = float(kelly_fraction if kelly_fraction is not None else risk.kelly_fraction)
    if not candidates or bankroll <= 0 or kf <= 0:
        return LadderSizing()

    # ---- 1. naive per-strike full-Kelly direction ------------------------------------
    raw: list[tuple[float, float, float]] = [
        naive_contracts(c, bankroll, kelly_fraction=1.0) for c in candidates
    ]
    d = np.array([c.signed_direction * r[0] for c, r in zip(candidates, raw)], dtype=float)

    q = [c.fair_prob for c in candidates]
    p_eff = np.array([r[2] for r in raw], dtype=float)
    # Per-contract expected P&L in the LONG-YES frame; the sign lives in d, not in m.
    m = np.asarray(q, dtype=float) - p_eff
    cov = ladder_covariance(q)

    denom = float(d @ cov @ d)
    numer = float(d @ m)
    if denom <= 0.0 or numer <= 0.0 or not math.isfinite(denom) or not math.isfinite(numer):
        # No dispersion (every q is 0 or 1) or the direction has no aggregate edge. Either
        # way the joint objective has nothing to optimise; fall back to naive.
        scale = 1.0
    else:
        scale = float(bankroll) * numer / denom
        if not allow_scale_up:
            scale = min(scale, 1.0)
        scale = max(0.0, scale)

    v = d * scale * kf

    portfolio_edge = float(v @ m)
    portfolio_var = float(v @ cov @ v)
    portfolio_std = math.sqrt(max(0.0, portfolio_var))

    # ---- 2. hard limits, applied per order -------------------------------------------
    orders: list[SizedOrder] = []
    naive_total = 0
    joint_total = 0
    total_risk = Decimal("0")
    remaining_bankroll = bankroll

    for c, r, want, naive_n in zip(candidates, raw, v, d):
        clamps: list[str] = []
        n_naive = int(abs(naive_n) * kf)  # what naive fractional Kelly would have done
        n = int(abs(want))  # truncate toward zero: never round a bet UP

        if n > risk.max_contracts_per_order:
            n = risk.max_contracts_per_order
            clamps.append("per_order")

        # Position limit is on the ABSOLUTE resulting position, so an order that reduces
        # an existing position is never blocked by it.
        target = c.existing_position + c.signed_direction * n
        if abs(target) > risk.max_position_per_strike:
            room = risk.max_position_per_strike - abs(c.existing_position)
            allowed = max(0, room) if abs(target) > abs(c.existing_position) else n
            if allowed < n:
                n = allowed
                clamps.append("per_strike")

        stake_per = Decimal(str(r[2])) if c.action is Action.BUY else (
            Decimal("1") - Decimal(str(r[2]))
        )
        stake_per = max(Decimal("0"), stake_per)
        cost = (stake_per * n).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        if cost > remaining_bankroll:
            affordable = int(remaining_bankroll / stake_per) if stake_per > 0 else 0
            n = min(n, max(0, affordable))
            cost = (stake_per * n).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
            clamps.append("bankroll")

        remaining_bankroll -= cost
        total_risk += cost
        naive_total += n_naive
        joint_total += n
        orders.append(
            SizedOrder(
                candidate=c,
                contracts=n,
                naive_contracts=n_naive,
                kelly_fraction_full=r[1],
                effective_price=r[2],
                capital_at_risk=cost,
                clamps=tuple(clamps),
            )
        )

    return LadderSizing(
        orders=orders,
        naive_total=naive_total,
        joint_total=joint_total,
        ladder_scale=scale,
        portfolio_edge=portfolio_edge,
        portfolio_std=portfolio_std,
        capital_at_risk=total_risk,
    )


__all__ = [
    "LadderSizing",
    "MIN_MEANINGFUL_EDGE",
    "SizedOrder",
    "SizingCandidate",
    "kelly_fraction_binary",
    "kelly_fraction_for",
    "ladder_covariance",
    "naive_contracts",
    "size_ladder",
]
