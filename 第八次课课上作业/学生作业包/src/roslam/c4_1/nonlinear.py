"""C4-1: scalar Gauss-Newton and Levenberg-Marquardt.

``residual(x)`` and ``jacobian(x)`` are callables provided by the caller.
Return a ``SolveTrace`` with the full iteration history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


ResidualFn = Callable[[float], float]
JacobianFn = Callable[[float], float]


@dataclass(frozen=True)
class SolveTrace:
    xs: tuple[float, ...]
    costs: tuple[float, ...]
    deltas: tuple[float, ...]


def _cost(residual: float) -> float:
    return float(residual * residual)


def gauss_newton_1d(
    x0: float,
    residual: ResidualFn,
    jacobian: JacobianFn,
    *,
    max_iters: int = 20,
    tol: float = 1e-10,
) -> SolveTrace:
    """Scalar GN: H=J², b=J·e, Δx=-b/H, x←x+Δx until |Δx|<tol or max_iters."""
    xs = [x0]
    costs = [_cost(residual(x0))]
    deltas = []

    x = x0
    for _ in range(max_iters):
        e = residual(x)
        J = jacobian(x)
        H = J * J
        b = J * e

        if H == 0.0:
            deltas.append(0.0)
            break

        delta = -b / H
        deltas.append(delta)
        x = x + delta

        xs.append(x)
        costs.append(_cost(residual(x)))

        if abs(delta) < tol:
            break

    return SolveTrace(xs=tuple(xs), costs=tuple(costs), deltas=tuple(deltas))


def levenberg_marquardt_1d(
    x0: float,
    residual: ResidualFn,
    jacobian: JacobianFn,
    *,
    max_iters: int = 30,
    tol: float = 1e-10,
    lambda0: float = 1e-3,
) -> SolveTrace:
    """Scalar LM: (H+λ)Δx=-b; accept only cost-decreasing steps.

    On success: λ ← max(λ/10, 1e-12). On failure: λ ← min(λ*10, 1e12),
    record delta=0 and keep x unchanged (still append to xs/costs).
    """
    xs = [x0]
    init_e = residual(x0)
    costs = [_cost(init_e)]
    deltas = []

    x = x0
    lam = lambda0

    for _ in range(max_iters):
        e = residual(x)
        J = jacobian(x)
        H = J * J
        b = J * e

        delta = -b / (H + lam)
        x_candidate = x + delta
        cost_candidate = _cost(residual(x_candidate))
        current_cost = _cost(e)

        if cost_candidate < current_cost:
            # 接受步，更新x、缩小lambda
            x = x_candidate
            lam = max(lam / 10.0, 1e-12)
            deltas.append(delta)
        else:
            # 拒绝步，x不变，delta=0，放大lambda
            lam = min(lam * 10.0, 1e12)
            delta = 0.0
            deltas.append(delta)

        xs.append(x)
        costs.append(_cost(residual(x)))

        if abs(delta) < tol:
            break

    return SolveTrace(xs=tuple(xs), costs=tuple(costs), deltas=tuple(deltas))
