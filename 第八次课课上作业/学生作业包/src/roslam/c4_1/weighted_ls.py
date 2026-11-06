"""C4-1: 1D weighted least squares (two-radar blackboard example)."""

from __future__ import annotations

from typing import Sequence


def weighted_mean_1d(measurements: Sequence[float], weights: Sequence[float]) -> float:
    """Return x* = argmin_x Σ w_i (z_i - x)^2.

    Requirements:
    * same length; at least one measurement
    * weights ≥ 0; total weight > 0 else ValueError
    """
    if len(measurements) != len(weights):
        raise ValueError("measurements and weights length mismatch")
    if len(measurements) == 0:
        raise ValueError("empty input")

    sum_w = 0.0
    sum_wz = 0.0
    for z, w in zip(measurements, weights):
        if w < 0:
            raise ValueError("negative weight not allowed")
        sum_w += w
        sum_wz += w * z

    if sum_w <= 0.0:
        raise ValueError("total weight must be > 0")

    return sum_wz / sum_w
