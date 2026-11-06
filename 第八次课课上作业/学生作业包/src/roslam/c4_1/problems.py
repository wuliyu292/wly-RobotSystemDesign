"""C4-1: classify localization / mapping / SLAM from known vs estimate.

Implement ``classify_scenario``. Do not hard-code the classroom scene names;
derive the kind from the two sets only.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class ProblemKind(str, Enum):
    LOCALIZATION = "localization"
    MAPPING = "mapping"
    SLAM = "slam"


_KNOWN_ALIASES = {
    "m": "map",
    "map": "map",
    "地图": "map",
    "x": "pose",
    "pose": "pose",
    "位姿": "pose",
    "轨迹": "pose",
    "z": "obs",
    "obs": "obs",
    "observation": "obs",
    "观测": "obs",
    "激光": "obs",
    "u": "ctrl",
    "ctrl": "ctrl",
    "control": "ctrl",
    "控制": "ctrl",
    "里程计": "ctrl",
}


def _normalize(items: Iterable[str]) -> frozenset[str]:
    out: set[str] = set()
    for raw in items:
        key = _KNOWN_ALIASES.get(str(raw).strip().lower(), str(raw).strip().lower())
        out.add(key)
    return frozenset(out)


def classify_scenario(known: Iterable[str], estimate: Iterable[str]) -> ProblemKind:
    """Return ProblemKind from known inputs and quantities to estimate.

    Rules:
    * localization — known has map, not pose; estimate has pose, not map
    * mapping — known has pose, not map; estimate has map, not pose
    * slam — known has neither map nor pose; estimate has both
    * otherwise raise ValueError
    """
    k = _normalize(known)
    e = _normalize(estimate)

    has_map_known = "map" in k
    has_pose_known = "pose" in k
    has_map_est = "map" in e
    has_pose_est = "pose" in e

    if has_map_known and (not has_pose_known) and has_pose_est and (not has_map_est):
        return ProblemKind.LOCALIZATION
    elif has_pose_known and (not has_map_known) and has_map_est and (not has_pose_est):
        return ProblemKind.MAPPING
    elif (not has_map_known) and (not has_pose_known) and has_map_est and has_pose_est:
        return ProblemKind.SLAM
    else:
        raise ValueError("cannot classify scenario")
