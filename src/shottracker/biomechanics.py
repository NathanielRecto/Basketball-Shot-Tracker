"""Shooting-form measurements from 2D pose keypoints.

Keypoints are named ``{l,r}_{shoulder,elbow,wrist,hip,knee,ankle}`` and given
as ``(x, y, visibility)`` in pixels. Angles are 2D projections, so they are
only meaningful for a roughly side-on camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .geometry import joint_angle

MIN_VISIBILITY = 0.5


@dataclass(frozen=True)
class PoseFrame:
    t: float
    keypoints: Dict[str, Tuple[float, float, float]]

    def point(self, name: str) -> Optional[Tuple[float, float]]:
        kp = self.keypoints.get(name)
        if kp is None or kp[2] < MIN_VISIBILITY:
            return None
        return kp[0], kp[1]


def _angle(pose: PoseFrame, side: str, a: str, b: str, c: str) -> Optional[float]:
    pts = [pose.point(f"{side}_{n}") for n in (a, b, c)]
    if any(p is None for p in pts):
        return None
    ang = joint_angle(*pts)
    return None if math.isnan(ang) else ang


def elbow_angle(pose: PoseFrame, side: str) -> Optional[float]:
    return _angle(pose, side, "shoulder", "elbow", "wrist")


def knee_angle(pose: PoseFrame, side: str) -> Optional[float]:
    return _angle(pose, side, "hip", "knee", "ankle")


def wrist_ball_distance(pose: PoseFrame, ball_xy: Tuple[float, float]) -> Optional[float]:
    dists = []
    for side in ("l", "r"):
        w = pose.point(f"{side}_wrist")
        if w is not None:
            dists.append(math.hypot(w[0] - ball_xy[0], w[1] - ball_xy[1]))
    return min(dists) if dists else None


def shooting_side(pose: PoseFrame, ball_xy: Tuple[float, float]) -> Optional[str]:
    """The side whose wrist is nearer the ball."""
    best, best_d = None, math.inf
    for side in ("l", "r"):
        w = pose.point(f"{side}_wrist")
        if w is None:
            continue
        d = math.hypot(w[0] - ball_xy[0], w[1] - ball_xy[1])
        if d < best_d:
            best, best_d = side, d
    return best


def ball_in_hand(pose: PoseFrame, ball_xy: Tuple[float, float], ball_diameter: float, factor: float = 1.5) -> bool:
    d = wrist_ball_distance(pose, ball_xy)
    return d is not None and d <= factor * max(ball_diameter, 1.0)


def analyse_release(
    poses: Sequence[PoseFrame],
    t_release: float,
    ball_xy: Tuple[float, float],
    lookback_s: float = 0.8,
    max_offset_s: float = 0.25,
) -> Dict[str, float]:
    """Elbow angle at release and the deepest knee bend in the ``lookback_s`` before it."""
    if not poses:
        return {}
    at = min(poses, key=lambda p: abs(p.t - t_release))
    if abs(at.t - t_release) > max_offset_s:
        return {}
    side = shooting_side(at, ball_xy)
    if side is None:
        return {}
    out: Dict[str, float] = {"shooting_side": 0.0 if side == "l" else 1.0}
    elbow = elbow_angle(at, side)
    if elbow is not None:
        out["elbow_at_release_deg"] = elbow
    knees = [
        k for p in poses if t_release - lookback_s <= p.t <= t_release
        for k in [knee_angle(p, side)] if k is not None
    ]
    if knees:
        out["knee_min_deg"] = min(knees)
    return out
