"""Geometry and trajectory-fitting helpers (numpy only).

Image coordinates throughout: x grows right, y grows *down*, so "above the rim"
means a smaller y.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

Sample = Tuple[float, float, float]  # (t, x, y)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def joint_angle(a, b, c) -> float:
    """Angle in degrees at vertex ``b`` formed by points a-b-c (nan if degenerate)."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cos = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos))


@dataclass(frozen=True)
class FlightFit:
    """Projectile model fitted against time: x(t) linear, y(t) quadratic.

    With t0 the first sample time and tau = t - t0:
        x = x_poly[0] * tau + x_poly[1]
        y = y_poly[0] * tau**2 + y_poly[1] * tau + y_poly[2]
    """

    t0: float
    x_poly: Tuple[float, float]
    y_poly: Tuple[float, float, float]
    rms: float
    n: int

    @property
    def is_physical(self) -> bool:
        # y points down, so gravity must show up as positive curvature.
        return self.y_poly[0] > 0

    def x_at(self, t: float) -> float:
        return self.x_poly[0] * (t - self.t0) + self.x_poly[1]

    def y_at(self, t: float) -> float:
        tau = t - self.t0
        c2, c1, c0 = self.y_poly
        return c2 * tau * tau + c1 * tau + c0

    def velocity_at(self, t: float) -> Tuple[float, float]:
        tau = t - self.t0
        return self.x_poly[0], 2 * self.y_poly[0] * tau + self.y_poly[1]

    def time_at_y(self, y: float) -> Optional[float]:
        """Time at which the *descending* branch reaches ``y`` (None if never)."""
        c2, c1, c0 = self.y_poly
        if c2 <= 0:
            return None
        disc = c1 * c1 - 4 * c2 * (c0 - y)
        if disc < 0:
            return None
        return self.t0 + (-c1 + math.sqrt(disc)) / (2 * c2)


def fit_flight(samples: Sequence[Sample], min_points: int = 4) -> Optional[FlightFit]:
    if len(samples) < min_points:
        return None
    arr = np.asarray(samples, dtype=float)
    t0 = float(arr[0, 0])
    tau = arr[:, 0] - t0
    if tau[-1] - tau[0] <= 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            xp = np.polyfit(tau, arr[:, 1], 1)
            yp = np.polyfit(tau, arr[:, 2], 2)
        except (np.linalg.LinAlgError, ValueError):
            return None
    if not (np.all(np.isfinite(xp)) and np.all(np.isfinite(yp))):
        return None
    resid = np.hypot(arr[:, 1] - np.polyval(xp, tau), arr[:, 2] - np.polyval(yp, tau))
    return FlightFit(
        t0=t0,
        x_poly=(float(xp[0]), float(xp[1])),
        y_poly=(float(yp[0]), float(yp[1]), float(yp[2])),
        rms=float(np.sqrt(np.mean(resid ** 2))),
        n=len(arr),
    )
