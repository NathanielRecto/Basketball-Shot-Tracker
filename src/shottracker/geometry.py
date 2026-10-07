"""Geometry and trajectory-fitting helpers (numpy only).

Image coordinates throughout: x grows right, y grows *down*, so "above the rim"
means a smaller y.
"""
from __future__ import annotations

import math
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


def dist(dx: float, dy: float) -> float:
    """Length of (dx, dy), spelled out as sqrt(dx*dx + dy*dy) so another language computes the same bits.

    ``math.hypot`` is more careful but rounds differently from JavaScript's ``Math.hypot``, and the phone
    app's port must reproduce this code exactly (docs/app_port.md).
    """
    return math.sqrt(dx * dx + dy * dy)


def _det3(a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float, i: float) -> float:
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def fit_flight(samples: Sequence[Sample], min_points: int = 4) -> Optional[FlightFit]:
    """Least-squares x(tau) = a*tau + b and y(tau) = c2*tau^2 + c1*tau + c0, with tau = t - t0.

    Solved from the normal equations with plain sums in sample order and Cramer's rule, so the phone
    app's port can reproduce it bit for bit (numpy's polyfit cannot be copied exactly). Over the short,
    well-spread time spans of a shot this agrees with polyfit to far below a pixel.
    """
    if len(samples) < min_points:
        return None
    t0 = float(samples[0][0])
    pts = [(float(s[0]) - t0, float(s[1]), float(s[2])) for s in samples]
    if pts[-1][0] - pts[0][0] <= 0:
        return None
    n = len(pts)
    s1 = s2 = s3 = s4 = sx = stx = sy = sty = stty = 0.0
    for tau, x, y in pts:
        t2 = tau * tau
        s1 += tau
        s2 += t2
        s3 += t2 * tau
        s4 += t2 * t2
        sx += x
        stx += tau * x
        sy += y
        sty += tau * y
        stty += t2 * y
    den = n * s2 - s1 * s1
    det = _det3(s4, s3, s2, s3, s2, s1, s2, s1, n)
    if den == 0 or det == 0:
        return None
    a = (n * stx - s1 * sx) / den
    b = (sx - a * s1) / n
    c2 = _det3(stty, s3, s2, sty, s2, s1, sy, s1, n) / det
    c1 = _det3(s4, stty, s2, s3, sty, s1, s2, sy, n) / det
    c0 = _det3(s4, s3, stty, s3, s2, sty, s2, s1, sy) / det
    if not all(math.isfinite(v) for v in (a, b, c2, c1, c0)):
        return None
    sq = 0.0
    for tau, x, y in pts:
        ex = x - (a * tau + b)
        ey = y - ((c2 * tau + c1) * tau + c0)
        sq += ex * ex + ey * ey
    return FlightFit(t0=t0, x_poly=(a, b), y_poly=(c2, c1, c0), rms=math.sqrt(sq / n), n=n)
