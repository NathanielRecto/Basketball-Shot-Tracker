"""Turns raw per-frame detections into one stable ball observation and one stable hoop box."""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Sequence

import numpy as np

from .geometry import Box, dist
from .types import BallObs, Detection


class StaticSuppressor:
    """Drops "ball" detections that stay in one place, since a ball in play moves.

    Catches persistent false positives (e.g. a round rim-mounting bracket) and spare balls
    lying on the floor. A detection is static when, over the last ``window_s`` seconds, the
    same spot (within ``radius`` x its size) had a ball detection in at least ``min_presence``
    of the frames, spread over at least ``min_static_s`` seconds. Once flagged, a spot stays
    suppressed for ``hold_s`` seconds, because weak false positives flicker on and off.

    Calibration (dev video_01): the rim-bracket false positive was present in 12-55% of
    frames (median per 8 s window); the busiest real-ball spot, where the shooter holds the
    ball before free throws, peaked at 8%.
    """

    def __init__(self, window_s: float = 8.0, min_presence: float = 0.15, radius: float = 0.6,
                 min_static_s: float = 3.0, hold_s: float = 30.0):
        self.window_s = window_s
        self.min_presence = min_presence
        self.radius = radius
        self.min_static_s = min_static_s
        self.hold_s = hold_s
        self._frames: Deque[float] = deque()
        self._hist: Deque[tuple] = deque()  # (t, cx, cy)
        self._spots: List[list] = []  # [cx, cy, r, until]

    def filter(self, t: float, detections: Sequence[Detection]) -> List[Detection]:
        self._frames.append(t)
        while self._frames and t - self._frames[0] > self.window_s:
            self._frames.popleft()
        while self._hist and t - self._hist[0][0] > self.window_s:
            self._hist.popleft()
        balls = [d for d in detections if d.label == "ball"]
        for d in balls:
            self._hist.append((t, d.box.cx, d.box.cy))
        self._spots = [s for s in self._spots if s[3] > t]
        n = len(self._frames)
        out: List[Detection] = []
        for d in detections:
            if d.label == "ball":
                if any(abs(d.box.cx - sx) <= sr and abs(d.box.cy - sy) <= sr for sx, sy, sr, _ in self._spots):
                    continue
                r = self.radius * max(d.box.w, d.box.h)
                seen = {ht for ht, hx, hy in self._hist if abs(hx - d.box.cx) <= r and abs(hy - d.box.cy) <= r}
                if len(seen) / n >= self.min_presence and max(seen) - min(seen) >= self.min_static_s:
                    self._spots.append([d.box.cx, d.box.cy, r, t + self.hold_s])
                    continue
            out.append(d)
        return out


class _Tracklet:
    """A short side track for a detection the main track did not take (see BallTracker)."""

    __slots__ = ("t", "x", "y", "d", "vx", "vy", "hits")

    def __init__(self, t: float, x: float, y: float, d: float):
        self.t, self.x, self.y, self.d = t, x, y, d
        self.vx = self.vy = 0.0
        self.hits = 1

    def step(self, t: float, x: float, y: float, d: float) -> None:
        dt = t - self.t
        nvx, nvy = (x - self.x) / dt, (y - self.y) / dt
        a = 1.0 if self.hits == 1 else 0.6
        self.vx, self.vy = a * nvx + (1 - a) * self.vx, a * nvy + (1 - a) * self.vy
        self.t, self.x, self.y, self.d = t, x, y, d
        self.hits += 1

    def speed(self) -> float:
        """Ball diameters per second."""
        return dist(self.vx, self.vy) / max(self.d, 12.0)


class BallTracker:
    """Follows the single ball in play with a constant-velocity gate.

    A frame can contain several ball detections (spare balls, false positives
    on a round object). The one closest to where the track *should* be wins;
    detections far outside the gate are ignored. After ``lost_s`` seconds
    without a match the track resets: a confident detection (>= ``init_conf``) starts a new
    track right away; a weaker one only does when it is confirmed by motion, i.e. a detection
    in a recent frame between ``min_move`` and ``max_speed`` ball-diameters away. In flight the
    ball is often only weakly detected; static false positives never move.

    Switching: false positives that move a little (heads, legs, hands when people stand around
    the hoop) are detected in nearly every frame, so a track that locked onto one would never
    be lost and never let go. Detections the main track does not take are therefore followed
    as short side tracks; when one has risen fast for ``switch_hits`` frames (a released shot:
    at least ``switch_speed`` diameters/s, mostly upward) while the main track moves at under
    ``switch_ratio`` of that speed, the main track jumps to it. ``switched`` is True on the
    frame where that happened, so the shot logic can drop the history of the old target.

    Leaving the top of the frame: a high shot is often out of view (or lost against ceiling
    lights) around its apex for longer than ``lost_s``. When the track was heading up fast
    enough to be above the frame by now, it waits up to ``exit_wait_s`` for the ball to come
    back down near where it left (horizontally) instead of starting over on whatever is
    detected meanwhile.
    """

    def __init__(self, init_conf: float = 0.4, lost_s: float = 0.5, gate_diams: float = 2.5, min_gate_px: float = 25.0,
                 confirm_s: float = 0.15, min_move: float = 0.3, max_speed: float = 120.0,
                 switch_speed: float = 12.0, switch_hits: int = 4, switch_ratio: float = 0.5, max_tracklets: int = 8,
                 exit_wait_s: float = 1.5, switch_min_up: float = 0.5, exit_horizon_s: float = 0.3):
        self.init_conf = init_conf
        self.lost_s = lost_s
        self.gate_diams = gate_diams
        self.min_gate_px = min_gate_px
        self.confirm_s = confirm_s
        self.min_move = min_move  # ball diameters between the two frames
        self.max_speed = max_speed  # ball diameters per second
        self._last: Optional[BallObs] = None
        self._vx = 0.0
        self._vy = 0.0
        self._recent: Deque[tuple] = deque()  # (t, cx, cy) of recent unassigned detections
        self.switch_speed = switch_speed
        self.switch_hits = switch_hits
        self.switch_ratio = switch_ratio
        self.max_tracklets = max_tracklets
        self._side: List[_Tracklet] = []
        self.switched = False
        self.exit_wait_s = exit_wait_s
        self.switch_min_up = switch_min_up  # upward share of the speed a side track needs to take over
        self.exit_horizon_s = exit_horizon_s

    def _exited_top(self, last: BallObs) -> bool:
        """The track was rising fast enough to be above the top of the frame within ``exit_horizon_s``."""
        return self._vy < 0 and last.y + self._vy * self.exit_horizon_s < 0

    def _reacquire(self, t: float, last: BallObs, balls: Sequence[Detection], hoop: Optional[Box] = None) -> Optional[Detection]:
        """A detection where a ball that left through the top edge would come back into view.

        The constant-velocity guess overshoots when the ball flies away from the camera: perspective
        slows it on screen while it is out of view (own session 2: predicted 400-900 px past where it
        came back). A ball heading for the hoop comes back down over it, so with a hoop the search
        also covers everything between the exit point and one hoop-width beyond the hoop.
        """
        dt = t - last.t
        diam = max(last.diameter, 12.0)
        px = last.x + self._vx * dt
        tol_x = max(4 * diam, 0.3 * abs(self._vx) * dt)

        def near(x: float) -> bool:
            if abs(x - px) <= tol_x:
                return True
            if hoop is None or (hoop.cx - last.x) * self._vx <= 0:  # no hoop, or not heading towards it
                return False
            lo, hi = (hoop.x1 - hoop.w, last.x) if hoop.cx < last.x else (last.x, hoop.x2 + hoop.w)
            return lo <= x <= hi

        # It comes back in from above, so it reappears no lower than where it was lost.
        cands = [d for d in balls if near(d.box.cx) and d.box.cy <= last.y + diam]
        return min(cands, key=lambda d: abs(d.box.cx - px)) if cands else None

    def _gate(self, x: float, y: float, d: float, vx: float, vy: float, dt: float) -> tuple:
        """(predicted x, predicted y, gate radius) for a constant-velocity track."""
        diam = max(d, 12.0)
        gate = max(self.min_gate_px, self.gate_diams * diam) + 0.5 * dist(vx, vy) * dt
        return x + vx * dt, y + vy * dt, gate

    def _update_side(self, t: float, balls: Sequence[Detection]) -> None:
        """Associate the detections the main track did not take with the side tracks."""
        self._side = [k for k in self._side if t - k.t <= self.lost_s]
        free = list(balls)
        pairs = []
        for ki, k in enumerate(self._side):
            px, py, gate = self._gate(k.x, k.y, k.d, k.vx, k.vy, t - k.t)
            for di, d in enumerate(free):
                gap = dist(d.box.cx - px, d.box.cy - py)
                if gap <= gate:
                    pairs.append((gap, ki, di))
        used_k, used_d = set(), set()
        for _, ki, di in sorted(pairs):
            if ki not in used_k and di not in used_d:
                used_k.add(ki)
                used_d.add(di)
                d = free[di]
                self._side[ki].step(t, d.box.cx, d.box.cy, max(d.box.w, d.box.h))
        for di, d in enumerate(free):
            if di not in used_d and len(self._side) < self.max_tracklets:
                self._side.append(_Tracklet(t, d.box.cx, d.box.cy, max(d.box.w, d.box.h)))

    def _riser(self, t: float, main_speed: float) -> Optional[_Tracklet]:
        best = None
        for k in self._side:
            sp = k.speed()
            if (k.t == t and k.hits >= self.switch_hits and sp >= self.switch_speed and -k.vy >= self.switch_min_up * dist(k.vx, k.vy)
                    and main_speed < self.switch_ratio * sp and (best is None or sp > best.speed())):
                best = k
        return best

    def _motion_confirmed(self, t: float, balls: Sequence[Detection]) -> Optional[Detection]:
        for d in sorted(balls, key=lambda d: -d.conf):
            size = max(d.box.w, d.box.h, 12.0)
            for pt, px, py in self._recent:
                dt = t - pt
                if 0 < dt <= self.confirm_s:
                    gap = dist(d.box.cx - px, d.box.cy - py)
                    if self.min_move * size <= gap <= self.max_speed * size * dt:
                        self._vx, self._vy = (d.box.cx - px) / dt, (d.box.cy - py) / dt
                        return d
        return None

    def update(self, t: float, detections: Sequence[Detection], hoop: Optional[Box] = None) -> Optional[BallObs]:
        self.switched = False
        balls = [d for d in detections if d.label == "ball"]
        while self._recent and t - self._recent[0][0] > self.confirm_s:
            self._recent.popleft()
        if not balls:
            return None
        last = self._last
        if last is not None and self.lost_s < t - last.t <= self.exit_wait_s and self._exited_top(last):
            back = self._reacquire(t, last, balls, hoop)
            if back is None:
                return None
            self._vy = 0.0  # it is coming down now; the gate re-learns the speed
            self._last = BallObs(t, back.box.cx, back.box.cy, max(back.box.w, back.box.h))
            return self._last
        if last is None or t - last.t > self.lost_s:
            cands = [d for d in balls if d.conf >= self.init_conf]
            if cands:
                best = max(cands, key=lambda d: d.conf)
                self._vx = self._vy = 0.0
            else:
                best = self._motion_confirmed(t, balls)
                self._recent.extend((t, d.box.cx, d.box.cy) for d in balls)
                if best is None:
                    return None
                last = None  # keep the velocity from the confirming pair
        else:
            dt = t - last.t
            px, py, gate = self._gate(last.x, last.y, last.diameter, self._vx, self._vy, dt)
            diam = max(last.diameter, 12.0)
            scored = []
            for d in balls:
                gap = dist(d.box.cx - px, d.box.cy - py)
                if gap <= gate:
                    scored.append((d.conf / (1.0 + gap / diam), d))
            best = max(scored, key=lambda s: s[0])[1] if scored else None
            self._update_side(t, [d for d in balls if d is not best])
            riser = self._riser(t, dist(self._vx, self._vy) / diam)
            if riser is not None:
                self._side.remove(riser)
                if best is not None:  # the old target becomes a side track
                    k = _Tracklet(last.t, last.x, last.y, last.diameter)
                    k.vx, k.vy = self._vx, self._vy
                    k.step(t, best.box.cx, best.box.cy, max(best.box.w, best.box.h))
                    self._side.append(k)
                self._vx, self._vy = riser.vx, riser.vy
                self._last = BallObs(t, riser.x, riser.y, riser.d)
                self.switched = True
                return self._last
            if best is None:
                return None

        obs = BallObs(t, best.box.cx, best.box.cy, max(best.box.w, best.box.h))
        if last is not None and 0 < t - last.t <= self.lost_s:
            dt = t - last.t
            nvx, nvy = (obs.x - last.x) / dt, (obs.y - last.y) / dt
            self._vx = 0.6 * nvx + 0.4 * self._vx
            self._vy = 0.6 * nvy + 0.4 * self._vy
        self._last = obs
        return obs


class HoopLocator:
    """Median of recent hoop detections; robust to flicker and occasional bad boxes.

    Assumes a fixed camera. For a handheld camera, shrink ``window`` so the box
    follows the hoop faster.
    """

    def __init__(self, window: int = 60, min_samples: int = 5, min_conf: float = 0.4):
        self.min_samples = min_samples
        self.min_conf = min_conf
        self._boxes: Deque[Box] = deque(maxlen=window)

    def update(self, detections: Sequence[Detection]) -> Optional[Box]:
        hoops = [d for d in detections if d.label == "hoop" and d.conf >= self.min_conf]
        if hoops:
            self._boxes.append(max(hoops, key=lambda d: d.conf).box)
        if len(self._boxes) < self.min_samples:
            return None
        arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in self._boxes])
        return Box(*np.median(arr, axis=0).tolist())
