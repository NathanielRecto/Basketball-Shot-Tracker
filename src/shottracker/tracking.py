"""Turns raw per-frame detections into one stable ball observation and one stable hoop box."""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Sequence

import numpy as np

from .geometry import Box
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


class BallTracker:
    """Follows the single ball in play with a constant-velocity gate.

    A frame can contain several ball detections (spare balls, false positives
    on a round object). The one closest to where the track *should* be wins;
    detections far outside the gate are ignored. After ``lost_s`` seconds
    without a match the track resets: a confident detection (>= ``init_conf``) starts a new
    track right away; a weaker one only does when it is confirmed by motion, i.e. a detection
    in a recent frame between ``min_move`` and ``max_speed`` ball-diameters away. In flight the
    ball is often only weakly detected; static false positives never move.
    """

    def __init__(self, init_conf: float = 0.4, lost_s: float = 0.5, gate_diams: float = 2.5, min_gate_px: float = 25.0,
                 confirm_s: float = 0.15, min_move: float = 0.3, max_speed: float = 120.0):
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

    def _motion_confirmed(self, t: float, balls: Sequence[Detection]) -> Optional[Detection]:
        for d in sorted(balls, key=lambda d: -d.conf):
            size = max(d.box.w, d.box.h, 12.0)
            for pt, px, py in self._recent:
                dt = t - pt
                if 0 < dt <= self.confirm_s:
                    dist = math.hypot(d.box.cx - px, d.box.cy - py)
                    if self.min_move * size <= dist <= self.max_speed * size * dt:
                        self._vx, self._vy = (d.box.cx - px) / dt, (d.box.cy - py) / dt
                        return d
        return None

    def update(self, t: float, detections: Sequence[Detection]) -> Optional[BallObs]:
        balls = [d for d in detections if d.label == "ball"]
        while self._recent and t - self._recent[0][0] > self.confirm_s:
            self._recent.popleft()
        if not balls:
            return None
        last = self._last
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
            px, py = last.x + self._vx * dt, last.y + self._vy * dt
            diam = max(last.diameter, 12.0)
            gate = max(self.min_gate_px, self.gate_diams * diam) + 0.5 * math.hypot(self._vx, self._vy) * dt
            scored = []
            for d in balls:
                dist = math.hypot(d.box.cx - px, d.box.cy - py)
                if dist <= gate:
                    scored.append((d.conf / (1.0 + dist / diam), d))
            if not scored:
                return None
            best = max(scored, key=lambda s: s[0])[1]

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
