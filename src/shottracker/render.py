"""Overlay drawing for annotated output video (OpenCV)."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .pipeline import FrameResult
from .types import Outcome, ShotEvent

GREEN, RED, ORANGE, BLUE, PURPLE, WHITE = (80, 190, 60), (60, 60, 230), (40, 130, 255), (230, 120, 20), (200, 90, 170), (245, 245, 245)
_BONES = [("shoulder", "elbow"), ("elbow", "wrist"), ("hip", "knee"), ("knee", "ankle"), ("shoulder", "hip")]


class Renderer:
    def __init__(self, trail_len: int = 25, banner_s: float = 1.2):
        self.trail: Deque[Tuple[int, int]] = deque(maxlen=trail_len)
        self.banner_s = banner_s
        self.done: List[ShotEvent] = []
        self.last: Optional[ShotEvent] = None
        self.last_t = -1e9
        self.made = self.attempts = 0

    def draw(self, frame: np.ndarray, fr: FrameResult) -> np.ndarray:
        out = frame.copy()
        for ev in fr.events:
            self.done.append(ev)
            self.last, self.last_t = ev, fr.t
            self.attempts += 1
            self.made += ev.outcome is Outcome.MADE

        for ev in self.done:  # faint record of every shot so far
            self._trace(out, ev, thick=2)

        if fr.hoop is not None:
            h = fr.hoop
            cv2.rectangle(out, (int(h.x1), int(h.y1)), (int(h.x2), int(h.y2)), ORANGE, 2)

        if fr.ball is not None:
            self.trail.append((int(fr.ball.x), int(fr.ball.y)))
            cv2.circle(out, self.trail[-1], max(6, int(fr.ball.diameter / 2)), BLUE, 2)
        if len(self.trail) > 1:
            cv2.polylines(out, [np.array(self.trail, np.int32)], False, PURPLE, 2, cv2.LINE_AA)

        if fr.pose is not None:
            self._skeleton(out, fr.pose)
        self._hud(out, fr.t)
        return out

    def _trace(self, img: np.ndarray, ev: ShotEvent, thick: int) -> None:
        color = GREEN if ev.outcome is Outcome.MADE else RED
        pts = np.array([(int(x), int(y)) for _, x, y in ev.trajectory], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)
        if ev.fit is not None and ev.fit.is_physical and len(ev.trajectory) > 1:
            ts = np.linspace(ev.trajectory[0][0], ev.t_cross, 40)
            curve = np.array([(int(ev.fit.x_at(t)), int(ev.fit.y_at(t))) for t in ts], np.int32)
            cv2.polylines(img, [curve], False, color, 1, cv2.LINE_AA)

    def _skeleton(self, img: np.ndarray, pose) -> None:
        for side in ("l", "r"):
            for a, b in _BONES:
                pa, pb = pose.point(f"{side}_{a}"), pose.point(f"{side}_{b}")
                if pa is not None and pb is not None:
                    cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), WHITE, 2, cv2.LINE_AA)

    def _hud(self, img: np.ndarray, t: float) -> None:
        pct = f" ({100 * self.made // self.attempts}%)" if self.attempts else ""
        self._text(img, f"Made {self.made}/{self.attempts}{pct}", (20, 40), 1.0, WHITE)
        ev = self.last
        if ev is None:
            return
        lines = []
        if ev.release_angle_deg is not None:
            lines.append(f"Release {ev.release_angle_deg:.0f} deg")
        if ev.entry_angle_deg is not None:
            lines.append(f"Entry {ev.entry_angle_deg:.0f} deg")
        for k, label in (("elbow_at_release_deg", "Elbow"), ("knee_min_deg", "Knee")):
            if k in ev.biomechanics:
                lines.append(f"{label} {ev.biomechanics[k]:.0f} deg")
        for i, s in enumerate(lines):
            self._text(img, s, (20, 75 + 30 * i), 0.7, WHITE)
        if t - self.last_t <= self.banner_s:
            ok = ev.outcome is Outcome.MADE
            h = ev.hoop
            self._text(img, "MADE" if ok else "MISSED", (int(h.cx) - 90, int(h.y1) - 30), 1.6, GREEN if ok else RED, 3)

    @staticmethod
    def _text(img, s, org, scale, color, thick=2):
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
