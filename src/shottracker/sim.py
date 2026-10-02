"""Synthetic shot generator.

Produces physically plausible ball trajectories (parabolic flight, optional rim
bounces, net occlusion, noise, dropped detections) so the judging logic can be
tested, tuned and demoed without any video or trained model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import ShotConfig
from .geometry import Box
from .types import BallObs, Detection

Frame = Tuple[float, Optional[BallObs]]

DEFAULT_HOOP = Box(900, 150, 970, 200)  # 70 x 50 px
BALL_DIAM = 40.0
FRAME_H = 720


@dataclass
class ShotSpec:
    offset: float = 0.0  # where the ball crosses the rim line, in hoop half-widths from centre
    release: Tuple[float, float] = (250.0, 380.0)
    apex_above_rim: float = 140.0  # px
    g: float = 1400.0  # px / s^2
    hold_s: float = 0.6  # ball held before the release
    behaviour: str = "normal"  # normal | rim_out | rim_bounce | fall_past (centred-looking, misses the net) | rattle_in
    occlude: bool = False  # hide the ball around the rim (net / backboard)


def _rim_y(hoop: Box, cfg: Optional[ShotConfig] = None) -> float:
    return hoop.y1 + (cfg or ShotConfig()).rim_line_frac * hoop.h


class _Flight:
    """Closed-form trajectory for one shot, with an optional bounce."""

    def __init__(self, spec: ShotSpec, hoop: Box):
        self.spec = spec
        rim_y = _rim_y(hoop)
        x0, y0 = spec.release
        g = spec.g
        apex_y = rim_y - spec.apex_above_rim
        xt = hoop.cx + spec.offset * hoop.w / 2
        t_up = math.sqrt(2 * (y0 - apex_y) / g)
        t_dn = math.sqrt(2 * (rim_y - apex_y) / g)
        self.x0, self.y0 = x0, y0
        self.vy0 = -g * t_up
        self.T = t_up + t_dn  # time of the rim-line crossing
        self.vx = (xt - x0) / self.T
        self.bounce_t: Optional[float] = None
        # A clean make drops into the net, which brakes the ball for a moment.
        self.net_t: Optional[float] = None
        if spec.behaviour == "normal" and abs(spec.offset) <= 0.5:
            self.net_t, self.net_s = self.T, 0.3
            self.nx, self.ny = xt, rim_y
            self.nvx, self.nvy = 0.1 * self.vx, 0.2 * (self.vy0 + g * self.T)
        # Bounces pop the ball up a clearly visible 1.2-1.5 hoop heights; weaker ones are
        # indistinguishable from a ball simply dropping through the net.
        if spec.behaviour == "rim_out":
            # dips into the rim, then pops back out
            t_b = t_up + math.sqrt(2 * (rim_y + 0.5 * hoop.h - apex_y) / g)
            self._set_bounce(t_b, 1.5 * hoop.h, 0.3)
        elif spec.behaviour == "rim_bounce":
            # hits the rim from above and never gets below it
            t_b = t_up + math.sqrt(2 * (rim_y - 0.3 * hoop.h - apex_y) / g)
            self._set_bounce(t_b, 1.2 * hoop.h, 0.4)
        elif spec.behaviour == "rattle_in":
            # hits the rim, pops straight up, falls back through the hoop into the net
            t_b = t_up + math.sqrt(2 * (rim_y - 0.3 * hoop.h - apex_y) / g)
            self._set_bounce(t_b, 1.2 * hoop.h, 0.0)
            t_back = (-self.bvy + math.sqrt(self.bvy ** 2 + 2 * g * (rim_y - self.by))) / g  # bounce -> rim line
            self.net_t, self.net_s = t_b + t_back, 0.3
            self.nx, self.ny = self.bx, rim_y
            self.nvx, self.nvy = 0.0, 0.2 * (self.bvy + g * t_back)

    def _set_bounce(self, t_b: float, pop_height: float, vx_keep: float) -> None:
        self.bounce_t = t_b
        self.bx = self.x0 + self.vx * t_b
        self.by = self.y0 + self.vy0 * t_b + 0.5 * self.spec.g * t_b ** 2
        self.bvx, self.bvy = -vx_keep * self.vx, -math.sqrt(2 * self.spec.g * pop_height)

    def pos(self, tau: float) -> Tuple[float, float]:
        g = self.spec.g
        if self.net_t is not None and tau >= self.net_t:
            s = tau - self.net_t
            if s <= self.net_s:
                return self.nx + self.nvx * s, self.ny + self.nvy * s
            s2 = s - self.net_s
            return (self.nx + self.nvx * (self.net_s + s2),
                    self.ny + self.nvy * (self.net_s + s2) + 0.5 * g * s2 * s2)
        if self.bounce_t is not None and tau >= self.bounce_t:
            s = tau - self.bounce_t
            return self.bx + self.bvx * s, self.by + self.bvy * s + 0.5 * g * s * s
        return self.x0 + self.vx * tau, self.y0 + self.vy0 * tau + 0.5 * g * tau * tau


def simulate_stream(
    hoop: Box,
    shots: Sequence[ShotSpec],
    fps: float = 30.0,
    noise_px: float = 0.0,
    dropout: float = 0.0,
    seed: int = 0,
    lead_s: float = 0.5,
    gap_s: float = 0.8,
    tail_s: float = 1.0,
) -> List[Frame]:
    """One ``(t, BallObs | None)`` per frame; ``None`` where the ball was not detected."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    frames: List[Frame] = []
    t = 0.0

    def emit(pos: Optional[Tuple[float, float]]):
        nonlocal t
        obs = None
        if pos is not None and 0 <= pos[1] <= FRAME_H and rng.random() >= dropout:
            x = pos[0] + rng.normal(0, noise_px) if noise_px else pos[0]
            y = pos[1] + rng.normal(0, noise_px) if noise_px else pos[1]
            obs = BallObs(t, float(x), float(y), BALL_DIAM)
        frames.append((t, obs))
        t += dt

    def idle(duration: float, at: Tuple[float, float]):
        for _ in range(int(round(duration * fps))):
            emit(at)

    idle(lead_s, (200.0, 600.0))
    for spec in shots:
        fl = _Flight(spec, hoop)
        for _ in range(int(round(spec.hold_s * fps))):
            emit((spec.release[0], spec.release[1]))
        post = 0.9
        tau = 0.0
        while tau < fl.T + post:
            hidden = spec.occlude and fl.T - 0.1 <= tau <= fl.T + 0.15
            emit(None if hidden else fl.pos(tau))
            tau += dt
        idle(gap_s, (200.0, 600.0))
    idle(tail_s, (200.0, 600.0))
    return frames


def dribble_stream(duration_s: float = 4.0, fps: float = 30.0, floor_y: float = 640.0, amp: float = 170.0) -> List[Frame]:
    """A ball being dribbled well below rim height (should never register a shot)."""
    out: List[Frame] = []
    for i in range(int(duration_s * fps)):
        t = i / fps
        y = floor_y - amp * abs(math.sin(math.pi * 1.6 * t))
        out.append((t, BallObs(t, 300.0 + 20 * math.sin(t), y, BALL_DIAM)))
    return out


def detections_from_stream(stream: Sequence[Frame], hoop: Box, hoop_conf: float = 0.9) -> List[List[Detection]]:
    """Fake per-frame detector output (ball + hoop) for pipeline tests and the demo."""
    out: List[List[Detection]] = []
    for _, obs in stream:
        dets = [Detection("hoop", hoop_conf, hoop)]
        if obs is not None:
            r = obs.diameter / 2
            dets.append(Detection("ball", 0.85, Box(obs.x - r, obs.y - r, obs.x + r, obs.y + r)))
        out.append(dets)
    return out


class ScriptedDetector:
    """Replays pre-computed detections, one list per call."""

    def __init__(self, per_frame: Sequence[Sequence[Detection]]):
        self._it = iter(per_frame)

    def __call__(self, frame) -> Sequence[Detection]:
        return next(self._it, [])
