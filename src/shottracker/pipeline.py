"""Glue: detector -> trackers -> shot judging -> (optional) pose analysis."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Sequence

from . import biomechanics as bio
from .config import ShotConfig
from .geometry import Box
from .shot_logic import ShotDetector
from .stats import SessionSummary, summarize
from .tracking import BallTracker, HoopLocator, StaticSuppressor
from .types import BallObs, Detection, ShotEvent

Detector = Callable[[object], Sequence[Detection]]
PoseFn = Callable[[object, float], Optional[bio.PoseFrame]]


@dataclass
class FrameResult:
    t: float
    ball: Optional[BallObs]
    hoop: Optional[Box]
    pose: Optional[bio.PoseFrame]
    events: List[ShotEvent] = field(default_factory=list)


class ShotPipeline:
    def __init__(
        self,
        detector: Detector,
        pose: Optional[PoseFn] = None,
        shot_cfg: Optional[ShotConfig] = None,
        hand_factor: float = 1.5,
        pose_buffer_s: float = 4.0,
        hoop: Optional[Box] = None,
        suppress_static: bool = True,
    ):
        """``hoop``: a fixed hoop box (calibrated or hand-marked); per-frame hoop detection is then ignored.
        ``suppress_static``: drop ball detections that never move (see StaticSuppressor)."""
        self.detector = detector
        self.fixed_hoop = hoop
        self.static = StaticSuppressor() if suppress_static else None
        self.pose = pose
        self.tracker = BallTracker()
        self.hoops = HoopLocator()
        self.shots = ShotDetector(shot_cfg)
        self.hand_factor = hand_factor
        self.pose_buffer_s = pose_buffer_s
        self.events: List[ShotEvent] = []
        self._poses: Deque[bio.PoseFrame] = deque()

    def step(self, t: float, frame) -> FrameResult:
        dets = self.detector(frame)
        if self.static is not None:
            dets = self.static.filter(t, dets)
        ball = self.tracker.update(t, dets)
        if self.tracker.switched:
            self.shots.note_track_switch(t)
        hoop = self.fixed_hoop if self.fixed_hoop is not None else self.hoops.update(dets)

        pose = self.pose(frame, t) if self.pose is not None else None
        if pose is not None:
            self._poses.append(pose)
            while self._poses and t - self._poses[0].t > self.pose_buffer_s:
                self._poses.popleft()
            if ball is not None and bio.ball_in_hand(pose, (ball.x, ball.y), ball.diameter, self.hand_factor):
                self.shots.note_ball_in_hand(t)

        events = self.shots.update(t, ball, hoop)
        for ev in events:
            if self._poses and ev.trajectory:
                _, x0, y0 = ev.trajectory[0]
                ev.biomechanics = bio.analyse_release(self._poses, ev.t_release, (x0, y0))
        self.events.extend(events)
        return FrameResult(t, ball, hoop, pose, events)

    def summary(self) -> SessionSummary:
        return summarize(self.events)
