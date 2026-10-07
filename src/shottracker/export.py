"""Hand the tracking + judging logic to another implementation (the phone app) and prove it matches.

Two products, both plain JSON:

* ``params()``: every tunable setting of the static filter, ball tracker and shot judge, read from the
  code's own defaults, so the copy can never drift from the Python.
* ``golden_case()``: per-frame detector output in, and what this Python code decided out (the tracked
  ball each frame, track switches, every shot call and the decision log). A port is correct when it
  replays the inputs and reproduces the expectations exactly.

Inputs are rounded BEFORE the expectations are computed, so the port receives exactly the numbers
the Python saw. Expected values are written at full float precision (JSON round-trips doubles), and
the port should match them exactly; Python floats and JavaScript numbers are the same IEEE doubles.
Only detections (numbers) are exported, never images.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ShotConfig
from .geometry import Box
from .pipeline import ShotPipeline
from .tracking import BallTracker, StaticSuppressor
from .types import Detection

FORMAT = "shottracker-golden/1"
PARAMS_FORMAT = "shottracker-params/1"
DETECTOR_CONF = 0.15  # the confidence the tracker is fed at (run_eval_videos.py --conf)
DECIMALS = 2  # box corners are rounded to this before anything is computed

Frame = Tuple[float, Sequence[Tuple[float, float, float, float, float]]]  # t, [(x1, y1, x2, y2, conf), ...]


def _defaults(cls) -> Dict[str, object]:
    return {k: p.default for k, p in inspect.signature(cls.__init__).parameters.items()
            if p.default is not inspect.Parameter.empty}


def params() -> Dict[str, object]:
    """Every setting the port needs, from the code's defaults."""
    return {
        "format": PARAMS_FORMAT,
        "detector": {"conf": DETECTOR_CONF},
        "static_filter": _defaults(StaticSuppressor),
        "tracker": _defaults(BallTracker),
        "shot": asdict(ShotConfig()),
    }


def round_frames(frames: Sequence[Frame], conf: float = DETECTOR_CONF) -> List[Frame]:
    """What the port is given: ball boxes at or above ``conf``, corners rounded to DECIMALS."""
    out = []
    for t, balls in frames:
        kept = [tuple(round(v, DECIMALS) for v in b[:4]) + (round(b[4], 3),) for b in balls if b[4] >= conf]
        out.append((round(t, 6), kept))
    return out


def golden_case(name: str, hoop: Sequence[float], frames: Sequence[Frame], source: str) -> Dict[str, object]:
    """Run the pipeline over ``frames`` (already rounded) and record what it decided."""
    pipe = ShotPipeline(lambda dets: dets, hoop=Box(*hoop))
    rows = []
    for t, balls in frames:
        dets = [Detection("ball", c, Box(x1, y1, x2, y2)) for x1, y1, x2, y2, c in balls]
        fr = pipe.step(t, dets)
        ball = None if fr.ball is None else [fr.ball.x, fr.ball.y, fr.ball.diameter]
        rows.append({"t": t, "balls": [list(b) for b in balls],
                     "expect": {"ball": ball, "switched": pipe.tracker.switched}})
    events = [{"t_release": e.t_release, "t_apex": e.t_apex, "t_cross": e.t_cross, "outcome": e.outcome.value,
               "reason": e.reason, "method": e.method, "cross_offset": e.cross_offset} for e in pipe.events]
    return {
        "format": FORMAT,
        "name": name,
        "source": source,
        "hoop": [float(v) for v in hoop],
        "frames": rows,
        "expect": {"events": events, "log": [[t, m] for t, m in pipe.shots.log]},
    }


def replay_matches(case: Dict[str, object]) -> Optional[str]:
    """Replay a case through THIS code; None if every expectation holds, else the first difference."""
    fresh = golden_case(case["name"], case["hoop"], [(f["t"], [tuple(b) for b in f["balls"]]) for f in case["frames"]],
                        case["source"])
    for a, b in zip(case["frames"], fresh["frames"]):
        if a["expect"] != b["expect"]:
            return f"t={a['t']}: expected {a['expect']}, got {b['expect']}"
    if case["expect"] != fresh["expect"]:
        return "events or decision log differ"
    return None


def sim_frames(seed: int = 7, fps: float = 30.0) -> Tuple[List[float], List[Frame]]:
    """A synthetic stream covering every shot type, with noise, dropped frames and distractors.

    Distractors: a static decoy next to the rim (the static filter must drop it) and a slowly
    wobbling "head" detected every frame (the tracker must switch away from it when a shot rises).
    """
    import math

    import numpy as np

    from .sim import DEFAULT_HOOP, ShotSpec, simulate_stream

    specs = [ShotSpec(offset=0.1), ShotSpec(offset=1.6, release=(300, 400)), ShotSpec(offset=0.0, behaviour="rim_out"),
             ShotSpec(offset=0.3, behaviour="rim_bounce"), ShotSpec(offset=0.2, behaviour="fall_past", occlude=True),
             ShotSpec(offset=0.1, behaviour="rattle_in"), ShotSpec(offset=-1.8, release=(420, 360))]
    stream = simulate_stream(DEFAULT_HOOP, specs, fps=fps, noise_px=1.5, dropout=0.1, seed=seed, lead_s=4.0)
    rng = np.random.default_rng(seed)
    h = DEFAULT_HOOP
    frames: List[Frame] = []
    for t, obs in stream:
        balls = []
        if obs is not None:
            r = obs.diameter / 2
            held = obs.y > 500  # a ball in someone's hands is only weakly detected, so the head wins at first
            conf = float(rng.uniform(0.2, 0.35)) if held else float(rng.uniform(0.3, 0.9))
            balls.append((obs.x - r, obs.y - r, obs.x + r, obs.y + r, conf))
        balls.append((h.x2 - 17.5, h.y1 - 9.5, h.x2 + 7.5, h.y1 + 15.5, 0.5))  # static decoy on the bracket
        hx, hy = 620 + 100 * math.sin(t * 0.5), 560 + 6 * math.sin(t * 1.9)  # a head that wanders (too much for the static filter)
        balls.append((hx - 18, hy - 18, hx + 18, hy + 18, float(rng.uniform(0.55, 0.75))))
        frames.append((t, balls))
    return [h.x1, h.y1, h.x2, h.y2], frames
