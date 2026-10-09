"""Locate the hoop once per video for a fixed camera.

Per-frame hoop detection can fail on footage unlike the training data (e.g. a see-through
backboard against bright sky). With a fixed camera the hoop never moves, so it is enough to
find it a few times anywhere in the video: sample frames, accept weak detections, and keep the
position that the most detections agree on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import Box


def consensus_box(cands: Sequence[Tuple[Box, float]], min_support: int = 3, radius: float = 0.5) -> Optional[Box]:
    """Median box of the largest cluster of candidates, weighted by confidence.

    A candidate supports another when their centres are within ``radius`` box-widths.
    Returns None when the best cluster has fewer than ``min_support`` members.
    """
    best_score, best_members = 0.0, []
    for b, _ in cands:
        reach = radius * max(b.w, 1.0)
        members = [(o, c) for o, c in cands if math.hypot(o.cx - b.cx, o.cy - b.cy) <= reach]
        score = sum(c for _, c in members)
        if score > best_score:
            best_score, best_members = score, members
    if len(best_members) < min_support:
        return None
    arr = np.array([[o.x1, o.y1, o.x2, o.y2] for o, _ in best_members])
    return Box(*np.median(arr, axis=0).tolist())


# Rim (rim_only box) -> rim + hanging net (the hoop box the shot logic uses), in rim widths:
# (x1, y1, x2 offsets from the rim box; y2 below the rim's top). Learned from hand-marked hoops on the
# own DEV sessions (1-5, 9-11); in a second gym it matched the hand-marked box at IoU 0.82-0.93.
RIM_TO_HOOP = (0.0, -0.05, 0.02, 1.27)


@dataclass
class HoopFind:
    box: Optional[Box]
    source: str  # "hoop" (hoop-class consensus), "rim" (biggest rim_only box + net), or "none"
    candidates: int
    # area of the second-biggest rim over the chosen one, median over frames that saw two or more rims:
    # near 1 = two similar hoops in view, so a person should confirm; None = only one rim ever seen
    ambiguity: Optional[float] = None


def rim_to_hoop(rim: Box) -> Box:
    w = rim.w
    a, b, c, d = RIM_TO_HOOP
    return Box(rim.x1 + a * w, rim.y1 + b * w, rim.x2 + c * w, rim.y1 + d * w)


def hoop_from_rims(per_frame: Sequence[Sequence[Box]], min_frames: int = 3) -> Tuple[Optional[Box], Optional[float]]:
    """The nearest hoop from rim boxes: the biggest rim in each frame, median over frames, plus the net.

    The hoop the camera is set up for is the closest one, so its rim looks biggest; the median ignores
    the odd frame where a player blocks it. Returns (hoop box, ambiguity).
    """
    biggest = [max(rims, key=lambda b: b.w * b.h) for rims in per_frame if rims]
    if len(biggest) < min_frames:
        return None, None
    arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in biggest])
    rim = Box(*np.median(arr, axis=0).tolist())
    ratios = []
    for rims in per_frame:
        if not rims:
            continue
        top = max(rims, key=lambda b: b.w * b.h)
        # another hoop = a rim at least one rim width away (a second box on the same rim does not count)
        others = [b.w * b.h for b in rims if math.hypot(b.cx - top.cx, b.cy - top.cy) > top.w]
        if others:
            ratios.append(max(others) / (top.w * top.h))
    return rim_to_hoop(rim), (float(np.median(ratios)) if ratios else None)


def find_hoop(video: str, detector, samples: int = 60, conf: float = 0.05, rim_conf: float = 0.25) -> HoopFind:
    """Locate the hoop once for a fixed camera from ``samples`` frames spread over the video.

    Detectors trained on gym footage often see the rim (``rim_only``) but not the full hoop class; then
    the biggest rim plus the net is used. When the hoop class itself is found, its consensus is kept
    (the behaviour every earlier result was produced with).
    """
    import cv2

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cands: List[Tuple[Box, float]] = []
    rims: List[List[Box]] = []
    try:
        for idx in np.linspace(0, max(n - 1, 0), samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            dets = detector(frame, conf=conf)
            cands += [(d.box, d.conf) for d in dets if d.label == "hoop"]
            rims.append([d.box for d in dets if d.label == "rim_only" and d.conf >= rim_conf])
    finally:
        cap.release()
    box = consensus_box(cands)
    if box is not None:
        return HoopFind(box, "hoop", len(cands))
    rim_box, ambiguity = hoop_from_rims(rims)
    if rim_box is not None:
        return HoopFind(rim_box, "rim", sum(len(r) for r in rims), ambiguity)
    return HoopFind(None, "none", len(cands))


def calibrate_hoop(video: str, detector, samples: int = 60, conf: float = 0.05) -> Tuple[Optional[Box], int]:
    """Sample ``samples`` frames, run ``detector(frame, conf=...)`` and return (hoop box, n candidates)."""
    found = find_hoop(video, detector, samples, conf)
    return found.box, found.candidates


def save_hoop_preview(video: str, box: Optional[Box], path: str, t: float = 5.0) -> bool:
    """Write a frame with the hoop box drawn on it, so a person can confirm the calibration."""
    import cv2

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    if box is not None:
        cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 4)
    else:
        cv2.putText(frame, "HOOP NOT FOUND", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
    scale = min(1.0, 960 / frame.shape[0])
    return cv2.imwrite(path, cv2.resize(frame, None, fx=scale, fy=scale))


def parse_box(text: str) -> Box:
    """``"x1,y1,x2,y2"`` -> Box."""
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4 or vals[2] <= vals[0] or vals[3] <= vals[1]:
        raise ValueError(f"expected x1,y1,x2,y2 with x2>x1 and y2>y1, got {text!r}")
    return Box(*vals)
