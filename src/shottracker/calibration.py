"""Locate the hoop once per video for a fixed camera.

Per-frame hoop detection can fail on footage unlike the training data (e.g. a see-through
backboard against bright sky). With a fixed camera the hoop never moves, so it is enough to
find it a few times anywhere in the video: sample frames, accept weak detections, and keep the
position that the most detections agree on.
"""
from __future__ import annotations

import math
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


def calibrate_hoop(video: str, detector, samples: int = 60, conf: float = 0.05) -> Tuple[Optional[Box], int]:
    """Sample ``samples`` frames, run ``detector(frame, conf=...)`` and return (hoop box, n candidates)."""
    import cv2

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cands: List[Tuple[Box, float]] = []
    try:
        for idx in np.linspace(0, max(n - 1, 0), samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            cands += [(d.box, d.conf) for d in detector(frame, conf=conf) if d.label == "hoop"]
    finally:
        cap.release()
    return consensus_box(cands), len(cands)


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
