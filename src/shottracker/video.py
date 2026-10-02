"""Video I/O helpers (OpenCV)."""
from __future__ import annotations

from typing import Iterator, Tuple

import cv2
import numpy as np


def probe(path: str) -> Tuple[float, int, int, int]:
    """(fps, width, height, frame_count)"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = (cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def read_frames(path: str, stride: int = 1) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, time_s, frame)``; time comes from the frame index, not wall clock."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def open_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise RuntimeError(f"Cannot open video writer for: {path}")
    return w
