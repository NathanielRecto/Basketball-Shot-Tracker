"""Pose estimator wrapper (MediaPipe Pose, Apache-2.0).

MediaPipe >= 1.0 only ships the Tasks API, which needs a ``pose_landmarker_*.task`` model
file (see README: "Pose model"). Older MediaPipe (0.10.x) bundles its model and exposes
``mp.solutions.pose``; that path is used automatically when available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .biomechanics import PoseFrame

# MediaPipe landmark indices for the joints we use.
_LANDMARKS = {
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_full.task"


class MediaPipePose:
    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
    ):
        try:
            import mediapipe as mp
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the ML extras first: pip install -r requirements-ml.txt") from e
        self._mp = mp
        self._legacy = None
        self._landmarker = None
        self._last_ts = -1
        if hasattr(mp, "solutions"):
            self._legacy = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=min_detection_conf, min_tracking_confidence=min_tracking_conf,
            )
            return
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.is_file():
            raise FileNotFoundError(
                f"Pose model not found at {path}. Download a MediaPipe Pose Landmarker .task file "
                "(see README, 'Pose model') and pass --pose-model or place it there."
            )
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, frame_bgr, t: float) -> Optional[PoseFrame]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._legacy is not None:
            res = self._legacy.process(rgb)
            lms = res.pose_landmarks.landmark if res.pose_landmarks else None
        else:
            ts = max(int(t * 1000), self._last_ts + 1)  # Tasks API needs strictly increasing ms timestamps
            self._last_ts = ts
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self._landmarker.detect_for_video(img, ts)
            lms = res.pose_landmarks[0] if res.pose_landmarks else None
        if lms is None:
            return None
        return PoseFrame(t, {n: (lms[i].x * w, lms[i].y * h, lms[i].visibility) for n, i in _LANDMARKS.items()})

    def close(self) -> None:
        if self._legacy is not None:
            self._legacy.close()
        if self._landmarker is not None:
            self._landmarker.close()
