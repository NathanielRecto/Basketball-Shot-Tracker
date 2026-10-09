"""Object detector wrapper (Ultralytics YOLO, AGPL-3.0). Weights are NOT shipped: train your own."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .geometry import Box
from .types import Detection

# Class names found in common public basketball datasets -> our labels. "rim_only" (a rim box without the
# net, from the Hooper data) is only used to find the hoop once per video (calibration.find_hoop).
DEFAULT_ALIASES: Dict[str, str] = {
    "basketball": "ball", "ball": "ball", "bball": "ball", "sports ball": "ball",
    "hoop": "hoop", "rim": "hoop", "basket": "hoop", "basketball hoop": "hoop",
    "basketball-hoop": "hoop", "basketball_hoop": "hoop", "ring": "hoop",
    "rim_only": "rim_only",
}


class YoloDetector:
    def __init__(
        self,
        weights: str,
        conf: float = 0.3,
        imgsz: int = 960,
        device: Optional[str] = None,
        aliases: Optional[Dict[str, str]] = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the ML extras first: pip install -r requirements-ml.txt") from e
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.aliases = {k.lower(): v for k, v in (aliases or DEFAULT_ALIASES).items()}

    def __call__(self, frame, conf: Optional[float] = None) -> Sequence[Detection]:
        res = self.model.predict(frame, conf=self.conf if conf is None else conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out: List[Detection] = []
        names = res.names
        for xyxy, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            label = self.aliases.get(str(names[int(k)]).lower())
            if label is not None:
                out.append(Detection(label, float(c), Box(*xyxy)))
        return out
