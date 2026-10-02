"""Plain data types shared across the package."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .geometry import Box, FlightFit


@dataclass(frozen=True)
class Detection:
    label: str  # normalised: "ball" or "hoop"
    conf: float
    box: Box


@dataclass(frozen=True)
class BallObs:
    t: float
    x: float
    y: float
    diameter: float = 0.0


class Outcome(str, Enum):
    MADE = "made"
    MISSED = "missed"


@dataclass
class ShotEvent:
    index: int
    outcome: Outcome
    reason: str  # through_hoop | off_target | rim_out | rim_bounce
    method: str  # interpolated | fit | extrapolated | rebound
    t_release: float
    t_apex: float
    t_cross: float
    cross_offset: float  # signed, in hoop half-widths (0 = dead centre)
    hoop: Box
    trajectory: List[Tuple[float, float, float]]
    fit: Optional[FlightFit] = None
    release_angle_deg: Optional[float] = None
    entry_angle_deg: Optional[float] = None
    biomechanics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "method": self.method,
            "t_release": round(self.t_release, 3),
            "t_apex": round(self.t_apex, 3),
            "t_cross": round(self.t_cross, 3),
            "cross_offset": round(self.cross_offset, 3),
            "release_angle_deg": _r(self.release_angle_deg),
            "entry_angle_deg": _r(self.entry_angle_deg),
            "biomechanics": {k: round(v, 1) for k, v in self.biomechanics.items()},
        }


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 1)
