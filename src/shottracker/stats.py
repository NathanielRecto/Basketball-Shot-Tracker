"""Session-level statistics."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .types import Outcome, ShotEvent


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


@dataclass
class SessionSummary:
    attempts: int
    made: int
    missed: int
    fg_pct: Optional[float]
    avg_release_angle_deg: Optional[float]
    avg_entry_angle_deg: Optional[float]
    avg_elbow_at_release_deg: Optional[float]
    avg_knee_min_deg: Optional[float]
    shots: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self, **kw) -> str:
        return json.dumps(self.__dict__, **kw)


def summarize(events: Sequence[ShotEvent]) -> SessionSummary:
    made = sum(e.outcome is Outcome.MADE for e in events)
    n = len(events)
    return SessionSummary(
        attempts=n,
        made=made,
        missed=n - made,
        fg_pct=round(100.0 * made / n, 1) if n else None,
        avg_release_angle_deg=_mean([e.release_angle_deg for e in events]),
        avg_entry_angle_deg=_mean([e.entry_angle_deg for e in events]),
        avg_elbow_at_release_deg=_mean([e.biomechanics.get("elbow_at_release_deg") for e in events]),
        avg_knee_min_deg=_mean([e.biomechanics.get("knee_min_deg") for e in events]),
        shots=[e.to_dict() for e in events],
    )
