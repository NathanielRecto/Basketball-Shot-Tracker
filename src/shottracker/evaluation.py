"""Compare predicted shots against hand-labelled ground truth.

Ground truth is one row per shot attempt: when the ball reached the hoop and whether it
went in. Predictions are matched to labels one-to-one by time, then scored.

Metrics
-------
* Attempt detection: precision / recall / F1 (did we find the shots, without inventing any?).
* Outcome accuracy among matched shots (made vs missed).
* End-to-end accuracy: correct outcomes / labelled shots. A shot we never found counts as wrong.
* Per-video count error: |predicted makes - labelled makes|, the metric the original author reports.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MADE, MISSED = "made", "missed"
_MADE_WORDS = {"made", "make", "m", "1", "yes", "y", "in", "score", "scored", "goal"}
_MISS_WORDS = {"missed", "miss", "x", "0", "no", "n", "out"}
SHOT_TYPES = ("FT", "mid", "3PT")
_SHOT_TYPE_WORDS = {
    "ft": "FT", "1": "FT", "free throw": "FT", "freethrow": "FT",
    "mid": "mid", "2": "mid", "mid-range": "mid", "midrange": "mid", "2pt": "mid",
    "3pt": "3PT", "3": "3PT", "three": "3PT", "3-pointer": "3PT",
}


@dataclass(frozen=True)
class LabeledShot:
    t: float
    outcome: str
    note: str = ""
    shot_type: str = ""  # FT / mid / 3PT, or "" when not tagged


@dataclass(frozen=True)
class PredictedShot:
    t: float
    outcome: str
    reason: str = ""


def parse_time(value: str) -> float:
    """Seconds from ``12.5``, ``1:23``, ``1:23.4`` or ``0:01:23``."""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad time: {value!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def normalize_outcome(value: str) -> str:
    v = value.strip().lower()
    if v in _MADE_WORDS:
        return MADE
    if v in _MISS_WORDS:
        return MISSED
    raise ValueError(f"outcome must be made or missed, got {value!r}")


def normalize_shot_type(value: str) -> str:
    v = value.strip().lower()
    if not v:
        return ""
    if v in _SHOT_TYPE_WORDS:
        return _SHOT_TYPE_WORDS[v]
    raise ValueError(f"shot_type must be FT, mid or 3PT (or empty), got {value!r}")


def load_labels(path: Path) -> List[LabeledShot]:
    """CSV with header ``time_s,outcome[,shot_type][,note]``. Blank lines and lines starting with # are ignored."""
    rows: List[LabeledShot] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    for i, rec in enumerate(csv.DictReader(lines), start=2):
        if not (rec.get("time_s") or "").strip() and not (rec.get("outcome") or "").strip():
            continue  # a half-empty template row
        try:
            rows.append(LabeledShot(parse_time(rec["time_s"]), normalize_outcome(rec["outcome"]), (rec.get("note") or "").strip(),
                                    normalize_shot_type(rec.get("shot_type") or "")))
        except (KeyError, ValueError) as e:
            raise ValueError(f"{path.name}: data row {i - 1}: {e}") from e
    return sorted(rows, key=lambda s: s.t)


def load_predictions(path: Path) -> List[PredictedShot]:
    """Reads the ``summary.json`` written by ``shottracker run``."""
    data = json.loads(Path(path).read_text())
    return sorted((PredictedShot(s["t_cross"], s["outcome"], s.get("reason", "")) for s in data["shots"]), key=lambda s: s.t)


def match_shots(
    labels: Sequence[LabeledShot], preds: Sequence[PredictedShot], tol_s: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """One-to-one matching, closest in time first. Returns (pairs, unmatched label ids, unmatched pred ids)."""
    cands = sorted(
        (abs(l.t - p.t), li, pi)
        for li, l in enumerate(labels) for pi, p in enumerate(preds) if abs(l.t - p.t) <= tol_s
    )
    used_l, used_p, pairs = set(), set(), []
    for _, li, pi in cands:
        if li not in used_l and pi not in used_p:
            used_l.add(li)
            used_p.add(pi)
            pairs.append((li, pi))
    pairs.sort()
    return (
        pairs,
        [i for i in range(len(labels)) if i not in used_l],
        [i for i in range(len(preds)) if i not in used_p],
    )


@dataclass
class VideoResult:
    video: str
    labeled: int
    predicted: int
    matched: int
    false_neg: int
    false_pos: int
    correct: int  # matched shots with the right outcome
    made_labeled: int
    made_predicted: int
    errors: List[Dict] = field(default_factory=list)
    by_shot_type: Dict[str, List[int]] = field(default_factory=dict)  # type -> [correct, labelled]

    @property
    def make_count_error(self) -> int:
        return abs(self.made_predicted - self.made_labeled)

    @property
    def attempt_count_error(self) -> int:
        return abs(self.predicted - self.labeled)


def evaluate_video(video: str, labels: Sequence[LabeledShot], preds: Sequence[PredictedShot], tol_s: float = 1.5) -> VideoResult:
    pairs, miss_l, extra_p = match_shots(labels, preds, tol_s)
    errors: List[Dict] = []
    correct = 0
    right = set()
    for li, pi in pairs:
        if labels[li].outcome == preds[pi].outcome:
            correct += 1
            right.add(li)
        else:
            errors.append(dict(video=video, kind="wrong_outcome", t=round(labels[li].t, 2), label=labels[li].outcome,
                               predicted=preds[pi].outcome, reason=preds[pi].reason, shot_type=labels[li].shot_type,
                               note=labels[li].note))
    for li in miss_l:
        errors.append(dict(video=video, kind="missed_shot", t=round(labels[li].t, 2), label=labels[li].outcome,
                           predicted="", reason="", shot_type=labels[li].shot_type, note=labels[li].note))
    for pi in extra_p:
        errors.append(dict(video=video, kind="extra_shot", t=round(preds[pi].t, 2), label="",
                           predicted=preds[pi].outcome, reason=preds[pi].reason, shot_type="", note=""))
    by_type: Dict[str, List[int]] = {}
    for li, l in enumerate(labels):
        if l.shot_type:
            g = by_type.setdefault(l.shot_type, [0, 0])
            g[0] += li in right
            g[1] += 1
    errors.sort(key=lambda e: (e["video"], e["t"]))
    return VideoResult(
        video=video, labeled=len(labels), predicted=len(preds), matched=len(pairs),
        false_neg=len(miss_l), false_pos=len(extra_p), correct=correct,
        made_labeled=sum(l.outcome == MADE for l in labels), made_predicted=sum(p.outcome == MADE for p in preds),
        errors=errors, by_shot_type=by_type,
    )


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ratio(a: int, b: int) -> Optional[float]:
    return a / b if b else None


@dataclass
class Summary:
    videos: int
    labeled: int
    predicted: int
    matched: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    outcome_accuracy: Optional[float]  # among matched shots
    end_to_end_accuracy: Optional[float]  # over all labelled shots
    end_to_end_ci95: Tuple[float, float]
    mean_abs_make_error: Optional[float]
    mean_abs_attempt_error: Optional[float]
    made_to_missed: int  # labelled made, called missed
    missed_to_made: int  # labelled missed, called made


def summarize(results: Sequence[VideoResult]) -> Summary:
    lab = sum(r.labeled for r in results)
    pred = sum(r.predicted for r in results)
    mat = sum(r.matched for r in results)
    cor = sum(r.correct for r in results)
    prec, rec = _ratio(mat, pred), _ratio(mat, lab)
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    m2x = sum(1 for r in results for e in r.errors if e["kind"] == "wrong_outcome" and e["label"] == MADE)
    x2m = sum(1 for r in results for e in r.errors if e["kind"] == "wrong_outcome" and e["label"] == MISSED)
    n = len(results)
    return Summary(
        videos=n, labeled=lab, predicted=pred, matched=mat, precision=prec, recall=rec, f1=f1,
        outcome_accuracy=_ratio(cor, mat), end_to_end_accuracy=_ratio(cor, lab), end_to_end_ci95=wilson_interval(cor, lab),
        mean_abs_make_error=sum(r.make_count_error for r in results) / n if n else None,
        mean_abs_attempt_error=sum(r.attempt_count_error for r in results) / n if n else None,
        made_to_missed=m2x, missed_to_made=x2m,
    )


def load_csv_by_video(path: Path) -> Dict[int, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {int(r["video"]): r for r in csv.DictReader(f)}


def video_number(name: str) -> Optional[int]:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None
