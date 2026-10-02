import pytest

from shottracker.calibration import consensus_box, parse_box
from shottracker.geometry import Box
from shottracker.pipeline import ShotPipeline
from shottracker.sim import DEFAULT_HOOP, ScriptedDetector, ShotSpec, detections_from_stream, simulate_stream
from shottracker.types import Outcome


def box(cx, cy, w=40, h=70):
    return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_consensus_prefers_the_agreeing_cluster_over_a_confident_outlier():
    cands = [(box(500 + d, 300 + d), 0.15) for d in (-2, -1, 0, 1, 2, 3)] + [(box(100, 900), 0.9)]
    b = consensus_box(cands)
    assert (b.cx, b.cy) == pytest.approx((500.5, 300.5), abs=1.5)


def test_consensus_needs_enough_support():
    assert consensus_box([(box(500, 300), 0.9), (box(501, 301), 0.9)]) is None
    assert consensus_box([]) is None


def test_parse_box():
    assert parse_box("10,20,50,90") == Box(10, 20, 50, 90)
    with pytest.raises(ValueError):
        parse_box("50,20,10,90")  # x2 < x1
    with pytest.raises(ValueError):
        parse_box("1,2,3")


def test_pipeline_uses_a_fixed_hoop_when_detection_never_finds_it():
    stream = simulate_stream(DEFAULT_HOOP, [ShotSpec(offset=0.0), ShotSpec(offset=1.5, release=(300, 400))], seed=4)
    no_hoop = [[d for d in frame if d.label != "hoop"] for frame in detections_from_stream(stream, DEFAULT_HOOP)]

    blind = ShotPipeline(ScriptedDetector(no_hoop))
    for t, _ in stream:
        blind.step(t, None)
    assert blind.events == []  # no hoop, no shots

    calibrated = ShotPipeline(ScriptedDetector(no_hoop), hoop=DEFAULT_HOOP)
    for t, _ in stream:
        calibrated.step(t, None)
    assert [e.outcome for e in calibrated.events] == [Outcome.MADE, Outcome.MISSED]
