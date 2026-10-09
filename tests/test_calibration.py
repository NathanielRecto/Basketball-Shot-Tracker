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


def test_the_nearest_hoop_is_the_biggest_rim_and_one_blocked_frame_does_not_move_it():
    from shottracker.calibration import hoop_from_rims, rim_to_hoop

    near, far = Box(400, 300, 500, 330), Box(1500, 400, 1540, 412)  # 100 px rim close by, 40 px on the far wall
    frames = [[near, far]] * 6 + [[far]]  # in one frame a player blocks the near rim
    hoop, ambiguity = hoop_from_rims(frames)
    assert hoop == rim_to_hoop(near)
    assert 0.0 < ambiguity < 0.3  # the far rim is much smaller: a clear choice
    assert hoop.y2 > near.y2 + 50 and hoop.x1 == near.x1  # the box reaches down over the net


def test_two_hoops_of_similar_size_are_flagged_and_too_few_frames_give_nothing():
    from shottracker.calibration import hoop_from_rims

    a, b = Box(400, 300, 500, 330), Box(1300, 300, 1395, 329)
    _, ambiguity = hoop_from_rims([[a, b]] * 5)
    assert ambiguity > 0.8
    assert hoop_from_rims([[a]] * 2) == (None, None)


def test_find_hoop_keeps_the_hoop_class_and_falls_back_to_the_rim(tmp_path):
    import cv2
    import numpy as np

    from shottracker.calibration import find_hoop, rim_to_hoop
    from shottracker.types import Detection

    video = str(tmp_path / "v.mp4")
    out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(12):
        out.write(np.zeros((240, 320, 3), np.uint8))
    out.release()
    rim = Box(100, 50, 140, 62)
    hoop = Box(98, 48, 142, 100)
    with_hoop = lambda frame, conf=None: [Detection("hoop", 0.8, hoop), Detection("rim_only", 0.9, rim)]  # noqa: E731
    rim_only = lambda frame, conf=None: [Detection("rim_only", 0.9, rim)]  # noqa: E731
    found = find_hoop(video, with_hoop, samples=6)
    assert (found.source, found.box) == ("hoop", hoop)
    found = find_hoop(video, rim_only, samples=6)
    assert (found.source, found.box) == ("rim", rim_to_hoop(rim))
    assert find_hoop(video, lambda frame, conf=None: [], samples=6).source == "none"
