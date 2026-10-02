import math

import pytest

from shottracker import biomechanics as bio
from shottracker.geometry import Box, fit_flight, joint_angle
from shottracker.pipeline import ShotPipeline
from shottracker.sim import (
    DEFAULT_HOOP, BALL_DIAM, ScriptedDetector, ShotSpec, detections_from_stream, simulate_stream,
)
from shottracker.stats import summarize
from shottracker.tracking import BallTracker, HoopLocator
from shottracker.types import Detection, Outcome


def det(label, cx, cy, size=40, conf=0.9):
    r = size / 2
    return Detection(label, conf, Box(cx - r, cy - r, cx + r, cy + r))


# ---- geometry ---------------------------------------------------------------------------

def test_joint_angle():
    assert joint_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90)
    assert joint_angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180)
    assert math.isnan(joint_angle((0, 0), (0, 0), (1, 1)))


def test_fit_flight_recovers_parabola():
    g, vy0, vx = 1400.0, -600.0, 300.0
    samples = [(t, 100 + vx * t, 400 + vy0 * t + 0.5 * g * t * t) for t in [i / 30 for i in range(25)]]
    fit = fit_flight(samples)
    assert fit.is_physical
    assert fit.rms < 1e-6
    assert fit.velocity_at(fit.t0)[1] == pytest.approx(vy0, rel=1e-6)
    # descending branch reaches y=300 at the larger root
    t = fit.time_at_y(300.0)
    assert fit.y_at(t) == pytest.approx(300.0, abs=1e-6)
    assert fit.velocity_at(t)[1] > 0


def test_fit_flight_needs_enough_points():
    assert fit_flight([(0, 0, 0), (1, 1, 1)]) is None


# ---- tracking ---------------------------------------------------------------------------

def test_tracker_follows_the_ball_not_a_distractor():
    tr = BallTracker()
    assert tr.update(0.0, [det("ball", 100, 100)]) is not None
    obs = tr.update(1 / 30, [det("ball", 112, 100), det("ball", 900, 500, conf=0.99)])
    assert (round(obs.x), round(obs.y)) == (112, 100)


def test_tracker_ignores_low_confidence_when_starting():
    assert BallTracker().update(0.0, [det("ball", 10, 10, conf=0.2)]) is None


def test_tracker_resets_after_losing_the_ball():
    tr = BallTracker()
    tr.update(0.0, [det("ball", 100, 100)])
    assert tr.update(0.1, [det("ball", 900, 500)]) is None  # outside the gate, track still alive
    obs = tr.update(1.0, [det("ball", 900, 500)])  # long gap -> re-acquire anywhere
    assert obs is not None and obs.x == 900


def test_tracker_starts_from_weak_detections_only_when_they_move():
    moving = BallTracker()
    out = [moving.update(i / 30, [det("ball", 500 + 40 * i, 560 - 5 * i, conf=0.2)]) for i in range(4)]
    assert out[0] is None and all(o is not None for o in out[1:])  # confirmed on the 2nd frame
    assert moving._vx == pytest.approx(1200, rel=0.05)  # velocity carried over from the confirming pair

    still = BallTracker()
    assert all(still.update(i / 30, [det("ball", 820, 686, conf=0.25)]) is None for i in range(30))


def test_tracker_rejects_impossible_jumps_when_starting_weakly():
    tr = BallTracker()
    tr.update(0.0, [det("ball", 100, 100, conf=0.2)])
    assert tr.update(1 / 30, [det("ball", 1000, 900, conf=0.2)]) is None  # too far for one frame


def test_hoop_locator_median_ignores_outliers():
    loc = HoopLocator(min_samples=3)
    good = [det("hoop", 935, 175, size=50)] * 9
    boxes = None
    for d in good + [det("hoop", 200, 400, size=50)]:
        boxes = loc.update([d])
    assert boxes.cx == pytest.approx(935)


# ---- biomechanics -----------------------------------------------------------------------

def _pose(t, elbow_xy=(0, 0)):
    kp = {
        "r_shoulder": (0, 0, 1), "r_elbow": (0, 100, 1), "r_wrist": (100, 100, 1),  # 90 degree elbow
        "r_hip": (0, 200, 1), "r_knee": (0, 300, 1), "r_ankle": (0, 400, 1),  # straight leg
        "l_wrist": (500, 500, 1),
    }
    return bio.PoseFrame(t, kp)


def test_elbow_and_knee_angles():
    p = _pose(0.0)
    assert bio.elbow_angle(p, "r") == pytest.approx(90)
    assert bio.knee_angle(p, "r") == pytest.approx(180)


def test_shooting_side_and_in_hand():
    p = _pose(0.0)
    assert bio.shooting_side(p, (110, 100)) == "r"
    assert bio.ball_in_hand(p, (110, 100), 40)
    assert not bio.ball_in_hand(p, (400, 100), 40)


def test_low_visibility_joints_are_ignored():
    kp = dict(_pose(0.0).keypoints)
    kp["r_elbow"] = (0, 100, 0.1)
    assert bio.elbow_angle(bio.PoseFrame(0.0, kp), "r") is None


def test_analyse_release_reports_elbow_and_deepest_knee():
    poses = [_pose(t / 10) for t in range(10)]
    out = bio.analyse_release(poses, t_release=0.5, ball_xy=(110, 100))
    assert out["elbow_at_release_deg"] == pytest.approx(90)
    assert out["knee_min_deg"] == pytest.approx(180)
    assert bio.analyse_release(poses, t_release=5.0, ball_xy=(110, 100)) == {}  # no pose near that time


# ---- pipeline ---------------------------------------------------------------------------

def test_pipeline_end_to_end_with_scripted_detector():
    hoop = DEFAULT_HOOP
    specs = [ShotSpec(offset=0.0), ShotSpec(offset=1.4, release=(300, 400)), ShotSpec(offset=-0.2, occlude=True)]
    stream = simulate_stream(hoop, specs, noise_px=1.0, dropout=0.05, seed=1)
    pipe = ShotPipeline(ScriptedDetector(detections_from_stream(stream, hoop)))
    for t, _ in stream:
        pipe.step(t, None)
    s = pipe.summary()
    assert (s.attempts, s.made, s.missed) == (3, 2, 1)
    assert s.fg_pct == pytest.approx(66.7)
    assert [x["outcome"] for x in s.shots] == ["made", "missed", "made"]


def test_pipeline_attaches_pose_metrics_and_release_hint():
    hoop = DEFAULT_HOOP
    spec = ShotSpec(offset=0.0)
    stream = simulate_stream(hoop, [spec], seed=2)

    def pose_fn(frame, t):  # right wrist sits on the ball until the release, then drops away
        return _pose_with_wrist(t, ball_obs.get(round(t, 4)))

    ball_obs = {round(t, 4): o for t, o in stream}

    def _pose_with_wrist(t, obs):
        kp = dict(_pose(t).keypoints)
        if obs is not None:
            kp["r_wrist"] = (obs.x + 5, obs.y + 5, 1.0)
        return bio.PoseFrame(t, kp)

    pipe = ShotPipeline(ScriptedDetector(detections_from_stream(stream, hoop)), pose=pose_fn)
    for t, _ in stream:
        pipe.step(t, None)
    ev = pipe.events[0]
    assert ev.outcome is Outcome.MADE
    assert "knee_min_deg" in ev.biomechanics
    assert summarize(pipe.events).avg_knee_min_deg is not None
