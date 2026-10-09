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


def test_find_videos_keys_numbered_videos_by_number(tmp_path):
    from shottracker.video import find_videos

    for name in ["video_07.mp4", "session_02.MOV", "session_10.mov", "notes.txt", "clip.mp4", "session_03.csv"]:
        (tmp_path / name).write_text("")
    assert {k: v.name for k, v in find_videos(tmp_path).items()} == {
        2: "session_02.MOV", 7: "video_07.mp4", 10: "session_10.mov"}


def _ball(cx, cy, conf=0.6, size=30):
    return Detection("ball", conf, Box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2))


def test_tracker_switches_from_a_wobbling_false_positive_to_a_rising_shot():
    # A "ball" on someone's head drifts slowly and is detected every frame, so the track never
    # gets lost; the real ball then rises fast a few hundred pixels away.
    tr = BallTracker()
    fps = 30
    for i in range(30):
        tr.update(i / fps, [_ball(800 + i, 600, conf=0.7)])
    got = []
    for j in range(10):
        t = (30 + j) / fps
        got.append(tr.update(t, [_ball(830, 600, conf=0.7), _ball(1300 - 20 * j, 600 - 40 * j, conf=0.5)]))
    assert got[-1].x < 1200 and got[-1].y < 400  # following the shot, not the head
    assert not got[0].x > 1000  # no instant jump on the very first sighting


def test_tracker_waits_for_a_ball_that_left_through_the_top_edge():
    tr = BallTracker()
    fps = 30
    for i in range(10):  # rising fast towards the top edge
        obs = tr.update(i / fps, [_ball(1200 - 25 * i, 400 - 40 * i)])
    assert obs is not None and obs.y < 50
    t = 10 / fps
    for k in range(20):  # out of view for ~0.65 s; a confident leg "ball" is the only detection
        assert tr.update(t + k / fps, [_ball(1400, 850, conf=0.8)]) is None
    back = tr.update(t + 20 / fps, [_ball(1400, 850, conf=0.8), _ball(1100, 40, conf=0.3), _ball(470, 50, conf=0.3)])
    assert back is not None and (round(back.x), round(back.y)) == (470, 50)  # where its sideways motion carried it


def test_shot_logic_does_not_give_up_on_a_ball_above_the_frame():
    from shottracker.shot_logic import ShotDetector
    from shottracker.types import BallObs

    hoop = Box(500, 300, 620, 460)
    sd = ShotDetector()
    fps = 30
    for i in range(8):  # rising past the arm line and out through the top
        sd.update(i / fps, BallObs(i / fps, 900 - 20 * i, 330 - 45 * i, 30), hoop)
    for k in range(1, 25):  # 0.8 s with no ball: longer than lost_s, shorter than top_exit_wait_s
        sd.update(7 / fps + k / fps, None, hoop)
    assert not any(m.startswith("abort") for _, m in sd.log)


def test_tracker_finds_a_ball_coming_back_down_over_the_hoop_after_it_slowed_out_of_view():
    # Own session 2: shot from near the camera, the ball flies away from it, so on screen it slows
    # down while above the frame and comes back far short of the constant-velocity guess.
    hoop = Box(500, 300, 620, 460)
    for with_hoop, expect in ((False, None), (True, (650, 30))):
        tr = BallTracker()
        fps = 30
        for i in range(10):  # leaving through the top at 1500 px/s sideways
            obs = tr.update(i / fps, [_ball(1700 - 50 * i, 400 - 40 * i)], hoop if with_hoop else None)
        assert obs is not None and obs.y < 50
        t = 10 / fps
        for k in range(30):  # 1 s out of view; a leg "ball" is the only detection
            tr.update(t + k / fps, [_ball(1400, 850, conf=0.8)], hoop if with_hoop else None)
        back = tr.update(t + 30 / fps, [_ball(1400, 850, conf=0.8), _ball(650, 30, conf=0.5)], hoop if with_hoop else None)
        got = None if back is None else (round(back.x), round(back.y))
        assert got == expect


def test_shot_logic_judges_an_arc_that_was_above_the_frame_as_one_flight():
    from shottracker.shot_logic import ShotDetector
    from shottracker.types import BallObs

    hoop = Box(500, 300, 620, 460)  # rim line at y = 324
    sd = ShotDetector()
    fps = 30
    x = lambda t: 900 - 341 * t  # noqa: E731  reaches the hoop centre at the rim line
    y = lambda t: 330 - 2000 * t + 2000 * t * t  # noqa: E731  apex 170 px above the frame, out of view 0.58 s
    events = []
    for k in range(60):
        t = k / fps
        seen = y(t) >= 0 and k <= 31  # after the rim line it disappears into the net
        events += sd.update(t, BallObs(t, x(t), y(t), 30) if seen else None, hoop)
    assert not any(m.startswith("abort") or "rim_bounce" in m for _, m in sd.log), sd.log
    assert [e.outcome for e in events] == [Outcome.MADE]


def test_shot_logic_calls_a_ball_that_bounces_back_out_at_net_height_a_miss():
    # Own footage: a ball crossing the rim line near the centre, then knocked back towards the shooter
    # off the back of the rim, falls away beside the net. It must not count as a make.
    from shottracker.shot_logic import ShotDetector
    from shottracker.types import BallObs

    hoop = Box(500, 300, 620, 460)  # rim line y = 324, centre x = 560
    fps = 60
    for back_px, want in ((0, Outcome.MADE), (110, Outcome.MISSED)):  # 110 px = 0.92 hoop widths
        sd, events = ShotDetector(), []
        for k in range(120):
            t = k / fps
            if t <= 1.0:  # falling from an apex at t = 0, through the rim line at the centre at t = 1.0
                r = 1.0 - t
                obs = BallObs(t, 560 + 300 * r, 324 - 600 * r + 300 * r * r, 30)
            elif t <= 1.3:
                u = (t - 1.0) / 0.3
                obs = BallObs(t, 560 + back_px * u, 324 + 120 * u, 30)  # slowed in the net, or knocked back out
            else:
                obs = None
            events += sd.update(t, obs, hoop)
        assert [e.outcome for e in events] == [want], (back_px, sd.log)


def test_a_make_whose_sideways_motion_the_net_stopped_is_not_called_fell_past_the_rim():
    # Own footage from the sideline: a shot reaches the rim moving fast sideways and keeps falling fast
    # (a loose net barely slows it). If the net stopped its sideways motion it went in; if it kept
    # moving sideways it fell past the rim.
    from shottracker.shot_logic import ShotDetector
    from shottracker.types import BallObs

    hoop = Box(500, 300, 620, 460)  # rim line y = 324, centre x = 560, 120 px wide
    fps = 60
    for keep_vx, want in ((0.0, Outcome.MADE), (1.0, Outcome.MISSED)):
        sd, events = ShotDetector(), []
        for k in range(24, 120):
            t = k / fps
            if t <= 1.0:  # coming down from the right at 800 px/s sideways (6.7 hoop widths/s)
                r = 1.0 - t
                obs = BallObs(t, 560 + 800 * r, 324 - 600 * r + 300 * r * r, 30)
            elif t <= 1.3:  # after the rim line: falling at 800 px/s, sideways speed stopped or kept
                u = t - 1.0
                obs = BallObs(t, 560 - 800 * keep_vx * u, 324 + 800 * u, 30)
            else:
                obs = None
            events += sd.update(t, obs, hoop)
        assert [e.outcome for e in events] == [want], (keep_vx, sd.log)


def test_a_shot_off_the_front_rim_is_counted_not_thrown_away():
    # Test session 6, 2:07: the ball hits the front rim and comes back towards the shooter. It is deflected
    # sideways, not bounced up, so the tracker follows it on and arc + deflection is no parabola. The arc up
    # to the rim is one, so it is a shot: here it falls outside the hoop, a miss.
    from shottracker.shot_logic import ShotDetector
    from shottracker.types import BallObs

    hoop = Box(500, 300, 620, 460)  # rim line y = 324, centre x = 560, 120 px wide
    sd, events, fps = ShotDetector(), [], 60
    for k in range(39, 150):  # the ball is first seen 0.35 s before it reaches the rim
        t = k / fps
        if t <= 1.0:  # coming down from the right onto the front of the rim
            r = 1.0 - t
            obs = BallObs(t, 625 + 300 * r, 309 - 600 * r + 300 * r * r, 30)
        elif t <= 1.6:  # knocked back towards the shooter, then falling outside the hoop
            s = t - 1.0
            obs = BallObs(t, 625 + 250 * s, 309 - 100 * s + 300 * s * s, 30)
        else:
            obs = None
        events += sd.update(t, obs, hoop)
    assert not any(m.startswith("abort") for _, m in sd.log), sd.log
    assert [(e.outcome, e.reason) for e in events] == [(Outcome.MISSED, "off_target")]
