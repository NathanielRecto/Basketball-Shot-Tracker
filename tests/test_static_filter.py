from shottracker.geometry import Box
from shottracker.pipeline import ShotPipeline
from shottracker.sim import DEFAULT_HOOP, ScriptedDetector, ShotSpec, detections_from_stream, simulate_stream
from shottracker.tracking import BallTracker, StaticSuppressor
from shottracker.types import Detection, Outcome

FPS = 30


def ball(cx, cy, size=30, conf=0.2):
    return Detection("ball", conf, Box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2))


def test_a_detection_that_never_moves_is_dropped_after_a_few_seconds():
    f = StaticSuppressor()
    kept = [bool(f.filter(i / FPS, [ball(820, 685)])) for i in range(10 * FPS)]
    assert all(kept[: 2 * FPS])  # not enough evidence yet
    assert not any(kept[4 * FPS:])  # clearly static by now


def test_a_moving_ball_is_never_dropped():
    f = StaticSuppressor()
    kept = [bool(f.filter(i / FPS, [ball(100 + 15 * i, 500 + 5 * (i % 40))])) for i in range(10 * FPS)]
    assert all(kept)


def test_a_ball_held_still_before_a_shot_is_kept():
    f = StaticSuppressor()
    kept = []
    for i in range(int(1.5 * FPS)):  # held still for 1.5 s at the start of the video
        kept.append(bool(f.filter(i / FPS, [ball(300, 900)])))
    assert all(kept)


def test_non_ball_detections_pass_through():
    f = StaticSuppressor()
    hoop = Detection("hoop", 0.9, Box(0, 0, 50, 80))
    for i in range(10 * FPS):
        out = f.filter(i / FPS, [hoop])
    assert out == [hoop]


def _decoy_run(suppress_static, **tracker_kw):
    # Reproduces video_01: the real ball goes undetected for a while, the tracker latches onto a
    # permanent "ball" on the rim bracket, and then ignores the real ball once it reappears.
    stream = simulate_stream(DEFAULT_HOOP, [ShotSpec(offset=0.0), ShotSpec(offset=1.5, release=(300, 400))],
                             seed=5, lead_s=6.0)
    decoy = ball(DEFAULT_HOOP.x2 - 5, DEFAULT_HOOP.y1 + 3, size=25, conf=0.5)
    dets = []
    for (t, _), frame in zip(stream, detections_from_stream(stream, DEFAULT_HOOP)):
        if t < 6.0:  # real ball not detected during the first seconds
            frame = [d for d in frame if d.label != "ball"]
        dets.append(frame + [decoy])
    pipe = ShotPipeline(ScriptedDetector(dets), hoop=DEFAULT_HOOP, suppress_static=suppress_static)
    pipe.tracker = BallTracker(**tracker_kw)
    for t, _ in stream:
        pipe.step(t, None)
    return [e.outcome for e in pipe.events]


def test_static_decoy_next_to_the_rim_hides_shots_without_the_filter_or_track_switching():
    assert _decoy_run(suppress_static=False, switch_speed=float("inf")) == []


def test_track_switching_alone_escapes_the_static_decoy():
    # The released shot rises fast while the decoy never moves, so the tracker jumps to it.
    assert _decoy_run(suppress_static=False) == [Outcome.MADE, Outcome.MISSED]


def test_pipeline_ignores_a_static_decoy_next_to_the_rim():
    assert _decoy_run(suppress_static=True) == [Outcome.MADE, Outcome.MISSED]
