import math

import pytest

from shottracker.shot_logic import ShotDetector
from shottracker.sim import DEFAULT_HOOP, ShotSpec, dribble_stream, simulate_stream
from shottracker.types import Outcome

HOOP = DEFAULT_HOOP


def run(stream, hoop=HOOP, hand_hint=None):
    det = ShotDetector()
    events = []
    for t, obs in stream:
        events += det.update(t, obs, hoop)
    return events, det


def test_centered_make():
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=0.0)]))
    assert [e.outcome for e in events] == [Outcome.MADE]
    assert abs(events[0].cross_offset) < 0.05
    assert events[0].method == "interpolated"


@pytest.mark.parametrize("offset", [-0.45, 0.45])
def test_make_off_centre(offset):
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=offset)]))
    assert [e.outcome for e in events] == [Outcome.MADE]
    assert events[0].cross_offset == pytest.approx(offset, abs=0.08)


@pytest.mark.parametrize("offset", [-1.4, -1.0, 1.0, 1.4, 2.5])
def test_miss_off_target(offset):
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=offset)]))
    assert [e.outcome for e in events] == [Outcome.MISSED]
    assert events[0].reason == "off_target"


def test_rim_out_is_a_miss():
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=0.1, behaviour="rim_out")]))
    assert [e.outcome for e in events] == [Outcome.MISSED]
    assert events[0].reason == "rim_out"


def test_bounce_off_rim_without_crossing_is_a_miss():
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=-0.3, behaviour="rim_bounce")]))
    assert [e.outcome for e in events] == [Outcome.MISSED]
    assert events[0].reason == "rim_bounce"


@pytest.mark.parametrize("offset,expected", [(0.0, Outcome.MADE), (0.3, Outcome.MADE), (1.3, Outcome.MISSED)])
def test_ball_hidden_behind_net_is_judged_from_the_arc(offset, expected):
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=offset, occlude=True)]))
    assert [e.outcome for e in events] == [expected]
    assert events[0].method in {"fit", "extrapolated"}


def test_two_shots_in_a_row():
    stream = simulate_stream(HOOP, [ShotSpec(offset=0.0), ShotSpec(offset=1.5, release=(300, 400))])
    events, _ = run(stream)
    assert [e.outcome for e in events] == [Outcome.MADE, Outcome.MISSED]
    assert [e.index for e in events] == [1, 2]


def test_dribbling_never_counts():
    events, _ = run(dribble_stream(6.0))
    assert events == []


def test_no_hoop_no_events():
    det = ShotDetector()
    events = []
    for t, obs in simulate_stream(HOOP, [ShotSpec()]):
        events += det.update(t, obs, None)
    assert events == []


def test_far_away_toss_is_not_a_shot():
    # Ball thrown up far from the hoop. (Real long misses land up to ~3 hoop widths past the
    # rim and DO count as attempts, so this toss comes down ~6 hoop widths away.)
    hoop = HOOP
    stream = simulate_stream(hoop, [ShotSpec(offset=-12.0, release=(100, 400))])
    events, _ = run(stream)
    assert events == []


def test_release_angle_matches_ground_truth():
    from shottracker.sim import _Flight

    spec = ShotSpec(offset=0.0)
    events, _ = run(simulate_stream(HOOP, [spec]))
    ev = events[0]
    assert ev.release_angle_deg is not None
    fl = _Flight(spec, HOOP)
    truth = math.degrees(math.atan2(-fl.vy0, abs(fl.vx)))
    assert ev.release_angle_deg == pytest.approx(truth, abs=3.0)


def test_entry_angle_is_steeper_for_higher_arcs():
    flat = run(simulate_stream(HOOP, [ShotSpec(apex_above_rim=60)]))[0][0]
    high = run(simulate_stream(HOOP, [ShotSpec(apex_above_rim=220)]))[0][0]
    assert high.entry_angle_deg > flat.entry_angle_deg + 8


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_frame_rate_independent(fps):
    stream = simulate_stream(HOOP, [ShotSpec(offset=0.2), ShotSpec(offset=1.3, release=(300, 400))], fps=fps)
    events, _ = run(stream)
    assert [e.outcome for e in events] == [Outcome.MADE, Outcome.MISSED]


def test_noisy_detections_stay_accurate():
    """Judge many randomised shots with pixel noise and 10% missed detections."""
    import numpy as np

    rng = np.random.default_rng(7)
    correct = total = 0
    for seed in range(60):
        made = bool(rng.integers(0, 2))
        side = rng.choice([-1, 1])
        offset = float(rng.uniform(0.0, 0.35)) * side if made else float(rng.uniform(1.0, 2.2)) * side
        spec = ShotSpec(
            offset=offset,
            apex_above_rim=float(rng.uniform(80, 200)),
            release=(float(rng.uniform(150, 450)), float(rng.uniform(330, 430))),
            occlude=bool(rng.integers(0, 2)),
        )
        events, _ = run(simulate_stream(HOOP, [spec], noise_px=1.5, dropout=0.10, seed=seed))
        total += 1
        if len(events) == 1 and (events[0].outcome is Outcome.MADE) == made:
            correct += 1
    assert correct / total >= 0.93, f"only {correct}/{total} correct"


def test_long_miss_past_the_rim_still_counts_as_an_attempt():
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=5.5)]))
    assert [(e.outcome, e.reason) for e in events] == [(Outcome.MISSED, "off_target")]


def test_centred_ball_that_keeps_free_falling_is_a_miss():
    # From a side camera a ball dropping just in front of the rim looks centred; it is not slowed
    # by the net, so it must not count as a make.
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=0.1, behaviour="fall_past")]))
    assert [(e.outcome, e.reason) for e in events] == [(Outcome.MISSED, "fell_past_rim")]


def test_centred_ball_braked_by_the_net_is_a_make():
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=0.1)]))
    assert [(e.outcome, e.reason) for e in events] == [(Outcome.MADE, "through_hoop")]


def test_ball_that_bounces_on_the_rim_and_drops_in_is_one_make():
    events, _ = run(simulate_stream(HOOP, [ShotSpec(offset=0.1, behaviour="rattle_in")]))
    assert [(e.outcome, e.reason) for e in events] == [(Outcome.MADE, "rattled_in")]


@pytest.mark.parametrize("behaviour", ["rim_out", "rim_bounce"])
def test_ball_leaving_the_rim_is_one_miss_not_two_shots(behaviour):
    stream = simulate_stream(HOOP, [ShotSpec(offset=0.1, behaviour=behaviour)], tail_s=3.0)
    events, _ = run(stream)
    assert [e.outcome for e in events] == [Outcome.MISSED]
