import importlib.util
from pathlib import Path

from shottracker.export import sim_frames
from shottracker.geometry import Box
from shottracker.pipeline import ShotPipeline
from shottracker.types import Detection

ROOT = Path(__file__).resolve().parent.parent


def _module():
    spec = importlib.util.spec_from_file_location("import_phone_session", ROOT / "scripts" / "import_phone_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _session(frames, hoop, calls, hoops):
    """A format /2 phone session from (t, [(x1, y1, x2, y2, conf)]) frames; balls stored as cx, cy, w, h, conf."""
    rows = [{"t": t, "balls": [[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, c] for x1, y1, x2, y2, c in bs],
             "rims": [], "judged": i > 0} for i, (t, bs) in enumerate(frames)]
    shots = [{"t_release": e.t_release, "t_apex": e.t_apex, "t_cross": e.t_cross, "outcome": e.outcome.value,
              "reason": e.reason, "method": e.method, "cross_offset": e.cross_offset} for e in calls]
    return {"format": "shottracker-phone-session/2", "conf": 0.15, "hoops": hoops, "shots": shots, "frames": rows}


def _phone_calls(frames, hoop):
    """What the phone does: the hoop found on frame 0 (not judged), every later frame through one pipeline."""
    pipe, calls = ShotPipeline(lambda d: d, hoop=Box(*hoop)), []
    for t, bs in frames[1:]:
        dets = [Detection("ball", c, Box(x1, y1, x2, y2)) for x1, y1, x2, y2, c in bs if c >= 0.15]
        calls += pipe.step(t, dets).events
    return calls


def test_replay_matches_the_calls_it_was_given():
    m = _module()
    hoop, frames = sim_frames()
    calls = _phone_calls(frames, hoop)
    assert len(calls) >= 5
    s = _session(frames, hoop, calls, [{"t": frames[0][0], "box": hoop, "kind": "found"}])
    replayed, notes = m.replay_like_phone(s)
    assert m.compare(s["shots"], replayed) == [] and notes == []


def test_a_reset_starts_a_new_judge_and_a_difference_is_reported():
    m = _module()
    hoop, frames = sim_frames()
    calls = _phone_calls(frames, hoop)
    s = _session(frames, hoop, calls, [{"t": frames[0][0], "box": hoop, "kind": "found"}])
    # A reset in the middle of a shot loses that shot: the replay must notice the phone did not do that.
    mid = calls[2].t_release + 0.1
    s["hoops"].append({"t": mid, "box": hoop, "kind": "reset"})
    replayed, _ = m.replay_like_phone(s)
    assert m.compare(s["shots"], replayed) != []
