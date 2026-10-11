"""Import a session recorded by the iPhone app, replay it through this code, and (with labels) score it.

    python scripts/import_phone_session.py G:/Downloads/ShotTracker-2026-10-10_19-49-11.zip
    python scripts/import_phone_session.py path/to/session_folder --labels data/phone/labels/session_12.csv

Reads session.json (format shottracker-phone-session/1 or /2) from the app's Export session zip, an unzipped
folder, or the file itself, and writes to outputs/phone_sessions/<name>/:

  debug.json   the phone's ball detections in scripts/debug_video.py's format (replay_debug.py and
               replay_eval.py read it; its fixed "hoop" is the first one the phone found)
  calls.csv    the phone's shot calls

Then it replays the detections through the Python pipeline the way the phone ran it (a new shot judge where the phone
found the hoop or was reset, the hoop moved at the same frames) and checks that every call matches: the phone runs
a golden-tested Swift port of this code, so a difference is a bug, except within ~40 s of an "initial" hoop (the
phone was already judging before the recording started, so its earlier state is not in the file; app builds from
2026-10-11 restart the judge when recording starts, so their recordings replay exactly). Format /1 files lack
the per-frame "judged" flag and hoop kinds, so their replay is approximate. With --labels (time_s,outcome,...
like data/own_footage/labels) it scores the phone's calls with the same evaluation as every other result.
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shottracker.config import ShotConfig  # noqa: E402
from shottracker.evaluation import PredictedShot, evaluate_video, load_labels, summarize  # noqa: E402
from shottracker.geometry import Box  # noqa: E402
from shottracker.pipeline import ShotPipeline  # noqa: E402
from shottracker.tracking import StaticSuppressor  # noqa: E402
from shottracker.types import Detection  # noqa: E402

TIME_TOL = 1e-6  # the phone's times are relative to its first recorded frame, so they differ in the last bits


def load_session(path: Path):
    """(session dict, default name)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next((n for n in z.namelist() if n.endswith("session.json")), None)
            if member is None:
                sys.exit(f"{path}: no session.json inside")
            return json.loads(z.read(member)), path.stem.replace("ShotTracker-", "")
    if path.is_dir():
        path = path / "session.json"
    s = json.loads(path.read_text())
    stamp = s.get("started", "")[:19].replace(":", "-").replace("T", "_")
    return s, (path.parent.name if path.parent.name[:4].isdigit() else f"phone_{stamp}")


def balls(frame, conf):
    return [Detection("ball", c, Box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)) for x, y, w, h, c in frame["balls"] if c >= conf]


def replay_like_phone(s):
    """Python's calls on the phone's detections, with the shot judge started / reset / moved where the phone did."""
    v2 = s["format"].endswith("/2")
    hoops = [dict(h) for h in s["hoops"]]
    for i, h in enumerate(hoops):  # format /1: the first hoop was already judging, later ones moved it
        h.setdefault("kind", None)
        if h["kind"] is None:
            h["kind"] = "initial" if i == 0 else "moved"
    pipe, calls, hi, notes, warmup = None, [], 0, [], []
    # How long a judge that started before the recording can still differ from a fresh one: the static filter's history
    # window plus how long it keeps a spot suppressed, plus the shot judge's own history.
    sf = StaticSuppressor()
    memory = sf.window_s + sf.hold_s + ShotConfig().history_s
    for f in s["frames"]:
        while hi < len(hoops) and hoops[hi]["t"] <= f["t"]:
            h = hoops[hi]
            hi += 1
            if h["kind"] in ("found", "reset", "initial"):
                pipe = ShotPipeline(lambda dets: dets, hoop=Box(*h["box"]))
                if h["kind"] == "initial":
                    notes.append(f"{h['t']:.2f} s: already judging when the recording started; calls released before "
                                 f"{h['t'] + memory:.0f} s may differ (state before the recording is unknown)")
                    warmup.append((h["t"], h["t"] + memory))
            elif h["kind"] == "moved" and pipe is not None:
                pipe.fixed_hoop = Box(*h["box"])
        judged = f.get("judged", True) if v2 else pipe is not None
        if pipe is not None and judged:
            calls += pipe.step(f["t"], balls(f, s["conf"])).events
    return calls, notes, warmup


def compare(phone, python, warmup=()):
    """(differences, expected differences) between the phone's calls and Python's replay; both empty = identical.

    A difference is expected when the shot was released before the end of a `warmup` window (after an "initial"
    hoop): the phone's judge and static filter remember things from before the recording, Python cannot.
    """
    def warm(t_release):
        return any(t_release <= hi for _, hi in warmup)

    diffs, expected, used = [], [], set()
    for i, a in enumerate(phone):  # pair each phone call with Python's call at about the same time (shots are seconds apart)
        j = next((j for j, b in enumerate(python) if j not in used and abs(b.t_cross - a["t_cross"]) <= 1.0), None)
        if j is None:
            (expected if warm(a["t_release"]) else diffs).append(
                f"shot {i + 1}: phone {a['t_cross']:.3f} s {a['outcome']} {a['reason']}, no call in Python")
            continue
        used.add(j)
        b = python[j]
        same = (a["outcome"] == b.outcome.value and a["reason"] == b.reason and a["method"] == b.method
                and abs(a["t_cross"] - b.t_cross) <= TIME_TOL and abs(a["cross_offset"] - b.cross_offset) <= TIME_TOL)
        if not same:
            (expected if warm(a["t_release"]) else diffs).append(
                f"shot {i + 1}: phone {a['t_cross']:.3f} s {a['outcome']} {a['reason']} ({a['method']}) vs "
                f"Python {b.t_cross:.3f} s {b.outcome.value} {b.reason} ({b.method})")
    for j, b in enumerate(python):
        if j not in used:
            (expected if warm(b.t_release) else diffs).append(
                f"Python {b.t_cross:.3f} s {b.outcome.value} {b.reason}, no call on the phone")
    return diffs, expected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="Export session zip, its unzipped folder, or session.json")
    ap.add_argument("--name", help="output folder name (default: the session's date and time)")
    ap.add_argument("--labels", type=Path, help="hand labels for this video (time_s,outcome,...), to score the phone")
    ap.add_argument("--tol", type=float, default=1.5, help="seconds between a label and a call to count as the same shot")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "phone_sessions")
    a = ap.parse_args()

    s, default_name = load_session(a.session)
    if not s.get("format", "").startswith("shottracker-phone-session/"):
        sys.exit(f"{a.session}: not a phone session (format {s.get('format')!r})")
    name = a.name or default_name
    out = a.out / name
    out.mkdir(parents=True, exist_ok=True)
    frames, shots = s["frames"], s["shots"]
    app = s.get("app") or {}
    print(f"{name}: {s['format']}, app {app.get('commit', 'unknown')[:7]}{' +changes' if app.get('dirty') else ''}, "
          f"model {s['model']['name']} {s['model']['precision']} {s['model']['weights_sha256'][:12]}, zoom {s['lens']}")
    if len(frames) > 1:
        span = frames[-1]["t"] - frames[0]["t"]
        print(f"  {len(frames)} detector frames over {span:.1f} s ({(len(frames) - 1) / span:.1f} fps), "
              f"{len(s['hoops'])} hoop events, {len(shots)} calls, {s.get('video_frames_dropped', 0)} video frames dropped")

    # debug.json for the existing replay tools (fixed hoop: the first one found).
    first = Box(*s["hoops"][0]["box"]) if s["hoops"] else None
    cfg = ShotConfig()
    debug = {
        "video": s.get("video", "video.mov"), "frame_size": s.get("frame_size"), "conf": s["conf"], "low_conf": s["conf"],
        "weights": s["model"]["weights_sha256"], "imgsz": 960, "stride": "phone",
        "hoop": None if first is None else [first.x1, first.y1, first.x2, first.y2],
        "rim_y": None if first is None else first.y1 + cfg.rim_line_frac * cfg.hoop_aspect * first.w,
        "arm_y": None if first is None else first.y1 + (cfg.rim_line_frac - cfg.arm_margin) * cfg.hoop_aspect * first.w,
        "frames": [{"t": f["t"], "balls": f["balls"]} for f in frames], "events": [],
    }
    (out / "debug.json").write_text(json.dumps(debug))
    with open(out / "calls.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_cross", "outcome", "reason", "method", "cross_offset"])
        for c in shots:
            w.writerow([round(c["t_cross"], 3), c["outcome"], c["reason"], c["method"], round(c["cross_offset"], 3)])
    print(f"  wrote {out / 'debug.json'} and {out / 'calls.csv'}")

    python, notes, warmup = replay_like_phone(s)
    diffs, expected = compare(shots, python, warmup)
    for n in notes:
        print(f"  note: {n}")
    for d in expected:
        print(f"  expected difference (flight began before the recording): {d}")
    if diffs:
        print(f"  REPLAY DIFFERS from the phone ({len(diffs)}):")
        for d in diffs:
            print(f"    {d}")
    else:
        same = len(shots) - len(expected)
        print(f"  replay matches the phone: {same} of {len(shots)} calls identical in Python"
              + (f", {len(expected)} started before the recording" if expected else ""))

    if a.labels:
        r = evaluate_video(name, load_labels(a.labels), [PredictedShot(c["t_cross"], c["outcome"], c["reason"]) for c in shots], a.tol)
        sm = summarize([r])
        lo, hi = sm.end_to_end_ci95
        print(f"  vs labels: labelled {sm.labeled}  found {sm.matched}  extra {sm.predicted - sm.matched}  | recall "
              f"{100 * sm.recall:.1f}%  precision {100 * sm.precision:.1f}%  outcome {100 * sm.outcome_accuracy:.1f}%  "
              f"| END-TO-END {100 * sm.end_to_end_accuracy:.1f}% (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)")
        for e in r.errors:
            print(f"    {e['t']:7.1f} s {e['kind']:<13} label={e['label'] or '-':<6} phone={e['predicted'] or '-':<6} {e['reason']}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
