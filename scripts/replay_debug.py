"""Re-run tracking + shot judging from a saved debug.json in seconds (no detector needed).

    python scripts/replay_debug.py outputs/debug/video_01/debug.json --conf 0.15

Lets you compare tracker settings quickly on development videos. debug.json comes from
scripts/debug_video.py and holds every ball detection down to its --low-conf.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shottracker.geometry import Box  # noqa: E402
from shottracker.pipeline import ShotPipeline  # noqa: E402
from shottracker.types import Detection  # noqa: E402


def _balls(f):
    for b in f["balls"]:
        yield b if len(b) == 5 else (b[0], b[1], b[2], b[2], b[3])  # older files had no height


def replay(debug: dict, conf: float, suppress_static: bool = True):
    hoop = Box(*debug["hoop"])
    pipe = ShotPipeline(lambda dets: dets, hoop=hoop, suppress_static=suppress_static)
    tracked = 0
    for f in debug["frames"]:
        dets = [Detection("ball", c, Box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)) for x, y, w, h, c in _balls(f) if c >= conf]
        fr = pipe.step(f["t"], dets)
        f["_tracked"] = None if fr.ball is None else (fr.ball.x, fr.ball.y)
        tracked += fr.ball is not None
    return pipe.events, tracked / max(len(debug["frames"]), 1), pipe.shots.log


def find_arcs(debug: dict, min_conf: float = 0.1, gap_s: float = 1.0):
    """Stretches where any ball detection rose above the arm line, away from the static bracket spot.

    A rough, detector-derived view of where shots probably are: for debugging only, not ground truth.
    """
    hoop = Box(*debug["hoop"])
    arm_y = debug["arm_y"]
    pts = [(f["t"], x, y, c) for f in debug["frames"] for x, y, w, h, c in _balls(f)
           if c >= min_conf and y < arm_y and not (hoop.x1 - 5 <= x <= hoop.x2 + 5 and y > hoop.y1 - hoop.h)]
    arcs, cur = [], []
    for p in sorted(pts):
        if cur and p[0] - cur[-1][0] > gap_s:
            arcs.append(cur)
            cur = []
        cur.append(p)
    if cur:
        arcs.append(cur)
    return [a for a in arcs if len(a) >= 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("debug_json")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.3], help="one or more ball confidence thresholds to compare")
    ap.add_argument("--no-static", action="store_true", help="disable the static-detection filter")
    ap.add_argument("--arcs", action="store_true", help="list visible ball arcs and what the tracker decided for each")
    a = ap.parse_args()
    debug = json.loads(Path(a.debug_json).read_text())
    if min(a.conf) < debug.get("low_conf", 0.1) - 1e-9:
        print("note: debug.json only holds detections down to its --low-conf")
    for c in a.conf:
        events, frac, log = replay(debug, c, not a.no_static)
        made = sum(e.outcome.value == "made" for e in events)
        print(f"conf>={c:<5} static filter {'off' if a.no_static else 'on '}: {len(events):>2} shots ({made} made), ball tracked in {100 * frac:.0f}% of frames")
        print("   " + "  ".join(f"{e.t_cross:.1f}{'M' if e.outcome.value == 'made' else 'x'}" for e in events))
        if not a.arcs:
            continue
        hoop = Box(*debug["hoop"])
        arcs = find_arcs(debug)
        print(f"\n   {len(arcs)} visible arcs (any detection >= 0.1 above the arm line):")
        for arc in arcs:
            t0, t1 = arc[0][0], arc[-1][0]
            xs = [p[1] for p in arc]
            strong = sum(p[3] >= c for p in arc)
            trk = sum(1 for f in debug["frames"] if t0 <= f["t"] <= t1 and f["_tracked"] and f["_tracked"][1] < debug["arm_y"])
            shot = [e for e in events if t0 - 0.5 <= e.t_cross <= t1 + 2.5]
            notes = [f"{t:.1f}:{m}" for t, m in log if t0 - 1.0 <= t <= t1 + 2.5]
            dx = (min(xs) - hoop.cx) / hoop.w, (max(xs) - hoop.cx) / hoop.w
            print(f"   {t0:6.1f}-{t1:5.1f}s  pts {len(arc):3d} (conf>={c}: {strong:3d})  tracked-above-arm {trk:3d}  "
                  f"x from hoop {dx[0]:+.1f}..{dx[1]:+.1f} widths  -> {'SHOT ' + shot[0].outcome.value if shot else 'no shot'}  {' '.join(notes)}")


if __name__ == "__main__":
    sys.exit(main())
