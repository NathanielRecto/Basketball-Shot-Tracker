"""Score the CURRENT tracking/judging code against hand labels in seconds, using saved detections.

    python scripts/replay_eval.py                 # dev videos 1 4 5 8 9 12 13 16
    python scripts/replay_eval.py --errors        # also list every error with context

Needs outputs/debug/video_XX/debug.json (scripts/debug_video.py) and labels. Development
videos only: never point this at the test set while tuning.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from replay_debug import replay  # noqa: E402
from shottracker.evaluation import PredictedShot, evaluate_video, load_labels, summarize  # noqa: E402

DEV = [1, 4, 5, 8, 9, 12, 13, 16]


def run(videos, conf, tol, show_errors):
    results = []
    for v in videos:
        name = f"video_{v:02d}"
        dbg = ROOT / "outputs" / "debug" / name / "debug.json"
        lab = ROOT / "data" / "eval_videos" / "labels" / f"{name}.csv"
        if not dbg.is_file() or not lab.is_file():
            print(f"skip {name}: missing debug.json or labels")
            continue
        events, _, log = replay(json.loads(dbg.read_text()), conf)
        preds = [PredictedShot(e.t_cross, e.outcome.value, e.reason) for e in events]
        r = evaluate_video(name, load_labels(lab), preds, tol)
        results.append(r)
        if show_errors:
            by_t = {round(e.t_cross, 2): e for e in events}
            for err in r.errors:
                e = by_t.get(err["t"])
                extra = f" offset={e.cross_offset:+.2f} entry={e.entry_angle_deg} method={e.method}" if e else ""
                ctx = " ".join(f"{t:.1f}:{m}" for t, m in log if abs(t - err["t"]) <= 2.5 and m != "arm")
                print(f"  {name} {err['t']:6.1f}s {err['kind']:<13} label={err['label'] or '-':<6} pred={err['predicted'] or '-':<6} "
                      f"{err['reason']:<12}{extra}  | {ctx}")
    s = summarize(results)
    lo, hi = s.end_to_end_ci95
    print(f"\nconf {conf}: labelled {s.labeled}  found {s.matched}  extra {s.predicted - s.matched}  "
          f"| recall {100 * s.recall:.1f}%  precision {100 * s.precision:.1f}%  outcome {100 * s.outcome_accuracy:.1f}%  "
          f"| END-TO-END {100 * s.end_to_end_accuracy:.1f}% (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=int, nargs="*", default=DEV)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--errors", action="store_true")
    a = ap.parse_args()
    run(a.videos, a.conf, a.tol, a.errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
