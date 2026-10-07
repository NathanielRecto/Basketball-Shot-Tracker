"""Score several prediction runs (e.g. old vs new detector) against the same labels, side by side.

    python scripts/compare_runs.py --labels data/own_footage/labels --videos 2 3 4 \
        --run baseline=outputs/own_eval_baseline --run finetuned=outputs/own_eval_finetuned

Each run folder is what run_eval_videos.py wrote (<run>/<video>/summary.json). Only videos that
are labelled AND present in every run are compared, so the numbers are on identical shots.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from shottracker.evaluation import evaluate_video, load_labels, load_predictions, summarize, video_number  # noqa: E402


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR", help="repeat for each run")
    ap.add_argument("--videos", type=int, nargs="*", help="only these video numbers (e.g. the test set)")
    ap.add_argument("--tol", type=float, default=1.5)
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        name, sep, folder = spec.partition("=")
        if not sep:
            ap.error(f"--run must look like NAME=DIR, got {spec!r}")
        runs.append((name, Path(folder)))

    labels = {}
    for lf in sorted(Path(a.labels).glob("*_*.csv")):
        v = video_number(lf.stem)
        if v is None or (a.videos and v not in set(a.videos)):
            continue
        rows = load_labels(lf)
        if rows:
            labels[lf.stem] = rows
    common = [n for n in labels if all((d / n / "summary.json").is_file() for _, d in runs)]
    for n in sorted(set(labels) - set(common)):
        print(f"left out {n}: not in every run")
    if not common:
        print("Nothing to compare.")
        return 1

    print(f"videos: {', '.join(common)}   (match tolerance +/-{a.tol}s)")
    seen = {}
    for _, d in runs:
        for n in common:
            sp = d / n / "settings.json"
            if sp.is_file():
                st = json.loads(sp.read_text())
                for key in ("imgsz", "stride"):
                    seen.setdefault((n, key), set()).add(st.get(key))
    clash = sorted({f"{key} on {n}" for (n, key), vals in seen.items() if len(vals) > 1})
    if clash:
        print("WARNING: the runs used different settings (" + ", ".join(clash) + "); "
              "differences may come from that, not from the change being tested")
    print()
    print(f"{'run':<16} {'shots':>5} {'end-to-end':>11} {'95% CI':>15} {'precision':>9} {'recall':>7} {'made/miss':>9}")
    cis = []
    for name, d in runs:
        s = summarize([evaluate_video(n, labels[n], load_predictions(d / n / "summary.json"), a.tol) for n in common])
        lo, hi = s.end_to_end_ci95
        cis.append((lo, hi))
        print(f"{name:<16} {s.labeled:>5} {pct(s.end_to_end_accuracy):>11} {pct(lo) + '-' + pct(hi):>15} "
              f"{pct(s.precision):>9} {pct(s.recall):>7} {pct(s.outcome_accuracy):>9}")
    overlap = max(lo for lo, _ in cis) <= min(hi for _, hi in cis)
    print("\nThe runs share the same shots, so differences come from the runs, not the sample. "
          + ("Their 95% CIs overlap: treat a small gap with care." if overlap else "Their 95% CIs do not overlap."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
