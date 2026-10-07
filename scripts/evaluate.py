"""Score the shot tracker against hand-labelled evaluation videos.

    python scripts/run_eval_videos.py --weights runs/detect/baseline/weights/best.pt
    python scripts/evaluate.py

Expects
    data/eval_videos/labels/video_XX.csv      your labels (see docs/labeling.md)
    outputs/eval/video_XX/summary.json        written by run_eval_videos.py

Own footage:

    python scripts/evaluate.py --labels data/own_footage/labels --pred outputs/own_eval --videos 2 3 4

Videos without labels (or without predictions) are skipped and listed. Writes
outputs/eval/evaluation.json and outputs/eval/errors.csv (every miss/extra/wrong call, to review).
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pathlib import Path  # noqa: E402

from shottracker.evaluation import (  # noqa: E402
    SHOT_TYPES, evaluate_video, load_csv_by_video, load_labels, load_predictions, summarize, video_number, wilson_interval,
)

ROOT = Path(__file__).resolve().parent.parent


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(ROOT / "data/eval_videos/labels"))
    ap.add_argument("--pred", default=str(ROOT / "outputs/eval"))
    ap.add_argument("--meta", default=None, help="optional: setup info per video (default: listed_counts.csv next to the labels folder)")
    ap.add_argument("--author", default=None, help="optional: other system's results (default: author_algorithm_results.csv next to the labels folder)")
    ap.add_argument("--tol", type=float, default=1.5, help="seconds within which a prediction matches a label")
    ap.add_argument("--out", default=None, help="where to write evaluation.json / errors.csv (default: --pred)")
    ap.add_argument("--videos", type=int, nargs="*", help="only these video numbers (e.g. the test set)")
    a = ap.parse_args()

    labels_dir, pred_dir = Path(a.labels), Path(a.pred)
    out_dir = Path(a.out) if a.out else pred_dir
    # Looked up next to the labels folder so another dataset never picks up these per-number files.
    meta_path = Path(a.meta) if a.meta else labels_dir.parent / "listed_counts.csv"
    author_path = Path(a.author) if a.author else labels_dir.parent / "author_algorithm_results.csv"
    meta = load_csv_by_video(meta_path) if meta_path.is_file() else {}
    author = load_csv_by_video(author_path) if author_path.is_file() else {}

    results, skipped = [], []
    for lf in sorted(f for f in labels_dir.glob("*_*.csv") if video_number(f.stem) is not None):
        name = lf.stem
        if a.videos and video_number(name) not in set(a.videos):
            continue
        labels = load_labels(lf)
        if not labels:
            skipped.append((name, "no labels yet"))
            continue
        sp = pred_dir / name / "summary.json"
        if not sp.is_file():
            skipped.append((name, "no predictions (run scripts/run_eval_videos.py)"))
            continue
        results.append(evaluate_video(name, labels, load_predictions(sp), a.tol))

    for name, why in skipped:
        print(f"skipped {name}: {why}")
    if not results:
        print("\nNothing to evaluate yet.")
        return 1

    print(f"\nmatch tolerance: +/-{a.tol}s\n")
    print(f"{'video':<10} {'labeled':>7} {'found':>5} {'extra':>5} {'missed':>6} {'wrong':>5} {'makes lab/pred':>15} {'|d makes|':>9}")
    for r in results:
        wrong = r.matched - r.correct
        print(f"{r.video:<10} {r.labeled:>7} {r.matched:>5} {r.false_pos:>5} {r.false_neg:>6} {wrong:>5} "
              f"{r.made_labeled:>7}/{r.made_predicted:<7} {r.make_count_error:>9}")

    s = summarize(results)
    lo, hi = s.end_to_end_ci95
    print("\n=== Overall ===")
    print(f"videos {s.videos} | labelled shots {s.labeled} | predicted {s.predicted} | matched {s.matched}")
    print(f"shot detection   precision {pct(s.precision)}   recall {pct(s.recall)}   F1 {pct(s.f1)}")
    print(f"make/miss on matched shots: {pct(s.outcome_accuracy)}   (made->missed {s.made_to_missed}, missed->made {s.missed_to_made})")
    print(f"END-TO-END accuracy: {pct(s.end_to_end_accuracy)}   95% CI {pct(lo)} - {pct(hi)}")
    print(f"mean |makes error| per video {s.mean_abs_make_error:.2f}   mean |attempts error| per video {s.mean_abs_attempt_error:.2f}")
    if s.labeled < 100:
        print(f"note: only {s.labeled} labelled shots, so the interval above is wide")

    groups = defaultdict(lambda: [0, 0])
    for r in results:
        v = video_number(r.video)
        if v in meta:
            for key in ("env", "camera_height", "throw_type"):
                g = groups[f"{key}={meta[v][key]}"]
                g[0] += r.correct
                g[1] += r.labeled
    if groups:
        print("\n=== By setup (end-to-end accuracy) ===")
        for k in sorted(groups):
            c, n = groups[k]
            glo, ghi = wilson_interval(c, n)
            print(f"{k:<28} {c:>3}/{n:<3} {pct(c / n):>7}   95% CI {pct(glo)} - {pct(ghi)}")

    types = defaultdict(lambda: [0, 0])
    for r in results:
        for k, (c, n) in r.by_shot_type.items():
            types[k][0] += c
            types[k][1] += n
    if types:
        print("\n=== By shot type (end-to-end accuracy) ===")
        for k in [t for t in SHOT_TYPES if t in types]:
            c, n = types[k]
            glo, ghi = wilson_interval(c, n)
            print(f"{k:<28} {c:>3}/{n:<3} {pct(c / n):>7}   95% CI {pct(glo)} - {pct(ghi)}")
        untagged = s.labeled - sum(n for _, n in types.values())
        if untagged:
            print(f"({untagged} labelled shots have no shot type)")

    both = [r for r in results if video_number(r.video) in author]
    if both:
        ours_make = sum(r.make_count_error for r in both) / len(both)
        ours_att = sum(r.attempt_count_error for r in both) / len(both)
        theirs_make = sum(abs(int(author[video_number(r.video)]["predicted_score"]) - int(author[video_number(r.video)]["actual_score"])) for r in both) / len(both)
        theirs_att = sum(abs(int(author[video_number(r.video)]["predicted_throws"]) - int(author[video_number(r.video)]["actual_throws"])) for r in both) / len(both)
        print(f"\n=== Count error vs the original author's system ({len(both)} shared videos) ===")
        print(f"mean |makes error|     ours {ours_make:.2f}   theirs {theirs_make:.2f}")
        print(f"mean |attempts error|  ours {ours_att:.2f}   theirs {theirs_att:.2f}")
        print("(counts only: they can hide offsetting errors; the end-to-end figure above is stricter)")

    out_dir.mkdir(parents=True, exist_ok=True)
    errors = [e for r in results for e in r.errors]
    with open(out_dir / "errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "kind", "t", "label", "predicted", "reason", "shot_type", "note"])
        w.writeheader()
        w.writerows(errors)
    (out_dir / "evaluation.json").write_text(json.dumps({
        "tolerance_s": a.tol,
        "summary": s.__dict__,
        "videos": [{k: v for k, v in r.__dict__.items() if k != "errors"} for r in results],
        "skipped": skipped,
    }, indent=2))
    print(f"\n{len(errors)} errors written to {out_dir / 'errors.csv'}; details in evaluation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
