"""Run the trained detector + shot tracker over the evaluation videos.

    python scripts/run_eval_videos.py --weights runs/detect/baseline/weights/best.pt

Writes outputs/eval/video_XX/summary.json (and annotated.mp4 unless --no-video), which
scripts/evaluate.py then scores against your labels. Use --videos 1 5 to run a subset.
Own footage (data/own_footage/raw/session_XX.mov):

    python scripts/run_eval_videos.py --weights ... --videos-dir data/own_footage/raw --out outputs/own_eval \
        --calibrate-hoop --hoop-overrides data/own_footage/hoop_overrides.json
Run it AFTER training: inference on the GPU while training would slow both.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shottracker.calibration import calibrate_hoop, save_hoop_preview  # noqa: E402
from shottracker.cli import analyse_video  # noqa: E402
from shottracker.detector import YoloDetector  # noqa: E402
from shottracker.geometry import Box  # noqa: E402
from shottracker.video import find_videos, probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--videos-dir", default=str(ROOT / "data/eval_videos"))
    ap.add_argument("--out", default=str(ROOT / "outputs/eval"))
    ap.add_argument("--videos", type=int, nargs="*", help="video numbers to run (default: all found)")
    ap.add_argument("--conf", type=float, default=0.15, help="detection confidence; weak ball detections are vetted by the tracker")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference size on the long side (default: 1280 for portrait video, 960 for landscape; "
                         "the published results on the portrait eval videos used 1280)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (2 halves the time at 60 fps)")
    ap.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4 (faster)")
    ap.add_argument("--pose", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--calibrate-hoop", action="store_true", help="locate the hoop once per video (fixed camera)")
    ap.add_argument("--hoop-overrides", default=None,
                    help='JSON file {"video_07": [x1, y1, x2, y2], ...} for videos where calibration fails')
    a = ap.parse_args()

    found = find_videos(a.videos_dir)
    files = [f for v, f in sorted(found.items()) if not a.videos or v in set(a.videos)]
    if not files:
        print(f"no videos found in {a.videos_dir}")
        return 1

    pose = None
    if a.pose:
        from shottracker.pose import MediaPipePose

        pose = MediaPipePose()
    detector = YoloDetector(a.weights, conf=a.conf, imgsz=a.imgsz or 960, device=a.device)
    overrides = json.loads(Path(a.hoop_overrides).read_text()) if a.hoop_overrides and Path(a.hoop_overrides).is_file() else {}
    for f in files:
        t0 = time.time()
        out_dir = os.path.join(a.out, f.stem)
        os.makedirs(out_dir, exist_ok=True)
        _, w, h, _ = probe(str(f))
        detector.imgsz = a.imgsz or (1280 if h > w else 960)
        Path(out_dir, "settings.json").write_text(json.dumps(
            {"weights": a.weights, "imgsz": detector.imgsz, "stride": a.stride, "conf": a.conf}, indent=2))
        hoop, source, n_cands = None, "per-frame detection", None
        if f.stem in overrides:
            hoop, source = Box(*overrides[f.stem]), "manual override"
        elif a.calibrate_hoop:
            hoop, n_cands = calibrate_hoop(str(f), detector)
            source = "calibrated" if hoop else "calibration failed; per-frame detection"
        save_hoop_preview(str(f), hoop, os.path.join(out_dir, "hoop_check.jpg"))
        Path(out_dir, "hoop.json").write_text(json.dumps(
            {"source": source, "box": None if hoop is None else [round(v, 1) for v in (hoop.x1, hoop.y1, hoop.x2, hoop.y2)],
             "candidates": n_cands}, indent=2))
        pipe = analyse_video(str(f), detector, pose, out_dir, a.stride, annotate=not a.no_video, quiet=True, hoop=hoop)
        s = pipe.summary()
        print(f"{f.stem}: {s.attempts} shots, {s.made} made | hoop: {source} ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nnext: python scripts/evaluate.py --pred {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
