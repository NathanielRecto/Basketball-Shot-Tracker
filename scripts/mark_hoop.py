"""Mark the hoop by hand for fixed-camera videos (when auto-calibration fails or picks the wrong thing).

    python scripts/mark_hoop.py 3 7 8

A window opens on a frame of each video. Drag a box around the rim AND the hanging net
(top edge at the top of the rim, bottom edge at the bottom of the net), then press ENTER
or SPACE. Press C to skip a video. If the hoop is blocked in that frame, rerun with --t 10
to use a frame 10 seconds in.
Boxes are saved to data/eval_videos/hoop_overrides.json, which run_eval_videos.py reads via
--hoop-overrides. For your own footage:

    python scripts/mark_hoop.py 1 2 3 4 --videos-dir data/own_footage/raw --out data/own_footage/hoop_overrides.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shottracker.video import find_videos  # noqa: E402


def read_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=int, nargs="+", help="video numbers, e.g. 3 7 8")
    ap.add_argument("--videos-dir", default=str(ROOT / "data/eval_videos"))
    ap.add_argument("--out", default=str(ROOT / "data/eval_videos/hoop_overrides.json"))
    ap.add_argument("--t", type=float, default=5.0, help="time in seconds of the frame to show")
    ap.add_argument("--max-height", type=int, default=900, help="display height in pixels")
    a = ap.parse_args()

    out = Path(a.out)
    boxes = json.loads(out.read_text()) if out.exists() else {}
    found = find_videos(a.videos_dir)
    for v in a.videos:
        if v not in found:
            print(f"no video number {v} in {a.videos_dir}")
            continue
        name = found[v].stem
        cap = cv2.VideoCapture(str(found[v]))
        t = a.t
        frame = read_frame(cap, t)
        if frame is None:
            print(f"{name}: could not read video")
            continue
        scale = min(1.0, a.max_height / frame.shape[0])
        title = f"{name}: drag a box around the rim + net, then press ENTER (C = skip)"
        disp = cv2.resize(frame, None, fx=scale, fy=scale)
        x, y, w, h = cv2.selectROI(title, disp, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        cap.release()
        if w == 0 or h == 0:
            print(f"{name}: skipped")
            continue
        box = [round(x / scale, 1), round(y / scale, 1), round((x + w) / scale, 1), round((y + h) / scale, 1)]
        boxes[name] = box
        out.write_text(json.dumps(boxes, indent=2))
        print(f"{name}: saved {box}")
    print(f"\noverrides file: {out}")


if __name__ == "__main__":
    main()
