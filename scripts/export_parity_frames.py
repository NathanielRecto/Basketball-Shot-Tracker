"""Frames + PyTorch detections for checking the iPhone's Core ML detector against this pipeline.

    python scripts/export_parity_frames.py                       # 3 frames from each dev session
    python scripts/export_parity_frames.py --sessions 1 5 --per-session 6

Picks frames where detector v2 saw a ball (from outputs/debug960_det2/session_XX/debug.json), spread over
each session, saves them as PNG (lossless, so phone and PC decode identical pixels) and runs the PyTorch
model on exactly those pixels with the predict() settings the evaluation uses (imgsz 960, conf 0.15).
Writes <out>/frames/*.png and <out>/reference.json. The iOS app (scripts/sync_model.sh copies this in)
runs its own detector on the same PNGs and compares. Dev sessions only (sessions.csv); the frames show
people, so they stay local (exports/ is gitignored).
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "combined_v1" / "weights" / "best.pt"  # detector v2
DEV_SESSIONS = [1, 2, 3, 4, 5, 9, 10, 11]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_frames(debug: dict, n: int, min_conf: float):
    """n frame indices spread evenly over the frames with a ball detection >= min_conf."""
    fps = 1 / (debug["frames"][1]["t"] - debug["frames"][0]["t"])
    cands = [round(f["t"] * fps) for f in debug["frames"] if any(b[-1] >= min_conf for b in f["balls"])]
    if not cands:
        return []
    step = len(cands) / n
    return sorted({cands[int(step * (i + 0.5))] for i in range(n)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, nargs="+", default=DEV_SESSIONS)
    ap.add_argument("--per-session", type=int, default=3)
    ap.add_argument("--min-ball-conf", type=float, default=0.3, help="only frames where the ball was seen at least this well")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--footage", type=Path, default=ROOT / "data" / "own_footage")
    ap.add_argument("--debug-root", type=Path, default=ROOT / "outputs" / "debug960_det2")
    ap.add_argument("--out", type=Path, default=ROOT / "exports" / "parity")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("sbf", ROOT / "scripts" / "sample_box_frames.py")
    sbf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sbf)
    splits = sbf.session_splits(a.footage / "sessions.csv")
    bad = [s for s in a.sessions if splits.get(s) != "dev"]
    if bad:
        print(f"refused: sessions {bad} are not 'dev' in sessions.csv (test footage stays unseen)")
        return 1

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(a.weights))
    classes = [model.names[i] for i in sorted(model.names)]
    frames_dir = a.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    records = []
    for s in a.sessions:
        name = f"session_{s:02d}"
        debug = json.loads((a.debug_root / name / "debug.json").read_text())
        video = ROOT / debug["video"]
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"skip {name}: cannot open {video}")
            continue
        for idx in pick_frames(debug, a.per_session, a.min_ball_conf):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"skip {name} frame {idx}: read failed")
                continue
            file = f"{name}_{idx:05d}.png"
            cv2.imwrite(str(frames_dir / file), frame)
            frame = cv2.imread(str(frames_dir / file))  # exactly the pixels the phone will decode
            res = model.predict(frame, imgsz=a.imgsz, conf=a.conf, iou=a.iou, verbose=False)[0]
            dets = [{"label": classes[int(k)], "conf": round(float(c), 5), "box": [round(v, 2) for v in xyxy]}
                    for xyxy, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]
            h, w = frame.shape[:2]
            records.append({"file": file, "session": s, "frame": idx, "width": w, "height": h, "dets": dets})
            print(f"{file}: {len(dets)} detections ({', '.join(d['label'] for d in dets)})")
        cap.release()

    ref = {
        "weights": a.weights.relative_to(ROOT).as_posix() if a.weights.is_relative_to(ROOT) else str(a.weights),
        "weights_sha256": sha256(a.weights), "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou, "classes": classes,
        "date": date.today().isoformat(), "frames": records,
    }
    (a.out / "reference.json").write_text(json.dumps(ref, indent=1) + "\n")
    print(f"wrote {len(records)} frames to {frames_dir} and {a.out / 'reference.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
