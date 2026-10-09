"""Pick frames from own DEV sessions for box labelling, with the current detector's boxes pre-drawn.

    python scripts/sample_box_frames.py --sessions 1 2 3 4

For each session, about --per-session frames are chosen, weighted towards what the detector finds
hard (detection dumps from scripts/debug_video.py say where it did and did not see a ball):

    top        a ball touching the top edge of the frame (high arcs leaving the picture)
    missed     inside a shot window, but no ball detected (the ball in flight it did not see)
    flight     inside a shot window: from 2 s before the labelled shot time to 0.6 s after
    extra      away from any shot, two or more "balls", or one the tracker did not follow
               (heads, hands, spare balls)
    random     anywhere, so ordinary frames and empty courts are covered too

Chosen frames are at least --min-gap frames apart, so near-identical neighbours are not labelled
twice. Writes <out>/images/*.jpg, <out>/labels/*.txt (YOLO: class cx cy w h, 0-1), <out>/classes.txt
and <out>/manifest.csv. Pre-labels are ball detections >= --min-conf plus the session's hoop box;
an existing label file is never overwritten, so corrections are safe if this is rerun.

Correct the boxes with scripts/box_labeler.py. Sessions whose split is "test" in sessions.csv are
refused: test footage must never become training data.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CLASSES = ["ball", "hoop", "rim_only"]  # same order as data/combined_v1/data.yaml
QUOTAS = [("top", 0.10), ("missed", 0.15), ("flight", 0.30), ("extra", 0.20), ("random", 0.25)]


def categorize(frames: Sequence[dict], shot_times: Sequence[float], fps: float, conf: float = 0.15,
               before_s: float = 2.0, after_s: float = 0.6, top_px: float = 40.0) -> Dict[str, List[int]]:
    """Frame indices per category (a frame can be in several)."""
    cats: Dict[str, List[int]] = {k: [] for k, _ in QUOTAS}
    for f in frames:
        i = round(f["t"] * fps)
        balls = [b for b in f["balls"] if b[4] >= conf]
        in_shot = any(t - before_s <= f["t"] <= t + after_s for t in shot_times)
        tracked = f.get("tracked")
        if any(b[1] - b[3] / 2 <= top_px for b in balls):
            cats["top"].append(i)
        if in_shot:
            cats["flight"].append(i)
            if not balls:
                cats["missed"].append(i)
        elif len(balls) >= 2 or (balls and tracked is None):
            cats["extra"].append(i)
        cats["random"].append(i)
    return cats


def pick(cats: Dict[str, List[int]], n: int, min_gap: int, seed: int) -> List[Tuple[int, str]]:
    """Up to ``n`` (frame index, category) pairs, filled category by category, ``min_gap`` frames apart."""
    rng = random.Random(seed)
    chosen: List[Tuple[int, str]] = []
    for name, share in QUOTAS:
        want = n - len(chosen) if name == QUOTAS[-1][0] else round(share * n)
        pool = list(cats.get(name, []))
        rng.shuffle(pool)
        got = 0
        for i in pool:
            if got >= want:
                break
            if all(abs(i - j) >= min_gap for j, _ in chosen):
                chosen.append((i, name))
                got += 1
    return sorted(chosen)


def yolo_lines(boxes: Sequence[Tuple[int, float, float, float, float]], width: int, height: int) -> List[str]:
    """(class, x1, y1, x2, y2) in pixels -> YOLO text lines, clipped to the image."""
    out = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append(f"{c} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return out


def session_splits(path: Path) -> Dict[int, str]:
    rows = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    return {int(r["session"]): r["split"].strip() for r in csv.DictReader(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, nargs="+", required=True)
    ap.add_argument("--per-session", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "data" / "own_boxes_v1"))
    ap.add_argument("--footage", default=str(ROOT / "data" / "own_footage"), help="holds raw/, labels/, sessions.csv, hoop_overrides.json")
    ap.add_argument("--debug-root", default=str(ROOT / "outputs" / "debug960_det2"))
    ap.add_argument("--min-conf", type=float, default=0.3, help="ball detections at or above this become pre-labels")
    ap.add_argument("--min-gap", type=int, default=8, help="frames between any two picked frames")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import cv2

    from shottracker.evaluation import load_labels
    from shottracker.video import find_videos, probe, read_frames

    footage = Path(a.footage)
    splits = session_splits(footage / "sessions.csv")
    bad = [s for s in a.sessions if splits.get(s) != "dev"]
    if bad:
        print(f"refused: sessions {bad} are not 'dev' in sessions.csv (test footage must never be trained on)")
        return 2
    hoops = json.loads((footage / "hoop_overrides.json").read_text())
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    manifest = out / "manifest.csv"
    rows = list(csv.DictReader(open(manifest))) if manifest.exists() else []
    done = {r["file"] for r in rows}

    for s in a.sessions:
        name = f"session_{s:02d}"
        video = find_videos(footage / "raw")[s]  # .mov or .mp4
        fps, width, height, _ = probe(str(video))
        dbg = json.loads((Path(a.debug_root) / name / "debug.json").read_text())
        shots = [lab.t for lab in load_labels(footage / "labels" / f"{name}.csv")]
        picked = dict(pick(categorize(dbg["frames"], shots, fps), a.per_session, a.min_gap, a.seed + s))
        by_index = {round(f["t"] * fps): f for f in dbg["frames"]}
        hx1, hy1, hx2, hy2 = hoops[name]
        counts: Dict[str, int] = {}
        for i, t, frame in read_frames(str(video)):
            if i not in picked:
                continue
            stem = f"{name}_f{i:06d}"
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lab = out / "labels" / f"{stem}.txt"
            if not lab.exists():
                f = by_index.get(i, {"balls": []})
                boxes = [(0, x - w / 2, y - h / 2, x + w / 2, y + h / 2) for x, y, w, h, c in f["balls"] if c >= a.min_conf]
                boxes.append((1, hx1, hy1, hx2, hy2))
                lab.write_text("\n".join(yolo_lines(boxes, width, height)) + "\n")
            if f"{stem}.jpg" not in done:
                rows.append({"file": f"{stem}.jpg", "session": s, "frame": i, "t": f"{t:.3f}", "reason": picked[i]})
                done.add(f"{stem}.jpg")
            counts[picked[i]] = counts.get(picked[i], 0) + 1
        print(f"{name}: {sum(counts.values())} frames " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    with open(manifest, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "session", "frame", "t", "reason"])
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["file"]))
    print(f"wrote {out} ({len(rows)} frames in manifest.csv); correct them with scripts/box_labeler.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
