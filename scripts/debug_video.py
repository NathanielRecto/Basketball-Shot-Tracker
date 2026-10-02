"""Frame-by-frame diagnostics for one video: where the ball is detected, where tracking drops it,
and which arcs turned into shots.

    python scripts/debug_video.py data/eval_videos/video_01.mp4 --weights runs/detect/baseline/weights/best.pt \
        --hoop-json outputs/eval/video_01/hoop.json --out outputs/debug/video_01

Writes timeline.png (ball height over time vs. the rim, with detected shots marked) and
debug.json (every ball detection, including weak ones below --conf). Only use this on
development videos, never on the held-out test set.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shottracker.calibration import calibrate_hoop, parse_box  # noqa: E402
from shottracker.config import ShotConfig  # noqa: E402
from shottracker.detector import YoloDetector  # noqa: E402
from shottracker.geometry import Box  # noqa: E402
from shottracker.pipeline import ShotPipeline  # noqa: E402
from shottracker.types import Outcome  # noqa: E402
from shottracker.video import probe, read_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.15, help="confidence the tracker uses")
    ap.add_argument("--low-conf", type=float, default=0.1, help="also log weaker detections down to this")
    ap.add_argument("--hoop", default=None, help="x1,y1,x2,y2")
    ap.add_argument("--hoop-json", default=None, help="hoop.json written by run_eval_videos.py")
    ap.add_argument("--row-seconds", type=float, default=30.0, help="seconds per row in the timeline plot")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = YoloDetector(a.weights, conf=a.low_conf, imgsz=a.imgsz)
    if a.hoop:
        hoop = parse_box(a.hoop)
    elif a.hoop_json:
        hoop = Box(*json.loads(Path(a.hoop_json).read_text())["box"])
    else:
        hoop, _ = calibrate_hoop(a.video, det)
    if hoop is None:
        print("no hoop: pass --hoop or --hoop-json")
        return 1
    cfg = ShotConfig()
    rim_y = hoop.y1 + cfg.rim_line_frac * hoop.h
    arm_y = rim_y - cfg.arm_margin * hoop.h

    # The pipeline gets the already-computed detections passed in place of the frame.
    pipe = ShotPipeline(lambda dets: dets, hoop=hoop, shot_cfg=cfg)
    fps, w, h, n = probe(a.video)
    frames, events = [], []
    for idx, t, frame in read_frames(a.video, a.stride):
        all_dets = det(frame)
        balls = [d for d in all_dets if d.label == "ball"]
        fr = pipe.step(t, [d for d in all_dets if d.conf >= a.conf])
        frames.append({
            "t": round(t, 3),
            "balls": [[round(d.box.cx, 1), round(d.box.cy, 1), round(d.box.w, 1), round(d.box.h, 1), round(d.conf, 3)] for d in balls],
            "tracked": None if fr.ball is None else [round(fr.ball.x, 1), round(fr.ball.y, 1)],
        })
        events += [e.to_dict() for e in fr.events]
        if idx % 600 == 0:
            print(f"t={t:.0f}s / {n / fps:.0f}s", flush=True)

    (out / "debug.json").write_text(json.dumps({
        "video": a.video, "hoop": [hoop.x1, hoop.y1, hoop.x2, hoop.y2], "rim_y": rim_y, "arm_y": arm_y,
        "frame_size": [w, h], "conf": a.conf, "low_conf": a.low_conf, "frames": frames, "events": events,
    }))

    # ---- stats ---------------------------------------------------------------------------
    nf = len(frames)
    strong = sum(any(b[-1] >= a.conf for b in f["balls"]) for f in frames)
    weak_only = sum(bool(f["balls"]) and not any(b[-1] >= a.conf for b in f["balls"]) for f in frames)
    tracked = sum(f["tracked"] is not None for f in frames)
    above = sum(f["tracked"] is not None and f["tracked"][1] < rim_y for f in frames)
    print(f"\nframes analysed: {nf}")
    print(f"ball detected >= {a.conf}: {strong} ({100 * strong / nf:.0f}%)   only weaker ({a.low_conf}-{a.conf}): {weak_only} ({100 * weak_only / nf:.0f}%)")
    print(f"tracker had the ball: {tracked} ({100 * tracked / nf:.0f}%)   ball above rim line: {above} frames")
    print(f"shots detected: {len(events)} ({sum(e['outcome'] == 'made' for e in events)} made)")
    for e in events:
        print(f"  t_cross={e['t_cross']:6.1f}s {e['outcome']:6s} {e['reason']:12s} {e['method']}")

    # ---- timeline plot -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dur = frames[-1]["t"] if frames else 0
    rows = max(1, int(dur // a.row_seconds) + 1)
    fig, axes = plt.subplots(rows, 1, figsize=(16, 3.2 * rows), squeeze=False)
    for r in range(rows):
        ax = axes[r][0]
        t0, t1 = r * a.row_seconds, (r + 1) * a.row_seconds
        fs = [f for f in frames if t0 <= f["t"] < t1]
        weak = [(f["t"], b[1]) for f in fs for b in f["balls"] if b[-1] < a.conf]
        strong_pts = [(f["t"], b[1]) for f in fs for b in f["balls"] if b[-1] >= a.conf]
        trk = [(f["t"], f["tracked"][1] if f["tracked"] else float("nan")) for f in fs]
        if weak:
            ax.scatter(*zip(*weak), s=6, c="#bbbbbb", label=f"ball conf<{a.conf}")
        if strong_pts:
            ax.scatter(*zip(*strong_pts), s=8, c="#1f77b4", label=f"ball conf>={a.conf}")
        if trk:
            ax.plot(*zip(*trk), lw=1, c="#ff7f0e", label="tracked ball")
        ax.axhline(rim_y, c="red", lw=1, ls="--", label="rim line")
        ax.axhline(arm_y, c="purple", lw=0.8, ls=":", label="arm line")
        ax.axhline(0, c="black", lw=0.6)
        for e in events:
            if t0 <= e["t_cross"] < t1:
                ax.axvline(e["t_cross"], c="green" if e["outcome"] == Outcome.MADE.value else "red", lw=1.5, alpha=0.7)
                ax.text(e["t_cross"], 0, e["reason"], fontsize=7, rotation=90, va="bottom")
        ax.set_xlim(t0, t1)
        ax.set_ylim(h, -50)  # image y grows downward; top of frame at the top
        ax.set_ylabel("ball y (px)")
        if r == 0:
            ax.legend(loc="lower right", fontsize=7, ncol=6)
    axes[-1][0].set_xlabel("time (s)   vertical lines = detected shots (green made / red missed)")
    fig.tight_layout()
    fig.savefig(out / "timeline.png", dpi=90)
    print(f"\nwrote {out / 'timeline.png'} and {out / 'debug.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
