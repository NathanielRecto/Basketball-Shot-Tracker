"""Command line entry points.

    python -m shottracker demo --out outputs/demo      # synthetic shots, no model needed
    python -m shottracker run clip.mp4 --weights best.pt --pose --out outputs/clip
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import cv2
import numpy as np

from .pipeline import ShotPipeline
from .render import Renderer
from .sim import BALL_DIAM, DEFAULT_HOOP, ScriptedDetector, ShotSpec, detections_from_stream, simulate_stream
from .video import open_writer, probe, read_frames


def _write_outputs(pipe: ShotPipeline, out_dir: str, quiet: bool = False) -> None:
    summary = pipe.summary()
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        f.write(summary.to_json(indent=2))
    if quiet:
        return
    pct = "n/a" if summary.fg_pct is None else f"{summary.fg_pct}%"
    print(f"\nAttempts {summary.attempts} | made {summary.made} | missed {summary.missed} | FG {pct}")
    print(f"Avg release {summary.avg_release_angle_deg} deg | avg entry {summary.avg_entry_angle_deg} deg")
    for s in summary.shots:
        print(f"  #{s['index']} {s['outcome']:6s} ({s['reason']}, {s['method']}) offset={s['cross_offset']:+.2f}"
              f" release={s['release_angle_deg']} entry={s['entry_angle_deg']} {s['biomechanics']}")


def analyse_video(video: str, detector, pose, out_dir: str, stride: int = 1, annotate: bool = True,
                  quiet: bool = False, hoop=None) -> ShotPipeline:
    """Run the pipeline over one video; writes summary.json (and annotated.mp4 when ``annotate``).

    ``hoop``: optional fixed hoop Box (from calibration or marked by hand).
    """
    os.makedirs(out_dir, exist_ok=True)
    fps, w, h, n = probe(video)
    pipe = ShotPipeline(detector, pose, hoop=hoop)
    rend = Renderer() if annotate else None
    writer = open_writer(os.path.join(out_dir, "annotated.mp4"), fps / stride, (w, h)) if annotate else None
    for idx, t, frame in read_frames(video, stride):
        fr = pipe.step(t, frame)
        if writer is not None:
            writer.write(rend.draw(frame, fr))
        if not quiet and idx % 300 == 0:
            print(f"frame {idx}/{n}", end="\r")
    if writer is not None:
        writer.release()
    _write_outputs(pipe, out_dir, quiet)
    return pipe


def cmd_run(a: argparse.Namespace) -> int:
    from .detector import YoloDetector

    pose = None
    if a.pose:
        from .pose import MediaPipePose

        pose = MediaPipePose(a.pose_model)
    detector = YoloDetector(a.weights, conf=a.conf, imgsz=a.imgsz, device=a.device)
    hoop = None
    if a.hoop == "auto":
        from .calibration import calibrate_hoop

        hoop, n_cands = calibrate_hoop(a.video, detector)
        print(f"hoop calibration: {hoop if hoop else 'not found'} ({n_cands} candidate detections)")
    elif a.hoop:
        from .calibration import parse_box

        hoop = parse_box(a.hoop)
    analyse_video(a.video, detector, pose, a.out, a.stride, hoop=hoop)
    return 0


def _demo_background(hoop, size=(1280, 720)) -> np.ndarray:
    bg = np.full((size[1], size[0], 3), (70, 60, 55), np.uint8)
    cv2.rectangle(bg, (0, 600), (size[0], size[1]), (60, 105, 150), -1)  # floor
    cv2.rectangle(bg, (int(hoop.x2) + 5, int(hoop.y1) - 70), (int(hoop.x2) + 15, int(hoop.y2) + 20), (220, 220, 220), -1)
    cv2.line(bg, (int(hoop.x1), int(hoop.y1 + 20)), (int(hoop.x2), int(hoop.y1 + 20)), (40, 100, 230), 4)
    return bg


def cmd_demo(a: argparse.Namespace) -> int:
    """Run the full pipeline on simulated shots (scripted detections + real renderer)."""
    os.makedirs(a.out, exist_ok=True)
    hoop = DEFAULT_HOOP
    shots: List[ShotSpec] = [
        ShotSpec(offset=0.05),
        ShotSpec(offset=1.4, release=(300, 400)),
        ShotSpec(offset=-0.2, release=(260, 360), occlude=True, apex_above_rim=190),
        ShotSpec(offset=0.1, behaviour="rim_out", release=(280, 390)),
        ShotSpec(offset=-0.3, behaviour="rim_bounce", release=(320, 380)),
    ]
    stream = simulate_stream(hoop, shots, noise_px=1.5, dropout=0.05, seed=3)
    pipe = ShotPipeline(ScriptedDetector(detections_from_stream(stream, hoop)))
    rend = Renderer()
    bg = _demo_background(hoop)
    writer = open_writer(os.path.join(a.out, "annotated.mp4"), 30.0, (1280, 720))
    first_event_frame = None
    for i, (t, obs) in enumerate(stream):
        frame = bg.copy()
        if obs is not None:
            cv2.circle(frame, (int(obs.x), int(obs.y)), int(BALL_DIAM / 2), (40, 140, 240), -1)
        fr = pipe.step(t, frame)
        out = rend.draw(frame, fr)
        writer.write(out)
        if fr.events and first_event_frame is None:
            first_event_frame = i
            cv2.imwrite(os.path.join(a.out, "preview.png"), out)
    writer.release()
    _write_outputs(pipe, a.out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="shottracker", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse a video with a trained detector")
    r.add_argument("video")
    r.add_argument("--weights", required=True, help="YOLO weights trained on ball + hoop")
    r.add_argument("--out", default="outputs/run")
    r.add_argument("--pose", action="store_true", help="also estimate elbow/knee angles with MediaPipe")
    r.add_argument("--pose-model", default=None, help="MediaPipe pose_landmarker .task file (default: models/pose_landmarker_full.task)")
    r.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    r.add_argument("--hoop", default=None,
                   help="fixed camera: 'auto' to locate the hoop once for the whole video, or x1,y1,x2,y2 in pixels")
    r.add_argument("--conf", type=float, default=0.15)
    r.add_argument("--imgsz", type=int, default=960)
    r.add_argument("--device", default=None)
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("demo", help="run on synthetic shots; needs no model")
    d.add_argument("--out", default="outputs/demo")
    d.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
