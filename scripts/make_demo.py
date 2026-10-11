"""Real-footage demo for the README: replay a session's saved detections through the current tracker and shot
judge, and draw what they see and decide on the video.

    python scripts/make_demo.py --session 9                        # list the calls to choose from
    python scripts/make_demo.py --session 9 --shots 19 20 21 --out docs/images/demo_real.webp

--shots are calls of the replay, numbered from 1. Each one plays at real speed from just before the release to just
after the call; the time between shots is cut. Writes animated WebP (about 3 MB for 14 s; a GIF of real video is
7 times larger) or GIF, by the --out extension. Needs debug.json from scripts/debug_video.py with every frame
(--stride 1) and the video. The frames show whoever was on court: only publish footage of people who agreed to it.
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from replay_debug import _balls  # noqa: E402
from shottracker.geometry import Box  # noqa: E402
from shottracker.pipeline import ShotPipeline  # noqa: E402
from shottracker.types import Detection, Outcome  # noqa: E402
from shottracker.video import read_frames  # noqa: E402

# The app's banner wording (iOS LiveView.reasonText).
REASONS = {
    "through_hoop": "through the hoop", "rattled_in": "rattled in", "off_target": "off target",
    "rim_out": "in and out", "rim_bounce": "off the rim", "fell_past_rim": "fell past the rim",
}
GREEN, RED, ORANGE, HOOP = (52, 199, 89), (255, 59, 48), (255, 149, 0), (120, 230, 120)
TRAIL_S = 0.6  # seconds of tracked ball path drawn behind the ball


def replay(debug: dict, conf: float):
    """Per saved frame: (t, tracked ball or None, shot events decided on that frame), as replay_debug.replay runs it."""
    pipe = ShotPipeline(lambda dets: dets, hoop=Box(*debug["hoop"]))
    out = []
    for f in debug["frames"]:
        dets = [Detection("ball", c, Box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)) for x, y, w, h, c in _balls(f) if c >= conf]
        fr = pipe.step(f["t"], dets)
        out.append((f["t"], fr.ball, fr.events))
    return out


def font(size: int, bold: bool = False):
    names = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--shots", type=int, nargs="*", help="calls to show, numbered from 1; omit to list them")
    ap.add_argument("--dumps", default=str(ROOT / "outputs" / "debug960_det2"), help="folder with session_XX/debug.json")
    ap.add_argument("--out", default=str(ROOT / "docs" / "images" / "demo_real.webp"))
    ap.add_argument("--quality", type=int, default=80, help="WebP quality")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--before", type=float, default=1.0, help="seconds shown before each release")
    ap.add_argument("--after", type=float, default=1.8, help="seconds shown after each call")
    a = ap.parse_args()

    debug = json.loads((Path(a.dumps) / f"session_{a.session:02d}" / "debug.json").read_text())
    if debug.get("stride", 1) != 1:
        sys.exit("debug.json must hold every frame (scripts/debug_video.py --stride 1)")
    frames = replay(debug, a.conf)
    calls = [(t, ev) for t, _, evs in frames for ev in evs]
    if not a.shots:
        for n, (t, ev) in enumerate(calls, 1):
            print(f"{n:3d}  released {ev.t_release:6.1f}s  called {t:6.1f}s  {ev.outcome.value:<6} {ev.reason}")
        return 0

    picked = [calls[n - 1] for n in a.shots]
    segments = [(ev.t_release - a.before, t + a.after) for t, ev in picked]
    chosen = {id(ev) for _, ev in picked}
    video = Path(debug["video"])
    video = video if video.is_absolute() else ROOT / video
    hoop = Box(*debug["hoop"])
    W, H = debug["frame_size"]
    s = a.width / W
    size = (a.width, round(H * s))
    step = max(1, round(1 / ((frames[-1][0] - frames[0][0]) / (len(frames) - 1)) / a.fps))
    f_score, f_call, f_reason = font(round(a.width / 30), True), font(round(a.width / 19), True), font(round(a.width / 38))

    images, trail = [], deque()
    made = attempts = 0
    banner = shown = seg_now = None
    for idx, t, frame in read_frames(str(video), 1):
        if t > segments[-1][1]:
            break
        ft, ball, events = frames[idx]
        if abs(ft - t) > 0.002:
            sys.exit(f"frame {idx}: video time {t:.3f}s but debug.json has {ft:.3f}s")
        if ball is not None:
            trail.append((t, ball.x * s, ball.y * s))
        while trail and t - trail[0][0] > TRAIL_S:
            trail.popleft()
        for ev in events:
            if id(ev) in chosen:
                made += ev.outcome is Outcome.MADE
                attempts += 1
                banner = ev
        seg = next((i for i, (t0, t1) in enumerate(segments) if t0 <= t <= t1), None)
        if seg is None or idx % step:
            continue
        if seg != seg_now:
            seg_now, banner, shown = seg, None, None
        shown = banner or shown

        img = Image.fromarray(cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([hoop.x1 * s, hoop.y1 * s, hoop.x2 * s, hoop.y2 * s], outline=HOOP, width=2)
        if shown is not None:  # the arc the judge used, in the call's colour
            pts = [(x * s, y * s) for _, x, y in shown.trajectory]
            d.line(pts, fill=GREEN if shown.outcome is Outcome.MADE else RED, width=3, joint="curve")
        elif len(trail) > 1:
            d.line([(x, y) for _, x, y in trail], fill=ORANGE, width=3, joint="curve")
        if ball is not None:
            r = max(6, ball.diameter * s / 2 + 2)
            d.ellipse([ball.x * s - r, ball.y * s - r, ball.x * s + r, ball.y * s + r], outline=ORANGE, width=2)

        pad = a.width // 60
        score = f"{made} / {attempts}"
        box = d.textbbox((0, 0), score, font=f_score)
        d.rounded_rectangle([pad, pad, pad * 3 + box[2], pad * 2.6 + box[3]], radius=pad, fill=(0, 0, 0, 150))
        d.text((pad * 2, pad * 1.6), score, font=f_score, fill="white")
        if shown is not None:
            ok = shown.outcome is Outcome.MADE
            title, why = ("MADE" if ok else "MISSED"), REASONS.get(shown.reason, shown.reason)
            tb, wb = d.textbbox((0, 0), title, font=f_call), d.textbbox((0, 0), why, font=f_reason)
            bw, bh = max(tb[2], wb[2]) + pad * 3, tb[3] + wb[3] + pad * 2.5
            x0, y1 = pad * 1.5, size[1] - pad * 1.5
            d.rounded_rectangle([x0, y1 - bh, x0 + bw, y1], radius=pad * 1.2, fill=(*(GREEN if ok else RED), 225))
            d.text((x0 + pad * 1.5, y1 - bh + pad * 0.6), title, font=f_call, fill="white")
            d.text((x0 + pad * 1.5, y1 - wb[3] - pad * 1.1), why, font=f_reason, fill="white")
        images.append(img)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = {"quality": a.quality, "method": 6} if out.suffix.lower() == ".webp" else {"optimize": True}
    images[0].save(out, save_all=True, append_images=images[1:], duration=round(1000 / a.fps), loop=0, **fmt)
    print(f"wrote {out}: {len(images)} frames, {len(images) / a.fps:.1f} s, {size[0]}x{size[1]}, "
          f"{out.stat().st_size / 1e6:.1f} MB; calls shown {made} made / {attempts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
