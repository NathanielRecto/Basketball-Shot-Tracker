"""Picture of every labelled shot: what the detector saw, what the tracker followed, and the call vs your label.

    python scripts/shot_report.py --sessions 6 7 8 --dumps outputs/report_test_gym2/dumps --out outputs/report_test_gym2
    python scripts/shot_report.py ... --src <copy of an older src/> # judge with a frozen version of the code

Needs debug.json per session (scripts/debug_video.py), the shot labels and the videos. For each shot it
writes a card (PNG): the area around the hoop with every ball detection in the 2 s before the hoop (dots,
blue = early, yellow = late), the path the tracker followed (orange), the hoop box (green), the rim line
(red) and where the ball crossed it (white x), under a header with your label and the system's call.
Mistakes also get a slow-motion clip (animated WebP). index.html shows a summary, the mistakes first,
then every correct shot. The frames show people: keep the output on this PC (outputs/ is not in git).
"""
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REASONS = {
    "through_hoop": "went through the hoop", "rattled_in": "hit the rim, then dropped in",
    "off_target": "missed the hoop", "rim_out": "went in, then popped back out",
    "rim_bounce": "bounced off the rim or board", "fell_past_rim": "looked centred but fell past the rim (not slowed by the net)",
}
ABORTS = {
    "not_parabolic": "its path did not look like one clean arc (the tracker probably jumped between objects)",
    "short_flight": "only a short piece of the flight was seen", "lost_too_few_points": "the ball was lost with too little of the arc seen",
    "lost_no_crossing": "the ball was lost and its arc did not reach the rim", "lost_bad_fit": "the ball was lost and its arc did not fit",
    "left_hoop_area": "the ball left the area around the hoop", "timeout": "the flight lasted too long",
    "too_far_from_hoop": "the arc ended too far from the hoop", "gap_too_long": "the ball was out of view too long near the rim",
}
WINDOW = (2.0, 0.7)  # seconds before / after the shot time shown on a card


def explain(log, t):
    """The shot logic's reason for a shot it did not count, from its decision log near ``t``."""
    near = [m for lt, m in log if abs(lt - t) <= 2.5 and m.startswith("abort:")]
    if not near:
        return "the ball was never followed up to the hoop (no shot attempt started)"
    key = near[-1][6:].split("(")[0]
    return ABORTS.get(key, key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, nargs="+", required=True)
    ap.add_argument("--dumps", required=True, help="folder with session_XX/debug.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=str(ROOT / "data" / "own_footage" / "labels"))
    ap.add_argument("--videos-dir", default=str(ROOT / "data" / "own_footage" / "raw"))
    ap.add_argument("--src", default=str(ROOT / "src"), help="shottracker package folder to judge with (e.g. a frozen copy)")
    ap.add_argument("--title", default="Shot report")
    ap.add_argument("--note", default="", help="a line under the title, e.g. which code version scored the run")
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--public", action="store_true", help="crop every picture tightly to the hoop (no people in frame), smaller "
                    "JPEG cards, and also write README.md (a page GitHub shows) next to index.html")
    ap.add_argument("--pred", default=None, help="folder with session_XX/summary.json from the scored run: the calls shown come "
                    "from there (a replay of debug.json can differ by a shot), the pictures from debug.json")
    a = ap.parse_args()
    sys.path.insert(0, a.src)

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from shottracker.evaluation import PredictedShot, load_labels, match_shots
    from shottracker.geometry import Box
    from shottracker.pipeline import ShotPipeline
    from shottracker.types import Detection
    from shottracker.video import find_videos

    out = Path(a.out)
    (out / "cards").mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
        bold = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 18)
    except OSError:
        font = bold = ImageFont.load_default()
    videos = find_videos(a.videos_dir)
    card_w = 360 if a.public else 640
    shots, summary = [], []

    for s in a.sessions:
        name = f"session_{s:02d}"
        dbg = json.loads((Path(a.dumps) / name / "debug.json").read_text())
        hoop = Box(*dbg["hoop"])
        pipe = ShotPipeline(lambda d: d, hoop=hoop)
        track = {}
        for f in dbg["frames"]:
            fr = pipe.step(f["t"], [Detection("ball", c, Box(x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                                    for x, y, w, h, c in f["balls"] if c >= dbg["conf"]])
            if fr.ball is not None:
                track[f["t"]] = (fr.ball.x, fr.ball.y)
        events, log = pipe.events, pipe.shots.log
        if a.pred:
            from types import SimpleNamespace
            official = json.loads((Path(a.pred) / name / "summary.json").read_text())["shots"]
            events = [SimpleNamespace(t_cross=e["t_cross"], outcome=SimpleNamespace(value=e["outcome"]), reason=e["reason"],
                                      cross_offset=e["cross_offset"]) for e in sorted(official, key=lambda e: e["t_cross"])]
        rim_y = pipe.shots._rim_y()  # from the code doing the judging (frozen or current)
        labels = load_labels(Path(a.labels) / f"{name}.csv")
        pairs, unmatched_l, unmatched_p = match_shots(labels, [PredictedShot(e.t_cross, e.outcome.value, e.reason) for e in events], 1.5)
        by_label = {li: pi for li, pi in pairs}
        cap = cv2.VideoCapture(str(videos[s]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        items = [("label", li) for li in range(len(labels))] + [("extra", pi) for pi in unmatched_p]
        n_ok = 0
        for kind, idx in items:
            if kind == "label":
                lab = labels[idx]
                ev = events[by_label[idx]] if idx in by_label else None
                t = ev.t_cross if ev else lab.t
                you = lab.outcome.upper()
                if ev is None:
                    status, system, why = "not found", "did not count this shot", explain(log, lab.t)
                else:
                    system = f"{ev.outcome.value.upper()}: {REASONS.get(ev.reason, ev.reason)}"
                    status = "correct" if ev.outcome.value == lab.outcome else "wrong call"
                    why = ""
                stype = lab.shot_type or "-"
            else:
                ev = events[idx]
                t, you, stype = ev.t_cross, "no shot", "-"
                status, system, why = "extra", f"counted a {ev.outcome.value.upper()} shot here", ""
            n_ok += status == "correct"
            lo, hi = t - WINDOW[0], t + WINDOW[1]
            dets = [(f["t"], b[0], b[1], b[2]) for f in dbg["frames"] if lo <= f["t"] <= hi for b in f["balls"] if b[4] >= dbg["conf"]]
            path = [(tt, *xy) for tt, xy in sorted(track.items()) if lo <= tt <= hi]
            xs = [p[1] for p in path] + [hoop.x1, hoop.x2]
            ys = [p[2] for p in path] + [hoop.y1, hoop.y2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t - 0.05) * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]
            pad = 1.2 * hoop.w
            x0, x1_ = max(0, int(min(xs) - pad)), min(W, int(max(xs) + pad))
            y0, y1_ = max(0, int(min(ys) - pad)), min(H, int(max(ys) + pad))
            if a.public:  # the hoop and just around it: the rim, the net and the ball's last stretch, no players
                x0, x1_ = max(0, int(hoop.x1 - 1.4 * hoop.w)), min(W, int(hoop.x2 + 1.4 * hoop.w))
                y0, y1_ = max(0, int(hoop.y1 - 1.3 * hoop.w)), min(H, int(hoop.y2 + 1.6 * hoop.w))
            elif x1_ - x0 < 480:
                c = (x0 + x1_) // 2
                x0, x1_ = max(0, c - 240), min(W, c + 240)
            if y1_ - y0 < 300 and not a.public:
                c = (y0 + y1_) // 2
                y0, y1_ = max(0, c - 150), min(H, c + 150)

            def overlay(img, upto):
                cv2.rectangle(img, (int(hoop.x1), int(hoop.y1)), (int(hoop.x2), int(hoop.y2)), (60, 200, 60), 2)
                cv2.line(img, (int(hoop.x1 - 0.5 * hoop.w), int(rim_y)), (int(hoop.x2 + 0.5 * hoop.w), int(rim_y)), (60, 60, 230), 1)
                for tt, x, y, d in dets:
                    if tt <= upto:
                        k = (tt - lo) / (hi - lo)
                        r = max(2, int(d / 14)) if a.public else max(3, int(d / 6))  # small dots on the tight hoop crop
                        cv2.circle(img, (int(x), int(y)), r, (int(255 * (1 - k)), int(200 * k + 40), int(230 * k)), -1)
                pts = np.array([(int(x), int(y)) for tt, x, y in path if tt <= upto], np.int32)
                if len(pts) > 1:
                    cv2.polylines(img, [pts], False, (40, 140, 255), 2)
                if ev is not None and upto >= t:
                    cx = hoop.cx + ev.cross_offset * hoop.w / 2
                    cv2.drawMarker(img, (int(cx), int(rim_y)), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
                return img

            color = {"correct": (34, 160, 90), "wrong call": (210, 60, 60), "not found": (210, 120, 30), "extra": (150, 80, 200)}[status]
            mm, ss = divmod(int(t), 60)
            lines = [(f"Session {s}  |  {stype}  |  {mm}:{ss:02d}  |  {status.upper()}", bold, color),
                     (f"You: {you}     System: {system}", font, (30, 30, 30))]
            if why:
                lines.append((f"Why: {why}", font, (90, 90, 90)))

            def wrapped(items, width=card_w - 28):
                out_lines = []
                for text, f_, c_ in items:
                    words, cur = text.split(" "), ""
                    for w_ in words:
                        trial = (cur + " " + w_).strip()
                        if f_.getlength(trial) <= width or not cur:
                            cur = trial
                        else:
                            out_lines.append((cur, f_, c_))
                            cur = "   " + w_
                    out_lines.append((cur, f_, c_))
                return out_lines

            lines = wrapped(lines)

            def card(img_bgr):
                crop = cv2.cvtColor(img_bgr[y0:y1_, x0:x1_], cv2.COLOR_BGR2RGB)
                scale = card_w / crop.shape[1]
                crop = cv2.resize(crop, (card_w, int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                head = 14 + 26 * len(lines)
                im = Image.new("RGB", (card_w, head + crop.shape[0]), (250, 250, 248))
                im.paste(Image.fromarray(crop), (0, head))
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, 6, head], fill=color)
                for i, (text, f_, c_) in enumerate(lines):
                    dr.text((16, 8 + 26 * i), text, font=f_, fill=c_)
                return im

            stem = f"s{s:02d}_{int(t * 100):06d}"
            card_file = f"cards/{stem}.jpg" if a.public else f"cards/{stem}.png"
            card(overlay(frame.copy(), hi)).save(out / card_file, **({"quality": 82} if a.public else {}))
            clip = None
            if status != "correct" and not a.no_clips:
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t - 1.6) * fps)))
                tt = t - 1.6
                while tt <= t + 1.0:
                    ok, fr_ = cap.read()
                    if not ok:
                        break
                    if int(round(tt * fps)) % 2 == 0:
                        frames.append(card(overlay(fr_, tt)))
                    tt += 1 / fps
                if frames:
                    clip = f"clips/{stem}.webp"
                    frames[0].save(out / clip, save_all=True, append_images=frames[1:], duration=66, loop=0, quality=70)
            shots.append(dict(session=s, type=stype, t=round(t, 2), status=status, card=card_file, clip=clip))
        cap.release()
        summary.append((s, len(labels), n_ok, len(unmatched_l), len(unmatched_p)))
        print(f"{name}: {len(labels)} labelled, {n_ok} correct, {len(unmatched_l)} not found, {len(unmatched_p)} extra")

    # ---- page --------------------------------------------------------------------------------------
    def block(sh):
        media = f'<img src="{sh["card"]}" loading="lazy" alt="shot card">'
        if sh["clip"]:
            media += f'<details><summary>Slow-motion clip</summary><img src="{sh["clip"]}" loading="lazy" alt="slow-motion clip"></details>'
        return f"<figure>{media}</figure>"

    types = sorted({sh["type"] for sh in shots if sh["status"] != "extra"})
    rows = "".join(f"<tr><td>Session {s}</td><td>{n}</td><td>{ok}</td><td>{n - ok - nf}</td><td>{nf}</td><td>{ex}</td>"
                   f"<td><b>{100 * ok / n:.1f}%</b></td></tr>" for s, n, ok, nf, ex in summary)
    trows = ""
    for ty in types:
        g = [sh for sh in shots if sh["type"] == ty and sh["status"] != "extra"]
        ok = sum(sh["status"] == "correct" for sh in g)
        trows += f"<tr><td>{html.escape(ty)}</td><td>{len(g)}</td><td>{ok}</td><td><b>{100 * ok / len(g):.1f}%</b></td></tr>"
    mistakes = [sh for sh in shots if sh["status"] != "correct"]
    sections = "".join(f'<h3>Session {s}</h3><div class="grid">' + "".join(block(sh) for sh in shots if sh["session"] == s and sh["status"] == "correct")
                       + "</div>" for s in a.sessions)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(a.title)}</title><style>
:root {{ --bg:#f6f6f3; --fg:#1d1d1b; --muted:#5d5d58; --card:#ffffff; --line:#dcdcd5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17181a; --fg:#ecece8; --muted:#a9a9a2; --card:#222326; --line:#35363a; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 "Segoe UI", system-ui, sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1360px; margin:0 auto; }} h1 {{ margin:0 0 4px; }} p {{ color:var(--muted); max-width:860px; }}
table {{ border-collapse:collapse; margin:8px 24px 16px 0; background:var(--card); display:inline-table; vertical-align:top; }}
td, th {{ border:1px solid var(--line); padding:5px 12px; text-align:right; }} th:first-child, td:first-child {{ text-align:left; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:14px; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
figure img {{ width:100%; display:block; }} summary {{ cursor:pointer; padding:6px 10px; color:var(--muted); }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:5px; margin:0 5px 0 12px; }}
</style></head><body><main>
<h1>{html.escape(a.title)}</h1>
<p>{html.escape(a.note)}</p>
<p>Every labelled shot, as the system saw it. On each picture:
<span class="dot" style="background:#3a6df0"></span>ball detections, early
<span class="dot" style="background:#e6d23c"></span>late
<span class="dot" style="background:#ff8c28"></span>path the tracker followed
<span class="dot" style="background:#3cc83c"></span>hoop box
<span class="dot" style="background:#e63c3c"></span>rim line; white x = where it crossed the rim line.</p>
<table><tr><th>Session</th><th>Shots</th><th>Correct</th><th>Wrong call</th><th>Not found</th><th>Extra</th><th>Accuracy</th></tr>{rows}</table>
<table><tr><th>Shot type</th><th>Shots</th><th>Correct</th><th>Accuracy</th></tr>{trows}</table>
<h2>Mistakes ({len(mistakes)})</h2><div class="grid">{"".join(block(sh) for sh in mistakes)}</div>
<h2>Correct calls ({len(shots) - len(mistakes)})</h2>{sections}
</main></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    if a.public:
        def md_cell(sh):
            cell = f'<img src="{sh["card"]}" width="340" alt="shot card">'
            if sh["clip"]:
                cell += f'<br><details><summary>slow motion</summary><img src="{sh["clip"]}" width="340" alt="slow-motion clip"></details>'
            return cell

        def md_grid(items, cols=3):
            rows_ = ["<table>"]
            for i in range(0, len(items), cols):
                rows_.append("<tr>" + "".join(f'<td valign="top">{md_cell(sh)}</td>' for sh in items[i:i + cols]) + "</tr>")
            return "\n".join(rows_ + ["</table>"])

        lines_md = [f"# {a.title}", "", *([a.note, ""] if a.note else []),
                    "Every labelled shot, as the system saw it, cropped to the hoop so no players are shown. On each picture:",
                    "dots = ball detections (blue early, yellow late), orange = the path the tracker followed, green = hoop box,",
                    "red = rim line, white x = where the ball crossed it. Header: your label against the system's call.", "",
                    "| Session | Shots | Correct | Wrong call | Not found | Extra | Accuracy |", "|---|---|---|---|---|---|---|"]
        lines_md += [f"| Session {s_} | {n} | {ok} | {n - ok - nf} | {nf} | {ex} | **{100 * ok / n:.1f}%** |" for s_, n, ok, nf, ex in summary]
        lines_md += ["", "| Shot type | Shots | Correct | Accuracy |", "|---|---|---|---|"]
        for ty in types:
            g = [sh for sh in shots if sh["type"] == ty and sh["status"] != "extra"]
            ok = sum(sh["status"] == "correct" for sh in g)
            lines_md.append(f"| {ty} | {len(g)} | {ok} | **{100 * ok / len(g):.1f}%** |")
        lines_md += ["", f"## Mistakes ({len(mistakes)})", "", md_grid(mistakes), "", f"## Correct calls ({len(shots) - len(mistakes)})", ""]
        for s_ in a.sessions:
            g = [sh for sh in shots if sh["session"] == s_ and sh["status"] == "correct"]
            lines_md += [f"<details><summary>Session {s_} ({len(g)} shots)</summary>", "", md_grid(g), "", "</details>", ""]
        lines_md += ["Made with `scripts/shot_report.py --public`. Full-frame cards (whole court and players) can be made locally",
                     "without `--public`."]
        (out / "README.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    print(f"wrote {out / 'index.html'} ({len(shots)} shots, {len(mistakes)} mistakes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
