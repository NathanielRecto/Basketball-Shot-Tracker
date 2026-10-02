"""Label shots by watching the video and pressing a key: no typing timestamps.

    python scripts/label_shots.py 1 4 5 8 9 12 13 16

The video plays in a window. Each time a shot reaches the hoop, press:
    M  = made        X  = missed        Z  = undo the last label
    SPACE = pause / play                S  = slow motion on / off
    LEFT / RIGHT arrow (or A / D) = back / forward 2 seconds
    , / .  = one frame back / forward (while paused)
    ENTER = save and go to the next video     ESC = save and quit
Labels are saved to data/eval_videos/labels/video_XX.csv after every key press, so nothing
is lost if the window is closed. Re-opening a video keeps its existing labels.

This tool never shows the tracker's predictions, so labels stay unbiased.
"""
import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data" / "eval_videos" / "labels"
VIDEOS_DIR = ROOT / "data" / "eval_videos"

KEYS_LEFT = {2424832, 65361, ord("a"), ord("A")}  # Windows / Linux arrow codes from waitKeyEx
KEYS_RIGHT = {2555904, 65363, ord("d"), ord("D")}


def read_labels(path: Path):
    """(header comment lines, [(time_s, outcome, note), ...])"""
    if not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8-sig")
    comments = [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
    body = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    rows = []
    for r in csv.DictReader(body):
        if (r.get("time_s") or "").strip() and (r.get("outcome") or "").strip():
            rows.append((float(r["time_s"]), r["outcome"].strip(), (r.get("note") or "").strip()))
    return comments, rows


def write_labels(path: Path, comments, rows) -> None:
    buf = io.StringIO()
    for c in comments:
        buf.write(c + "\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["time_s", "outcome", "note"])
    for t, outcome, note in sorted(rows):
        w.writerow([f"{t:.2f}", outcome, note])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")


def label_video(video: Path, label_path: Path, max_height: int) -> str:
    """Returns 'next' or 'quit'."""
    import cv2

    comments, rows = read_labels(label_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open {video}")
        return "next"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = f"{video.stem}  |  M made   X missed   Z undo   SPACE pause   S slow   arrows +/-2s   ENTER next   ESC quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    idx, paused, slow, flash, flash_until = 0, False, False, "", -1
    ok, frame = cap.read()
    if not ok:
        return "next"

    def seek(i):
        nonlocal idx, frame
        idx = max(0, min(n - 1, i))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        good, f = cap.read()
        if good:
            frame = f

    result = "quit"
    while True:
        t = idx / fps
        scale = min(1.0, max_height / frame.shape[0])
        disp = cv2.resize(frame, None, fx=scale, fy=scale)
        made = sum(r[1] == "made" for r in rows)
        status = f"{int(t // 60)}:{t % 60:05.2f}   made {made}  missed {len(rows) - made}"
        status += "   PAUSED" if paused else ("   SLOW" if slow else "")
        if idx >= n - 1:
            status += "   END - ENTER for next video"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(disp, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if idx <= flash_until:
            color = (60, 200, 60) if flash.startswith("MADE") else (60, 60, 230) if flash.startswith("MISSED") else (200, 200, 200)
            cv2.putText(disp, flash, (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        cv2.imshow(win, disp)

        delay = 30 if paused or idx >= n - 1 else max(1, int(1000 / fps * (3 if slow else 1)))
        k = cv2.waitKeyEx(delay)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with the X button
        if k == -1:
            if not paused and idx < n - 1:
                good, f = cap.read()
                if good:
                    frame, idx = f, idx + 1
                else:
                    idx = n - 1
            continue
        key = k & 0xFF
        if key in (ord("m"), ord("M"), ord("x"), ord("X")):
            outcome = "made" if key in (ord("m"), ord("M")) else "missed"
            rows.append((round(t, 2), outcome, ""))
            write_labels(label_path, comments, rows)
            flash, flash_until = f"{outcome.upper()} at {t:.1f}s", idx + int(fps)
        elif key in (ord("z"), ord("Z")) and rows:
            last = max(rows)
            rows.remove(last)
            write_labels(label_path, comments, rows)
            flash, flash_until = f"undid {last[1]} at {last[0]:.1f}s", idx + int(fps)
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("s"), ord("S")):
            slow = not slow
        elif k in KEYS_LEFT:
            seek(idx - int(2 * fps))
        elif k in KEYS_RIGHT:
            seek(idx + int(2 * fps))
        elif key == ord(",") and paused:
            seek(idx - 1)
        elif key == ord(".") and paused:
            seek(idx + 1)
        elif key == 13:  # ENTER
            result = "next"
            break
        elif key == 27:  # ESC
            result = "quit"
            break
    cap.release()
    cv2.destroyAllWindows()
    write_labels(label_path, comments, rows)
    made = sum(r[1] == "made" for r in rows)
    print(f"{video.stem}: saved {len(rows)} shots ({made} made, {len(rows) - made} missed) -> {label_path}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=int, nargs="+", help="video numbers, e.g. 1 4 5 8")
    ap.add_argument("--max-height", type=int, default=900, help="window height in pixels")
    a = ap.parse_args()
    for v in a.videos:
        name = f"video_{v:02d}"
        if label_video(VIDEOS_DIR / f"{name}.mp4", LABELS_DIR / f"{name}.csv", a.max_height) == "quit":
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
