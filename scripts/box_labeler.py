"""Check and correct ball / hoop boxes on frames, one image at a time. Everything stays on this PC.

    python scripts/box_labeler.py                      # data/own_boxes_v1 (from sample_box_frames.py)
    python scripts/box_labeler.py --dir data/own_boxes_v1 --width 1600

Each image opens with its current boxes (orange = ball, green = hoop). Fix them so that every
basketball and every hoop in the picture has a tight box, and nothing else does:

    drag (left mouse)        draw a new box of the current class
    click (left mouse)       select the smallest box under the cursor (drawn white)
    right click              delete the smallest box under the cursor
    1 / 2                    current class ball / hoop (also changes the selected box)
    DEL / BACKSPACE / X      delete the selected box
    Z                        undo the last change on this image
    H                        hide / show boxes, to see the ball clearly
    N / SPACE / RIGHT        done with this image: save, mark reviewed, next
    P / LEFT                 previous image
    F                        jump to the first image not reviewed yet
    Q / ESC                  save and quit (rerun later to continue)

What to box: every ball, including spare balls on the floor and balls in hands; box only the
visible part of a ball cut off by the frame edge or hidden behind the net, and skip a ball less
than about a third visible. The hoop being shot at as rim plus hanging net; far-wall hoops are not boxed.
Labels are saved after every change, and reviewed.txt records which images are done.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["ball", "hoop"]
COLORS = [(42, 116, 232), (80, 200, 60)]  # BGR: orange ball, green hoop
SELECTED = (255, 255, 255)
Box = List[float]  # [class, x1, y1, x2, y2] in image pixels

KEYS_LEFT = {2424832, 65361}
KEYS_RIGHT = {2555904, 65363}
KEYS_DELETE = {3014656, 65535, 8, ord("x"), ord("X")}


def read_yolo(path: Path, width: int, height: int) -> List[Box]:
    boxes: List[Box] = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), float(p[1]) * width, float(p[2]) * height, float(p[3]) * width, float(p[4]) * height
        boxes.append([c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return boxes


def write_yolo(path: Path, boxes: List[Box], width: int, height: int) -> None:
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            lines.append(f"{int(c)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def box_at(boxes: List[Box], x: float, y: float) -> Optional[int]:
    """Index of the smallest box containing (x, y): a ball inside the hoop box wins over the hoop."""
    hits = [i for i, b in enumerate(boxes) if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
    return min(hits, key=lambda i: (boxes[i][3] - boxes[i][1]) * (boxes[i][4] - boxes[i][2])) if hits else None


class Labeler:
    def __init__(self, folder: Path, width: int):
        self.dir = folder
        self.images = sorted((folder / "images").glob("*.jpg"))
        if not self.images:
            raise SystemExit(f"no images in {folder / 'images'} (run scripts/sample_box_frames.py first)")
        self.reviewed_path = folder / "reviewed.txt"
        self.reviewed = set(self.reviewed_path.read_text().split()) if self.reviewed_path.exists() else set()
        reasons = {}
        manifest = folder / "manifest.csv"
        if manifest.exists():
            import csv
            reasons = {r["file"]: (r["reason"], r["t"]) for r in csv.DictReader(open(manifest))}
        self.reasons = reasons
        self.view_w = width
        self.cls = 0
        self.idx = self.first_unreviewed()
        self.drag: Optional[Tuple[float, float]] = None
        self.cursor: Tuple[float, float] = (0.0, 0.0)
        self.hidden = False

    def first_unreviewed(self) -> int:
        return next((i for i, p in enumerate(self.images) if p.name not in self.reviewed), len(self.images) - 1)

    def load(self) -> None:
        import cv2
        self.img = cv2.imread(str(self.images[self.idx]))
        self.h, self.w = self.img.shape[:2]
        self.scale = self.view_w / self.w
        self.label_path = self.dir / "labels" / (self.images[self.idx].stem + ".txt")
        self.boxes = read_yolo(self.label_path, self.w, self.h)
        self.history: List[List[Box]] = []
        self.sel: Optional[int] = None

    def change(self, new: List[Box]) -> None:
        self.history.append([b[:] for b in self.boxes])
        self.boxes = new
        write_yolo(self.label_path, self.boxes, self.w, self.h)

    def on_mouse(self, event, x, y, flags, param) -> None:
        import cv2
        ix, iy = x / self.scale, y / self.scale
        self.cursor = (ix, iy)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            x0, y0 = self.drag
            self.drag = None
            if abs(ix - x0) * self.scale < 4 and abs(iy - y0) * self.scale < 4:
                self.sel = box_at(self.boxes, ix, iy)
                return
            self.change(self.boxes + [[self.cls, min(x0, ix), min(y0, iy), max(x0, ix), max(y0, iy)]])
            self.sel = len(self.boxes) - 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = box_at(self.boxes, ix, iy)
            if i is not None:
                self.change([b for k, b in enumerate(self.boxes) if k != i])
                self.sel = None

    def render(self):
        import cv2
        view = cv2.resize(self.img, (self.view_w, round(self.h * self.scale)), interpolation=cv2.INTER_AREA)
        s = self.scale
        if not self.hidden:
            for k, (c, x1, y1, x2, y2) in enumerate(self.boxes):
                color = SELECTED if k == self.sel else COLORS[int(c) % 2]
                cv2.rectangle(view, (round(x1 * s), round(y1 * s)), (round(x2 * s), round(y2 * s)), color, 2 if k != self.sel else 3)
        if self.drag is not None:
            (x0, y0), (x1, y1) = self.drag, self.cursor
            cv2.rectangle(view, (round(x0 * s), round(y0 * s)), (round(x1 * s), round(y1 * s)), COLORS[self.cls], 1)
        name = self.images[self.idx].name
        reason, t = self.reasons.get(name, ("", ""))
        mark = "reviewed" if name in self.reviewed else "new"
        top = f"{self.idx + 1}/{len(self.images)}  done {len(self.reviewed)}  {name}  t={t}s  [{reason}]  {mark}   class: {NAMES[self.cls]}"
        bottom = "drag=new box  click=select  right-click=delete  1 ball  2 hoop  X delete  Z undo  H hide  N next  P back  F first new  Q quit"
        for text, y in ((top, 24), (bottom, view.shape[0] - 12)):
            cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    def mark_reviewed(self) -> None:
        self.reviewed.add(self.images[self.idx].name)
        self.reviewed_path.write_text("\n".join(sorted(self.reviewed)) + "\n")

    def run(self) -> None:
        import cv2
        win = "box labeler"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        self.load()
        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKeyEx(30)
            if key == -1:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            ch = key & 0xFF if key < 256 else None
            if ch in (ord("q"), ord("Q"), 27):
                break
            if ch in (ord("1"), ord("2")):
                self.cls = ch - ord("1")
                if self.sel is not None:
                    new = [b[:] for b in self.boxes]
                    new[self.sel][0] = self.cls
                    self.change(new)
            elif key in KEYS_DELETE and self.sel is not None:
                self.change([b for k, b in enumerate(self.boxes) if k != self.sel])
                self.sel = None
            elif ch in (ord("z"), ord("Z")) and self.history:
                self.boxes = self.history.pop()
                write_yolo(self.label_path, self.boxes, self.w, self.h)
                self.sel = None
            elif ch in (ord("h"), ord("H")):
                self.hidden = not self.hidden
            elif ch in (ord("n"), ord("N"), ord(" ")) or key in KEYS_RIGHT:
                write_yolo(self.label_path, self.boxes, self.w, self.h)
                self.mark_reviewed()
                if self.idx < len(self.images) - 1:
                    self.idx += 1
                    self.load()
                else:
                    print("that was the last image")
            elif (ch in (ord("p"), ord("P")) or key in KEYS_LEFT) and self.idx > 0:
                self.idx -= 1
                self.load()
            elif ch in (ord("f"), ord("F")):
                self.idx = self.first_unreviewed()
                self.load()
        cv2.destroyAllWindows()
        print(f"{len(self.reviewed)}/{len(self.images)} images reviewed; labels in {self.dir / 'labels'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "data" / "own_boxes_v1"))
    ap.add_argument("--width", type=int, default=1600, help="window width in pixels (the image is scaled to it)")
    a = ap.parse_args()
    Labeler(Path(a.dir), a.width).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
