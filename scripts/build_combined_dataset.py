"""Build one YOLO training set from several labelled sources, with one shared class list.

    python scripts/build_combined_dataset.py --hooper-zip "G:/Downloads/Basketball Detection v6.v1i.yolov8.zip"

Classes of the combined set:
    0 ball   1 hoop (rim + hanging net, the convention of the hotshot labels and of ShotConfig.rim_line_frac)
    2 rim_only (rim only: the Hooper "Basketball Detection v6" labels; kept as its own class so the two
              box conventions never mix; the pipeline ignores it)

Sources
    hotshot   data/hotshot (Roboflow export, CC BY 4.0): its own train / valid split is kept.
    hooper    "Basketball Detection v6" by Hooper (CC BY 4.0), read straight from the export zip:
              phone video of indoor gyms with many people on court. Every --hooper-stride-th frame is
              used; frames are numbered in order, so validation takes whole blocks of --block
              consecutive numbers (neighbouring frames never land on both sides). Its "holder" and
              "scorer" (people) boxes are dropped: people are background for ball / hoop / rim.

Writes <out>/{train,valid}/{images,labels}, <out>/data.yaml and <out>/sources.csv (where every
image came from). Never add frames from held-out test footage here.
"""
import argparse
import csv
import random
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["ball", "hoop", "rim_only"]  # "rim" would alias to hoop in detector.DEFAULT_ALIASES
HOTSHOT_MAP = {0: 0, 1: 1}  # basketball, basketball-hoop
HOOPER_MAP = {0: 0, 2: 2}  # ball, hoop (rim only); holder / scorer dropped


def remap(text: str, mapping: dict) -> str:
    out = []
    for row in text.splitlines():
        parts = row.split()
        if len(parts) >= 5 and int(parts[0]) in mapping:
            out.append(" ".join([str(mapping[int(parts[0])])] + parts[1:5]))
    return "\n".join(out) + ("\n" if out else "")


def add_hotshot(src: Path, out: Path, log) -> None:
    for split in ("train", "valid"):
        for img in sorted((src / split / "images").iterdir()):
            lab = src / split / "labels" / (img.stem + ".txt")
            name = "hotshot_" + img.name
            shutil.copy2(img, out / split / "images" / name)
            text = lab.read_text() if lab.is_file() else ""
            (out / split / "labels" / (Path(name).stem + ".txt")).write_text(remap(text, HOTSHOT_MAP))
            log.writerow(["hotshot", split, name, f"{split}/images/{img.name}"])


def add_hooper(zip_path: Path, out: Path, stride: int, block: int, val_frac: float, seed: int, log) -> None:
    z = zipfile.ZipFile(zip_path)
    frames = {}
    for n in z.namelist():
        m = re.match(r"train/images/(\d+)_", n)
        if m:
            frames[int(m.group(1))] = n
    ids = sorted(frames)
    blocks = sorted({i // block for i in ids})
    rng = random.Random(seed)
    val_blocks = set(rng.sample(blocks, max(1, round(val_frac * len(blocks)))))
    for k, i in enumerate(ids):
        if k % stride:
            continue
        split = "valid" if i // block in val_blocks else "train"
        n = frames[i]
        name = f"hooper_{i:07d}.jpg"
        (out / split / "images" / name).write_bytes(z.read(n))
        text = z.read(n.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt").decode()
        (out / split / "labels" / f"hooper_{i:07d}.txt").write_text(remap(text, HOOPER_MAP))
        log.writerow(["hooper", split, name, n])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "combined_v1"))
    ap.add_argument("--hotshot", default=str(ROOT / "data" / "hotshot"))
    ap.add_argument("--hooper-zip", required=True)
    ap.add_argument("--hooper-stride", type=int, default=8, help="use every Nth Hooper frame")
    ap.add_argument("--block", type=int, default=2000, help="consecutive frame numbers kept together in one split")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "valid"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
    with open(out / "sources.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["source", "split", "file", "original"])
        add_hotshot(Path(a.hotshot), out, log)
        add_hooper(Path(a.hooper_zip), out, a.hooper_stride, a.block, a.val_frac, a.seed, log)
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: valid/images\n\nnc: {len(NAMES)}\nnames: {NAMES}\n"
        "# sources: hotshot (CC BY 4.0), Basketball Detection v6 by Hooper (CC BY 4.0); see sources.csv\n")
    with open(out / "sources.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for src in ("hotshot", "hooper"):
        for split in ("train", "valid"):
            print(f"{src:<8} {split:<6} {sum(r['source'] == src and r['split'] == split for r in rows):>6} images")
    print(f"wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
