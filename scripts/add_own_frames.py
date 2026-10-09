"""Build a training set: an existing combined set plus hand-checked frames from own DEV sessions.

    python scripts/add_own_frames.py --sessions 1 2 4 --repeat 3

Own frames come from data/own_boxes_v1 (scripts/sample_box_frames.py + scripts/box_labeler.py); only
frames marked reviewed are used, and only from sessions whose split is "dev" in sessions.csv. Each is
included --repeat times (hard links, no extra disk space) so a few hundred frames still carry weight
next to thousands of others. Validation stays the base set's, so scores compare with earlier detectors.
Writes <out>/own_train/{images,labels}, <out>/data.yaml and <out>/sources.csv.

Sessions left out (here 3) stay unseen, so the new detector can be compared fairly on them.
"""
import argparse
import csv
import importlib.util
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, nargs="+", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--boxes", default=str(ROOT / "data" / "own_boxes_v1"))
    ap.add_argument("--base", default=str(ROOT / "data" / "combined_v1"))
    ap.add_argument("--out", default=str(ROOT / "data" / "combined_v2"))
    ap.add_argument("--sessions-csv", default=str(ROOT / "data" / "own_footage" / "sessions.csv"))
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("sbf", ROOT / "scripts" / "sample_box_frames.py")
    sbf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sbf)
    splits = sbf.session_splits(Path(a.sessions_csv))
    bad = [s for s in a.sessions if splits.get(s) != "dev"]
    if bad:
        print(f"refused: sessions {bad} are not 'dev' in sessions.csv (test footage must never be trained on)")
        return 2

    boxes, base, out = Path(a.boxes), Path(a.base), Path(a.out)
    reviewed = set((boxes / "reviewed.txt").read_text().split())
    rows = [r for r in csv.DictReader(open(boxes / "manifest.csv")) if int(r["session"]) in a.sessions]
    unchecked = [r["file"] for r in rows if r["file"] not in reviewed]
    if unchecked:
        print(f"refused: {len(unchecked)} frames from these sessions are not reviewed yet (first: {unchecked[0]})")
        return 2
    for sub in ("images", "labels"):
        (out / "own_train" / sub).mkdir(parents=True, exist_ok=True)
    sources = []
    for r in rows:
        stem = Path(r["file"]).stem
        for k in range(a.repeat):
            name = f"own_{stem}_r{k}"
            link_or_copy(boxes / "images" / f"{stem}.jpg", out / "own_train" / "images" / f"{name}.jpg")
            link_or_copy(boxes / "labels" / f"{stem}.txt", out / "own_train" / "labels" / f"{name}.txt")
            sources.append({"file": f"{name}.jpg", "session": r["session"], "copy": k})
    with open(out / "sources.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "session", "copy"])
        w.writeheader()
        w.writerows(sources)
    base_yaml = (base / "data.yaml").read_text()
    names = next(ln for ln in base_yaml.splitlines() if ln.startswith("names:"))
    (out / "data.yaml").write_text(
        f"# {base.name} + own dev frames from sessions {a.sessions} x{a.repeat} (scripts/add_own_frames.py)\n"
        f"train:\n  - {(base / 'train' / 'images').as_posix()}\n  - {(out / 'own_train' / 'images').as_posix()}\n"
        f"val: {(base / 'valid' / 'images').as_posix()}\n\nnc: 3\n{names}\n")
    n_base = len(list((base / "train" / "images").glob("*")))
    print(f"{out}: {n_base} base + {len(rows)} own frames x{a.repeat} = {n_base + len(sources)} training images; "
          f"validation unchanged ({len(list((base / 'valid' / 'images').glob('*')))} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
