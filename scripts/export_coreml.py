"""Convert the YOLO detector to Core ML for the native iPhone app.

    ~/coreml-env/bin/python scripts/export_coreml.py            # inside WSL / Linux / macOS
    ~/coreml-env/bin/python scripts/export_coreml.py --imgsz 544 960 --weights models/detector_v2_best.pt

coremltools has no Windows wheels, so run this from WSL (a Python 3.12 env with the same Ultralytics
version as .venv, numpy <= 2.3.5: 2.4 breaks coremltools). Writes <out>/<name>.mlpackage and <out>/<name>.json
(source weights and SHA-256, input size, precision, classes, thresholds, letterbox, versions). FP16 (default)
runs on the Neural Engine; --quantize 32 --name DetectorV2_fp32 gives a reference copy. Exported WITHOUT NMS: its
raw output is [1, 4 + classes, anchors] (box centre x, y, w, h in input pixels, then class scores), and
the app decodes it and runs per-class NMS the way Ultralytics predict() does, so phone and Python agree.

Default input 544 x 960 (h x w) is what predict(imgsz=960) uses on 1920 x 1080 footage: scale 0.5 to
960 x 540, centred, padded to a multiple of 32 with grey 114. Weights are not shipped: the .mlpackage
stays local (exports/ is gitignored), like the .pt it comes from.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "combined_v1" / "weights" / "best.pt"  # detector v2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("H", "W"))
    ap.add_argument("--name", default="DetectorV2")
    ap.add_argument("--out", type=Path, default=ROOT / "exports" / "coreml")
    ap.add_argument("--conf", type=float, default=0.15, help="threshold the app should use (same as run_eval_videos.py)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU the app should use (Ultralytics predict default)")
    ap.add_argument("--quantize", type=int, choices=[16, 32], default=16,
                    help="16 runs on the iPhone's Neural Engine; 32 is slower (GPU/CPU), a reference for parity checks")
    a = ap.parse_args()

    try:
        import coremltools
        import numpy
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"{e}: run this from the WSL export env (pip install ultralytics coremltools)")

    if not a.weights.exists():
        sys.exit(f"weights not found: {a.weights}")
    model = YOLO(str(a.weights))
    classes = [model.names[i] for i in sorted(model.names)]
    exported = Path(model.export(format="coreml", imgsz=a.imgsz, nms=False, quantize=a.quantize))

    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / f"{a.name}.mlpackage"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(exported), dest)

    h, w = a.imgsz
    info = {
        "name": a.name,
        "weights": a.weights.relative_to(ROOT).as_posix() if a.weights.is_relative_to(ROOT) else str(a.weights),
        "weights_sha256": sha256(a.weights),
        "input_hw": [h, w],
        "precision": f"fp{a.quantize}",
        "classes": classes,
        "conf": a.conf,
        "iou": a.iou,
        "max_det": 300,
        "letterbox": {"pad_value": 114, "center": True, "stride": 32},
        "output": f"[1, {4 + len(classes)}, anchors]: cx, cy, w, h (input pixels), then per-class scores",
        "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                     "coremltools": coremltools.__version__, "numpy": numpy.__version__},
        "date": date.today().isoformat(),
    }
    info_path = a.out / f"{a.name}.json"
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {dest} and {info_path}")


if __name__ == "__main__":
    main()
