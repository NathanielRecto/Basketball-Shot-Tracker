"""Fine-tune a YOLO detector on a basketball + hoop dataset (Ultralytics, AGPL-3.0).

Baseline on the Roboflow "hotshot" export (640x640 stretched images):

    python scripts/train_detector.py --data data/hotshot/data_local.yaml --epochs 50

Quick check that everything works (1 epoch on 5% of the data):

    python scripts/train_detector.py --data data/hotshot/data_local.yaml --smoke

Best weights land in runs/detect/<name>/weights/best.pt; pass that to
`python -m shottracker run --weights ...`.

NOTE: the hotshot test split shares source videos with its train split, so its mAP is
optimistic. Judge the model on held-out footage and the make/miss evaluation clips.

A GTX 1660 SUPER (6 GB) handles yolov8s at imgsz 640 with batch 16. If you run out of
memory, lower --batch or use yolov8n.pt.
"""
import argparse
from pathlib import Path

RUNS_DIR = str(Path(__file__).resolve().parent.parent / "runs" / "detect")  # absolute: a relative path gets nested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the dataset yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="8 fits a 6 GB GPU in fp32 (AMP is auto-disabled on GTX 16xx)")
    ap.add_argument("--workers", type=int, default=0, help="0 is ~5x faster than 2 on this Windows machine (worker spawn contention)")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on this fraction of the training images")
    ap.add_argument("--patience", type=int, default=15, help="stop if validation does not improve for this many epochs")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 5%% of the data, to verify the setup")
    ap.add_argument("--resume", default=None, help="path to last.pt of an interrupted run to continue it")
    args = ap.parse_args()

    from ultralytics import YOLO  # pip install -r requirements-ml.txt

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        extra = (dict(epochs=1, fraction=0.05, name=args.name + "_smoke") if args.smoke
                 else dict(epochs=args.epochs, fraction=args.fraction, name=args.name))
        model.train(
            data=args.data, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
            patience=args.patience, device=args.device, project=RUNS_DIR, exist_ok=True, **extra,
        )
    import yaml

    splits = yaml.safe_load(Path(args.data).read_text())
    for split in ("val", "test"):
        if not splits.get(split):
            continue  # e.g. the combined set has no test split: held-out footage is the test
        m = model.val(data=args.data, split=split, imgsz=args.imgsz, device=args.device)
        print(f"[{split}] mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")


if __name__ == "__main__":
    main()
