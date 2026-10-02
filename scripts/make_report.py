"""Build the results report: training curves, confusion matrices, precision/recall/F1, per-video
accuracy and uncertainty estimates. Writes figures to docs/images/ and metrics to
docs/results_metrics.json.

    python scripts/make_report.py

Inputs: runs/detect/baseline/ (Ultralytics training output), outputs/eval_final/ (tracker
predictions from run_eval_videos.py) and data/eval_videos/labels/ (hand labels).
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shottracker.evaluation import load_labels, load_predictions, match_shots, wilson_interval  # noqa: E402

RUN = ROOT / "runs" / "detect" / "baseline"
PRED = ROOT / "outputs" / "eval_final"
LABELS = ROOT / "data" / "eval_videos" / "labels"
META = ROOT / "data" / "eval_videos" / "listed_counts.csv"
IMG = ROOT / "docs" / "images"
SPLITS = {"dev": [1, 4, 5, 8, 9, 12, 13, 16], "test": [2, 3, 6, 7, 10, 11, 14, 15]}

# Reference palette (dataviz skill, light mode): categorical slots in fixed order + chart chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASE,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "legend.labelcolor": INK2,
    })
    return plt


# ---- detector training -------------------------------------------------------------------

def training(plt):
    rows = list(csv.DictReader(open(RUN / "results.csv")))
    ep = np.array([int(r["epoch"]) for r in rows])
    f = lambda k: np.array([float(r[k]) for r in rows])
    train_loss = f("train/box_loss") + f("train/cls_loss") + f("train/dfl_loss")
    val_loss = f("val/box_loss") + f("val/cls_loss") + f("val/dfl_loss")
    metrics = {"mAP50": f("metrics/mAP50(B)"), "mAP50-95": f("metrics/mAP50-95(B)"),
               "precision": f("metrics/precision(B)"), "recall": f("metrics/recall(B)")}
    fitness = 0.1 * metrics["mAP50"] + 0.9 * metrics["mAP50-95"]  # Ultralytics' checkpoint criterion
    best = int(ep[np.argmax(fitness)])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for y, name, c in ((train_loss, "train", SERIES[0]), (val_loss, "validation", SERIES[1])):
        ax.plot(ep, y, color=c, lw=2)
        ax.annotate(name, (ep[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2)
    ax.set_title("Loss per epoch (box + class + DFL)")
    ax.set_xlabel("epoch")
    ax.set_xlim(1, ep[-1] + 6)
    ax = axes[1]
    for (name, y), c in zip(metrics.items(), SERIES):
        ax.plot(ep, y, color=c, lw=2)
    # direct labels at the line ends, nudged apart so they never overlap
    ends = sorted(((float(y[-1]), name, float(y[best - 1])) for name, y in metrics.items()), reverse=True)
    placed = []
    for yv, name, at_best in ends:
        ty = min(yv, placed[-1] - 0.06) if placed else yv
        placed.append(ty)
        ax.annotate(f"{name} {at_best:.2f}", xy=(ep[-1], yv), xytext=(ep[-1] + 1.2, ty), textcoords="data",
                    va="center", color=INK2, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
    ax.axvline(best, color=MUTED, lw=1, ls="--")
    ax.text(best, 1.02, f"best epoch {best}", transform=ax.get_xaxis_transform(), ha="center", color=MUTED, fontsize=9)
    ax.set_title("Validation metrics per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, ep[-1] + 10)
    fig.tight_layout()
    fig.savefig(IMG / "training_curves.png", dpi=130)
    plt.close(fig)

    for src, dst in (("BoxPR_curve.png", "detector_pr_curve.png"), ("BoxF1_curve.png", "detector_f1_curve.png"),
                     ("confusion_matrix_normalized.png", "detector_confusion_matrix.png")):
        shutil.copy(RUN / src, IMG / dst)

    args = {}
    for line in open(RUN / "args.yaml"):
        if ":" in line:
            k, v = line.split(":", 1)
            args[k.strip()] = v.strip()
    return {
        "epochs": int(ep[-1]), "best_epoch": best, "patience": int(args.get("patience", 0)),
        "batch": int(args.get("batch", 0)), "imgsz": int(args.get("imgsz", 0)), "model": args.get("model"),
        "hours": round(float(rows[-1]["time"]) / 3600, 2),
        "best": {k: round(float(v[best - 1]), 3) for k, v in metrics.items()},
        "final_train_loss": round(float(train_loss[-1]), 3), "final_val_loss": round(float(val_loss[-1]), 3),
    }


# ---- shot-level evaluation ---------------------------------------------------------------

def shot_level(split, tol=1.5):
    meta = {int(r["video"]): r for r in csv.DictReader(open(META))}
    cm = {(l, p): 0 for l in ("made", "missed") for p in ("made", "missed", "not found")}
    extra = {"made": 0, "missed": 0}
    per_video = []
    for v in SPLITS[split]:
        labels = load_labels(LABELS / f"video_{v:02d}.csv")
        preds = load_predictions(PRED / f"video_{v:02d}" / "summary.json")
        pairs, miss, ext = match_shots(labels, preds, tol)
        correct = 0
        for li, pi in pairs:
            cm[(labels[li].outcome, preds[pi].outcome)] += 1
            correct += labels[li].outcome == preds[pi].outcome
        for li in miss:
            cm[(labels[li].outcome, "not found")] += 1
        for pi in ext:
            extra[preds[pi].outcome] += 1
        per_video.append({"video": v, "env": meta[v]["env"], "throw": meta[v]["throw_type"],
                          "camera": meta[v]["camera_height"], "labelled": len(labels), "correct": correct})
    return cm, extra, per_video


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def summarize(cm, extra, per_video, rng):
    labelled = sum(cm.values())
    found = labelled - cm[("made", "not found")] - cm[("missed", "not found")]
    predicted = found + extra["made"] + extra["missed"]
    det_p, det_r, det_f1 = prf(found, predicted - found, labelled - found)
    tp, fp, fn, tn = cm[("made", "made")], cm[("missed", "made")], cm[("made", "missed")], cm[("missed", "missed")]
    mk_p, mk_r, mk_f1 = prf(tp, fp, fn)
    # end to end, "made" as the positive class: a shot never found counts as not predicted made
    e2e_p, e2e_r, e2e_f1 = prf(tp, fp + extra["made"], fn + cm[("made", "not found")])
    correct = tp + tn
    lo, hi = wilson_interval(correct, labelled)
    # video-level (cluster) bootstrap: shots in one video are not independent
    c = np.array([pv["correct"] for pv in per_video])
    n = np.array([pv["labelled"] for pv in per_video])
    idx = rng.integers(0, len(c), size=(20000, len(c)))
    boot = c[idx].sum(1) / n[idx].sum(1)
    r3 = lambda x: round(float(x), 3)
    return {
        "labelled": labelled, "found": found, "predicted": predicted, "extra": extra["made"] + extra["missed"],
        "shot_detection": {"precision": r3(det_p), "recall": r3(det_r), "f1": r3(det_f1)},
        "make_on_found": {"accuracy": r3((tp + tn) / found), "precision": r3(mk_p), "recall": r3(mk_r), "f1": r3(mk_f1)},
        "make_end_to_end": {"precision": r3(e2e_p), "recall": r3(e2e_r), "f1": r3(e2e_f1)},
        "end_to_end_accuracy": r3(correct / labelled),
        "ci95_wilson_shots": [r3(lo), r3(hi)],
        "ci95_bootstrap_videos": [r3(np.percentile(boot, 2.5)), r3(np.percentile(boot, 97.5))],
        "per_video_accuracy": {f"video_{pv['video']:02d}": r3(pv["correct"] / pv["labelled"]) for pv in per_video},
        "per_video_range": [r3((c / n).min()), r3((c / n).max())],
        "confusion": {f"{l} -> {p}": v for (l, p), v in cm.items()},
        "extra_shots": extra,
    }


def confusion_figure(plt, results):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    cols = ["made", "missed", "not found"]
    for ax, split in zip(axes, ("dev", "test")):
        cm = results[split]["cm"]
        m = np.array([[cm[(l, p)] for p in cols] for l in ("made", "missed")], dtype=float)
        frac = m / m.sum(1, keepdims=True)
        ax.imshow(frac, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f"{int(m[i, j])}\n{100 * frac[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                        color="#ffffff" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), [f"called {c}" if c != "not found" else "not found" for c in cols])
        ax.set_yticks(range(2), ["labelled made", "labelled missed"])
        ax.tick_params(colors=INK2, length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ex = results[split]["extra"]
        ax.set_title(f"{split.capitalize()} videos ({int(m.sum())} labelled shots)")
        ax.set_xlabel(f"plus {ex['made'] + ex['missed']} invented shot(s) with no label", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "shot_confusion_matrix.png", dpi=130)
    plt.close(fig)


def per_video_figure(plt, results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, split in zip(axes, ("dev", "test")):
        pv = results[split]["per_video"]
        x = np.arange(len(pv))
        acc = [100 * p["correct"] / p["labelled"] for p in pv]
        colors = [SERIES[0] if p["env"] == "Indoors" else SERIES[1] for p in pv]
        ax.bar(x, acc, color=colors, width=0.72, edgecolor=SURFACE, linewidth=2)
        for xi, a, p in zip(x, acc, pv):
            ax.text(xi, a + 2, f"{a:.0f}%", ha="center", color=INK2, fontsize=9)
        ax.set_xticks(x, [f"{p['video']}\n{'FT' if p['throw'].startswith('Free') else '3PT'}" for p in pv])
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
        overall = 100 * results[split]["summary"]["end_to_end_accuracy"]
        ax.axhline(overall, color=MUTED, lw=1, ls="--")
        ax.set_title(f"{split.capitalize()} videos (dashed line = overall {overall:.1f}%)")
        ax.set_xlabel("video number / shot type")
    axes[0].set_ylabel("shots found and called right (%)")
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=SERIES[0], label="indoor"), Patch(color=SERIES[1], label="outdoor")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(IMG / "per_video_accuracy.png", dpi=130)
    plt.close(fig)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    plt = style()
    rng = np.random.default_rng(0)
    out = {"detector_training": training(plt)}
    results = {}
    for split in ("dev", "test"):
        cm, extra, pv = shot_level(split)
        results[split] = {"cm": cm, "extra": extra, "per_video": pv, "summary": summarize(cm, extra, pv, rng)}
        out[split] = results[split]["summary"]
    confusion_figure(plt, results)
    per_video_figure(plt, results)
    (ROOT / "docs" / "results_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
