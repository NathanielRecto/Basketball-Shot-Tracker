# Results

Two models are evaluated separately:

1. **The detector** (YOLOv8s) finds the ball and hoop in each frame. Standard object-detection
   metrics: loss curves, precision, recall, mAP, PR / F1 curves, confusion matrix.
2. **The full shot tracker** (detector + tracking + physics-based judging) turns a video into a
   list of shots with made / missed calls. Scored against hand labels: confusion matrix,
   precision / recall / F1, accuracy and confidence intervals.

Regenerate every figure and number on this page with `python scripts/make_report.py`
(raw numbers: [`results_metrics.json`](results_metrics.json)).

## Data splits

| Data | Split | Notes |
|---|---|---|
| Detector dataset ("hotshot", CC BY 4.0) | 3,023 train / 190 validation / 412 test images | Publisher's split. The test images share source videos with the training images (checked by file name), so only validation metrics are reported as headline detector numbers |
| Evaluation videos | 8 dev / 8 test videos, 98 / 103 hand-labelled shots | Split by whole video before any tuning, one video of every setup per half. The detector never trained on any of them |

| Video | Split | Court | Camera | Shot type | Length | Shots labelled (made) |
|---|---|---|---|---|---|---|
| 1 | Dev | Outdoors | ground, right side | free throws | 1:54 | 16 (10) |
| 4 | Dev | Outdoors | ground, left side | three-pointers | 1:43 | 13 (1) |
| 5 | Dev | Outdoors | tripod (1.5 m), right side | free throws | 1:30 | 13 (8) |
| 8 | Dev | Outdoors | tripod (1.5 m), left side | three-pointers | 1:22 | 8 (2) |
| 9 | Dev | Indoors | ground, right side | free throws | 1:56 | 15 (7) |
| 12 | Dev | Indoors | ground, left side | three-pointers | 1:00 | 7 (1) |
| 13 | Dev | Indoors | tripod (1.5 m), right side | free throws | 1:32 | 19 (10) |
| 16 | Dev | Indoors | tripod (1.5 m), left side | three-pointers | 1:00 | 7 (2) |
| 2 | Test | Outdoors | ground, right side | three-pointers | 1:39 | 12 (4) |
| 3 | Test | Outdoors | ground, left side | free throws | 1:41 | 12 (9) |
| 6 | Test | Outdoors | tripod (1.5 m), right side | three-pointers | 2:02 | 11 (4) |
| 7 | Test | Outdoors | tripod (1.5 m), left side | free throws | 1:48 | 16 (5) |
| 10 | Test | Indoors | ground, right side | three-pointers | 1:19 | 10 (5) |
| 11 | Test | Indoors | ground, left side | free throws | 1:58 | 14 (4) |
| 14 | Test | Indoors | tripod (1.5 m), right side | three-pointers | 1:43 | 12 (4) |
| 15 | Test | Indoors | tripod (1.5 m), left side | free throws | 1:55 | 16 (8) |

## 1. Detector training

| Setting | Value |
|---|---|
| Model | YOLOv8s (11.1 M parameters), fine-tuned from COCO weights |
| Data | "hotshot" basketball dataset, CC BY 4.0: 3,023 train / 190 validation images, classes `basketball`, `basketball-hoop` |
| Epochs | 30 (early-stopping patience 10, not triggered) |
| Best epoch | **29** (Ultralytics fitness = 0.1·mAP50 + 0.9·mAP50-95) |
| Batch / image size | 8 / 640 px, fp32 |
| Training time | 3.8 h on a GTX 1660 SUPER (6 GB) |

![Training curves](images/training_curves.png)

* Training loss falls steadily; validation loss falls until about epoch 25 and then flattens:
  no sign of the model memorizing the training set (validation loss rising) within 30 epochs.
* Validation metrics were still creeping up at the end, so a few more epochs might add a little,
  but more varied data matters more (see section 3).

Best-epoch validation metrics:

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All classes | 0.891 | 0.849 | 0.882 | 0.404 |
| Ball | 0.897 | 0.726 | 0.812 | 0.375 |
| Hoop | 0.884 | 0.972 | 0.951 | 0.431 |

The ball is the harder class: it is small (about 23 px wide at 640 px) and blurred in flight.

| Precision-recall curve | F1 vs confidence | Confusion matrix (normalized) |
|---|---|---|
| ![PR curve](images/detector_pr_curve.png) | ![F1 curve](images/detector_f1_curve.png) | ![Detector confusion matrix](images/detector_confusion_matrix.png) |

*The dataset's own test split (mAP50 0.970) is not reported as a headline number: its frames
come from the same source videos as the training frames, which inflates it.*

## 2. Shot tracker: made / missed on real video

16 phone videos (1080×1920, ~60 fps): indoor and outdoor, camera on the ground or a tripod,
free throws and three-pointers. Every shot was labelled by hand (time + made / missed) before
looking at any tracker output. **Dev** videos (8) were used for debugging and tuning; **test**
videos (8) were scored **once**, after all tuning was frozen. Each half has one video of every
setup.

### Confusion matrix

![Shot confusion matrix](images/shot_confusion_matrix.png)

"Not found" means a labelled shot the tracker never detected. On the test videos the tracker
**never called a real make a miss**; its errors are 5 misses called makes and 18 shots not
found (most of them outdoors).

### Precision, recall, F1

| | Dev | **Test** |
|---|---|---|
| **Shot detection** (is there a shot?) | P 1.000 · R 0.929 · **F1 0.963** | P 0.988 · R 0.825 · **F1 0.899** |
| **Made vs missed, on shots found** ("made" = positive) | accuracy 0.956 · P 0.944 · R 0.944 · **F1 0.944** | accuracy 0.941 · P 0.872 · R 1.000 · **F1 0.932** |
| **Made, end to end** (unfound shots count as errors) | P 0.944 · R 0.829 · F1 0.883 | P 0.872 · R 0.791 · F1 0.829 |
| **End-to-end accuracy** (found *and* called right) | **88.8%** | **77.7%** |

### Uncertainty

| | Dev | **Test** |
|---|---|---|
| 95% CI, shots treated as independent (Wilson) | 81.0–93.6% | 68.7–84.6% |
| 95% CI, **resampling whole videos** (bootstrap, 20,000 resamples) | 81.6–95.1% | **61.2–89.9%** |
| Per-video accuracy range | 57–100% | 25–100% |

Shots from the same video share a camera angle, court and lighting, so they are not independent.
Resampling whole videos gives the more honest (wider) interval: with 8 test videos, the true
test accuracy could plausibly be anywhere from about 61% to 90%.

![Per-video accuracy](images/per_video_accuracy.png)

The weak spot is outdoor footage, above all video 2 (outdoor, camera on the ground,
three-pointers): 3 of 12 shots right, almost all of them never found.

## 3. Why no k-fold cross-validation?

Cross-validation estimates how a model generalizes by training and testing on several splits.
It is the right tool for a tabular model trained in seconds; here it was replaced by things
that answer the same question more honestly for this setup:

* **Frames are not independent samples.** Consecutive frames from one video are near
  duplicates. Splitting frames at random (as k-fold does by default) leaks near-copies between
  train and test and inflates the score, which is exactly what happened to the detector
  dataset's own test split. Any split must be **by video**.
* **The detector dataset has only ~4 source videos**, so grouped k-fold would mean 4 folds of
  very different content, each costing ~4 hours of GPU training. Fresh, independent test
  footage is a better use of that time (the roadmap item "own footage").
* **For the shot tracker**, the only tuning was a handful of physical thresholds and rules, chosen
  on the dev videos. The test videos were held out until the end and scored once, so the test
  number was never used to make a decision. The video-level bootstrap above provides the
  "how much would this move on other videos?" estimate that cross-validation would otherwise
  give.

## 4. Judging logic on simulated shots

The make/miss logic is also stress-tested on simulated shots (400 per row: makes, misses,
rim-outs, rim bounces, rattle-ins and "fall past" misses that look centred from the side; half
with the ball hidden near the rim; added pixel noise and dropped detections):

| Detection noise | Missed detections | FPS | Correct |
|---|---|---|---|
| 0 px | 0% | 30 | 96.0% |
| 1.5 px | 10% | 30 | 95.8% |
| 3 px | 20% | 30 | 95.5% |
| 5 px | 30% | 30 | 92.8% |

Nearly all simulated errors are "fall past" misses where the ball is hidden right after the rim:
the net-braking check needs to see the ball falling (100% caught when visible, 0% when hidden).
This measures the logic on the project's own simulator, not real-world accuracy.
