# Results

Two models are evaluated separately:

1. **The detector** (YOLOv8s) finds the ball and hoop in each frame. Standard object-detection
   metrics: loss curves, precision, recall, mAP, PR / F1 curves, confusion matrix.
2. **The full shot tracker** (detector + tracking + physics-based judging) turns a video into a
   list of shots with made / missed calls. Scored against hand labels: confusion matrix,
   precision / recall / F1, accuracy and confidence intervals.

Sections 1–4 cover the public phone videos with the original system; section 5 covers our own
iPhone footage; section 6 covers the current system (retrained detector) on both. Regenerate
the figures and numbers in sections 1–4 with `python scripts/make_report.py`
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

This section is the original detector (v1). The retrained detector v2 that the current system uses,
with its own per-epoch curves, is in [section 6](#detector-v2).

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

## 5. Own iPhone footage (October 2026)

The 16 public videos above are mostly side-on views of one shooter. To test the system on the
footage it is meant for, four indoor sessions were filmed with an iPhone 11 on a tripod
(1080p, 60 fps, landscape): **three people taking turns** shooting free throws, mid-range shots
and threes, with the camera on the **baseline beside the hoop, facing the court**. The raw
videos are not published (other people appear in them).

| Session | Split | Shots labelled (made) | FT / mid / 3PT | Notes |
|---|---|---|---|---|
| 1 | **Dev** | 43 (16) | 11 / 14 / 18 | two balls in play at times |
| 2 | **Test** | 41 (5) | 10 / 15 / 16 | one ball |
| 3 | **Test** | 39 (21) | 10 / 14 / 15 | one ball |
| 4 | **Test** | 39 (11) | 10 / 15 / 14 | one ball |

As before, every shot was labelled blind (time, made / missed, shot type) before any tracker
output was looked at, the split was fixed before any tuning, and the test sessions were scored
**once** after the code was frozen (label-file hashes and the exact code are kept with the run).
The detector was **not** retrained: it is the same baseline as above.

### What went wrong on this footage (found on the dev session)

![Ball height over time, dev session 1, old tracker](images/timeline_own_session01_old_tracker.png)

Blue dots are ball detections, the orange line is what the tracker followed, vertical lines are
shots it found. Real shots show up as clean arcs, so the detector does see the ball in flight,
but:

1. **Persistent false "balls" on people.** The detector fires on heads (dark hair, caps), calf
   sleeves and hands at 0.6–0.8 confidence (the flat bands at y ≈ 700 and 840). The tracker
   followed a single target, so once it latched onto a head it never let go.
2. **The ball vanishes near the apex.** From under the hoop, high arcs leave the top of the frame
   or are lost against the ceiling lights for ~0.5 s; the tracker then restarted on a leg.
3. **Arcs barely clear the arm line.** The camera is low and close, so the hoop box is tall and
   many arcs peak just above the rim: half a hoop-height above it was too strict.

### Tracker v2 (tuned on dev session 1 only)

* **Side tracks and switching:** detections the main track does not take are followed as short
  side tracks; one that rises fast for 4 frames (a released shot) takes over from a main track
  that is barely moving (a head).
* **Waiting for a ball that left through the top:** when the track was heading out of the frame
  it waits up to 1.5 s for the ball to come back down near where it left, instead of restarting
  on whatever is detected meanwhile. The shot logic waits as well.
* **Lower arm line:** 0.5 → 0.15 hoop-heights above the rim.

Each change was checked against the old dev videos so it would not break the side-on case
(replay score 87.8% → 89.8%), and the synthetic benchmark is unchanged.

### Results

| | Dev (session 1, 43 shots) | | **Test (sessions 2–4, 119 shots)** | |
|---|---|---|---|---|
| | old tracker | v2 | old tracker | **v2** |
| **End-to-end accuracy** | 9.3% | 62.8% | 13.4% (95% CI 8–21%) | **37.8% (95% CI 30–47%)** |
| Shot detection precision | 100% | 100% | 95.7% | **100%** |
| Shot detection recall | 16.3% | 69.8% | 18.5% | **44.5%** |
| Make/miss accuracy on found shots | 57.1% | 90.0% | 72.7% | **84.9%** |

Test by session (v2): 11/41, 15/39 and 19/39 shots found and called right (the old tracker found
**no** shots in session 2). Test by shot type (v2): free throws **60.0%** (18/30), mid-range
**38.6%** (17/44), three-pointers **22.2%** (10/45).

What this says:

* The two trackers were scored on the same shots and their intervals do not overlap: the
  improvement is real. But **37.8% is not good enough**, and the dev score (62.8%) overstates it,
  as expected when tuning on one session.
* When v2 finds a shot it is usually right (85%) and it **never invented a shot**. The weakness
  is **recall**: over half the shots are never followed to the hoop, mostly because of the false
  "balls" on people, and threes (highest arcs, longest out of view) suffer most.
* The next step is the detector, not the tracker: retrain it on frames from this kind of footage
  (never from test sessions 2–4) so heads and legs stop looking like balls, and film with more
  room above the rim.
* Sessions 2–4 are now **used**. Like the public test videos, they will not be tuned on, and
  new claims need new footage.
* Tracker v2 has not been re-scored on the public test videos; the 77.7% in section 2 is the
  original tracker. (Section 6 re-tests the current system, with the retrained detector, on both
  test sets.)

## 6. Current system: retrained detector + tracker v2 (October 2026)

The current system scores **92.2%** on the public test videos (was 77.7%) and **78.2%** on our own
test sessions (was 37.8% with the old detector). Both test sets had been scored before, so these
are **re-tests of a frozen system**, not first looks: nothing was tuned on them and the code,
detector and labels were fingerprinted before the runs, but the low 37.8% is what prompted the
retraining. A clean first-look number needs newly filmed, blind-labelled sessions.

### Detector v2

Fine-tuned from the original detector for 12 epochs (960 px, batch 2) on 8,347 images: the hotshot
set plus 5,324 frames of [Basketball Detection v6](https://universe.roboflow.com/hooper-ibdsr/basketball-detection-v6)
by Hooper (CC BY 4.0; phone video of gyms with many people on court; every 8th of 48,110 frames).
Its hoop boxes cover the rim only, so they became a separate `rim_only` class the tracker ignores;
its boxes around people were dropped (`scripts/build_combined_dataset.py`). Before retraining, the
original detector found only 53 of 242 labelled balls in 300 random Hooper frames at 0.5 confidence.

| Setting | Value |
|---|---|
| Model | YOLOv8s, fine-tuned from detector v1 (`runs/detect/baseline/weights/best.pt`) |
| Data | 8,347 train / 880 validation images (hotshot + Hooper), classes `ball`, `hoop`, `rim_only` |
| Epochs | 12 (early-stopping patience 5, not triggered) |
| Best epoch | **12**, the last one (same fitness criterion as section 1) |
| Batch / image size | 2 / 960 px, mixed precision (batch 4 ran out of the GPU's 6 GB) |
| Training time | 8.0 h on a GTX 1660 SUPER (6 GB) |

![Detector v2 training curves](images/detector_v2_training_curves.png)

* Both losses fall over all 12 epochs and validation loss never turns upward, so there is no sign
  of overfitting.
* The best epoch is the last one and validation mAP50-95 was still rising (0.35 to 0.43), so
  training was stopped by time, not because the model had converged. More epochs would likely add
  a little.

Best-epoch validation metrics (880 images, Hooper frames held out in whole 2,000-frame blocks):

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All classes | 0.871 | 0.823 | 0.851 | 0.433 |
| Ball | 0.865 | 0.637 | 0.747 | 0.441 |
| Hoop (rim + net) | 0.883 | 0.982 | 0.931 | 0.390 |
| Rim only | 0.867 | 0.850 | 0.876 | 0.466 |

These are not comparable with section 1: the validation set is different and much harder (small,
distant balls in busy gyms). What the change did to shot calls is in the re-tests below.

| Precision-recall curve | F1 vs confidence | Confusion matrix (normalized) |
|---|---|---|
| ![PR curve](images/detector_v2_pr_curve.png) | ![F1 curve](images/detector_v2_f1_curve.png) | ![Detector v2 confusion matrix](images/detector_v2_confusion_matrix.png) |

### Frozen re-test: public test videos (103 shots)

Same settings as section 2 (1280 px, every 2nd frame, same hoop boxes).

| | Original | **Current system** |
|---|---|---|
| End-to-end accuracy (95% CI) | 77.7% (69–85%) | **92.2% (85–96%)** |
| Shot detection precision / recall | 98.8% / 82.5% | 99.0% / 96.1% |
| Make/miss accuracy on found shots | 94.1% | 96.0% |
| Outdoor / indoor | 66.7% / 88.5% | 90.2% / 94.2% |
| Free throws / three-pointers | 84.5% / 68.9% | 89.7% / 95.6% |
| Mean abs. error in makes per video | 1.00 | 0.62 |

Video 2 (outdoor, ground camera, threes), the weakest before, went from 3 to 11 of 12 shots right.

### Frozen re-test: own sessions 2–4 (119 shots)

Same settings as section 5 (960 px, every frame).

| | Old detector, original tracker | Old detector, tracker v2 | **Current system** |
|---|---|---|---|
| End-to-end accuracy (95% CI) | 13.4% (8–21%) | 37.8% (30–47%) | **78.2% (70–85%)** |
| Shot detection precision / recall | 95.7% / 18.5% | 100% / 44.5% | 98.1% / 86.6% |
| Make/miss accuracy on found shots | 72.7% | 84.9% | 90.3% |
| Free throws / mid-range / threes | | 60.0% / 38.6% / 22.2% | 76.7% / 70.5% / 86.7% |

Shots found per session: 27/41, 37/39, 39/39; session 2 holds 14 of the 16 shots not found.

### Each tracker change, on dev footage

Replays with the new detector at 960 px:

| Tracker | Old dev videos (98 shots) | Own dev session 1 (43 shots) |
|---|---|---|
| Original | 89.8% | 62.8% |
| Tracker v2 | 89.8% | 86.0% |
| v2 without switching / top-exit wait / lower arm line | 89.8% / 89.8% / 89.8% | 74.4% / 76.7% / 79.1% |

On side-on footage the changes alter nothing (the same shots right and wrong); on footage from
under the hoop each adds 7–12 points. An earlier comparison that suggested tracker v2 lost 5 points
on the old videos had run those portrait videos at 960 px instead of 1280 px; run settings are now
saved per run and `compare_runs.py` warns when they differ.

### Speed vs accuracy (for a phone app, dev footage)

End-to-end accuracy, old dev videos (portrait) / own session 1 (landscape):

| Image size | 60 fps | 30 fps | 20 fps | PC time per frame |
|---|---|---|---|---|
| 640 px | 80.6% / 88.4% | 84.7% / 88.4% | 84.7% / 83.7% | 12.7 ms |
| 960 px | 89.8% / 86.0% | 88.8% / 88.4% | 90.8% / 88.4% | 19.4 ms |
| 1280 px | 88.8% / 88.4% | 94.9% / 86.0% | 91.8% / 88.4% | 31.6 ms |

Frame rate barely matters, and landscape footage holds at 640 px; portrait video needs at least
960 px. Target for the app: landscape filming, 960 px at 20–30 fps, with 640 px at 30 fps as the
fallback. Speed on the iPhone itself is not yet measured. Differences of 2–3 points here are one or
two shots.
