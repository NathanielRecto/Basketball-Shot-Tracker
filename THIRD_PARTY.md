# Third-party software, data and prior art

This project's source code is original. No code, trained weights, datasets or
binaries from other basketball-analysis projects are included.

## Dependencies (installed via pip, not vendored)

| Package | Use | License |
|---|---|---|
| NumPy | numerics | BSD-3-Clause |
| OpenCV (`opencv-python`) | video I/O, drawing | Apache-2.0 |
| Ultralytics YOLO | ball/hoop detection, training | **AGPL-3.0** |
| MediaPipe | pose estimation (and its `pose_landmarker_full.task` model, downloaded separately, not committed) | Apache-2.0 |
| PyTorch | training / inference backend for Ultralytics | BSD-3-Clause |
| pytest | tests | MIT |

Because Ultralytics is AGPL-3.0, this repository is published under AGPL-3.0 (see `LICENSE`).
If the detector is ever swapped for a permissively licensed one, the repo license can be
reconsidered.

## Deliberately not used

* **OpenPose** (Carnegie Mellon University): its license permits non-commercial
  academic research use only, forbids distribution, and assigns ownership of
  derivatives to CMU. MediaPipe Pose is used instead.

## Datasets

Record every dataset you train on here with its license and attribution. Trained weights
inherit those terms. Dataset images are NOT committed to this repository (`data/` is gitignored).

### "basketball detection" (v5) by the Roboflow user "hotshot"

* Source: https://universe.roboflow.com/hotshot/basketball-detection-tqwcs
* License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* Used for: training the ball/hoop detector (classes `basketball`, `basketball-hoop`).
* Changes: none to the annotations. Images were exported by Roboflow resized to 640x640 (stretch).
* Note: the images are frames from third-party videos (e.g. YouTube). CC BY covers the
  annotator's contribution; do not redistribute the images themselves.

### "Basketball Detection v6" (v1) by the Roboflow user "Hooper"

* Source: https://universe.roboflow.com/hooper-ibdsr/basketball-detection-v6
* License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* Used for: the second detector (`runs/detect/combined_v1`, built by
  `scripts/build_combined_dataset.py`), together with the hotshot dataset above. Phone video of
  indoor gyms with many people on court; every 8th of its 48,110 frames was used.
* Changes: its `ball` boxes are used as `ball`; its `hoop` boxes (which cover the rim only) became a
  separate `rim_only` class so they never mix with the rim-and-net `hoop` boxes; its `holder` and
  `scorer` (people) boxes were dropped. No boxes were edited.
* Note: do not redistribute the images themselves.

## Evaluation videos

* Source: the 16 raw test videos linked from the "Testing Dataset" table of
  [gbikushev/Basketball-Score-Detection-ComputerVision](https://github.com/gbikushev/Basketball-Score-Detection-ComputerVision).
* That repository has no license, so the videos are used **locally only** for evaluation and are
  **not redistributed** here (no videos, frames or crops are committed). The shot-by-shot labels
  used for scoring were made by hand for this project.

## Prior art / acknowledgements

The idea of tracking a ball and hoop to judge shots and measure release angle has
been explored by others, notably
[chonyy/AI-basketball-analysis](https://github.com/chonyy/AI-basketball-analysis).
That project's code and models were studied only to understand the problem;
this implementation uses a different design (time-based projectile fit,
interpolated rim-crossing, occlusion and rim-bounce handling, constant-velocity
ball tracking) and was written from scratch.
