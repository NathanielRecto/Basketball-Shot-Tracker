# Labeling the evaluation videos

The accuracy number is only as honest as these labels. Label **before** looking at any output
from the tracker, and label what you **see**, not what a table says the count should be.

## What to record

One row per **shot attempt**. A shot is a ball thrown toward the hoop (free throw, three-pointer,
airball, brick, bank shot, tip that misses, all count). Dribbling, passing and picking the ball up
do **not**.

```
time_s,outcome,note
12.4,made,
19.8,missed,hit the front rim
1:31.2,missed,airball
```

| Column | Meaning |
|---|---|
| `time_s` | When the ball **reaches the hoop**, not when it leaves the hand. Seconds (`12.4`) or `m:ss` / `m:ss.s` (`1:31.2`) both work. Within about a second is plenty accurate. |
| `outcome` | `made` or `missed`. (`make`, `miss`, `m`, `x`, `1`, `0`, `yes`, `no` are also accepted.) |
| `note` | Optional. Handy for later error analysis: `rim bounce`, `airball`, `bank`, `ball hidden by net`, `shooter blocked the view`. |

## How

1. Open `data/eval_videos/video_XX.mp4` in a player that shows the time (VLC, Windows Media Player).
   The videos are portrait, about 59 fps, 1 to 2 minutes each.
2. Scrub to each shot and note the player's timestamp when the ball reaches the rim.
3. Add a row to `data/eval_videos/labels/video_XX.csv` (lines starting with `#` are ignored).
4. If you cannot tell whether a shot went in, slow it down or step frame by frame. If it is still
   ambiguous, write `?` in the note and make your best call. Do not skip it.

## Rules that keep it fair

* Label the whole video, including shots near the start or end.
* Do not peek at the tracker's predictions or `author_algorithm_results.csv` first.
* The repo's own table has an error (video 7 lists 16 attempts, 6 makes and 11 misses). Treat the
  table as a rough sanity check only. If your count differs by one or two, trust your eyes.
* Only label a video once you have watched it all the way through.

## Checking your work

```
python scripts/evaluate.py
```

Videos with an empty label file are skipped, so you can label a few at a time and re-run. The
script writes `outputs/eval/errors.csv` listing every shot the tracker missed, invented or got
wrong, with timestamps, so you can jump straight to each one.

## Own iPhone footage

Your own sessions live in `data/own_footage/` (gitignored, never published):

| Path | What |
|---|---|
| `raw/session_XX.mov` | The videos, numbered in recording order |
| `sessions.csv` | Session log: original file name, time, setup, balls in play, and the **split** (dev / test), fixed before any tracker output was looked at |
| `labels/session_XX.csv` | Your labels |
| `hoop_overrides.json` | The hoop box per session (rim + net) |

Label with the same tool, pointed at these folders:

```
python scripts/label_shots.py 1 2 3 4 --videos-dir data/own_footage/raw --labels-dir data/own_footage/labels
```

The landscape videos are 1080p, so the window is scaled to `--max-height` (900 px by default).

**Shot type.** Right after pressing M or X, press **1** (free throw), **2** (mid-range) or **3**
(three-pointer). It tags the latest shot at or before the current moment, so you can also pause,
step back and fix a tag. The status bar counts untagged shots. Tags go in a `shot_type` column
(`FT`, `mid`, `3PT`), and `evaluate.py` then reports accuracy per shot type. Old label files
without the column still work.

Mark a shot's note (edit the CSV afterwards) when something unusual happens, e.g. `two balls`,
`blocked by player`, `ball out of frame`. It helps the error analysis later.

Then score and compare runs:

```
python scripts/run_eval_videos.py --weights runs/detect/baseline/weights/best.pt --videos-dir data/own_footage/raw \
    --out outputs/own_eval_baseline --hoop-overrides data/own_footage/hoop_overrides.json
python scripts/evaluate.py --labels data/own_footage/labels --pred outputs/own_eval_baseline --videos 2 3 4
python scripts/compare_runs.py --labels data/own_footage/labels --videos 2 3 4 \
    --run baseline=outputs/own_eval_baseline --run finetuned=outputs/own_eval_finetuned
```
