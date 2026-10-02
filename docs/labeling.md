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
