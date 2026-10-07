# Porting the tracker to the phone app

The app lives in its own repo: [Basketball-Shot-Tracker-App](https://github.com/NathanielRecto/Basketball-Shot-Tracker-App)
(Expo / React Native, TypeScript; port in `src/shottracker/`, comparison in `tests/golden.test.ts`).

The phone app re-implements the static filter, ball tracker and shot judge (`src/shottracker/tracking.py`,
`shot_logic.py`, `pipeline.py`) in its own language. This repo stays the reference: changes are made and
measured here first, then copied to the app. To keep the two identical, this repo exports **settings** and
**golden test cases**, and the app's tests must reproduce the cases exactly.

```
python scripts/export_app_fixtures.py
python scripts/export_app_fixtures.py --debug-json outputs/debug960_det2/session_01/debug.json --name own_session_01
```

writes `exports/app/` (gitignored here; copy the whole folder into the app repo):

| File | What it is |
|---|---|
| `params.json` | Every setting of the static filter, tracker and shot judge, read from this code's defaults |
| `golden/sim_all_shot_types.json` | Synthetic: every shot type (make, miss, rim-out, rim bounce, fall past, rattle-in), noise, dropped frames, a static decoy and a wandering "head" the tracker must switch away from |
| `golden/<name>.json` | Optional real cases from `debug.json` files (detections only, no images) |
| `manifest.json` | The Python commit the files came from, their sizes and SHA-256 hashes |

Regenerate and re-copy whenever the tracking or judging code, or a setting, changes. The app README should
name the Python commit its copy matches (from `manifest.json`).

## Golden case format (`shottracker-golden/1`)

```json
{
  "format": "shottracker-golden/1",
  "name": "sim_all_shot_types",
  "source": "synthetic: ...",
  "hoop": [x1, y1, x2, y2],
  "frames": [
    {"t": 4.633333,
     "balls": [[x1, y1, x2, y2, conf], ...],
     "expect": {"ball": [x, y, diameter] | null, "switched": true | false}}
  ],
  "expect": {
    "events": [{"t_release", "t_apex", "t_cross", "outcome": "made" | "missed", "reason", "method", "cross_offset"}],
    "log": [[t, "arm" | "track_switch" | "rattle:<reason>" | "shot:<outcome>:<reason>" | "abort:<reason>"], ...]
  }
}
```

* `balls` = the detector's ball boxes for that frame, already filtered to confidence >= `params.detector.conf`
  and rounded; the hoop box is fixed for the whole case.
* `expect.ball` = what the tracker returned that frame; `switched` = it jumped to a different object.
* `expect.events` / `expect.log` = every shot call and every decision the shot judge logged, in order.

## How the app's test should use it

For each case: build a fresh pipeline with `params.json` and the case's hoop, feed the frames in order, and
compare at every frame. Expected numbers are written at full double precision and both languages use IEEE
doubles, so compare **exactly**; when a value differs, the port's arithmetic differs (often an operation in
a different order), so fix the port rather than loosening the test. `scripts/export_app_fixtures.py` itself
refuses to write a case that this Python code cannot replay identically after a JSON round trip.

## Code rules that make exact matching possible

The tracking and judging code here avoids anything another language cannot reproduce bit for bit: the arc
fit is plain least squares (sums in sample order, Cramer's rule) instead of `numpy.polyfit`, distances use
`geometry.dist` (`sqrt(dx*dx + dy*dy)`) instead of `math.hypot`, and sums are explicit loops instead of
`sum()` (which changes its algorithm in Python 3.12) and `** 2`. Switching to these changed none of the
751 shot calls on the dev footage. Keep to them when editing `geometry.py`, `tracking.py` or
`shot_logic.py`; the app repo's `docs/porting.md` lists the same rules from the other side.

## Data rules

Only export footage you have the right to publish. Synthetic cases and our own sessions are fine; the public
evaluation videos have no licence, so the script refuses them. Exporting a test session here does not
change its role: it is still never tuned on.
