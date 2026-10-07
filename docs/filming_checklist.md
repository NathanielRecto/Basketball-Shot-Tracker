# Filming checklist

Footage from these sessions does two jobs: it **trains** the detector on the kind of video the
app will actually see, and it gives a **fresh held-out test set** to measure the improvement
honestly. Following the same setup every time is what makes both work.

## How much to film

| | |
|---|---|
| **One session** | **8–10 minutes** of continuous recording, **30–40 shots** (about one shot every 15 s, including getting your own rebound) |
| **Total** | **10 sessions** over 2–3 weeks, about 350 shots and 1.5 hours of video |
| **File size** | Roughly 0.5–1 GB per session at 1080p / 60 fps |

How the 10 sessions are used (decide **before** filming, never after seeing results):

| Split | Sessions | Shots (approx.) | Used for |
|---|---|---|---|
| **Train** | 5 | ~175 | Labelling frames to retrain the detector |
| **Dev** | 2 | ~70 | Checking and tuning while improving |
| **Test** | 3 | ~105 | Locked away, labelled blind, scored **once** at the end |

More test shots means a tighter result: ~105 shots gives roughly ±8 percentage points, ~200
gives about ±6. If you can, film 2 extra test sessions.

## Equipment

- [ ] iPhone 11, charged (a power bank for outdoor sessions)
- [ ] **Tripod or phone stand** with a phone clamp, about **1.2–1.5 m** tall (chest height)
- [ ] One basketball in play (spare balls lying around are fine, but note them)
- [ ] At least 5 GB free on the phone

## iPhone settings (once)

- [ ] **Settings → Camera → Record Video → 1080p HD at 60 fps**
- [ ] Settings → Camera → Formats: **High Efficiency** is fine
- [ ] Settings → Photos → Transfer to Mac or PC: **Keep Originals**
- [ ] Before each session: **Do Not Disturb** or **Airplane mode**, so a call can't stop the recording
- [ ] Use the normal **1×** lens. No zoom, no Slo-mo, no Cinematic mode

## Setting up the camera (every session)

- [ ] **Side view**: stand the phone to the **side** of the hoop, roughly level with the free-throw
      line or a bit closer to the baseline, so the ball travels **across** the picture, not
      toward the camera
- [ ] **Landscape** (phone sideways) for most sessions: the shooter, the full arc and the hoop
      fit side by side. Film **1–2 sessions in portrait** too, for variety
- [ ] **Far enough back** that the whole shot is in frame: shooter's feet, the hoop and backboard,
      and **plenty of space above the rim** for the top of the arc (high three-pointers too)
- [ ] Hoop in the **upper part** of the picture, not cut off at the edge
- [ ] Tripod **stable** and **not moved** during the session (if it gets bumped, stop and start a
      new clip)
- [ ] Tap and hold on the hoop to **lock focus and exposure** (AE/AF LOCK)
- [ ] Avoid pointing straight into the sun

## During the session

- [ ] Start recording, then say or show the **session number** (e.g. hold up fingers) so clips
      are easy to match to the log
- [ ] Shoot normally. **Don't** force makes or misses; a natural mix is what the app will see
- [ ] Vary shots within the session: free throws, mid-range, three-pointers, different spots on
      the floor
- [ ] Shoot from **both sides** of the hoop across sessions (left side and right side)
- [ ] Rim-outs, bounces and airballs are welcome; they are the hard cases
- [ ] One continuous recording per session is easiest; if you pause, keep the camera where it is

## Variety across the 10 sessions

Spread these out so that train, dev and test each get a mix, and keep at least one **test**
session at a court or time of day that train does not have (that tests how well it generalizes):

| Vary | Examples |
|---|---|
| Place | indoor gym, outdoor court(s), a different court if you can |
| Light | midday sun, cloudy, evening, indoor lights |
| Camera side | left of the hoop, right of the hoop |
| Distance / height | standard tripod height for most; one session lower (about 0.5 m) as a hard case |
| Orientation | mostly landscape; 1–2 sessions portrait |
| Background | quiet court; one session with other people around (only with their OK) |

## Session log

Keep a simple log (phone note or spreadsheet), one row per session, **filled in on the day**:

| session | date | place | indoor/outdoor | light | camera side | orientation | shots (approx.) | split | notes |
|---|---|---|---|---|---|---|---|---|---|
| 01 | | | | | | | | train | |

Assign the split **before** looking at any tracker output on that session.

## After the session

- [ ] Plug the iPhone into the PC (USB), open **Photos → Import** (or File Explorer → Apple iPhone
      → DCIM) and copy the video
- [ ] Save it as `data/own_footage/raw/session_01.mov` (the `data/` folder is never uploaded to
      GitHub)
- [ ] Fill in the log row
- [ ] **Test sessions**: label them blind later with `python scripts/label_shots.py`, before
      running the tracker on them

## Lessons from the first sessions (2026-10-05, indoor)

- **A second hoop in the background fools auto-calibration.** The far-wall hoop was picked in
  3 of 4 videos. If another hoop is visible, mark the hoop by hand
  (`scripts/mark_hoop.py ... --videos-dir data/own_footage/raw --out data/own_footage/hoop_overrides.json`).
  The app's "tap the hoop" setup step exists for exactly this reason.
- **Keep one ball in play for test sessions.** Session 1 had two balls going at once; it became
  the dev session instead.
- **The current detector sees "balls" on heads (dark hair, caps) and legs (calf sleeves)** in this
  gym, often at 60–80% confidence. That is a detector problem, not a filming one, and the reason
  own-footage frames are needed for training.

## Privacy

If other people appear in the video, ask them first, and don't publish clips of them without
their OK. Raw footage stays on your PC; only short demo clips of yourself should go online.
