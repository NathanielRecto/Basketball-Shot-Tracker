"""Stress-test the make/miss logic on simulated shots.

This measures the *judging logic* given imperfect ball detections. It says
nothing about the detector itself; real-video accuracy needs a labelled clip set.

    python scripts/eval_synthetic.py --n 500
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shottracker.shot_logic import ShotDetector  # noqa: E402
from shottracker.sim import DEFAULT_HOOP, ShotSpec, simulate_stream  # noqa: E402
from shottracker.types import Outcome  # noqa: E402


def one_trial(rng, seed, noise, dropout, fps):
    kind = rng.choice(["make", "miss", "rim_out", "rim_bounce", "fall_past", "rattle_in"], p=[0.35, 0.25, 0.1, 0.1, 0.1, 0.1])
    side = rng.choice([-1, 1])
    behaviour = kind if kind in ("rim_out", "rim_bounce", "fall_past", "rattle_in") else "normal"
    offset = {
        "make": rng.uniform(0.0, 0.35), "miss": rng.uniform(1.0, 2.2),
        "rim_out": rng.uniform(0.0, 0.3), "rim_bounce": rng.uniform(0.0, 0.6), "fall_past": rng.uniform(0.0, 0.4), "rattle_in": rng.uniform(0.0, 0.3),
    }[kind] * side
    spec = ShotSpec(
        offset=float(offset), behaviour=behaviour,
        apex_above_rim=float(rng.uniform(70, 160)),  # keeps the apex inside the 720 px frame
        release=(float(rng.uniform(150, 450)), float(rng.uniform(330, 430))),
        occlude=bool(rng.integers(0, 2)),
    )
    det, events = ShotDetector(), []
    for t, obs in simulate_stream(DEFAULT_HOOP, [spec], fps=fps, noise_px=noise, dropout=dropout, seed=seed):
        events += det.update(t, obs, DEFAULT_HOOP)
    want = Outcome.MADE if kind in ("make", "rattle_in") else Outcome.MISSED
    return kind, len(events) == 1 and events[0].outcome is want, len(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    print(f"{'noise px':>8} {'dropout':>8} {'fps':>4} | {'correct':>8} | by type")
    for noise, dropout, fps in [(0, 0, 30), (1.5, 0.1, 30), (3.0, 0.2, 30), (3.0, 0.2, 60), (5.0, 0.3, 30)]:
        rng = np.random.default_rng(0)
        tally = {}
        for i in range(args.n):
            kind, ok, _ = one_trial(rng, i, noise, dropout, fps)
            c, n = tally.get(kind, (0, 0))
            tally[kind] = (c + ok, n + 1)
        total_c = sum(c for c, _ in tally.values())
        by = "  ".join(f"{k}:{100 * c / n:.0f}%" for k, (c, n) in sorted(tally.items()))
        print(f"{noise:8.1f} {dropout:8.2f} {fps:4d} | {100 * total_c / args.n:7.1f}% | {by}")


if __name__ == "__main__":
    main()
