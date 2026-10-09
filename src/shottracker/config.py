"""Tunable thresholds for shot detection.

Distances are expressed relative to the detected hoop box (``h`` = height,
``w`` = width) so the same settings work at different resolutions and zoom
levels. Times are in seconds so they do not depend on the video frame rate.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ShotConfig:
    # Where the "rim line" sits inside the hoop box, as a fraction of box height from the top.
    # The hotshot dataset labels the rim AND the hanging net (boxes ~0.57 w/h), so the rim
    # itself is near the top of the box. Re-check this whenever the hoop labels change.
    rim_line_frac: float = 0.15

    # Arming: the ball must get this many hoop-heights above the rim line, and be
    # within this many hoop-widths of the hoop horizontally, to start a shot attempt.
    arm_margin: float = 0.15  # was 0.5; a low camera close to the hoop (own dev session 1) sees many arcs peak just above the rim
    max_dx: float = 5.0

    # Flight bookkeeping.
    history_s: float = 2.5
    gap_max_s: float = 0.50  # longest run of missed frames still treated as one continuous flight
    max_flight_s: float = 3.5
    lost_s: float = 0.35
    top_exit_wait_s: float = 1.5  # a ball that left through the top edge may be out of view this long
    max_extrap_s: float = 0.8
    min_fit_points: int = 5
    max_fit_rms: float = 0.6  # of hoop height; rejects non-parabolic "flights"
    trim_rms: float = 0.2  # of hoop height; leading points are dropped until the fit is this tight
    rise_tol: float = 0.15  # of hoop height; wobble tolerated while walking back to the release
    still_frac: float = 0.08  # of hoop height; below this per-frame motion the ball is "held"
    interp_max_gap_s: float = 0.10
    min_flight_s: float = 0.2

    # Judging.
    make_halfwidth: float = 0.75  # |offset| (hoop half-widths) still counted as through the hoop; dev makes reached ~0.74
    attempt_max_offset: float = 8.0  # half-widths; long misses can land 2-3 hoop widths past the rim
    confirm_s: float = 0.45
    # Net braking: a provisional make is demoted when the ball is seen for a while after the
    # crossing and is not slowed at all (no net in the way). Ratio 1.0 = 'no braking'.
    net_window_s: float = 0.25
    net_min_points: int = 4
    net_min_speed: float = 3.0  # hoop heights / s
    # was 1.0; own footage: some makes reached 1.1-1.35 through the net. 1.3 keeps the simulator's fall-past
    # detection intact at realistic noise (1.5 gained two dev shots but weakened it); see RESULTS.md section 7.
    net_brake_ratio: float = 1.3
    # Bounced back out: after the crossing, a ball that moves back towards where it came from by this many
    # hoop widths while still at net height hit the back of the rim or the board (dev makes reached 0.62).
    back_out_frac: float = 0.75
    # Caught by the net: a make is not demoted as 'fell past the rim' when the ball reached the rim moving sideways
    # at least net_catch_min_vx hoop widths/s and kept at most net_catch_ratio of that speed after the crossing.
    # Dev data: wrongly demoted makes came in at 6.5-8 widths/s and kept -0.2..0.35; balls really falling past kept ~1.3.
    net_catch_min_vx: float = 3.0
    net_catch_ratio: float = 0.5
    reentry_margin: float = 0.25  # of hoop height above rim line => ball came back out
    rebound_px: float = 0.35  # of hoop height reversal that signals a bounce off the rim
    rebound_zone: float = 1.0  # of hoop height; reversal must happen this close to the rim line
    cooldown_s: float = 0.7
    # Rim contact: wait this long for the ball to settle (drop in, or fall away) before calling it.
    rattle_s: float = 2.0
    rattle_max_dx: float = 3.0  # hoop widths; farther than this the ball has left the hoop
    max_rattles: int = 2
