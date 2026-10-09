"""Shot detection and make/miss judging.

Pure logic, no computer-vision dependencies: it consumes one ball observation
per frame (or ``None`` when the ball was not seen) plus the hoop box, and emits
``ShotEvent`` objects. That keeps it unit-testable with synthetic trajectories.

How a shot is judged
--------------------
A small state machine (IDLE -> FLIGHT -> CONFIRM) follows the ball:

* IDLE: wait for the ball to climb above the rim line near the hoop ("arming").
* FLIGHT: collect the ball path. The attempt resolves when the ball
    - crosses the rim line going down  -> judged from where it crossed,
    - disappears (net / backboard occlusion) -> the projectile fit is
      extrapolated to the rim line, or
    - reverses upward just above the rim -> a bounce off the rim, i.e. a miss.
* CONFIRM: for a short window after the crossing, a ball that pops back above
  the rim line is demoted to a miss ("rim_out").

The crossing point is estimated by interpolating between the two frames that
straddle the rim line, or, when frames are missing, by solving a projectile
model fitted against *time* (x linear in t, y quadratic in t).
"""
from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import Deque, List, Optional, Tuple

from .config import ShotConfig
from .geometry import Box, FlightFit, dist, fit_flight
from .types import BallObs, Outcome, ShotEvent


class _State(Enum):
    IDLE = 0
    FLIGHT = 1
    CONFIRM = 2
    RATTLE = 3  # ball hit the rim; waiting to see where it ends up


def _rising_out_of_top(a: BallObs, b: BallObs) -> bool:
    """a -> b rises fast enough to be above the top of the frame within 0.3 s."""
    if b.t <= a.t:
        return False
    vy = (b.y - a.y) / (b.t - a.t)
    return vy < 0 and b.y + vy * 0.3 < 0


class _Pending:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.rim_out = False
        self.bounced_out = False


class ShotDetector:
    def __init__(self, cfg: Optional[ShotConfig] = None):
        self.cfg = cfg or ShotConfig()
        self.discarded = 0
        self.log: List[Tuple[float, str]] = []  # (time, decision) for debugging: arm, abort:<reason>, cross, ...
        self._buf: Deque[BallObs] = deque()
        self._hoop: Optional[Box] = None
        self._state = _State.IDLE
        self._cooldown_until = -math.inf
        self._hand_t: Optional[float] = None
        self._armed_t = 0.0
        self._apex: Optional[BallObs] = None
        self._desc_max: Optional[BallObs] = None
        self._pending: Optional[_Pending] = None
        self._count = 0

    # ---- public API -------------------------------------------------------------------

    def note_ball_in_hand(self, t: float) -> None:
        """Optional hint from pose estimation: the ball was still in the shooter's hand at ``t``."""
        self._hand_t = t

    def note_track_switch(self, t: float) -> None:
        """The tracker jumped to a different object: the points so far belong to something else."""
        if self._state is _State.IDLE:
            self._buf.clear()
            self.log.append((t, "track_switch"))

    def update(self, t: float, ball: Optional[BallObs], hoop: Optional[Box]) -> List[ShotEvent]:
        if hoop is not None:
            self._hoop = hoop
        if ball is not None:
            self._buf.append(ball)
        while self._buf and t - self._buf[0].t > self.cfg.history_s:
            self._buf.popleft()
        if self._hoop is None:
            return []
        if self._state is _State.IDLE:
            return self._idle(t, ball)
        if self._state is _State.FLIGHT:
            return self._flight(t, ball)
        if self._state is _State.RATTLE:
            return self._rattle(t, ball)
        return self._confirm(t, ball)

    # ---- helpers ----------------------------------------------------------------------

    def _exited_top(self) -> bool:
        """The last two observations were rising fast enough to be above the frame within 0.3 s."""
        if len(self._buf) < 2:
            return False
        return _rising_out_of_top(self._buf[-2], self._buf[-1])

    def _back_from_top(self, a: BallObs, b: BallObs, c: BallObs) -> bool:
        """a -> b left through the top edge and c, after a gap a high arc explains, is back above the rim.

        Such a gap is one flight, not two: the ball was only out of view above the frame.
        """
        return _rising_out_of_top(a, b) and c.t - b.t <= self.cfg.top_exit_wait_s and c.y < self._rim_y()

    def _u(self) -> float:
        """The length unit of every distance setting: hoop_aspect rim widths (the hoop box's width).

        It used to be the hoop box's height, which depends on how long the net is and how it is boxed
        (0.97-1.60 widths across our videos); the rim is detected consistently.
        """
        return self.cfg.hoop_aspect * self._hoop.w

    def _rim_y(self) -> float:
        h = self._hoop
        return h.y1 + self.cfg.rim_line_frac * self._u()

    def _reset(self, t: float, cooldown: float) -> None:
        self._state = _State.IDLE
        self._pending = None
        self._apex = self._desc_max = None
        self._buf.clear()
        self._cooldown_until = t + cooldown

    def _abort(self, t: float, reason: str) -> List[ShotEvent]:
        self.discarded += 1
        self.log.append((t, f"abort:{reason}"))
        self._reset(t, 0.3)
        return []

    def _flight_points(self, exclude_last: bool = False) -> List[BallObs]:
        """Most recent continuous run of observations, trimmed back to where the ball left the hand."""
        cfg, h = self.cfg, self._hoop
        pts = list(self._buf)[:-1] if exclude_last else list(self._buf)
        start = len(pts) - 1
        while start > 0 and (pts[start].t - pts[start - 1].t <= cfg.gap_max_s
                             or (start >= 2 and self._back_from_top(pts[start - 2], pts[start - 1], pts[start]))):
            start -= 1
        seg = pts[start:]
        apex = min(range(len(seg)), key=lambda i: seg[i].y)
        tol, still = cfg.rise_tol * self._u(), cfg.still_frac * self._u()
        j = apex
        while j > 0:
            a, b = seg[j - 1], seg[j]
            if a.y < b.y - tol:  # it was higher earlier, so this motion began later
                break
            if dist(b.x - a.x, b.y - a.y) < still:  # held, not flying
                break
            j -= 1
        return seg[j:]

    def _analyse(self, flight: List[BallObs]) -> Tuple[float, Optional[float], Optional[FlightFit], float]:
        """Release time, release angle, projectile fit and apex time for a flight."""
        t_apex = min(flight, key=lambda o: o.y).t
        release_t = flight[0].t
        hand = self._hand_t
        if hand is not None and flight[0].t - 0.3 <= hand <= t_apex:
            release_t = max(hand, flight[0].t)
        pts = [(o.t, o.x, o.y) for o in flight if o.t >= release_t - 1e-9]
        fit = fit_flight(pts, self.cfg.min_fit_points)
        # Outlier rejection: "held ball" points before the release and stray false detections
        # picked up mid-flight both sit off the arc. Drop the worst-fitting point and refit,
        # until the arc fits cleanly and curves the way gravity does (at most half the points are dropped).
        trim = self.cfg.trim_rms * self._u()
        max_drop, dropped = len(pts) // 2, 0
        while fit is not None and (fit.rms > trim or not fit.is_physical) and len(pts) > self.cfg.min_fit_points and dropped < max_drop:
            f = fit
            worst = max(range(len(pts)), key=lambda i: dist(pts[i][1] - f.x_at(pts[i][0]), pts[i][2] - f.y_at(pts[i][0])))
            pts.pop(worst)
            dropped += 1
            fit = fit_flight(pts, self.cfg.min_fit_points)
        if pts and pts[0][0] > release_t:
            release_t = pts[0][0]
        angle = None
        if fit is not None and fit.is_physical:
            vx, vy = fit.velocity_at(fit.t0)
            angle = math.degrees(math.atan2(-vy, abs(vx)))
        return release_t, angle, fit, t_apex

    def _build_pending(self, flight, release_t, rel_angle, fit, t_apex, t_cross, x_cross, method, entry, deadline):
        h = self._hoop
        return _Pending(
            t_release=release_t, release_angle=rel_angle, fit=fit, t_apex=t_apex,
            t_cross=t_cross, x_cross=x_cross, offset=(x_cross - h.cx) / (h.w / 2), method=method,
            entry_angle=entry, deadline=deadline, hoop=h,
            trajectory=[(o.t, o.x, o.y) for o in flight if o.t >= release_t - 1e-9],
        )

    # ---- states -----------------------------------------------------------------------

    def _idle(self, t: float, ball: Optional[BallObs]) -> List[ShotEvent]:
        if ball is None or t < self._cooldown_until:
            return []
        h = self._hoop
        if ball.y < self._rim_y() - self.cfg.arm_margin * self._u() and abs(ball.x - h.cx) <= self.cfg.max_dx * h.w:
            self._state = _State.FLIGHT
            self._armed_t = t
            self.log.append((t, "arm"))
            self._apex, self._desc_max = ball, None
        return []

    def _flight(self, t: float, ball: Optional[BallObs]) -> List[ShotEvent]:
        cfg, h, rim_y = self.cfg, self._hoop, self._rim_y()
        if t - self._armed_t > cfg.max_flight_s:
            return self._abort(t, "timeout")
        if ball is None:
            if self._buf and t - self._buf[-1].t > cfg.lost_s:
                if t - self._buf[-1].t <= cfg.top_exit_wait_s and self._exited_top():
                    return []  # high arc above the frame: wait for it to come back down
                return self._lost(t)
            return []
        if abs(ball.x - h.cx) > cfg.max_dx * h.w:
            return self._abort(t, "left_hoop_area")
        if len(self._buf) >= 2:
            prev = self._buf[-2]
            if prev.y < rim_y <= ball.y:
                return self._cross(t, prev, ball)
            came_down = len(self._buf) >= 3 and self._back_from_top(self._buf[-3], prev, ball)
            if ball.t - prev.t > cfg.interp_max_gap_s and ball.y < rim_y - cfg.reentry_margin * self._u() and not came_down:
                ev = self._reappeared_above(t, ball)
                if ev:
                    return ev

        if ball.y < self._apex.y:
            self._apex, self._desc_max = ball, None
        elif self._desc_max is None or ball.y > self._desc_max.y:
            self._desc_max = ball
        else:
            d = self._desc_max
            if (
                d.y - ball.y > cfg.rebound_px * self._u()
                and d.y >= rim_y - cfg.rebound_zone * self._u()
                and d.y - self._apex.y > 0.5 * self._u()
                and ball.y < rim_y
            ):
                return self._rebound(t)
        return []

    def _cross(self, t: float, prev: BallObs, cur: BallObs) -> List[ShotEvent]:
        cfg, h, rim_y = self.cfg, self._hoop, self._rim_y()
        flight = self._flight_points()
        if len(flight) < 2 or cur.t - flight[0].t < cfg.min_flight_s:
            return self._abort(t, "short_flight")
        release_t, rel_angle, fit, t_apex = self._analyse(flight)
        if fit is not None and (not fit.is_physical or fit.rms > cfg.max_fit_rms * self._u()):
            return self._abort(t, f"not_parabolic(rms={fit.rms / self._u():.2f}h,n={fit.n},physical={fit.is_physical})")

        gap = cur.t - prev.t
        t_c: Optional[float] = None
        x_c = 0.0
        method = "interpolated"
        if gap > cfg.interp_max_gap_s and fit is not None:
            tc = fit.time_at_y(rim_y)
            if tc is not None and prev.t - 1e-6 <= tc <= cur.t + 1e-6:
                t_c, x_c, method = tc, fit.x_at(tc), "fit"
        if t_c is None:
            if gap > cfg.gap_max_s:
                return self._abort(t, "gap_too_long")
            f = (rim_y - prev.y) / (cur.y - prev.y)
            t_c, x_c = prev.t + f * gap, prev.x + f * (cur.x - prev.x)

        if fit is not None and fit.is_physical:
            vx, vy = fit.velocity_at(t_c)
        else:
            vx, vy = cur.x - prev.x, cur.y - prev.y
        entry = math.degrees(math.atan2(vy, abs(vx)))

        pend = self._build_pending(flight, release_t, rel_angle, fit, t_apex, t_c, x_c, method, entry, t_c + cfg.confirm_s)
        if abs(pend.offset) > cfg.attempt_max_offset:
            return self._abort(t, f"too_far_from_hoop(offset={pend.offset:+.1f})")
        self._pending, self._state = pend, _State.CONFIRM
        return []

    def _lost(self, t: float) -> List[ShotEvent]:
        """Ball vanished above the rim line; judge from the fitted arc if it is trustworthy."""
        cfg, h, rim_y = self.cfg, self._hoop, self._rim_y()
        flight = self._flight_points()
        if len(flight) < cfg.min_fit_points or flight[-1].t - flight[0].t < cfg.min_flight_s:
            return self._abort(t, f"lost_too_few_points(n={len(flight)})")
        release_t, rel_angle, fit, t_apex = self._analyse(flight)
        if fit is None or not fit.is_physical or fit.rms > cfg.max_fit_rms * self._u():
            return self._abort(t, "lost_bad_fit")
        tc = fit.time_at_y(rim_y)
        last = flight[-1]
        if tc is None or tc < last.t or tc - last.t > cfg.max_extrap_s:
            return self._abort(t, "lost_no_crossing")
        vx, vy = fit.velocity_at(tc)
        entry = math.degrees(math.atan2(vy, abs(vx)))
        pend = self._build_pending(
            flight, release_t, rel_angle, fit, t_apex, tc, fit.x_at(tc), "extrapolated", entry,
            max(tc + cfg.confirm_s, t),
        )
        if abs(pend.offset) > cfg.attempt_max_offset:
            return self._abort(t, "lost_too_far_from_hoop")
        self._pending, self._state = pend, _State.CONFIRM
        return []

    def _reappeared_above(self, t: float, ball: BallObs) -> List[ShotEvent]:
        """Ball was hidden near the rim and came back *above* the rim line.

        If the arc fitted before the gap says it should already have dropped below
        the rim by now, the ball must have been deflected out: a miss.
        """
        cfg, h, rim_y = self.cfg, self._hoop, self._rim_y()
        flight = self._flight_points(exclude_last=True)
        if len(flight) < cfg.min_fit_points:
            return []
        release_t, rel_angle, fit, t_apex = self._analyse(flight)
        if fit is None or not fit.is_physical or fit.rms > cfg.max_fit_rms * self._u():
            return []
        tc = fit.time_at_y(rim_y)
        if tc is None or tc < flight[-1].t or tc >= ball.t - 0.02:
            return []
        offset = (fit.x_at(tc) - h.cx) / (h.w / 2)
        if abs(offset) > cfg.attempt_max_offset:
            return []
        pend = self._build_pending(flight, release_t, rel_angle, fit, t_apex, tc, fit.x_at(tc), "extrapolated", None, t)
        return self._start_rattle(t, pend, "rim_bounce")

    def _rebound(self, t: float) -> List[ShotEvent]:
        """Ball came down to the rim, then reversed upward without ever passing below it."""
        h, d = self._hoop, self._desc_max
        flight = [o for o in self._flight_points() if o.t <= d.t]  # approach only, not the bounce
        if len(flight) < 2:
            return self._abort(t, "rebound_too_few_points")
        release_t, rel_angle, fit, t_apex = self._analyse(flight)
        offset = (d.x - h.cx) / (h.w / 2)
        if abs(offset) > self.cfg.attempt_max_offset:
            return self._abort(t, "rebound_too_far_from_hoop")
        pend = self._build_pending(flight, release_t, rel_angle, fit, t_apex, d.t, d.x, "rebound", None, t)
        return self._start_rattle(t, pend, "rim_bounce")

    def _start_rattle(self, t: float, p: "_Pending", reason: str) -> List[ShotEvent]:
        """The ball touched the rim and came back up. Wait for it to settle before calling the shot,
        so a bounce that drops in is a make and the ball's later fall is not counted twice."""
        p.rattle_reason = reason
        p.rattle_start = t
        p.rattle_deadline = t + self.cfg.rattle_s
        p.rattles = getattr(p, "rattles", 0) + 1
        self._pending, self._state = p, _State.RATTLE
        self.log.append((t, f"rattle:{reason}"))
        return []

    def _rattle(self, t: float, ball: Optional[BallObs]) -> List[ShotEvent]:
        cfg, h, p, rim_y = self.cfg, self._hoop, self._pending, self._rim_y()
        if ball is not None and len(self._buf) >= 2:
            prev = self._buf[-2]
            if prev.t >= p.rattle_start and prev.y < rim_y <= ball.y:  # coming down past the rim line
                f = (rim_y - prev.y) / (ball.y - prev.y)
                tc = prev.t + f * (ball.t - prev.t)
                off = (prev.x + f * (ball.x - prev.x) - h.cx) / (h.w / 2)
                if abs(off) <= cfg.make_halfwidth:
                    # Dropped in after the rim: confirm like any crossing (pop-out, net braking).
                    p.trajectory = p.trajectory + [(o.t, o.x, o.y) for o in self._buf if o.t > p.trajectory[-1][0]]
                    p.t_cross, p.offset, p.method, p.rim_out = tc, off, "rattle", False
                    p.x_cross, p.bounced_out = prev.x + f * (ball.x - prev.x), False
                    p.deadline = tc + cfg.confirm_s
                    self._state = _State.CONFIRM
                    return []
                return self._rattle_miss(t, tc)
            if abs(ball.x - h.cx) > cfg.rattle_max_dx * h.w:
                return self._rattle_miss(t, p.t_cross)
        if t >= p.rattle_deadline:
            return self._rattle_miss(t, p.t_cross)
        return []

    def _rattle_miss(self, t: float, t_cross: float) -> List[ShotEvent]:
        p = self._pending
        ev = self._event(
            Outcome.MISSED, p.rattle_reason, p.method, p.t_release, p.t_apex, t_cross, p.offset, p.hoop,
            p.trajectory, p.fit, p.release_angle, p.entry_angle,
        )
        self._reset(t, self.cfg.cooldown_s)
        return [ev]

    def _confirm(self, t: float, ball: Optional[BallObs]) -> List[ShotEvent]:
        cfg, h, p = self.cfg, self._hoop, self._pending
        if ball is not None and ball.t > p.t_cross:
            if ball.y < self._rim_y() - cfg.reentry_margin * self._u() and abs(ball.x - h.cx) <= 1.5 * h.w:
                p.rim_out = True
                if abs(p.offset) <= cfg.make_halfwidth and getattr(p, "rattles", 0) < cfg.max_rattles:
                    return self._start_rattle(t, p, "rim_out")
                return self._finish(t)
            if ball.y <= h.y2 and self._moved_back(p, ball):
                p.bounced_out = True
        if t >= p.deadline:
            return self._finish(t)
        return []

    def _vertical_speed(self, pts: List[Tuple[float, float]]) -> Optional[float]:
        """Least-squares downward speed of (t, y) points, in hoop heights per second."""
        s = self._slope(pts)
        return None if s is None else s / self._u()

    def _slope(self, pts: List[Tuple[float, float]]) -> Optional[float]:
        """Least-squares d(value)/dt of (t, value) points, in pixels per second."""
        if len(pts) < 2:
            return None
        # Plain loops, not sum() / ** 2: the phone app's port must reproduce these bits exactly.
        st = sy = 0.0
        for q in pts:
            st += q[0]
            sy += q[1]
        mt, my = st / len(pts), sy / len(pts)
        den = num = 0.0
        for q in pts:
            dt = q[0] - mt
            den += dt * dt
            num += dt * (q[1] - my)
        return None if den == 0 else num / den

    def _moved_back(self, p: _Pending, ball: BallObs) -> bool:
        """The ball is back towards where it came from by ``back_out_frac`` hoop widths since the crossing."""
        pre = [q for q in p.trajectory if p.t_cross - 0.15 <= q[0] < p.t_cross]
        if len(pre) < 2 or pre[-1][1] == pre[0][1]:
            return False
        back = (p.x_cross - ball.x) if pre[-1][1] > pre[0][1] else (ball.x - p.x_cross)
        return back / self._hoop.w >= self.cfg.back_out_frac

    def _fell_past_rim(self, p: _Pending) -> bool:
        """True when the ball is clearly seen falling after the crossing without being braked.

        A ball that goes through the net is slowed by it (or disappears into it). A ball that
        drops just in front of / behind the rim (it looks centred from a side camera) keeps
        falling at least as fast as it arrived.
        """
        cfg = self.cfg
        pre = [(q[0], q[2]) for q in p.trajectory if p.t_cross - 0.15 <= q[0] < p.t_cross]
        post = [(o.t, o.y) for o in self._buf if p.t_cross < o.t <= p.t_cross + cfg.net_window_s]
        if len(post) < cfg.net_min_points or post[-1][0] - post[0][0] < 0.12:
            return False
        v_pre, v_post = self._vertical_speed(pre), self._vertical_speed(post)
        min_pre = cfg.net_min_speed if p.method == "rattle" else 0.0
        if v_pre is None or v_post is None or v_pre <= min_pre:
            return False
        if not (v_post >= cfg.net_min_speed and v_post >= cfg.net_brake_ratio * v_pre):
            return False
        # Caught by the net: from the side, a shot reaches the rim moving fast sideways. The net stops that;
        # a ball falling past the rim keeps it. So a ball that lost most of its sideways speed went in.
        x_pre = [(q[0], q[1]) for q in p.trajectory if p.t_cross - 0.15 <= q[0] < p.t_cross]
        x_post = [(o.t, o.x) for o in self._buf if p.t_cross < o.t <= p.t_cross + cfg.net_window_s]
        sx_pre, sx_post = self._slope(x_pre), self._slope(x_post)
        if sx_pre is not None and sx_post is not None and abs(sx_pre) / self._hoop.w >= cfg.net_catch_min_vx:
            if abs(sx_post) <= cfg.net_catch_ratio * abs(sx_pre):
                return False
        return True

    def _finish(self, t: float) -> List[ShotEvent]:
        p, cfg = self._pending, self.cfg
        through = abs(p.offset) <= cfg.make_halfwidth
        if through and p.rim_out:
            outcome, reason = Outcome.MISSED, "rim_out"
        elif through and p.bounced_out:
            outcome, reason = Outcome.MISSED, "rim_bounce"
        elif through and self._fell_past_rim(p):
            outcome, reason = Outcome.MISSED, "fell_past_rim"
        elif through:
            outcome, reason = Outcome.MADE, "rattled_in" if p.method == "rattle" else "through_hoop"
        else:
            outcome, reason = Outcome.MISSED, "off_target"
        ev = self._event(
            outcome, reason, p.method, p.t_release, p.t_apex, p.t_cross, p.offset, p.hoop,
            p.trajectory, p.fit, p.release_angle, p.entry_angle,
        )
        self._reset(t, cfg.cooldown_s)
        return [ev]

    def _event(self, outcome, reason, method, t_release, t_apex, t_cross, offset, hoop, traj, fit, rel, entry):
        self._count += 1
        self.log.append((t_cross, f"shot:{outcome.value}:{reason}"))
        return ShotEvent(
            index=self._count, outcome=outcome, reason=reason, method=method,
            t_release=t_release, t_apex=t_apex, t_cross=t_cross, cross_offset=offset,
            hoop=hoop, trajectory=traj, fit=fit, release_angle_deg=rel, entry_angle_deg=entry,
        )
