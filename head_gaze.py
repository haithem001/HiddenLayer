"""
head_gaze.py — Humanoid gaze / look-around state machine.

The walker ALWAYS starts looking hard-left (offset = -SEARCH_HALF_ARC)
and sweeps rightward.  This guarantees he has to scan across before
the gaze ray reaches the target, even if he spawns facing it directly.
"""

import math
import time as _time

# ── Tunables ──────────────────────────────────────────────────────────

SEARCH_HALF_ARC = math.radians(170)   # sweep ±170° — nearly full circle
SEARCH_SPEED    = 0.55                # rad/s — slow, readable sweep (~10s full pass)
SWEEP_PAUSE     = 0.9                 # seconds pause at each extreme

# Very narrow FOV — must almost directly face the target to detect it
FOV_SEARCH_HALF = math.radians(10)    # ±10° = 20° total cone
FOV_TRACK_HALF  = math.radians(6)     # narrows once tracking

LOCK_THRESHOLD  = math.radians(4)     # head centred → start pursuing
LOST_HALF_ANGLE = math.radians(45)    # target this far from body axis → lost

TRACK_SPEED     = 1.8                 # rad/s toward target once detected
FOV_LERP_SPEED  = 2.0


class HeadGaze:
    SEARCHING = 'searching'
    TRACKING  = 'tracking'
    PURSUING  = 'pursuing'

    def __init__(self):
        self.state          = self.SEARCHING
        self.world_yaw      = 0.0
        self.fov_half_angle = FOV_SEARCH_HALF
        self.detected       = False

        # Always start at the leftmost extreme so scan must travel across
        self._sweep_offset  = -SEARCH_HALF_ARC
        self._sweep_dir     = 1.0          # start sweeping right
        self._pause_until   = 0.0

        self._last_wall_t   = _time.perf_counter()

    def update(self, torso_x, torso_z, walk_dx, walk_dz,
               target_x, target_z, dt=None):
        now = _time.perf_counter()
        dt  = min(now - self._last_wall_t, 0.1)
        self._last_wall_t = now

        forward_yaw    = math.atan2(walk_dx, -walk_dz)
        dx             = target_x - torso_x
        dz             = target_z - torso_z
        target_yaw     = math.atan2(dx, -dz)
        gaze_to_target = _angle_diff(target_yaw, self.world_yaw)
        fwd_to_target  = _angle_diff(target_yaw, forward_yaw)

        if self.state == self.SEARCHING:
            self._do_search(dt, forward_yaw, now)
            if abs(gaze_to_target) < self.fov_half_angle:
                self.detected = True
                self.state    = self.TRACKING
            self.fov_half_angle = _lerp(
                self.fov_half_angle, FOV_SEARCH_HALF, FOV_LERP_SPEED * dt)

        elif self.state == self.TRACKING:
            self._do_track(dt, target_yaw)
            if abs(gaze_to_target) < LOCK_THRESHOLD:
                self.state = self.PURSUING
            self.fov_half_angle = _lerp(
                self.fov_half_angle, FOV_TRACK_HALF, FOV_LERP_SPEED * dt)

        elif self.state == self.PURSUING:
            self._do_track(dt, target_yaw)
            self.fov_half_angle = _lerp(
                self.fov_half_angle, FOV_TRACK_HALF, FOV_LERP_SPEED * dt)
            if abs(fwd_to_target) > LOST_HALF_ANGLE:
                self.detected = False
                self.state    = self.SEARCHING

    def reset(self):
        """Reset to hard-left so the next scan always sweeps across."""
        self.state          = self.SEARCHING
        self.detected       = False
        self._sweep_offset  = -SEARCH_HALF_ARC   # ← always start hard-left
        self._sweep_dir     = 1.0                 # sweep rightward
        self._pause_until   = 0.0
        self.fov_half_angle = FOV_SEARCH_HALF
        self._last_wall_t   = _time.perf_counter()

    def _do_search(self, dt, forward_yaw, now):
        if now < self._pause_until:
            self.world_yaw = forward_yaw + self._sweep_offset
            return

        self._sweep_offset += SEARCH_SPEED * dt * self._sweep_dir

        if self._sweep_offset >= SEARCH_HALF_ARC:
            self._sweep_offset = SEARCH_HALF_ARC
            self._sweep_dir    = -1.0
            self._pause_until  = now + SWEEP_PAUSE
        elif self._sweep_offset <= -SEARCH_HALF_ARC:
            self._sweep_offset = -SEARCH_HALF_ARC
            self._sweep_dir    = 1.0
            self._pause_until  = now + SWEEP_PAUSE

        self.world_yaw = forward_yaw + self._sweep_offset

    def _do_track(self, dt, target_yaw):
        diff     = _angle_diff(target_yaw, self.world_yaw)
        max_step = TRACK_SPEED * dt
        if abs(diff) <= max_step:
            self.world_yaw = target_yaw
        else:
            self.world_yaw += math.copysign(max_step, diff)
        self.world_yaw = _wrap(self.world_yaw)


def _angle_diff(a, b):
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d

def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t
