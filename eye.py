"""
eye.py — Visual-field HUD for the walker's head.

Pupil position logic (per state):
  searching  → pupil drifts across D in sync with the head sweep angle.
               sweep_x/y (normalised -1..1) comes from HeadGaze._sweep_offset
               so the pupil mirrors the physical head rotation.
  tracking   → pupil moves toward the detected target position inside D.
               Position comes from retinal detection (retinal_x/y) mapped
               into D-pixel space.  No angular math.
  pursuing   → pupil locked on target, contracts with proximity + ambient lux.

Pupil radius:
  Driven by LightModel.pupil_dilation (0=contracted/bright, 1=dilated/dark)
  with a secondary squeeze for close targets in pursuing state.
"""

import math


# Angular extents of each zone (must match head_gaze.py constants)
D_HALF_YAW   = math.radians(145)
D_HALF_PITCH = math.radians(72)
C_HALF_YAW   = D_HALF_YAW   * 0.70
C_HALF_PITCH = D_HALF_PITCH * 0.70
B_HALF_YAW   = math.radians(10)
B_HALF_PITCH = math.radians(10)


class Eye:
    def __init__(self, D_width: float, D_height: float):
        self.D_width  = D_width
        self.D_height = D_height
        self.C_width  = D_width  * (C_HALF_YAW   / D_HALF_YAW)
        self.C_height = D_height * (C_HALF_PITCH  / D_HALF_PITCH)
        self.B_width  = D_width  * (B_HALF_YAW   / D_HALF_YAW)
        self.B_height = D_height * (B_HALF_PITCH  / D_HALF_PITCH)

        self.min_radius = min(self.B_width, self.B_height) * 0.12
        self.max_radius = min(self.B_width, self.B_height) * 0.50

        # Pupil state
        self.pupil_center_X = 0.0
        self.pupil_center_Y = 0.0
        self.pupil_radius   = self.max_radius
        self.pupil_depth_Z  = 0.0

        # Gaze state
        self.gaze_state    = 'searching'
        self.target_dist   = float('inf')
        self.target_in_fov = False

        # Light
        self.ambient_lux    = 0.5
        self.pupil_dilation = 0.5
        self.vignette_alpha = 80

        # Sweep position in normalised [-1, 1] (for Render.py sweep cursor)
        self.sweep_x = 0.0
        self.sweep_y = 0.0

    # ── Primary update — retinal pipeline ─────────────────────────────

    def update_from_retina(self,
                           retinal_x:      float,
                           retinal_y:      float,
                           detected:       bool,
                           gaze_state:     str,
                           target_dist:    float,
                           lux:            float,
                           pupil_dilation: float,
                           vignette_alpha: int = 80,
                           sweep_offset:   float = 0.0):
        """
        Update from retinal detection result.

        Parameters
        ----------
        retinal_x, retinal_y : centre-of-mass of detected pixels [-1, 1]
        detected             : True when target colour found in retinal image
        gaze_state           : from HeadGaze
        target_dist          : metres to target
        lux                  : ambient brightness from LightModel
        pupil_dilation       : 0=contracted ... 1=dilated from LightModel
        vignette_alpha       : edge darkness (0-255)
        sweep_offset         : HeadGaze._sweep_offset in radians
                               (used to animate pupil during searching)
        """
        self.gaze_state     = gaze_state
        self.target_dist    = target_dist
        self.ambient_lux    = lux
        self.pupil_dilation = pupil_dilation
        self.vignette_alpha = vignette_alpha
        self.target_in_fov  = detected

        if gaze_state == 'searching':
            # Pupil sweeps with the head across D
            SEARCH_HALF_ARC = math.radians(170)
            norm  = max(-1.0, min(1.0, sweep_offset / SEARCH_HALF_ARC))
            raw_x = norm * (self.D_width  / 2.0)
            raw_y = 0.0
            self.sweep_x = norm
            self.sweep_y = 0.0
            depth_z = 0.0

        elif gaze_state == 'tracking':
            # Pupil moves toward the detected target retinal position
            raw_x = retinal_x * (self.D_width  / 2.0)
            raw_y = retinal_y * (self.D_height / 2.0)
            self.sweep_x = retinal_x
            self.sweep_y = retinal_y
            depth_z = 0.35

        else:  # pursuing
            # Pupil locked on detected position, contracts as target nears
            raw_x = retinal_x * (self.D_width  / 2.0)
            raw_y = retinal_y * (self.D_height / 2.0)
            self.sweep_x = retinal_x
            self.sweep_y = retinal_y
            depth_z = 1.0 - max(0.0, min(1.0, (target_dist - 0.5) / 14.5))

        self.pupil_depth_Z = depth_z
        self._update_radius()
        self.pupil_center_X, self.pupil_center_Y = self._clamp(raw_x, raw_y)

    # ── Fallback — angular math (overview / non-FPV mode) ─────────────

    def update_from_walker(self,
                           head_yaw:      float,
                           head_pitch:    float,
                           target_yaw:    float,
                           target_pitch:  float,
                           gaze_state:    str,
                           target_dist:   float,
                           fov_half:      float,
                           sweep_offset:  float = 0.0):
        """
        Angular-math update used in overview mode (no retinal render).
        Pupil follows the head sweep or the target angular offset.
        """
        self.gaze_state  = gaze_state
        self.target_dist = target_dist

        delta_yaw   = _angle_diff(target_yaw,   head_yaw)
        delta_pitch = _angle_diff(target_pitch,  head_pitch)
        in_fov = (abs(delta_yaw) < fov_half and abs(delta_pitch) < fov_half)
        self.target_in_fov = in_fov

        if gaze_state == 'searching':
            SEARCH_HALF_ARC = math.radians(170)
            norm  = max(-1.0, min(1.0, sweep_offset / SEARCH_HALF_ARC))
            raw_x = norm * (self.D_width  / 2.0)
            raw_y = 0.0
            self.sweep_x = norm
            self.sweep_y = 0.0
            depth_z = 0.0
        else:
            px_per_rad_x = (self.D_width  / 2.0) / D_HALF_YAW
            px_per_rad_y = (self.D_height / 2.0) / D_HALF_PITCH
            raw_x = delta_yaw   * px_per_rad_x
            raw_y = delta_pitch * px_per_rad_y
            self.sweep_x = max(-1.0, min(1.0, delta_yaw   / D_HALF_YAW))
            self.sweep_y = max(-1.0, min(1.0, delta_pitch / D_HALF_PITCH))
            depth_z = 0.35 if gaze_state == 'tracking' else (
                    1.0 - max(0.0, min(1.0, (target_dist - 0.5) / 14.5))
            )

        self.pupil_depth_Z = depth_z
        self._update_radius()
        self.pupil_center_X, self.pupil_center_Y = self._clamp(raw_x, raw_y)

    # ── Geometry helpers ──────────────────────────────────────────────

    def _update_radius(self):
        dil    = max(0.0, min(1.0, self.pupil_dilation))
        base_r = self.min_radius + dil * (self.max_radius - self.min_radius)
        squeeze = max(0.0, min(1.0, self.pupil_depth_Z)) * 0.22
        self.pupil_radius = base_r * (1.0 - squeeze)

    def _clamp(self, x: float, y: float):
        r     = self.pupil_radius
        lim_x = self.D_width  / 2.0 - r
        lim_y = self.D_height / 2.0 - r
        return (max(-lim_x, min(lim_x, x)),
                max(-lim_y, min(lim_y, y)))

    # ── Snapshot ──────────────────────────────────────────────────────

    def get_render_data(self) -> dict:
        return {
            'cx':             self.pupil_center_X,
            'cy':             self.pupil_center_Y,
            'radius':         self.pupil_radius,
            'depth_z':        self.pupil_depth_Z,
            'state':          self.gaze_state,
            'target_dist':    self.target_dist,
            'target_in_fov':  self.target_in_fov,
            'ambient_lux':    self.ambient_lux,
            'pupil_dilation': self.pupil_dilation,
            'vignette_alpha': self.vignette_alpha,
            'sweep_x':        self.sweep_x,
            'sweep_y':        self.sweep_y,
            'D_w': self.D_width,  'D_h': self.D_height,
            'C_w': self.C_width,  'C_h': self.C_height,
            'B_w': self.B_width,  'B_h': self.B_height,
        }


def _angle_diff(a: float, b: float) -> float:
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d
