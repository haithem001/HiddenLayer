import math
import numpy as np


# ── Joint geometry ────────────────────────────────────────────────────
THIGH_LEN = 0.45
LEG_LEN   = 0.40
FOOT_LEN  = 0.20

TORQUE_SCALE = 10.0

HEALTHY_Y_MIN   = 0.8
HEALTHY_Y_MAX   = 2.0
HEALTHY_ANG_MAX = 1.2

# ── Reward weights ────────────────────────────────────────────────────
APPROACH_WEIGHT  = 4.0
ALIVE_BONUS      = 0.5
CTRL_COST_WEIGHT = 3e-4

PROXIMITY_RADIUS = 3.0
PROXIMITY_BONUS  = 2.0
REACH_RADIUS     = 1.5
REACH_BONUS      = 50.0

FALL_PENALTY     = -3.0
MAX_STEPS        = 6000

ANTI_PHASE_WEIGHT = 0.8
FOOT_LIFT_WEIGHT  = 0.4
FOOT_LIFT_MIN_Y   = 0.08
HIP_OSC_WEIGHT    = 0.5
HIP_OSC_SCALE     = 15.0


class WalkerEnv:

    class _ActionSpace:
        shape = (6,)
        def sample(self):
            return np.random.uniform(-1, 1, 6).astype(np.float32)

    action_space            = _ActionSpace()
    observation_space_size  = 17

    def __init__(self, points, velocities, constraints, joint_names, physics, target):
        self.joint_names = joint_names
        self.physics     = physics
        self.target      = target
        self.constraints = constraints
        self.points      = points
        self.velocities  = velocities
        self.initial_points     = [p.copy() for p in points]
        self.initial_velocities = [v.copy() for v in velocities]

        physics.torso_index   = joint_names['torso']
        physics.torso_fixed_y = points[joint_names['torso']]['y']
        physics.foot_indices  = [joint_names['r_foot'], joint_names['l_foot']]

        self.step_count        = 0
        self.fell              = False
        self.reach_bonus_given = False
        self.prev_torso_x      = points[joint_names['torso']]['x']
        self.angles            = np.zeros(6, dtype=np.float64)
        self.angular_velocities = np.zeros(6, dtype=np.float64)
        self.previous_angles   = np.zeros(6, dtype=np.float64)

        self._prev_dist = self._xz_dist()

    # ── Walk-direction helpers ─────────────────────────────────────────

    def _fwd(self):
        """Unit forward vector (walk direction) as (fx, fz)."""
        return self.physics.walk_dx, self.physics.walk_dz

    def _local_delta(self, pi, ci):
        """
        Vector from point pi to ci, projected into the walker's local frame.
        Returns (forward_component, up_component).
        Forward = walk_dx/walk_dz axis.  Up = world Y.
        Lateral component is intentionally discarded (2-D walker).
        """
        fx, fz = self._fwd()
        dx_w = self.points[ci]['x'] - self.points[pi]['x']
        dy_w = self.points[ci]['y'] - self.points[pi]['y']
        dz_w = self.points[ci].get('z', 0.0) - self.points[pi].get('z', 0.0)

        # Project world XZ offset onto forward axis
        fwd_component = dx_w * fx + dz_w * fz
        return fwd_component, dy_w

    # ── Horizontal distance (x,z plane) ──────────────────────────────

    def _xz_dist(self):
        ti = self.joint_names['torso']
        tx = self.points[ti]['x']
        tz = self.points[ti].get('z', 0.0)
        return math.sqrt(
            (tx - float(self.target[0]))**2 +
            (tz - float(self.target[2]))**2
        )

    def on_target_changed(self):
        self._prev_dist        = self._xz_dist()
        self.reach_bonus_given = False

    # ── Public API ────────────────────────────────────────────────────

    def reset(self):
        for i in range(len(self.points)):
            self.points[i]     = self.initial_points[i].copy()
            self.velocities[i] = self.initial_velocities[i].copy()

        noise = 5e-3
        for p in self.points:
            p['x'] += np.random.uniform(-noise, noise)
            p['y'] += np.random.uniform(-noise, noise)
        for v in self.velocities:
            v['vx'] += np.random.uniform(-noise, noise)
            v['vy'] += np.random.uniform(-noise, noise)

        self.step_count        = 0
        self.fell              = False
        self.reach_bonus_given = False
        self.prev_torso_x      = self.points[self.joint_names['torso']]['x']
        self.previous_angles   = self.compute_angles()
        self.angles            = self.previous_angles.copy()
        self.angular_velocities[:] = 0.0
        self._prev_dist        = self._xz_dist()

        return self.get_observator()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        self.apply_torques(action)
        self.points, self.velocities = self.physics.apply_physics_step(
            self.points, self.velocities, self.constraints
        )

        new_angles = self.compute_angles()
        dt = self.physics.dt
        self.angular_velocities = (new_angles - self.previous_angles) / dt
        self.previous_angles    = new_angles
        self.angles             = new_angles

        healthy   = self.is_healthy()
        ti        = self.joint_names['torso']
        torso_x   = self.points[ti]['x']
        torso_z   = self.points[ti].get('z', 0.0)
        torso_vx  = self.velocities[ti]['vx']

        curr_dist    = math.sqrt(
            (torso_x - float(self.target[0]))**2 +
            (torso_z - float(self.target[2]))**2
        )
        control_cost = CTRL_COST_WEIGHT * float(np.dot(action, action))

        approach_reward = 0.0
        proximity_bonus = 0.0
        reach_bonus     = 0.0
        gait_reward     = 0.0

        if healthy:
            approach_reward = APPROACH_WEIGHT * (self._prev_dist - curr_dist) / dt
            alive = ALIVE_BONUS

            if curr_dist < PROXIMITY_RADIUS:
                proximity_bonus = PROXIMITY_BONUS

            if curr_dist < REACH_RADIUS and not self.reach_bonus_given:
                reach_bonus            = REACH_BONUS
                self.reach_bonus_given = True

            jn = self.joint_names
            r_foot_y = self.points[jn['r_foot']]['y']
            l_foot_y = self.points[jn['l_foot']]['y']
            omega_r  = self.angular_velocities[0]
            omega_l  = self.angular_velocities[3]

            osc_raw           = -omega_r * omega_l
            anti_phase_reward = ANTI_PHASE_WEIGHT * min(max(osc_raw / HIP_OSC_SCALE, 0.0), 1.0)

            r_up = r_foot_y > FOOT_LIFT_MIN_Y
            l_up = l_foot_y > FOOT_LIFT_MIN_Y
            foot_lift_reward = FOOT_LIFT_WEIGHT if (r_up != l_up) else 0.0

            swing_mag      = (abs(omega_r) + abs(omega_l)) / 2.0
            hip_osc_reward = HIP_OSC_WEIGHT * min(swing_mag / HIP_OSC_SCALE, 1.0)

            gait_reward = anti_phase_reward + foot_lift_reward + hip_osc_reward

            reward = approach_reward + alive + proximity_bonus + reach_bonus + gait_reward - control_cost

        else:
            if not self.fell:
                self.fell = True
                reward = FALL_PENALTY
            else:
                reward = 0.0

        self._prev_dist   = curr_dist
        self.prev_torso_x = torso_x
        self.step_count  += 1
        done = self.step_count >= MAX_STEPS

        return self.get_observator(), reward, done, {
            'reward_approach':    approach_reward,
            'reward_ctrl':        -control_cost,
            'reward_gait':        gait_reward,
            'distance_to_target': curr_dist,
            'x_position':         torso_x,
            'x_velocity':         torso_vx,
            'is_healthy':         healthy,
        }

    def get_observator(self):
        jn  = self.joint_names
        pts = self.points
        vel = self.velocities
        ti  = jn['torso']

        # Forward velocity = component of torso velocity along walk direction
        fx, fz    = self._fwd()
        vx_w      = vel[ti]['vx']
        # physics drives x via vx*walk_dx and z via vx*walk_dz,
        # so vx IS already the scalar along the walk axis — use directly.
        fwd_vel   = vx_w          # scalar: positive = moving toward target
        vy        = vel[ti]['vy']

        return np.array([
            pts[ti]['y'],
            self.get_torso_pitch(),
            self.angles[0], self.angles[1], self.angles[2],
            self.angles[3], self.angles[4], self.angles[5],
            fwd_vel,          # was vel[ti]['vx'] — now explicitly forward speed
            vy,
            self.get_torso_pitch_rate(),
            self.angular_velocities[0], self.angular_velocities[1],
            self.angular_velocities[2], self.angular_velocities[3],
            self.angular_velocities[4], self.angular_velocities[5],
        ], dtype=np.float64)

    def is_healthy(self):
        y = self.points[self.joint_names['torso']]['y']
        return (HEALTHY_Y_MIN <= y <= HEALTHY_Y_MAX and
                abs(self.get_torso_pitch()) <= HEALTHY_ANG_MAX)

    # ── Internals ─────────────────────────────────────────────────────

    def apply_torques(self, action):
        """
        Apply joint torques as velocity impulses in the walker's LOCAL frame.

        The walker lives in a rotated XZ plane.  Torques must be decomposed
        along (walk_dx, walk_dz) — the forward axis — and world Y — the up
        axis.  This keeps the agent's learned policy invariant to world
        orientation: action[0]=+1 always means 'swing right thigh forward'.
        """
        jn       = self.joint_names
        pts      = self.points
        vel      = self.velocities
        dt       = self.physics.dt
        substeps = max(self.physics.substeps, 1)
        mdt      = dt / substeps

        fx, fz   = self._fwd()          # unit forward vector in world XZ
        MAX_DELTA_V = 4.0

        def hinge(pi, ci, torque, length):
            # Vector from parent to child in local frame
            fwd_c, up_c = self._local_delta(pi, ci)
            d = math.sqrt(fwd_c**2 + up_c**2) + 1e-6

            # Perpendicular in local (fwd, up) plane = rotate 90°
            # perp_fwd = -up_c/d,  perp_up = fwd_c/d
            perp_fwd = -up_c / d
            perp_up  =  fwd_c / d

            # Force magnitude
            f  = torque * TORQUE_SCALE / max(length, 0.1)

            ca = f / max(pts[ci]['mass'], 0.1)
            pa = f / max(pts[pi]['mass'], 0.1)

            cap = MAX_DELTA_V / substeps

            def clamp(v, lim): return max(-lim, min(lim, v))

            # Child: impulse along perpendicular, decomposed back to world XZ + Y
            dvci_fwd = clamp(ca * perp_fwd * mdt, cap)
            dvci_up  = clamp(ca * perp_up  * mdt, cap)
            # Parent: equal and opposite
            dvpi_fwd = clamp(pa * perp_fwd * mdt, cap)
            dvpi_up  = clamp(pa * perp_up  * mdt, cap)

            # Convert local-forward impulse back to world vx scalar
            # (physics integrates: x += vx*walk_dx*mdt, z += vx*walk_dz*mdt)
            # So world vx drives the forward axis — add fwd impulse directly to vx.
            vel[ci]['vx'] += dvci_fwd
            vel[ci]['vy'] += dvci_up
            vel[pi]['vx'] -= dvpi_fwd
            vel[pi]['vy'] -= dvpi_up

        hinge(jn['torso'],   jn['r_thigh'], action[0], THIGH_LEN)
        hinge(jn['r_thigh'], jn['r_leg'],   action[1], LEG_LEN)
        hinge(jn['r_leg'],   jn['r_foot'],  action[2], FOOT_LEN)
        hinge(jn['torso'],   jn['l_thigh'], action[3], THIGH_LEN)
        hinge(jn['l_thigh'], jn['l_leg'],   action[4], LEG_LEN)
        hinge(jn['l_leg'],   jn['l_foot'],  action[5], FOOT_LEN)

    def compute_angles(self):
        """
        Compute joint angles in the walker's LOCAL frame (forward/up plane).
        This makes angles invariant to world rotation — the agent always sees
        the same numbers regardless of which direction it's facing.
        """
        jn = self.joint_names

        def ang(pi, ci):
            fwd, up = self._local_delta(pi, ci)
            # atan2(forward, -up): 0 = hanging straight down, + = swung forward
            return math.atan2(fwd, -up)

        return np.array([
            ang(jn['torso'],   jn['r_thigh']),
            ang(jn['r_thigh'], jn['r_leg']),
            ang(jn['r_leg'],   jn['r_foot']),
            ang(jn['torso'],   jn['l_thigh']),
            ang(jn['l_thigh'], jn['l_leg']),
            ang(jn['l_leg'],   jn['l_foot']),
        ], dtype=np.float64)

    def get_torso_pitch(self):
        """Torso pitch in local frame: angle of r_thigh relative to torso, local fwd/up."""
        jn = self.joint_names
        fwd, up = self._local_delta(jn['torso'], jn['r_thigh'])
        return math.atan2(fwd, -up)

    def get_torso_pitch_rate(self):
        ti = self.joint_names['torso']
        return self.velocities[ti]['vx'] / max(self.points[ti]['y'], 0.5)
