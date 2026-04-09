"""
walker_physics.py — Constraint physics for Walker2D-style walker.

Kinematic points:
  Set physics.kinematic_indices = {idx, ...} to exclude joints from
  gravity, integration, damping and ground-contact.  The constraint
  solver also skips them (infinite mass = zero correction weight).
  Used to pin the head joint without any solver cost or drooping.
"""

import math


class WalkerPhysics:
    def __init__(self,
                 gravity=-9.81,
                 ground_y=0.0,
                 dt=0.008,
                 constraint_iterations=40,
                 ground_restitution=0.0,
                 ground_friction=0.05):

        self.gravity               = gravity
        self.ground_y              = ground_y
        self.dt                    = dt
        self.constraint_iterations = constraint_iterations
        self.ground_restitution    = ground_restitution
        self.ground_friction       = ground_friction

        self.torso_index   = None
        self.torso_y_min   = 0.8
        self.torso_y_max   = 2.0
        self.foot_indices  = []
        self.substeps      = 2

        self.walk_dx = 1.0
        self.walk_dz = 0.0

        # Indices that are pinned externally — skip gravity / integration / damping
        self.kinematic_indices: set = set()

    def set_walk_direction(self, dx: float, dz: float):
        mag = math.sqrt(dx * dx + dz * dz)
        if mag < 1e-6:
            self.walk_dx, self.walk_dz = 1.0, 0.0
        else:
            self.walk_dx = dx / mag
            self.walk_dz = dz / mag

    # ── Public entry point ────────────────────────────────────────────

    def apply_physics_step(self, points, velocities, constraints, muscles=None):
        for _ in range(self.substeps):
            self._micro_step(points, velocities, constraints)
        return points, velocities

    # ── Single micro-step ─────────────────────────────────────────────

    def _micro_step(self, points, velocities, constraints):
        mdt = self.dt / self.substeps
        n   = len(points)
        kin = self.kinematic_indices   # shorthand

        # 1. Gravity — skip kinematic points
        for i in range(n):
            if i in kin:
                continue
            velocities[i]['vy'] += self.gravity * mdt

        # 2. Integrate positions — skip kinematic points
        for i in range(n):
            if i in kin:
                continue
            vx = velocities[i]['vx']
            vy = velocities[i]['vy']

            points[i]['x'] += vx * self.walk_dx * mdt
            points[i]['z'] += vx * self.walk_dz * mdt
            points[i]['y'] += vy * mdt

            lat_x = -self.walk_dz
            lat_z =  self.walk_dx
            curr_lat = points[i]['x'] * lat_x + points[i]['z'] * lat_z
            base_lat = points[i].get('_base_lateral', 0.0)
            drift = curr_lat - base_lat
            points[i]['x'] -= drift * lat_x
            points[i]['z'] -= drift * lat_z

        # 3. Constraint solver — kinematic points treated as infinite mass
        self._solve_constraints(points, constraints)

        # 4. Velocity update + cap — skip kinematic points
        self._update_velocities_from_positions(points, velocities, mdt, kin)

        # 5. Ground contact — skip kinematic points
        self._ground_contact(points, velocities, kin)

        # 6. Damping — skip kinematic points
        damp_linear = 0.97
        for i in range(n):
            if i in kin:
                continue
            velocities[i]['vx'] *= damp_linear
            velocities[i]['vy'] *= damp_linear

        # 7. Torso Y clamp
        if self.torso_index is not None:
            idx = self.torso_index
            y   = points[idx]['y']
            if y < self.torso_y_min:
                points[idx]['y']      = self.torso_y_min
                velocities[idx]['vy'] = max(0.0, velocities[idx]['vy'])
            elif y > self.torso_y_max:
                points[idx]['y']      = self.torso_y_max
                velocities[idx]['vy'] = min(0.0, velocities[idx]['vy'])

    # ── PBD constraint solver — kinematic = infinite mass (w=0) ───────

    def _solve_constraints(self, points, constraints):
        kin = self.kinematic_indices
        for _ in range(self.constraint_iterations):
            for c in constraints:
                i1 = c['joint1']; i2 = c['joint2']
                if i1 >= len(points) or i2 >= len(points):
                    continue

                # Kinematic points have infinite mass → weight = 0
                kin1 = i1 in kin
                kin2 = i2 in kin
                if kin1 and kin2:
                    continue   # both fixed, nothing to solve


                p1 = points[i1]; p2 = points[i2]

                dx = p2['x'] - p1['x']
                dy = p2['y'] - p1['y']
                dz = p2.get('z', 0.0) - p1.get('z', 0.0)

                d = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d < 1e-9:
                    continue

                rest = c['rest_length']
                err  = (d - rest) / d
                s    = c.get('stiffness', 1.0) * 0.5

                m1   = 0.0 if kin1 else max(p1.get('mass', 1.0), 0.01)
                m2   = 0.0 if kin2 else max(p2.get('mass', 1.0), 0.01)
                w1   = 0.0 if kin1 else 1.0 / m1
                w2   = 0.0 if kin2 else 1.0 / m2
                wsum = w1 + w2
                if wsum < 1e-9:
                    continue

                corr = err * s
                f1   = (w1 / wsum) * corr
                f2   = (w2 / wsum) * corr

                if not kin1:
                    p1['x'] += f1 * dx
                    p1['y'] += f1 * dy
                    p1['z']  = p1.get('z', 0.0) + f1 * dz

                if not kin2:
                    p2['x'] -= f2 * dx
                    p2['y'] -= f2 * dy
                    p2['z']  = p2.get('z', 0.0) - f2 * dz

    # ── Velocity correction ───────────────────────────────────────────

    def _update_velocities_from_positions(self, points, velocities, mdt, kin=None):
        if kin is None:
            kin = set()
        for i, (p, v) in enumerate(zip(points, velocities)):
            if i in kin:
                continue
            vx = v['vx']; vy = v['vy']
            speed = math.sqrt(vx*vx + vy*vy)
            max_speed = 30.0
            if speed > max_speed:
                scale = max_speed / speed
                v['vx'] *= scale
                v['vy'] *= scale

    # ── Ground contact ────────────────────────────────────────────────

    def _ground_contact(self, points, velocities, kin=None):
        if kin is None:
            kin = set()
        foot_set = set(self.foot_indices)
        for i, (p, v) in enumerate(zip(points, velocities)):
            if i in kin:
                continue
            if p['y'] <= self.ground_y:
                p['y'] = self.ground_y
                if v['vy'] < 0:
                    v['vy'] *= -self.ground_restitution
                if i in foot_set:
                    v['vx'] = 0.0
                else:
                    v['vx'] *= self.ground_friction

    # ── Helpers ───────────────────────────────────────────────────────

    def get_center_of_mass(self, points):
        tm = sum(p.get('mass', 1.0) for p in points) or 1.0
        return {
            'x': sum(p['x'] * p.get('mass', 1.0) for p in points) / tm,
            'y': sum(p['y'] * p.get('mass', 1.0) for p in points) / tm,
            'z': sum(p.get('z', 0.0) * p.get('mass', 1.0) for p in points) / tm,
        }

    def get_com_velocity(self, points, velocities):
        tm = sum(p.get('mass', 1.0) for p in points) or 1.0
        return {
            'vx': sum(velocities[i]['vx'] * points[i].get('mass', 1.0)
                      for i in range(len(points))) / tm,
            'vy': sum(velocities[i]['vy'] * points[i].get('mass', 1.0)
                      for i in range(len(points))) / tm,
            'vz': 0.0,
        }

    def get_contact_cost(self):
        return 0.0

    def step(self, points, velocities, constraints, torques):
        return self.apply_physics_step(points, velocities, constraints)

    def solve_constraints(self, points, constraints):
        self._solve_constraints(points, constraints)

    def ground_contact(self, points, velocities):
        self._ground_contact(points, velocities)

    def micro_step(self, points, velocities, constraints):
        self._micro_step(points, velocities, constraints)
