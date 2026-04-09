import math
import numpy as np
from humanoid       import create_walker
from walker_physics import WalkerPhysics
from walker_env     import WalkerEnv
from head_gaze import HeadGaze

OBS_DIM = 20

def angle_to_target_xz(from_x, from_z, target_x, target_z):
    return math.atan2(target_z - from_z, target_x - from_x)


class _SingleWalker:
    def __init__(self, spawn_x, spawn_z, target):
        self.spawn_x = spawn_x
        self.spawn_z = spawn_z
        self.target  = np.asarray(target, dtype=np.float32)
        self.dead    = False
        self.dead_points = None
        self.dead_lines  = None
        self.physics = WalkerPhysics(dt=0.008)
        self.gaze = HeadGaze()

        self._build()

    def _build(self, spawn_x=None, spawn_z=None, face_angle=0.0):
        sx = spawn_x if spawn_x is not None else self.spawn_x
        sz = spawn_z if spawn_z is not None else self.spawn_z

        pts, lines, joint_names, constraints = create_walker(base_x=sx, base_z=sz)

        # Rotate all joints around spawn point to face target direction
        cos_a = math.cos(face_angle)
        sin_a = math.sin(face_angle)

        for p in pts:
            dx = p['x'] - sx
            dz = p.get('z', 0.0) - sz
            new_x = sx + dx * cos_a - dz * sin_a
            new_z = sz + dx * sin_a + dz * cos_a
            p['x'] = new_x
            p['z'] = new_z

            # Walking forward axis: (cos_a, sin_a)
            # Lateral axis (perpendicular): (-sin_a, cos_a)
            # _base_lateral = projection of this point onto the lateral axis
            # This is what the physics locks — keeps joints in the rotated plane
            lat_x = -sin_a
            lat_z =  cos_a
            p['_base_lateral'] = new_x * lat_x + new_z * lat_z

            # Keep _base_z for legacy compatibility (used by old Z-lock code path)
            p['_base_z'] = new_z

        velocities = [{'vx': 0.0, 'vy': 0.0, 'vz': 0.0} for _ in pts]
        self.points      = pts
        self.velocities  = velocities
        self.constraints = constraints
        self.lines       = lines
        self.joint_names = joint_names

        # Tell physics which direction to walk
        self.physics.set_walk_direction(cos_a, sin_a)

        self.env = WalkerEnv(pts, velocities, constraints,
                             joint_names, self.physics, self.target)
        self.gaze.reset()


    def reset(self, spawn_x=None, spawn_z=None, face_angle=0.0):
        self.dead        = False
        self.dead_points = None
        self.dead_lines  = None
        self._build(spawn_x=spawn_x, spawn_z=spawn_z, face_angle=face_angle)
        return self.env.reset()

    def on_target_changed(self):
        self.env.on_target_changed()

    def get_obs_extended(self):
        base = self.env.get_observator()  # (17,)

        ti  = self.joint_names['torso']
        tx  = self.points[ti]['x']
        tz  = self.points[ti].get('z', 0.0)

        dx_world = float(self.target[0]) - tx
        dz_world = float(self.target[2]) - tz
        dist     = math.sqrt(dx_world**2 + dz_world**2) + 1e-6

        # Forward direction = walking axis
        fwd_x = self.physics.walk_dx
        fwd_z = self.physics.walk_dz

        # Right = lateral axis
        rgt_x =  fwd_z
        rgt_z = -fwd_x

        local_fwd   = dx_world * fwd_x + dz_world * fwd_z
        local_right = dx_world * rgt_x + dz_world * rgt_z

        extra = np.array([
            local_fwd   / dist,
            local_right / dist,
            1.0 / (1.0 + dist),
            ], dtype=np.float32)

        return np.concatenate([base, extra]).astype(np.float32)
    def _pin_head(self):
        if 'head' not in self.joint_names:
            return
        ti = self.joint_names['torso']
        hi = self.joint_names['head']
        self.points[hi]['x'] = self.points[ti]['x']
        self.points[hi]['y'] = self.points[ti]['y'] + 0.22
        self.points[hi]['z'] = self.points[ti].get('z', 0.0)
        self.velocities[hi]['vx'] = 0.0
        self.velocities[hi]['vy'] = 0.0
        self.velocities[hi]['vz'] = 0.0
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._pin_head()

        if done and not self.dead:
            self.dead        = True
            self.dead_points = [p.copy() for p in self.points]
            self.dead_lines  = list(self.lines)
        return obs, reward, done, info

    @property
    def is_alive(self):
        return not self.dead
    def update_gaze(self, target):
        if self.dead:
            return
        ti = self.joint_names['torso']
        self.gaze.update(
            torso_x  = self.points[ti]['x'],
            torso_z  = self.points[ti].get('z', 0.0),
            walk_dx  = self.physics.walk_dx,
            walk_dz  = self.physics.walk_dz,
            target_x = float(target[0]),
            target_z = float(target[2]),
        )
    def get_render_data(self):
        if self.dead:
            return (self.dead_points or []), (self.dead_lines or [])
        return list(self.points), list(self.lines)


class Population:
    def __init__(self, n, target, spawn_spread=1.5):
        self.n      = n
        self.target = np.asarray(target, dtype=np.float32)

        z_positions = np.linspace(-(n // 2) * spawn_spread,
                                  (n // 2) * spawn_spread, n)
        self.humans = [
            _SingleWalker(spawn_x=0.0, spawn_z=float(z), target=target)
            for z in z_positions
        ]

        self.episode       = 0
        self.all_time_best = float('inf')
        self._reset_all()

    @property
    def n_alive(self):
        return sum(1 for h in self.humans if h.is_alive)

    def get_observator_batch(self):
        return np.stack([h.get_obs_extended() for h in self.humans],
                        axis=0).astype(np.float32)

    def set_target(self, x, y, z):
        self.target[:] = [x, y, z]
        for h in self.humans:
            h.target[:] = [x, y, z]
            h.on_target_changed()

    def get_render_data(self):
        all_points, all_lines = [], []
        for human in self.humans:
            pts, lns = human.get_render_data()
            if not pts:
                continue
            offset = len(all_points)
            if not human.is_alive:
                all_points.extend(p.copy() for p in pts)
                for ln in lns:
                    all_lines.append({
                        'from':   ln['from']  + offset,
                        'to':     ln['to']    + offset,
                        'color':  (0.3, 0.3, 0.3),
                        'radius': ln['radius'] * 0.6,
                    })
            else:
                all_points.extend(pts)
                for ln in lns:
                    all_lines.append({
                        'from':   ln['from']  + offset,
                        'to':     ln['to']    + offset,
                        'color':  ln['color'],
                        'radius': ln['radius'],
                    })
        return all_points, all_lines
    def update_gazes(self):
        """Advance every walker's gaze FSM. Wall-clock timed internally."""
        for h in self.humans:
            if not h.dead and hasattr(h, 'gaze'):
                h.update_gaze(self.target)
    def _reset_all(self, spawn_x=None, spawn_z=None, face_target=False):
        spread = 1.5
        n = len(self.humans)
        z_offsets = np.linspace(-(n // 2) * spread, (n // 2) * spread, n)

        for i, h in enumerate(self.humans):
            sx = spawn_x if spawn_x is not None else h.spawn_x
            sz = (spawn_z + float(z_offsets[i])) if spawn_z is not None else h.spawn_z

            face_angle = 0.0
            if face_target:
                face_angle = angle_to_target_xz(sx, sz,
                                                float(self.target[0]),
                                                float(self.target[2]))
                print(f"walker[{i}] face_angle={math.degrees(face_angle):.1f}°  "
                      f"spawn=({sx:.3f},{sz:.3f})  "
                      f"target=({float(self.target[0]):.3f},{float(self.target[2]):.3f})")

            h.reset(spawn_x=sx, spawn_z=sz, face_angle=face_angle)

    def close(self):
        pass
