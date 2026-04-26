"""
voxel_terrain.py — Chunked voxel terrain with procedural generation,
material embedding, greedy meshing, and PyOpenGL rendering.

Designed to integrate cleanly with the existing Walker/Render system.

Usage in Render.py:
    from voxel_terrain import VoxelWorld, TerrainRenderer

    terrain = VoxelWorld(seed=42)
    terrain.generate(center_x=0, center_z=0, radius=3)
    renderer = TerrainRenderer(terrain)

    # In render_frame, after draw_grid():
    renderer.draw()

Usage in main.py (physics/damage):
    from voxel_terrain import VoxelWorld

    def damage_voxel_at_ray(terrain, ray_origin, ray_dir, force=1.0):
        hit = terrain.raycast(ray_origin, ray_dir, max_dist=10.0)
        if hit:
            cx, cy, cz = hit
            terrain.damage(cx, cy, cz, force)
"""

import math
import numpy as np
import random
from typing import Dict, List, Optional, Tuple

# ── Try noise library, fall back to manual Perlin ──────────────────────
try:
    from noise import pnoise2, pnoise3
    _HAS_NOISE_LIB = True
except ImportError:
    _HAS_NOISE_LIB = False
    print("⚠️  'noise' library not found — using built-in simplex approximation")


# ══════════════════════════════════════════════════════════════════════
# Material registry
# ══════════════════════════════════════════════════════════════════════

AIR   = 0
ROCK  = 1
DIRT  = 2
GRASS = 3
IRON  = 4
GOLD  = 5
COAL  = 6
STONE = 7

MATERIALS: Dict[int, dict] = {
    AIR:   {"name": "air",   "hardness": 0,  "color": (0.0,  0.0,  0.0),  "emit": False},
    ROCK:  {"name": "rock",  "hardness": 5,  "color": (0.45, 0.43, 0.41), "emit": False},
    DIRT:  {"name": "dirt",  "hardness": 2,  "color": (0.42, 0.28, 0.18), "emit": False},
    GRASS: {"name": "grass", "hardness": 1,  "color": (0.25, 0.52, 0.18), "emit": False},
    IRON:  {"name": "iron",  "hardness": 10, "color": (0.65, 0.65, 0.72), "emit": False},
    GOLD:  {"name": "gold",  "hardness": 3,  "color": (0.95, 0.76, 0.15), "emit": True},
    COAL:  {"name": "coal",  "hardness": 4,  "color": (0.18, 0.18, 0.20), "emit": False},
    STONE: {"name": "stone", "hardness": 7,  "color": (0.55, 0.54, 0.53), "emit": False},
}

VOXEL_SIZE = 1.0   # world units per voxel


# ══════════════════════════════════════════════════════════════════════
# Simplex-like noise fallback (no dependency)
# ══════════════════════════════════════════════════════════════════════

class _SimplexFallback:
    """Cheap value-noise fallback if the 'noise' package isn't installed."""
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._table = rng.random((512,)).astype(np.float32)

    def _fade(self, t): return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a, b, t): return a + t * (b - a)

    def noise2(self, x: float, y: float) -> float:
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u  = self._fade(xf)
        v  = self._fade(yf)
        aa = self._table[(xi     + self._table[yi     & 255].astype(int)) & 255]
        ab = self._table[(xi     + self._table[(yi+1) & 255].astype(int)) & 255]
        ba = self._table[(xi + 1 + self._table[yi     & 255].astype(int)) & 255]
        bb = self._table[(xi + 1 + self._table[(yi+1) & 255].astype(int)) & 255]
        return float(self._lerp(self._lerp(aa, ba, u), self._lerp(ab, bb, u), v))

    def noise3(self, x: float, y: float, z: float) -> float:
        return self.noise2(x + z * 31.7, y + z * 17.3)


# ══════════════════════════════════════════════════════════════════════
# Chunk
# ══════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 16   # 16×16×16 voxels per chunk


class Chunk:
    """
    Stores material and integrity for a 16x16x16 region.

    Two-stage cache:
      mesh_dirty=True  -> CPU arrays need rebuilding (worker thread).
      vbo_dirty=True   -> GPU VBO needs re-uploading (render thread).

    Once uploaded, drawing costs one glDrawArrays call per chunk
    instead of thousands of glVertex3f calls.
    """
    def __init__(self, cx: int, cy: int, cz: int):
        self.cx, self.cy, self.cz = cx, cy, cz
        self.material  = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
        self.integrity = np.ones( (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)

        # CPU mesh -- rebuilt whenever mesh_dirty is True
        self.mesh_dirty  = True
        self.mesh_verts  = np.empty((0, 3), dtype=np.float32)
        self.mesh_colors = np.empty((0, 3), dtype=np.float32)
        self.mesh_norms  = np.empty((0, 3), dtype=np.float32)
        self.vert_count  = 0

        # GPU VBO handles (0 = not yet allocated)
        self.vbo_dirty = True
        self.vbo_vert  = 0
        self.vbo_col   = 0
        self.vbo_norm  = 0

    def world_origin(self) -> Tuple[float, float, float]:
        return (
            self.cx * CHUNK_SIZE * VOXEL_SIZE,
            self.cy * CHUNK_SIZE * VOXEL_SIZE,
            self.cz * CHUNK_SIZE * VOXEL_SIZE,
        )

    def free_vbos(self):
        """Release GPU memory. Call before discarding a chunk."""
        try:
            from OpenGL.GL import glDeleteBuffers
            ids = [i for i in (self.vbo_vert, self.vbo_col, self.vbo_norm) if i]
            if ids:
                glDeleteBuffers(len(ids), ids)
        except Exception:
            pass
        self.vbo_vert = self.vbo_col = self.vbo_norm = 0


# ══════════════════════════════════════════════════════════════════════
# VoxelWorld
# ══════════════════════════════════════════════════════════════════════

class VoxelWorld:
    """
    Manages a sparse dict of Chunks.
    Terrain height is driven by 2D Perlin noise; ore veins by 3D noise.
    """
    def __init__(self, seed: int = 0, terrain_height: int = 8,
                 surface_y: int = 0):
        self.seed           = seed
        self.terrain_height = terrain_height   # voxels above ground_y
        self.surface_y      = surface_y        # world-Y of ground surface
        self.chunks: Dict[Tuple[int,int,int], Chunk] = {}
        self._debris: List[dict] = []          # flying debris particles

        rng = random.Random(seed)
        ns  = rng.randint(0, 99999)
        if _HAS_NOISE_LIB:
            self._noise2 = lambda x, z, oct=4, pers=0.5, lac=2.0, base=ns: \
                pnoise2(x, z, octaves=oct, persistence=pers,
                        lacunarity=lac, base=base)
            self._noise3 = lambda x, y, z, oct=3, pers=0.5, lac=2.0, base=ns: \
                pnoise3(x, y, z, octaves=oct, persistence=pers,
                        lacunarity=lac, base=base)
        else:
            fb2 = _SimplexFallback(ns)
            fb3 = _SimplexFallback(ns + 1)
            self._noise2 = lambda x, z, **kw: fb2.noise2(x, z)
            self._noise3 = lambda x, y, z, **kw: fb3.noise3(x, y, z)

    # ── Coordinate helpers ─────────────────────────────────────────

    def _chunk_key(self, wx: int, wy: int, wz: int) -> Tuple[int,int,int]:
        return (wx // CHUNK_SIZE, wy // CHUNK_SIZE, wz // CHUNK_SIZE)

    def _local(self, w: int) -> int:
        return w % CHUNK_SIZE

    def get_chunk(self, cx: int, cy: int, cz: int) -> Optional[Chunk]:
        return self.chunks.get((cx, cy, cz))

    def get_or_create_chunk(self, cx: int, cy: int, cz: int) -> Chunk:
        key = (cx, cy, cz)
        if key not in self.chunks:
            self.chunks[key] = Chunk(cx, cy, cz)
        return self.chunks[key]

    def get_voxel(self, wx: int, wy: int, wz: int) -> int:
        key = self._chunk_key(wx, wy, wz)
        ch  = self.chunks.get(key)
        if ch is None:
            return AIR
        return int(ch.material[self._local(wx), self._local(wy), self._local(wz)])

    def set_voxel(self, wx: int, wy: int, wz: int, mat: int,
                  integrity: float = 1.0):
        key = self._chunk_key(wx, wy, wz)
        ch  = self.get_or_create_chunk(*key)
        lx, ly, lz = self._local(wx), self._local(wy), self._local(wz)
        ch.material [lx, ly, lz] = mat
        ch.integrity[lx, ly, lz] = integrity
        ch.mesh_dirty = True

    # ── Procedural generation ──────────────────────────────────────

    def _surface_height(self, wx: int, wz: int) -> int:
        """Flat terrain — every column has the same surface Y."""
        return self.surface_y

    def _ore_material(self, wx: int, wy: int, wz: int) -> int:
        """3D noise → ore type or ROCK."""
        n3 = self._noise3(wx * 0.06, wy * 0.06, wz * 0.06)
        n3 = (n3 + 1.0) * 0.5   # 0..1

        if n3 > 0.88:
            n_gold = self._noise3(wx * 0.14, wy * 0.14, wz * 0.14)
            if (n_gold + 1.0) * 0.5 > 0.92:
                return GOLD
            return IRON
        if n3 > 0.82:
            return COAL
        if n3 > 0.75:
            return STONE
        return ROCK

    def generate(self, center_x: float = 0.0, center_z: float = 0.0,
                 radius: int = 4):
        """
        Generate terrain in a square of `radius` chunks around (center_x, center_z).
        Call again when the player moves to load new chunks.
        """
        ccx = int(center_x) // CHUNK_SIZE
        ccz = int(center_z) // CHUNK_SIZE

        # We only generate Y = 0 layer of chunks (ground + a bit below)
        cy_range = range(-1, 1)   # chunks below and at surface

        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                for cy in cy_range:
                    cx = ccx + dx
                    cz = ccz + dz
                    key = (cx, cy, cz)
                    if key in self.chunks:
                        continue   # already generated
                    self._generate_chunk(cx, cy, cz)

    def _heightmap_batch(self, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Flat terrain — every column has the same surface Y."""
        return np.full(xs.shape, self.surface_y, dtype=np.int32)

    def _ore_batch(self, xs: np.ndarray, ys: np.ndarray,
                   zs: np.ndarray) -> np.ndarray:
        """
        Vectorized ore classification over (N,M,K) world-coord arrays.
        Returns uint8 material array of same shape.
        """
        vfunc3 = np.vectorize(
            lambda x, y, z: self._noise3(x * 0.06, y * 0.06, z * 0.06))
        n3 = (vfunc3(xs, ys, zs) + 1.0) * 0.5        # 0..1

        # Second noise for gold rarity
        vfunc_g = np.vectorize(
            lambda x, y, z: self._noise3(x * 0.14, y * 0.14, z * 0.14))
        ng = (vfunc_g(xs, ys, zs) + 1.0) * 0.5

        mat = np.full(xs.shape, ROCK, dtype=np.uint8)
        mat[n3 > 0.75] = STONE
        mat[n3 > 0.82] = COAL
        iron_mask = n3 > 0.88
        mat[iron_mask] = IRON
        mat[iron_mask & (ng > 0.92)] = GOLD
        return mat

    def _generate_chunk(self, cx: int, cy: int, cz: int):
        ch = self.get_or_create_chunk(cx, cy, cz)
        ox, oy, oz = cx * CHUNK_SIZE, cy * CHUNK_SIZE, cz * CHUNK_SIZE

        # Build (16,16,16) world-coord grids in one shot
        lx = np.arange(CHUNK_SIZE, dtype=np.int32)
        ly = np.arange(CHUNK_SIZE, dtype=np.int32)
        lz = np.arange(CHUNK_SIZE, dtype=np.int32)
        gx, gy, gz = np.meshgrid(lx + ox, ly + oy, lz + oz, indexing='ij')
        # gx/gy/gz shape: (16, 16, 16)

        # Heights: one value per X,Z column → shape (16, 16)
        heights = self._heightmap_batch(gx[:, 0, :], gz[:, 0, :])
        # Broadcast to (16, 1, 16) so comparisons work across the Y axis
        heights_3d = heights[:, np.newaxis, :]         # (16, 1, 16)

        # Start everything as ROCK, then carve / surface in NumPy
        mat = np.full((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), ROCK, dtype=np.uint8)

        air_mask   = gy > heights_3d
        grass_mask = gy == heights_3d
        dirt_mask  = (gy == heights_3d - 1) | (gy == heights_3d - 2)
        deep_mask  = ~air_mask & ~grass_mask & ~dirt_mask

        # Fill ores for deep voxels via batched noise
        if deep_mask.any():
            ore_mat = self._ore_batch(gx[deep_mask], gy[deep_mask], gz[deep_mask])
            mat[deep_mask] = ore_mat

        mat[dirt_mask]  = DIRT
        mat[grass_mask] = GRASS
        mat[air_mask]   = AIR

        ch.material  = mat
        ch.integrity = np.ones((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
        ch.mesh_dirty = True

    # ── Damage / destruction ───────────────────────────────────────

    def damage(self, wx: int, wy: int, wz: int, force: float = 1.0,
               spawn_debris: bool = True) -> bool:
        """
        Apply damage to voxel at world coords.
        Returns True if the voxel was destroyed.
        """
        key = self._chunk_key(wx, wy, wz)
        ch  = self.chunks.get(key)
        if ch is None:
            return False
        lx, ly, lz = self._local(wx), self._local(wy), self._local(wz)
        mat = int(ch.material[lx, ly, lz])
        if mat == AIR:
            return False

        hardness = MATERIALS[mat]["hardness"]
        ch.integrity[lx, ly, lz] -= force / max(hardness, 0.1)

        if ch.integrity[lx, ly, lz] <= 0.0:
            if spawn_debris:
                self._spawn_debris(wx + 0.5, wy + 0.5, wz + 0.5, mat)
            ch.material [lx, ly, lz] = AIR
            ch.integrity[lx, ly, lz] = 0.0
            ch.mesh_dirty = True
            # Mark neighbour chunks dirty too (face visibility change)
            for neighbour in self._neighbour_chunks(wx, wy, wz):
                nc = self.chunks.get(self._chunk_key(*neighbour))
                if nc:
                    nc.mesh_dirty = True
            return True
        else:
            ch.mesh_dirty = True   # crack visual update
            return False

    def _neighbour_chunks(self, wx, wy, wz):
        """World coords of 6 face-neighbour voxels."""
        return [
            (wx-1,wy,wz),(wx+1,wy,wz),
            (wx,wy-1,wz),(wx,wy+1,wz),
            (wx,wy,wz-1),(wx,wy,wz+1),
        ]

    # ── Debris ────────────────────────────────────────────────────

    def _spawn_debris(self, x: float, y: float, z: float, mat: int,
                      count: int = 6):
        color = MATERIALS[mat]["color"]
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(1.5, 4.0)
            self._debris.append({
                "pos": [x, y, z],
                "vel": [math.cos(angle) * speed,
                        random.uniform(2.0, 5.0),
                        math.sin(angle) * speed],
                "color": color,
                "life": random.uniform(0.5, 1.2),
                "size": random.uniform(0.05, 0.15),
            })

    def step_debris(self, dt: float):
        """Integrate debris particles. Call once per physics tick."""
        alive = []
        for d in self._debris:
            d["vel"][1] -= 9.81 * dt
            d["pos"][0] += d["vel"][0] * dt
            d["pos"][1] += d["vel"][1] * dt
            d["pos"][2] += d["vel"][2] * dt
            if d["pos"][1] < 0.0:
                d["pos"][1] = 0.0
                d["vel"][1] *= -0.3
                d["vel"][0] *= 0.5
                d["vel"][2] *= 0.5
            d["life"] -= dt
            if d["life"] > 0:
                alive.append(d)
        self._debris = alive


    # ── Footprint / Laplacian curvature deformation ────────────────

    # Height displacement map: world (x,z) → float depression depth
    # Stored as a flat dict so it works across chunks without extra structure.
    # Key: (wx, wz) integer grid coords. Value: depression depth 0..1.
    # Applied during mesh build to push the top grass face downward.
    _footprint_map: dict = {}          # class-level shared across all instances
    FOOTPRINT_RADIUS   = 2.0           # world units — foot imprint radius
    FOOTPRINT_DEPTH    = 0.18          # max depression in world units
    FOOTPRINT_DECAY    = 0.0004        # per physics tick recovery rate
    FOOTPRINT_LAPLACE  = 1            # one pass keeps ring shape intact

    def add_footprint(self, foot_x: float, foot_z: float, weight: float = 1.0):
        """
        Stamp a Laplacian-of-Gaussian (Mexican Hat) shaped imprint.

        The LoG kernel f(r) = (1 - r²/σ²) * exp(-r²/(2σ²)) produces:
          - negative (sunken) centre  — the foot depression
          - positive (raised) ring    — displaced earth around the rim
          - smooth falloff to zero    — no hard edges

        This is physically correct: stepping compresses soil downward
        at the contact patch and pushes it outward into a raised ring.

        After stamping, iterative Laplacian smoothing is applied so
        neighbouring cells blend naturally — creating the concentric
        circle curvature effect.
        """
        r     = self.FOOTPRINT_RADIUS
        sigma = r * 0.7           # depression zone ~70% of radius, ring at outer 30%
        depth = self.FOOTPRINT_DEPTH * weight

        # Sample region (extend by 1 beyond r to capture the raised rim)
        margin = int(math.ceil(r)) + 1
        ix0 = int(math.floor(foot_x)) - margin
        ix1 = int(math.floor(foot_x)) + margin
        iz0 = int(math.floor(foot_z)) - margin
        iz1 = int(math.floor(foot_z)) + margin

        s2 = sigma * sigma

        for ix in range(ix0, ix1 + 1):
            for iz in range(iz0, iz1 + 1):
                dx = ix + 0.5 - foot_x
                dz = iz + 0.5 - foot_z
                r2 = dx*dx + dz*dz

                # Skip cells well outside influence zone
                if r2 > (r + 1.0) ** 2:
                    continue

                # Laplacian-of-Gaussian (Mexican Hat):
                # positive = raised rim, negative = sunken centre
                log_val = (1.0 - r2 / s2) * math.exp(-r2 / (2.0 * s2))

                # Scale: centre goes down by `depth`, rim rises by ~depth*0.3
                delta = log_val * depth    # LoG peak at centre → depression (positive=down)

                key  = (ix, iz)
                prev = VoxelWorld._footprint_map.get(key, 0.0)
                # Clamp: depression max 1.0, rim raise max 0.25
                new_val = prev + delta
                new_val = max(-0.25, min(1.0, new_val))
                if abs(new_val) > 0.001:
                    VoxelWorld._footprint_map[key] = new_val
                elif key in VoxelWorld._footprint_map:
                    del VoxelWorld._footprint_map[key]

        # Iterative Laplacian smoothing — blends the ring into the
        # surrounding surface, producing smooth concentric curvature.
        for _ in range(self.FOOTPRINT_LAPLACE):
            self._laplacian_smooth_footprints(ix0-1, ix1+1, iz0-1, iz1+1)

        # Mark affected surface chunks dirty
        # surface voxels are at wy=-1, living in chunk cy=-1
        affected_chunks = set()
        for ix in range(ix0, ix1 + 1):
            for iz in range(iz0, iz1 + 1):
                affected_chunks.add((ix // CHUNK_SIZE, -1, iz // CHUNK_SIZE))
        for ck in affected_chunks:
            ch = self.chunks.get(ck)
            if ch:
                ch.mesh_dirty = True

    def _laplacian_smooth_footprints(self, x0, x1, z0, z1):
        """
        One Laplacian smoothing pass over the footprint displacement map.

        Blends every cell (positive=depression, negative=raised rim)
        with its 4 neighbours using a 5-point discrete Laplacian:
            new[i] = centre_weight * self + (1-centre_weight) * avg(neighbours)

        Using centre_weight=0.6 preserves the overall shape while
        spreading the curvature outward, producing smooth concentric rings.
        """
        fm   = VoxelWorld._footprint_map
        # Gather all non-zero cells in region, plus their neighbours
        candidates = set()
        for x in range(x0, x1+1):
            for z in range(z0, z1+1):
                if (x, z) in fm:
                    candidates.add((x, z))
                    for dx, dz in ((-1,0),(1,0),(0,-1),(0,1)):
                        candidates.add((x+dx, z+dz))
        if not candidates:
            return
        updates = {}
        cw = 0.8   # centre weight — high to preserve LoG ring shape
        for (x, z) in candidates:
            v     = fm.get((x, z), 0.0)
            n_avg = sum(fm.get((x+dx, z+dz), 0.0)
                        for dx, dz in ((-1,0),(1,0),(0,-1),(0,1))) * 0.25
            new_v = cw * v + (1.0 - cw) * n_avg
            if abs(new_v) > 0.0005:
                updates[(x, z)] = max(-0.25, min(1.0, new_v))
            elif (x, z) in fm:
                updates[(x, z)] = 0.0   # will be cleaned on next step_footprints
        fm.update(updates)

    def step_footprints(self, dt: float):
        """
        Slowly recover footprints back toward flat.
        Call once per physics tick alongside step_debris().
        """
        if not VoxelWorld._footprint_map:
            return
        decay = self.FOOTPRINT_DECAY
        to_delete = []
        # Track which chunks need redraw
        dirty_chunks = set()
        for key, val in VoxelWorld._footprint_map.items():
            # Move toward zero regardless of sign (recover depression AND rim)
            if val > 0:
                new_val = val - decay
            else:
                new_val = val + decay  # rim cells recover upward
            if abs(new_val) < 0.001:
                to_delete.append(key)
                dirty_chunks.add((key[0] // CHUNK_SIZE,-1, key[1] // CHUNK_SIZE))
            else:
                VoxelWorld._footprint_map[key] = new_val
        for k in to_delete:
            del VoxelWorld._footprint_map[k]
        # Only mark dirty if something actually changed
        if to_delete:
            for ck in dirty_chunks:
                ch = self.chunks.get(ck)
                if ch:
                    ch.mesh_dirty = True

    @staticmethod
    def get_footprint_depression(wx: int, wz: int) -> float:
        """Return the current depression depth (0=flat, positive=sunken) for cell (wx,wz)."""
        return VoxelWorld._footprint_map.get((wx, wz), 0.0)


    # ── Raycast ───────────────────────────────────────────────────

    def raycast(self, origin: Tuple[float,float,float],
                direction: Tuple[float,float,float],
                max_dist: float = 12.0) -> Optional[Tuple[int,int,int]]:
        """
        DDA voxel traversal. Returns (wx,wy,wz) of first solid voxel hit or None.
        """
        ox, oy, oz = origin
        dx, dy, dz = direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 1e-9:
            return None
        dx /= length; dy /= length; dz /= length

        x, y, z = int(math.floor(ox)), int(math.floor(oy)), int(math.floor(oz))
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        sz = 1 if dz > 0 else -1

        def tdelta(d): return abs(1.0 / d) if abs(d) > 1e-9 else 1e30
        def tcross(o, d, step):
            if abs(d) < 1e-9: return 1e30
            b = math.floor(o) + (1 if step > 0 else 0)
            return (b - o) / d

        tx = tcross(ox, dx, sx); ty = tcross(oy, dy, sy); tz = tcross(oz, dz, sz)
        dtx = tdelta(dx); dty = tdelta(dy); dtz = tdelta(dz)

        dist = 0.0
        while dist < max_dist:
            mat = self.get_voxel(x, y, z)
            if mat != AIR:
                return (x, y, z)
            if tx < ty and tx < tz:
                x += sx; dist = tx; tx += dtx
            elif ty < tz:
                y += sy; dist = ty; ty += dty
            else:
                z += sz; dist = tz; tz += dtz
        return None


# ══════════════════════════════════════════════════════════════════════
# Greedy mesher — produces quads per visible face
# ══════════════════════════════════════════════════════════════════════

_FACE_NORMALS = [
    np.array([ 1, 0, 0], dtype=np.float32),
    np.array([-1, 0, 0], dtype=np.float32),
    np.array([ 0, 1, 0], dtype=np.float32),
    np.array([ 0,-1, 0], dtype=np.float32),
    np.array([ 0, 0, 1], dtype=np.float32),
    np.array([ 0, 0,-1], dtype=np.float32),
]

# 4 vertices per face (CCW), unit cube at origin
_FACE_VERTS = [
    # +X
    np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]], dtype=np.float32),
    # -X
    np.array([[0,0,1],[0,1,1],[0,1,0],[0,0,0]], dtype=np.float32),
    # +Y
    np.array([[0,1,0],[0,1,1],[1,1,1],[1,1,0]], dtype=np.float32),
    # -Y
    np.array([[0,0,1],[0,0,0],[1,0,0],[1,0,1]], dtype=np.float32),
    # +Z
    np.array([[1,0,1],[1,1,1],[0,1,1],[0,0,1]], dtype=np.float32),
    # -Z
    np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]], dtype=np.float32),
]

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)


def build_chunk_mesh(chunk: Chunk, world: VoxelWorld) -> None:
    """
    Simple face-culling mesh (not full greedy merge, but very fast in NumPy).
    Skips faces shared with solid neighbours.  Writes into chunk arrays.
    """
    mat  = chunk.material
    intg = chunk.integrity
    ox, oy, oz = chunk.world_origin()

    verts_list  = []
    colors_list = []
    norms_list  = []

    offsets = [
        (1, 0, 0, 0), (-1, 0, 0, 1),
        (0, 1, 0, 2), (0, -1, 0, 3),
        (0, 0, 1, 4), (0, 0, -1, 5),
    ]

    for lx in range(CHUNK_SIZE):
        for ly in range(CHUNK_SIZE):
            for lz in range(CHUNK_SIZE):
                m = int(mat[lx, ly, lz])
                if m == AIR:
                    continue
                color = np.array(MATERIALS[m]["color"], dtype=np.float32)
                # Darken by damage
                dmg_factor = float(intg[lx, ly, lz])
                color = color * (0.6 + 0.4 * dmg_factor)

                wx = ox + lx * VOXEL_SIZE
                wy = oy + ly * VOXEL_SIZE
                wz = oz + lz * VOXEL_SIZE
                # Footprint displacement: depression (>0=sunken) or rim (<0=raised)
                _fp_depress = 0.0
                if m == GRASS or m == DIRT:
                    _fp_depress = VoxelWorld.get_footprint_depression(
                        int(ox) + lx, int(oz) + lz)
                    # Darken depressed cells to mark the footprint visually
                    if _fp_depress > 0.01:
                        dark = max(0.0, 1.0 - _fp_depress * 2.5)
                        color = color * dark

                for ndx, ndy, ndz, face_idx in offsets:
                    nlx, nly, nlz = lx + ndx, ly + ndy, lz + ndz
                    # Check if neighbour is solid (in-chunk fast path)
                    if 0 <= nlx < CHUNK_SIZE and 0 <= nly < CHUNK_SIZE and 0 <= nlz < CHUNK_SIZE:
                        if int(mat[nlx, nly, nlz]) != AIR:
                            continue
                    else:
                        # Cross-chunk neighbour lookup
                        nwx = int(ox / VOXEL_SIZE) + lx + ndx
                        nwy = int(oy / VOXEL_SIZE) + ly + ndy
                        nwz = int(oz / VOXEL_SIZE) + lz + ndz
                        if world.get_voxel(nwx, nwy, nwz) != AIR:
                            continue

                    # Emit 2 triangles for this face
                    fv  = _FACE_VERTS[face_idx].copy()
                    fv[:, 0] = fv[:, 0] * VOXEL_SIZE + wx
                    fv[:, 1] = fv[:, 1] * VOXEL_SIZE + wy
                    fv[:, 2] = fv[:, 2] * VOXEL_SIZE + wz
                    # Apply footprint displacement to top (+Y) face vertices only
                    # Positive _fp_depress → sunken, negative → raised rim
                    if face_idx == 2 and abs(_fp_depress) > 0.0005:
                        fv[:, 1] -= _fp_depress

                    tri_verts = fv[_QUAD_INDICES]  # 6 vertices
                    verts_list.append(tri_verts)
                    colors_list.append(np.tile(color, (6, 1)))
                    norms_list.append(np.tile(_FACE_NORMALS[face_idx], (6, 1)))

    if verts_list:
        chunk.mesh_verts  = np.concatenate(verts_list,  axis=0)
        chunk.mesh_colors = np.concatenate(colors_list, axis=0)
        chunk.mesh_norms  = np.concatenate(norms_list,  axis=0)
        chunk.vert_count  = len(chunk.mesh_verts)
    else:
        chunk.mesh_verts  = np.empty((0, 3), dtype=np.float32)
        chunk.mesh_colors = np.empty((0, 3), dtype=np.float32)
        chunk.mesh_norms  = np.empty((0, 3), dtype=np.float32)
        chunk.vert_count  = 0

    chunk.mesh_dirty = False
    chunk.vbo_dirty  = True   # signal renderer to re-upload to GPU


# ══════════════════════════════════════════════════════════════════════
# TerrainRenderer  (PyOpenGL immediate mode + VBO optional)
# ══════════════════════════════════════════════════════════════════════

class TerrainRenderer:
    """
    VBO-cached chunk renderer.

    Pipeline per frame:
      1. update_dirty_chunks()  -- rebuild CPU mesh for up to N dirty chunks
      2. _upload_vbo_chunks()   -- push newly rebuilt meshes to GPU (once per change)
      3. _draw_vbo_chunks()     -- one glDrawArrays call per chunk (fast path)
      4. _draw_debris()         -- point-sprite particles for broken voxels

    A chunk's CPU mesh is rebuilt on voxel change (mesh_dirty).
    The GPU VBO is re-uploaded only after a CPU rebuild (vbo_dirty).
    Every subsequent frame draws straight from the cached GPU buffer.
    """

    def __init__(self, world: VoxelWorld):
        self.world = world
        # Interleaved VBO layout: [x,y,z, nx,ny,nz, r,g,b] per vertex = 9 floats
        self._STRIDE    = 9 * 4          # bytes
        self._OFF_VERT  = 0              # byte offset of position
        self._OFF_NORM  = 3 * 4          # byte offset of normal
        self._OFF_COLOR = 6 * 4          # byte offset of color

    # ── CPU mesh management ───────────────────────────────────────────

    def update_dirty_chunks(self, max_rebuilds: int = 4):
        """Rebuild CPU mesh for up to max_rebuilds dirty chunks per frame."""
        rebuilt = 0
        for chunk in self.world.chunks.values():
            if chunk.mesh_dirty:
                build_chunk_mesh(chunk, self.world)
                rebuilt += 1
                if rebuilt >= max_rebuilds:
                    break

    # ── GPU upload ────────────────────────────────────────────────────

    def _upload_vbo_chunks(self):
        """
        For every chunk whose vbo_dirty flag is set, pack vertex data
        into an interleaved buffer and upload it to the GPU.
        Called once per frame BEFORE drawing.
        """
        from OpenGL.GL import (glGenBuffers, glBindBuffer, glBufferData,
                               GL_ARRAY_BUFFER, GL_STATIC_DRAW)

        for chunk in self.world.chunks.values():
            if not chunk.vbo_dirty:
                continue
            if chunk.vert_count == 0:
                chunk.vbo_dirty = False
                continue

            # Interleave: pos(3) + norm(3) + color(3) per vertex
            interleaved = np.concatenate([
                chunk.mesh_verts,   # (N, 3)
                chunk.mesh_norms,   # (N, 3)
                chunk.mesh_colors,  # (N, 3)
            ], axis=1).astype(np.float32)  # (N, 9) contiguous

            data = interleaved.tobytes()

            # Allocate GPU buffer if needed
            if chunk.vbo_vert == 0:
                chunk.vbo_vert = int(glGenBuffers(1))

            glBindBuffer(GL_ARRAY_BUFFER, chunk.vbo_vert)
            glBufferData(GL_ARRAY_BUFFER, len(data), data, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            chunk.vbo_dirty = False

    # ── GPU draw ──────────────────────────────────────────────────────

    def _draw_vbo_chunks(self):
        """
        Draw all chunks using cached VBOs.
        One glDrawArrays call per non-empty chunk.
        """
        from OpenGL.GL import (
            glBindBuffer, glDrawArrays, GL_ARRAY_BUFFER, GL_TRIANGLES,
            glEnableClientState, glDisableClientState,
            GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_COLOR_ARRAY,
            glVertexPointer, glNormalPointer, glColorPointer,
            GL_FLOAT, ctypes,
        )
        import ctypes as _ct

        stride = self._STRIDE

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        for chunk in self.world.chunks.values():
            if chunk.vert_count == 0 or chunk.vbo_vert == 0:
                continue
            try:
                glBindBuffer(GL_ARRAY_BUFFER, chunk.vbo_vert)
                glVertexPointer(3, GL_FLOAT, stride,
                                _ct.c_void_p(self._OFF_VERT))
                glNormalPointer(   GL_FLOAT, stride,
                                   _ct.c_void_p(self._OFF_NORM))
                glColorPointer( 3, GL_FLOAT, stride,
                                _ct.c_void_p(self._OFF_COLOR))
                glDrawArrays(GL_TRIANGLES, 0, chunk.vert_count)
            except Exception as e:
                print(f"VBO draw error chunk ({chunk.cx},{chunk.cy},{chunk.cz}): {e}")

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    # ── Main draw entry point ─────────────────────────────────────────

    def draw(self):
        """Call this inside render_frame after draw_grid()."""
        try:
            from OpenGL.GL import (glEnable, glDisable, glLightfv,
                                   GL_LIGHTING, GL_LIGHT0,
                                   GL_AMBIENT, GL_DIFFUSE, GL_POSITION,
                                   GL_COLOR_MATERIAL, glColorMaterial,
                                   GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        except ImportError:
            return

        # Step 1: rebuild dirty CPU meshes
        self.update_dirty_chunks()

        # Step 2: upload any newly rebuilt meshes to GPU
        self._upload_vbo_chunks()

        # Step 3: set up lighting, draw from GPU cache
        try:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glLightfv(GL_LIGHT0, GL_POSITION, (50.0, 200.0, 50.0, 0.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.85, 0.82, 0.75, 1.0))
            glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.25, 0.25, 0.28, 1.0))
        except Exception:
            pass

        self._draw_vbo_chunks()

        try:
            glDisable(GL_LIGHTING)
            glDisable(GL_COLOR_MATERIAL)
        except Exception:
            pass

        # Step 4: debris particles
        self._draw_debris()

    # ── Debris ────────────────────────────────────────────────────────

    def _draw_debris(self):
        if not self.world._debris:
            return
        try:
            from OpenGL.GL import (glBegin, glEnd, glVertex3f, glColor4f,
                                   glPointSize, GL_POINTS,
                                   glEnable, glDisable, GL_BLEND,
                                   glBlendFunc, GL_SRC_ALPHA,
                                   GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glPointSize(5.0)
            glBegin(GL_POINTS)
            for d in self.world._debris:
                alpha = min(1.0, d["life"] / 0.5)
                r, g, b = d["color"]
                glColor4f(r, g, b, alpha)
                glVertex3f(*d["pos"])
            glEnd()
            glDisable(GL_BLEND)
        except Exception:
            try:
                from OpenGL.GL import glEnd
                glEnd()
            except Exception:
                pass

    def destroy(self):
        """Free all GPU VBOs. Call on shutdown."""
        for chunk in self.world.chunks.values():
            chunk.free_vbos()


# ══════════════════════════════════════════════════════════════════════
# Integration helpers for main.py / Render.py
# ══════════════════════════════════════════════════════════════════════

def create_world(seed: int = 42, terrain_height: int = 6) -> VoxelWorld:
    """
    Convenience factory.  Call once at startup.
    terrain_height: how many voxels tall the hills can be above Y=0.
    """
    world = VoxelWorld(seed=seed, terrain_height=terrain_height, surface_y=-1)
    world.generate(center_x=0.0, center_z=0.0, radius=3)
    return world


def player_break_voxel(world: VoxelWorld,
                       ray_origin: Tuple[float,float,float],
                       ray_dir:    Tuple[float,float,float],
                       force: float = 2.0) -> Optional[dict]:
    """
    Call from your humanoid's interaction logic.
    Returns hit-info dict or None.
    """
    hit = world.raycast(ray_origin, ray_dir, max_dist=4.0)
    if hit is None:
        return None
    wx, wy, wz = hit
    mat  = world.get_voxel(wx, wy, wz)
    destroyed = world.damage(wx, wy, wz, force=force)
    return {
        "voxel":     (wx, wy, wz),
        "material":  MATERIALS[mat]["name"],
        "destroyed": destroyed,
    }
