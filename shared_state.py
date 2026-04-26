"""
shared_state.py
───────────────
Single source of truth for every piece of data that crosses thread/module
boundaries.  Import this module everywhere instead of passing globals around.

Nothing here imports from the project — only stdlib — so it can be the
first module loaded without circular-import risk.
"""

from __future__ import annotations

import threading
from typing import Optional


# ── AppState ──────────────────────────────────────────────────────────
class AppState:
    """
    Thin attribute bag shared across main, physics_loop, render_loop,
    and ws_handlers.  Attribute names are intentionally identical to the
    old module-level globals so call-sites need minimal edits.

    Objects that require construction (population, ppo_agent, eye, …) are
    injected by main.py after creation; everything else has a safe default.
    """

    # ── simulation objects (set by main.py) ───────────────────────────
    TARGET:             object = None   # np.ndarray [3]
    population:         object = None
    ppo_agent:          object = None
    eye:                object = None
    light:              object = None
    eye_camera:         object = None
    Population_number:  int    = 1

    # ── control flags ─────────────────────────────────────────────────
    start:             bool  = False
    pause:             bool  = False
    agent_active:      bool  = False
    first_person_mode: bool  = False
    mouse_look:        bool  = False
    cam_speed:         float = 0.5
    mouse_sens:        float = 0.005
    EP_MAX_STEPS:      int   = 3000

    # ── camera (read by render, written by render + ws) ───────────────
    cam_yaw:   float = -0.5
    cam_pitch: float = -0.4
    cam_roll:  float =  0.0
    cam_x:     float =  0.0
    cam_y:     float =  8.0
    cam_z:     float = 18.0
    FP_EYE_HEIGHT: float = 0.35

    # ── episode counters ──────────────────────────────────────────────
    total_episodes:       int   = 0
    all_time_best:        float = float('inf')
    targets_reached_count: int  = 0
    reached_x:            float = 0.0
    reached_z:            float = 0.0

    # ── physics caches (written by physics, read by ws_handlers) ──────
    cached_walker_dists: dict  = None   # {walker_index: float}
    trajectory_len:      int   = 0

    # ── async / WS ────────────────────────────────────────────────────
    ws_connections: set   = None   # set[WebSocket]
    async_loop:     object = None  # asyncio event loop

    # ── input ─────────────────────────────────────────────────────────
    keys_pressed: set = None   # set[str]

    # ── stop events (set by main.py) ──────────────────────────────────
    physics_stop: object = None   # threading.Event
    render_stop:  object = None   # threading.Event

    # ── callables injected by main.py ─────────────────────────────────
    # These let ws_handlers / physics call back into main helpers without
    # importing main (which would be circular).
    spawn_target_fn:      object = None
    clear_trajectory_fn:  object = None
    get_trajectory_fn:    object = None
    on_mouse_fn:          object = None
    read_snapshot_fn:     object = None
    get_retinal_snap_fn:  object = None
    get_eye_data_fn:      object = None


# Singleton — every module does `from shared_state import state`
state = AppState()
state.cached_walker_dists = {}
state.ws_connections      = set()
state.keys_pressed        = set()


# ── Snapshot ──────────────────────────────────────────────────────────
_snap_lock = threading.Lock()
_snapshot: dict = {
    'points': [], 'lines': [], 'target_pt': None, 'trajectory': [],
    'n_alive': 1, 'episode': 0, 'best': float('inf'), 'rot_y': 0.0,
    'dirty': False,
}


def write_snapshot(points, lines, target_pt, trajectory,
                   n_alive, episode, best, rot_y: float = 0.0) -> None:
    """
    Trajectory is stored by reference — the render thread only reads it
    under this lock, so no copy is needed here.  The physics thread must
    NOT mutate the list it passed after calling this; it should pass a new
    list each time (physics_loop already does list(_trajectory_points)).
    """
    with _snap_lock:
        _snapshot['points']     = points
        _snapshot['lines']      = lines
        _snapshot['target_pt']  = target_pt
        _snapshot['trajectory'] = trajectory
        _snapshot['n_alive']    = n_alive
        _snapshot['episode']    = episode
        _snapshot['best']       = best
        _snapshot['rot_y']      = rot_y
        _snapshot['dirty']      = True


def read_snapshot_if_dirty() -> tuple | None:
    """
    Single-lock check-and-read.  Returns None when nothing has changed,
    avoiding a second lock acquisition in the render loop.
    """
    with _snap_lock:
        if not _snapshot['dirty']:
            return None
        _snapshot['dirty'] = False
        return (
            _snapshot['points'],
            _snapshot['lines'],
            _snapshot['target_pt'],
            _snapshot['trajectory'],
            _snapshot['n_alive'],
            _snapshot['episode'],
            _snapshot['best'],
            _snapshot['rot_y'],
        )


def read_snapshot() -> tuple:
    """Unconditional read (used by ws_handlers)."""
    with _snap_lock:
        _snapshot['dirty'] = False
        return (
            _snapshot['points'],
            _snapshot['lines'],
            _snapshot['target_pt'],
            _snapshot['trajectory'],
            _snapshot['n_alive'],
            _snapshot['episode'],
            _snapshot['best'],
            _snapshot['rot_y'],
        )


# ── Frame buffer ──────────────────────────────────────────────────────
_frame_lock:  threading.Lock = threading.Lock()
_frame_bytes: Optional[bytes] = None


def write_frame(data: bytes) -> None:
    global _frame_bytes
    with _frame_lock:
        _frame_bytes = data


def read_frame() -> Optional[bytes]:
    with _frame_lock:
        return _frame_bytes


# ── Retinal result ────────────────────────────────────────────────────
_retinal_lock: threading.Lock = threading.Lock()
_retinal: dict = {
    'detected': False, 'retinal_x': 0.0, 'retinal_y': 0.0,
    'lux': 0.5,        'dilation':  0.5,
}


def write_retinal(detected: bool, rx: float, ry: float,
                  lux: float, dilation: float) -> None:
    with _retinal_lock:
        _retinal['detected']  = detected
        _retinal['retinal_x'] = rx
        _retinal['retinal_y'] = ry
        _retinal['lux']       = lux
        _retinal['dilation']  = dilation


def read_retinal() -> dict:
    """Return a fresh copy — safe to read outside the lock."""
    with _retinal_lock:
        return dict(_retinal)


# ── Eye data ──────────────────────────────────────────────────────────
# Stored as a single reference rather than a mutable dict so writers do
# one atomic assignment and readers get a consistent snapshot cheaply.
_eye_lock: threading.Lock = threading.Lock()
_eye_data: dict = {}


def write_eye_data(data: dict) -> None:
    """Replace eye data atomically — no per-key iteration."""
    with _eye_lock:
        # Rebind the global reference; readers that already hold a reference
        # to the old dict are unaffected (they finish reading the old copy).
        globals()['_eye_data'] = data


def read_eye_data() -> dict:
    with _eye_lock:
        return _eye_data   # caller gets the current reference — cheap
