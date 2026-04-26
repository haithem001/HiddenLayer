"""
render_loop.py
──────────────
Everything that touches OpenGL / Pygame lives here.

Public API
──────────
    run_render_loop()   — target for the render thread
    handle_camera(step) — called from the render thread each tick

Dependencies on shared_state
─────────────────────────────
    Reads : state.{first_person_mode, cam_*, keys_pressed, population, …}
    Writes: write_frame(), write_retinal(), write_eye_data()
    Reads : read_snapshot(), snapshot_dirty(), read_eye_data()
"""

from __future__ import annotations

import math
import time
import traceback

import numpy as np
import pygame

from Render import (
    render_frame,
    DOUBLEBUF, OPENGL, HIDDEN,
    glViewport, glMatrixMode, glLoadIdentity,
    gluPerspective, GL_PROJECTION, GL_MODELVIEW,
    glEnable, GL_DEPTH_TEST,
)

from shared_state import (
    state,
    read_snapshot_if_dirty,
    write_frame, read_eye_data,
    write_retinal, write_eye_data,
)

WIDTH, HEIGHT = 1600, 900
RENDER_FPS    = 120
RENDER_DT     = 1.0 / RENDER_FPS


# ── OpenGL init ───────────────────────────────────────────────────────

def _init_gl() -> bool:
    try:
        pygame.display.init()
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | HIDDEN)
        glViewport(0, 0, WIDTH, HEIGHT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(60.0, WIDTH / HEIGHT, 0.1, 2000.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glEnable(GL_DEPTH_TEST)
        print("✅ OpenGL ready")
        return True
    except Exception as e:
        print(f"❌ OpenGL: {e}")
        return False


# ── Camera helpers ────────────────────────────────────────────────────

def handle_camera(step: float) -> None:
    """Keyboard-driven free-camera movement.  Called each render tick."""
    keys = state.keys_pressed
    fx, fz = math.sin(state.cam_yaw), -math.cos(state.cam_yaw)
    rx, rz = math.cos(state.cam_yaw),  math.sin(state.cam_yaw)

    if "w" in keys: state.cam_y += step
    if "x" in keys: state.cam_y -= step
    if "z" in keys: state.cam_x += fx * step; state.cam_z += fz * step
    if "s" in keys: state.cam_x -= fx * step; state.cam_z -= fz * step
    if "d" in keys: state.cam_x += rx * step; state.cam_z += rz * step
    if "q" in keys: state.cam_x -= rx * step; state.cam_z -= rz * step

    state.cam_x     = float(np.clip(state.cam_x,     -500.,  500.))
    state.cam_y     = float(np.clip(state.cam_y,        1.,  200.))
    state.cam_z     = float(np.clip(state.cam_z,     -200.,  200.))
    state.cam_pitch = float(np.clip(state.cam_pitch,
                                    -math.pi / 2 + 0.01,
                                    math.pi / 2 - 0.01))


def _get_fp_head_info() -> dict | None:
    """
    Compute first-person camera position/orientation from the first alive walker.
    Also updates state.cam_{x,y,z,yaw,pitch} as a side-effect so the rest of
    the render call uses consistent values.
    """
    TARGET = state.TARGET
    try:
        alive = [h for h in state.population.humans if not h.dead]
        if not alive:
            return None
        h = alive[0]

        if 'head' in h.joint_names:
            hi          = h.joint_names['head']
            state.cam_x = float(h.points[hi]['x'])
            state.cam_y = float(h.points[hi]['y']) + 0.05
            state.cam_z = float(h.points[hi].get('z', 0.0))
        else:
            ti          = h.joint_names['torso']
            state.cam_x = float(h.points[ti]['x'])
            state.cam_y = float(h.points[ti]['y']) + state.FP_EYE_HEIGHT
            state.cam_z = float(h.points[ti].get('z', 0.0))

        if hasattr(h, 'gaze'):
            state.cam_yaw = h.gaze.world_yaw
            dx_t  = float(TARGET[0]) - state.cam_x
            dz_t  = float(TARGET[2]) - state.cam_z
            horiz = math.sqrt(dx_t * dx_t + dz_t * dz_t) + 1e-6
            pitch = math.atan2(1.25 - state.cam_y, horiz)
            state.cam_pitch = max(-math.pi/2 + 0.05, min(math.pi/2 - 0.05, pitch))
            fov_deg = max(10.0, min(170.0, math.degrees(h.gaze.fov_half_angle * 2.0)))
            dist    = horiz
            gstate  = h.gaze.state
        else:
            dx = float(TARGET[0]) - state.cam_x
            dz = float(TARGET[2]) - state.cam_z
            state.cam_yaw = math.atan2(dx, -dz)
            horiz = math.sqrt(dx * dx + dz * dz) + 1e-6
            pitch = math.atan2(1.25 - state.cam_y, horiz)
            state.cam_pitch = max(-math.pi/2 + 0.05, min(math.pi/2 - 0.05, pitch))
            fov_deg = 60.0
            dist    = horiz
            gstate  = 'searching'

        return dict(
            hx=state.cam_x, hy=state.cam_y, hz=state.cam_z,
            gy=state.cam_yaw, gp=state.cam_pitch,
            fov=fov_deg, dist=dist, gstate=gstate,
        )
    except Exception as e:
        print(f"⚠️  FP head info: {e}")
        return None


def _push_retinal_result(retinal_result: dict, fp_info: dict) -> None:
    """
    Write retinal detection into shared_state and update the eye model.
    Also pushes detection flag into the walker's HeadGaze FSM if supported.
    """
    det = retinal_result['detected']
    rx  = retinal_result['retinal_x']
    ry  = retinal_result['retinal_y']

    ld = state.light.get_render_data()
    write_retinal(detected=det, rx=rx, ry=ry,
                  lux=ld['ambient_lux'], dilation=ld['pupil_dilation'])

    alive = [h for h in state.population.humans if not h.dead]
    state.eye.update_from_retina(
        retinal_x      = rx,
        retinal_y      = ry,
        detected       = det,
        gaze_state     = fp_info['gstate'],
        target_dist    = fp_info['dist'],
        lux            = ld['ambient_lux'],
        pupil_dilation = ld['pupil_dilation'],
        vignette_alpha = ld['vignette_alpha'],
        sweep_offset   = _get_sweep_offset(alive),
    )
    write_eye_data(state.eye.get_render_data())

    # Notify HeadGaze FSM (reuse alive list)
    if alive and hasattr(alive[0], 'gaze'):
        g = alive[0].gaze
        if hasattr(g, 'set_retinal_detection'):
            g.set_retinal_detection(det)


def _get_sweep_offset(alive: list | None = None) -> float:
    try:
        if alive is None:
            alive = [h for h in state.population.humans if not h.dead]
        if alive and hasattr(alive[0], 'gaze'):
            return float(alive[0].gaze._sweep_offset)
    except Exception:
        pass
    return 0.0


# ── Main render loop ──────────────────────────────────────────────────

def run_render_loop() -> None:
    if not _init_gl():
        return
    print("Render thread started")
    next_t = time.perf_counter()

    while not state.render_stop.is_set():
        # Pace to RENDER_FPS
        sleep_t = next_t - time.perf_counter()
        if sleep_t > 0:
            time.sleep(sleep_t)
        next_t += RENDER_DT

        # Free-camera movement (overview only)
        if not state.first_person_mode:
            handle_camera(state.cam_speed)

        # Single lock: check dirty + read atomically — skip GPU work if unchanged
        snap = read_snapshot_if_dirty()
        if snap is None:
            continue
        points, lines, target_pt, trajectory, _, _, _, rot_y = snap
        if not points:
            continue

        # Build retinal kwargs only in FPV mode
        retinal_kwargs: dict = {}
        fp_info: dict | None = None
        if state.first_person_mode:
            fp_info = _get_fp_head_info()
            if fp_info:
                retinal_kwargs = dict(
                    eye_camera  = state.eye_camera,
                    light_model = state.light,
                    head_pos    = (fp_info['hx'], fp_info['hy'], fp_info['hz']),
                    gaze_yaw    = fp_info['gy'],
                    gaze_pitch  = fp_info['gp'],
                    fov_deg     = fp_info['fov'],
                )

        eye_overlay = read_eye_data() if state.first_person_mode else None

        try:
            frame, retinal_result = render_frame(
                state.cam_yaw, state.cam_pitch, state.cam_roll,
                state.cam_x,   state.cam_y,    state.cam_z,
                points, lines, target_pt, rot_y,
                trajectory  = trajectory,
                population  = state.population,
                eye_overlay = eye_overlay,
                **retinal_kwargs,
            )
            write_frame(frame)

            if retinal_result is not None and fp_info is not None:
                _push_retinal_result(retinal_result, fp_info)

        except Exception:
            print("⚠️  Render exception:")
            traceback.print_exc()

    state.eye_camera.destroy()
    print("✅ Render stopped")
