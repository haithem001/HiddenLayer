import json
import math
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import asyncio
import numpy as np
import pygame
import random as _random
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.templating import Jinja2Templates

from population_display import *
from ppo_agent import PPOAgent
from Render import *
from eye import Eye
from Eye_camera import EyeCamera
from Light_model import LightModel

WIDTH, HEIGHT     = 1600, 900
RENDER_FPS        = 60
RENDER_DT         = 1.0 / RENDER_FPS
Population_number = 1

TARGET = np.array([4.0, 1.25, 0.0], dtype=np.float32)

TARGET_REACH_DISTANCE  = 1.5
_target_reached_flag   = False
_targets_reached_count = 0

MAX_TRAJECTORY_LEN  = 500
_trajectory_points: list = []
_traj_lock = threading.Lock()


def _record_trajectory():
    try:
        cx = float(np.mean([h.points[h.joint_names['torso']]['x']
                            for h in population.humans if not h.dead]))
        cy = float(np.mean([h.points[h.joint_names['torso']]['y']
                            for h in population.humans if not h.dead]))
        cz = float(np.mean([h.points[h.joint_names['torso']].get('z', 0.)
                            for h in population.humans if not h.dead]))
    except Exception:
        return
    with _traj_lock:
        _trajectory_points.append({'x': cx, 'y': cy, 'z': cz})
        if len(_trajectory_points) > MAX_TRAJECTORY_LEN:
            _trajectory_points.pop(0)


def _get_trajectory_snapshot():
    with _traj_lock:
        return list(_trajectory_points)


def _clear_trajectory():
    global _trajectory_points
    with _traj_lock:
        _trajectory_points = []


def _spawn_target_and_reorient(spawn_x: float, spawn_z: float):
    global TARGET, _targets_reached_count
    _targets_reached_count += 1
    n        = _targets_reached_count
    distance = _random.uniform(5, 5)
    angle    = _random.uniform(-math.pi, math.pi)
    new_x    = spawn_x + distance * math.cos(angle)
    new_z    = spawn_z + distance * math.sin(angle)
    TARGET[:] = [new_x, 1.25, new_z]
    population.set_target(new_x, 1.25, new_z)
    ppo_agent.set_target(new_x, 1.25, new_z)
    # threading.Thread(target=ppo_agent.visualize_actor_critic,
    #                  daemon=True, name="plot").start()
    print(f"🎯 Target #{n}: ({new_x:.2f}, 1.25, {new_z:.2f})  "
          f"angle={math.degrees(angle):.1f}°")
    population.reset_all(spawn_x=spawn_x, spawn_z=spawn_z, face_target=True)


print("Initialising population…")
population = Population(n=Population_number, target=TARGET)

print("Initialising PPO agent…")
ppo_agent    = PPOAgent(target=tuple(TARGET), load_existing=True,
                        device="cpu", number_of_population=Population_number)
agent_active = False

# ── Eye system ────────────────────────────────────────────────────────
_eye        = Eye(D_width=float(WIDTH), D_height=float(HEIGHT))
_eye_camera = EyeCamera()
_light      = LightModel()
_eye_lock   = threading.Lock()
_eye_data:  dict = {}

# Retinal result shared between render thread → physics thread
_retinal_lock   = threading.Lock()
_retinal_result = {'detected': False, 'retinal_x': 0.0, 'retinal_y': 0.0,
                   'lux': 0.5, 'dilation': 0.5}


def _read_eye_data() -> dict:
    with _eye_lock:
        return dict(_eye_data)


def _get_sweep_offset() -> float:
    """Return the current head-sweep offset from the first alive walker."""
    try:
        alive = [h for h in population.humans if not h.dead]
        if alive and hasattr(alive[0], 'gaze'):
            return float(alive[0].gaze._sweep_offset)
    except Exception:
        pass
    return 0.0


def _update_eye_from_retinal(gaze_state: str, target_dist: float):
    """Called after render processes a retinal frame."""
    with _retinal_lock:
        snap = dict(_retinal_result)
    ld = _light.get_render_data()
    _eye.update_from_retina(
        retinal_x      = snap['retinal_x'],
        retinal_y      = snap['retinal_y'],
        detected       = snap['detected'],
        gaze_state     = gaze_state,
        target_dist    = target_dist,
        lux            = ld['ambient_lux'],
        pupil_dilation = ld['pupil_dilation'],
        vignette_alpha = ld['vignette_alpha'],
        sweep_offset   = _get_sweep_offset(),
    )
    with _eye_lock:
        _eye_data.update(_eye.get_render_data())


def _update_eye_overview():
    """Angular-math eye update for overview mode (no retinal render)."""
    try:
        alive = [h for h in population.humans if not h.dead]
        if not alive:
            return
        h  = alive[0]
        ti = h.joint_names['torso']
        if not hasattr(h, 'gaze'):
            return

        if 'head' in h.joint_names:
            hi = h.joint_names['head']
            hx = float(h.points[hi]['x'])
            hy = float(h.points[hi]['y'])
            hz = float(h.points[hi].get('z', 0.0))
        else:
            hx = float(h.points[ti]['x'])
            hy = float(h.points[ti]['y']) + 0.22
            hz = float(h.points[ti].get('z', 0.0))

        dx   = float(TARGET[0]) - hx
        dz   = float(TARGET[2]) - hz
        dist = math.sqrt(dx * dx + dz * dz) + 1e-6

        _eye.update_from_walker(
            head_yaw      = h.gaze.world_yaw,
            head_pitch    = 0.0,
            target_yaw    = math.atan2(dx, -dz),
            target_pitch  = math.atan2(float(TARGET[1]) - hy, dist),
            gaze_state    = h.gaze.state,
            target_dist   = dist,
            fov_half      = h.gaze.fov_half_angle,
            sweep_offset  = _get_sweep_offset(),
        )
        with _eye_lock:
            _eye_data.update(_eye.get_render_data())
    except Exception as e:
        print(f"⚠️  Eye overview: {e}")


def _get_fp_head_info():
    """Head camera info for the first alive walker. Returns dict or None."""
    global cam_x, cam_y, cam_z, cam_yaw, cam_pitch
    try:
        alive = [h for h in population.humans if not h.dead]
        if not alive:
            return None
        h = alive[0]

        if 'head' in h.joint_names:
            hi    = h.joint_names['head']
            cam_x = float(h.points[hi]['x'])
            cam_y = float(h.points[hi]['y']) + 0.05
            cam_z = float(h.points[hi].get('z', 0.0))
        else:
            ti    = h.joint_names['torso']
            cam_x = float(h.points[ti]['x'])
            cam_y = float(h.points[ti]['y']) + FP_EYE_HEIGHT
            cam_z = float(h.points[ti].get('z', 0.0))

        if hasattr(h, 'gaze'):
            cam_yaw   = h.gaze.world_yaw
            dx_t      = float(TARGET[0]) - cam_x
            dz_t      = float(TARGET[2]) - cam_z
            horiz     = math.sqrt(dx_t*dx_t + dz_t*dz_t) + 1e-6
            cam_pitch = math.atan2(1.25 - cam_y, horiz)
            cam_pitch = max(-math.pi/2 + 0.05, min(math.pi/2 - 0.05, cam_pitch))
            fov_deg   = math.degrees(h.gaze.fov_half_angle * 2.0)
            fov_deg   = max(10.0, min(170.0, fov_deg))
            dist      = horiz
            gstate    = h.gaze.state
        else:
            dx = float(TARGET[0]) - cam_x
            dz = float(TARGET[2]) - cam_z
            cam_yaw   = math.atan2(dx, -dz)
            horiz     = math.sqrt(dx*dx + dz*dz) + 1e-6
            cam_pitch = math.atan2(1.25 - cam_y, horiz)
            cam_pitch = max(-math.pi/2 + 0.05, min(math.pi/2 - 0.05, cam_pitch))
            fov_deg   = 60.0
            dist      = horiz
            gstate    = 'searching'

        return dict(hx=cam_x, hy=cam_y, hz=cam_z,
                    gy=cam_yaw, gp=cam_pitch,
                    fov=fov_deg, dist=dist, gstate=gstate)
    except Exception as e:
        print(f"⚠️  FP head info: {e}")
        return None


# ── Snapshot ──────────────────────────────────────────────────────────
snap_lock = threading.Lock()
snapshot  = {
    'points': [], 'lines': [], 'target_pt': None, 'trajectory': [],
    'n_alive': Population_number, 'episode': 0, 'best': float('inf'), 'rot_y': 0.0,
}


def write_snapshot(points, lines, target_pt, trajectory, n_alive, episode, best, rot_y=0.0):
    with snap_lock:
        snapshot.update(points=points, lines=lines, target_pt=target_pt,
                        trajectory=trajectory, n_alive=n_alive,
                        episode=episode, best=best, rot_y=rot_y)


def read_snapshot():
    with snap_lock:
        return (snapshot['points'], snapshot['lines'], snapshot['target_pt'],
                snapshot['trajectory'], snapshot['n_alive'], snapshot['episode'],
                snapshot['best'], snapshot.get('rot_y', 0.0))


# ── Frame ─────────────────────────────────────────────────────────────
frame_lock  = threading.Lock()
frame_bytes: Optional[bytes] = None


def write_frame(d):
    global frame_bytes
    with frame_lock:
        frame_bytes = d


def read_frame():
    with frame_lock:
        return frame_bytes


# ── Control ───────────────────────────────────────────────────────────
physics_stop = threading.Event()
render_stop  = threading.Event()
start = False
pause = False
keys_pressed = set()

cam_yaw, cam_pitch, cam_roll = -0.5, -0.4, 0.0
cam_x, cam_y, cam_z          = 0.0, 8.0, 18.0
mouse_look  = False
mouse_sens  = 0.005
cam_speed   = 0.5

first_person_mode = False
FP_EYE_HEIGHT     = 0.35

ws_connections: set = set()
async_loop: Optional[asyncio.AbstractEventLoop] = None

total_episodes = 0
all_time_best  = float('inf')
EP_MAX_STEPS   = 3000
reached_x, reached_z = 0.0, 0.0


# ── Physics thread ────────────────────────────────────────────────────

def physics_loop():
    global total_episodes, all_time_best, agent_active
    global _target_reached_flag, reached_x, reached_z

    print("Physics thread started")
    target_point = {
        'x': float(TARGET[0]), 'y': float(TARGET[1]),
        'z': float(TARGET[2]), 'mass': 50.0, 'radius': 0.4,
    }
    ep_step = 0

    while not physics_stop.is_set():
        try:
            if not start or pause:
                points, lines = population.get_render_data()
                target_point.update(x=float(TARGET[0]), y=float(TARGET[1]),
                                    z=float(TARGET[2]))
                write_snapshot(points, lines, target_point,
                               _get_trajectory_snapshot(),
                               population.n_alive, total_episodes, all_time_best)
                _update_eye_overview()
                time.sleep(0.008)
                continue

            # Gaze FSM always ticks
            population.update_gazes()

            # Keep eye alive in overview mode too
            if not first_person_mode:
                _update_eye_overview()

            # Gate: wait for walkers to detect target
            with _retinal_lock:
                retinal_detected = _retinal_result['detected']

            all_detected = all(
                (h.gaze.state != 'searching') if hasattr(h, 'gaze') else True
                for h in population.humans if not h.dead
            ) or (first_person_mode and retinal_detected)

            if not all_detected:
                points, lines = population.get_render_data()
                target_point.update(x=float(TARGET[0]), y=float(TARGET[1]),
                                    z=float(TARGET[2]))
                write_snapshot(points, lines, target_point,
                               _get_trajectory_snapshot(),
                               population.n_alive, total_episodes, all_time_best)
                time.sleep(0.002)
                continue

            # Episode step
            obs_batch = population.get_observator_batch()

            if agent_active:
                if ppo_agent.is_training:
                    actions, log_probs, values = ppo_agent.get_action_and_info(obs_batch)
                else:
                    actions   = ppo_agent.get_best_action(obs_batch)
                    log_probs = np.zeros(Population_number, np.float32)
                    values    = np.zeros(Population_number, np.float32)
            else:
                actions   = np.zeros((Population_number, 6), np.float32)
                log_probs = np.zeros(Population_number,      np.float32)
                values    = np.zeros(Population_number,      np.float32)

            rewards = np.zeros(Population_number, np.float32)
            dones   = np.zeros(Population_number, np.float32)

            for i, human in enumerate(population.humans):
                if human.dead:
                    dones[i] = 1.0
                    continue
                _, reward, done, _ = human.step(actions[i])
                rewards[i] = float(reward)
                dones[i]   = float(done)

            ep_step += 1
            _record_trajectory()

            if ep_step % 300 == 1:
                with _retinal_lock:
                    r_det = _retinal_result['detected']
                    r_lux = _retinal_result['lux']
                for i, h in enumerate(population.humans):
                    if h.dead:
                        continue
                    ti   = h.joint_names['torso']
                    tx   = h.points[ti]['x']
                    tz   = h.points[ti].get('z', 0.0)
                    vx   = h.velocities[ti]['vx']
                    vz   = h.velocities[ti].get('vz', 0.0)
                    dist = math.sqrt((float(TARGET[0])-tx)**2 +
                                     (float(TARGET[2])-tz)**2)
                    print(f"  [STEP {ep_step:4d}] torso=({tx:.3f},{tz:.3f})  "
                          f"vx={vx:.3f}  vz={vz:.3f}  dist={dist:.3f}m  "
                          f"gaze={h.gaze.state}  "
                          f"retinal_det={r_det}  lux={r_lux:.2f}")

            any_reached = any(
                math.sqrt(
                    (h.points[h.joint_names['torso']]['x']        - TARGET[0])**2 +
                    (h.points[h.joint_names['torso']].get('z', 0.) - TARGET[2])**2
                ) < TARGET_REACH_DISTANCE
                for h in population.humans if not h.dead
            )

            if any_reached and not _target_reached_flag:
                _target_reached_flag = True
                reached_x = float(np.mean([
                    h.points[h.joint_names['torso']]['x']
                    for h in population.humans if not h.dead]))
                reached_z = float(np.mean([
                    h.points[h.joint_names['torso']].get('z', 0.)
                    for h in population.humans if not h.dead]))
                print(f"✅ Target reached! Walker at ({reached_x:.3f}, {reached_z:.3f})")
                ep_step = 0
                _clear_trajectory()
                _spawn_target_and_reorient(reached_x, reached_z)
                for h in population.humans:
                    if hasattr(h, 'gaze'):
                        h.gaze.reset()
                _light.reset()
                target_point.update(x=float(TARGET[0]), y=float(TARGET[1]),
                                    z=float(TARGET[2]))
                print("👀 Searching for new target (retinal)…")

            elif not any_reached:
                _target_reached_flag = False

            if agent_active and ppo_agent.is_training:
                ppo_agent.record_step(obs_batch, actions, log_probs,
                                      rewards, values, dones)

            best_dist = min(
                (math.sqrt(
                    (h.points[h.joint_names['torso']]['x']        - TARGET[0])**2 +
                    (h.points[h.joint_names['torso']].get('z', 0.) - TARGET[2])**2
                ) for h in population.humans if not h.dead),
                default=float('inf'),
            )

            episode_done = (ep_step >= EP_MAX_STEPS or
                            all(h.dead for h in population.humans))
            if episode_done:
                if best_dist < all_time_best:
                    all_time_best = best_dist
                total_episodes          += 1
                population.episode      += 1
                population.all_time_best = all_time_best
                reason = "⏱ time" if ep_step >= EP_MAX_STEPS else "💀 all dead"
                print(f"🏁 Ep {total_episodes} [{reason}] | steps={ep_step} | "
                      f"best_dist={best_dist:.2f}m | "
                      f"targets_reached={_targets_reached_count}")
                ep_step = 0
                _clear_trajectory()
                _light.reset()
                population.reset_all(spawn_x=reached_x, spawn_z=reached_z,
                                      face_target=True)
                for h in population.humans:
                    if hasattr(h, 'gaze'):
                        h.gaze.reset()

            points, lines = population.get_render_data()
            target_point.update(x=float(TARGET[0]), y=float(TARGET[1]),
                                z=float(TARGET[2]))
            write_snapshot(points, lines, target_point,
                           _get_trajectory_snapshot(),
                           population.n_alive, population.episode,
                           population.all_time_best, rot_y=0.0)

        except Exception as e:
            print(f"⚠️  Physics: {e}")
            import traceback; traceback.print_exc()
            time.sleep(0.02)

    print("✅ Physics stopped")


# ── Render thread ─────────────────────────────────────────────────────

def init_gl():
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


def render_loop():
    if not init_gl():
        return
    print("Render thread started")
    next_t = time.perf_counter()

    while not render_stop.is_set():
        sleep_t = next_t - time.perf_counter()
        if sleep_t > 0:
            time.sleep(sleep_t)
        next_t += RENDER_DT

        if not first_person_mode:
            handle_camera(cam_speed)

        points, lines, target_pt, trajectory, n_alive, episode, best, rot_y = read_snapshot()
        if not points:
            continue

        # Build retinal kwargs only in FPV mode
        retinal_kwargs = {}
        fp_info        = None
        if first_person_mode:
            fp_info = _get_fp_head_info()
            if fp_info:
                retinal_kwargs = dict(
                    eye_camera  = _eye_camera,
                    light_model = _light,
                    head_pos    = (fp_info['hx'], fp_info['hy'], fp_info['hz']),
                    gaze_yaw    = fp_info['gy'],
                    gaze_pitch  = fp_info['gp'],
                    fov_deg     = fp_info['fov'],
                )

        eye_overlay = _read_eye_data() if first_person_mode else None

        try:
            # render_frame ALWAYS returns (bytes, retinal_result | None)
            frame, retinal_result = render_frame(
                cam_yaw, cam_pitch, cam_roll,
                cam_x, cam_y, cam_z,
                points, lines, target_pt, rot_y,
                trajectory  = trajectory,
                population  = population,
                eye_overlay = eye_overlay,
                **retinal_kwargs,
            )
            write_frame(frame)

            # Process retinal result
            if retinal_result is not None and fp_info is not None:
                ld = _light.get_render_data()
                with _retinal_lock:
                    _retinal_result.update(
                        detected  = retinal_result['detected'],
                        retinal_x = retinal_result['retinal_x'],
                        retinal_y = retinal_result['retinal_y'],
                        lux       = ld['ambient_lux'],
                        dilation  = ld['pupil_dilation'],
                    )
                _update_eye_from_retinal(fp_info['gstate'], fp_info['dist'])

                # Push retinal detection into HeadGaze FSM if it supports it
                alive = [h for h in population.humans if not h.dead]
                if alive and hasattr(alive[0], 'gaze'):
                    g = alive[0].gaze
                    if hasattr(g, 'set_retinal_detection'):
                        g.set_retinal_detection(retinal_result['detected'])

        except Exception as e:
            print(f"⚠️  Render: {e}")
            import traceback; traceback.print_exc()

    _eye_camera.destroy()
    print("✅ Render stopped")


def on_mouse(msg):
    global cam_yaw, cam_pitch
    if not first_person_mode:
        cam_yaw   += msg["x"] * mouse_sens
        cam_pitch -= msg["y"] * mouse_sens


def handle_camera(step):
    global cam_x, cam_y, cam_z, cam_yaw, cam_pitch
    fx, fz = math.sin(cam_yaw), -math.cos(cam_yaw)
    rx, rz = math.cos(cam_yaw),  math.sin(cam_yaw)
    if "w" in keys_pressed: cam_y += step
    if "x" in keys_pressed: cam_y -= step
    if "z" in keys_pressed: cam_x += fx * step; cam_z += fz * step
    if "s" in keys_pressed: cam_x -= fx * step; cam_z -= fz * step
    if "d" in keys_pressed: cam_x += rx * step; cam_z += rz * step
    if "q" in keys_pressed: cam_x -= rx * step; cam_z -= rz * step
    cam_x = float(np.clip(cam_x, -500., 500.))
    cam_y = float(np.clip(cam_y,    1., 200.))
    cam_z = float(np.clip(cam_z, -200., 200.))
    cam_pitch = float(np.clip(cam_pitch, -math.pi/2 + 0.01, math.pi/2 - 0.01))


# ── FastAPI ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global async_loop
    async_loop = asyncio.get_running_loop()

    def _on_plot_ready():
        if async_loop is None:
            return
        async def _broadcast():
            dead = set()
            for ws in list(ws_connections):
                try:
                    await ws.send_json({"type": "plot_updated"})
                except Exception:
                    dead.add(ws)
            ws_connections.difference_update(dead)
        asyncio.run_coroutine_threadsafe(_broadcast(), async_loop)

    ppo_agent.on_plot_updated = _on_plot_ready
    physics_stop.clear(); render_stop.clear()
    pt = threading.Thread(target=physics_loop, daemon=True, name="physics")
    rt = threading.Thread(target=render_loop,  daemon=True, name="render")
    pt.start(); rt.start()
    await asyncio.sleep(0.5)
    try:
        yield
    finally:
        ppo_agent.stop_training()
        physics_stop.set(); render_stop.set()
        pt.join(timeout=5.); rt.join(timeout=5.)
        population.close()
        ppo_agent.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "version": str(int(time.time()))})


@app.get("/frame")
def get_frame():
    f = read_frame()
    return Response(content=f or b"", media_type="image/jpeg",
                    status_code=200 if f else 503)


def _mjpeg():
    while not render_stop.is_set():
        f = read_frame()
        if f:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(RENDER_DT)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/reward_plot")
def get_reward_plot():
    p = "checkpoints/reward_curve.png"
    if os.path.exists(p):
        with open(p, "rb") as f:
            data = f.read()
        return Response(content=data, media_type="image/png",
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                                 "Pragma": "no-cache", "Expires": "0"})
    from fastapi.responses import JSONResponse
    return JSONResponse({"error": "No plot yet"}, status_code=404)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global start, pause, agent_active, first_person_mode
    global keys_pressed, mouse_look, cam_speed

    await ws.accept()
    ws_connections.add(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg   = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "start":
                    start = True; pause = False
                    await _status(ws, f"Started — {Population_number} walkers", "success")
                elif mtype == "pause":
                    pause = not pause
                    await _status(ws, "Paused" if pause else "▶ Resumed", "info")
                elif mtype == "dqn_activate":
                    agent_active = True
                    ppo_agent.disable_best_mode()
                    await _status(ws, "Agent ON (stochastic)", "success")
                elif mtype == "dqn_deactivate":
                    agent_active = False
                    await _status(ws, "Agent OFF", "info")
                elif mtype == "train_start":
                    ppo_agent.start_training()
                    agent_active = True
                    await _status(ws, "Training started", "success")
                elif mtype == "train_stop":
                    ppo_agent.stop_training()
                    await _status(ws, "Training stopped", "info")
                elif mtype == "save_model":
                    ppo_agent.save()
                    await _status(ws, "Saved walker_ppo.pt", "success")
                elif mtype == "load_best":
                    ppo_agent.stop_training()
                    ok = ppo_agent.load('checkpoints/walker_ppo_best.pt')
                    if ok:
                        ppo_agent.enable_best_mode()
                        agent_active = True
                        await _status(ws, "Best model loaded — deterministic mode ON", "success")
                    else:
                        await _status(ws, "No best checkpoint found", "error")
                elif mtype == "load_latest":
                    ok = ppo_agent.load('checkpoints/walker_ppo.pt')
                    if ok:
                        ppo_agent.disable_best_mode()
                        await _status(ws, "Latest checkpoint loaded", "success")
                    else:
                        await _status(ws, "No checkpoint found", "error")
                elif mtype == "spawn_random_target":
                    _spawn_target_and_reorient(reached_x, reached_z)
                    await _status(ws, f"Target → ({TARGET[0]:.1f}, {TARGET[2]:.1f})", "success")
                elif mtype == "reset":
                    for h in population.humans:
                        h.reset()
                    _clear_trajectory()
                    _light.reset()
                    await _status(ws, "All walkers reset", "success")
                elif mtype == "set_target":
                    x = float(msg.get("x", 10.0))
                    y = float(msg.get("y", 1.25))
                    z = float(msg.get("z",  0.0))
                    TARGET[:] = [x, y, z]
                    population.set_target(x, y, z)
                    ppo_agent.set_target(x, y, z)
                    await _status(ws, f"Target → ({x:.1f},{y:.1f},{z:.1f})", "success")
                elif mtype == "toggle_camera":
                    first_person_mode = not first_person_mode
                    mode = "👁 First-Person (retinal)" if first_person_mode else "🌍 Overview"
                    await _status(ws, f"Camera: {mode}", "info")
                    await ws.send_json({"type": "camera_mode",
                                        "first_person": first_person_mode})

                elif mtype == "dqn_status":
                    _, _, _, _, n_alive, episode, best, _ = read_snapshot()
                    with _retinal_lock:
                        r_snap = dict(_retinal_result)

                    walkers_hormones = []
                    for h in population.humans:
                        gaze_off = 0.0; dist_tgt = 0.0; in_fov = False
                        if not h.dead and hasattr(h, 'gaze'):
                            fwd_yaw  = math.atan2(h.physics.walk_dx, -h.physics.walk_dz)
                            gaze_off = float(h.gaze.world_yaw - fwd_yaw)
                            ti       = h.joint_names['torso']
                            tx       = h.points[ti]['x']
                            tz       = h.points[ti].get('z', 0.0)
                            dist_tgt = math.sqrt((float(TARGET[0])-tx)**2 +
                                                 (float(TARGET[2])-tz)**2)
                            in_fov   = _eye.target_in_fov

                        walkers_hormones.append({
                            "alive":           not h.dead,
                            "hormones":        h.isv.get_all_hormones_grouped(),
                            "gaze_state":      h.gaze.state if hasattr(h, 'gaze') else 'unknown',
                            "gaze_offset_yaw": gaze_off,
                            "target_dist":     round(dist_tgt, 3),
                            "target_in_fov":   in_fov,
                            "eye": {
                                "pupil_x":     round(_eye.pupil_center_X, 3),
                                "pupil_y":     round(_eye.pupil_center_Y, 3),
                                "pupil_r":     round(_eye.pupil_radius,   3),
                                "state":       _eye.gaze_state,
                                "in_fov":      _eye.target_in_fov,
                                "retinal_det": r_snap['detected'],
                                "retinal_x":   round(r_snap['retinal_x'], 3),
                                "retinal_y":   round(r_snap['retinal_y'], 3),
                                "ambient_lux": round(r_snap['lux'],       3),
                                "dilation":    round(r_snap['dilation'],   3),
                            },
                        })

                    gaze_states = {}
                    for h in population.humans:
                        if not h.dead and hasattr(h, 'gaze'):
                            s = h.gaze.state
                            gaze_states[s] = gaze_states.get(s, 0) + 1

                    await ws.send_json({"type": "dqn_status", "data": {
                        'initialized':      True,
                        'training':         ppo_agent.is_training,
                        'agent_active':     agent_active,
                        'deterministic':    ppo_agent._deterministic,
                        'episode':          episode,
                        'total_steps':      ppo_agent.total_steps,
                        'episode_reward':   float(np.mean(ppo_agent.episode_rewards[-5:])) if ppo_agent.episode_rewards else 0.0,
                        'best_distance':    round(all_time_best if all_time_best < 1e9 else 0., 3),
                        'target_position':  TARGET.tolist(),
                        'targets_reached':  _targets_reached_count,
                        'n_alive':          n_alive,
                        'epsilon':          0.,
                        'buffer_size':      Population_number,
                        'trajectory_len':   len(_trajectory_points),
                        'first_person':     first_person_mode,
                        'gaze_states':      gaze_states,
                        'walkers_hormones': walkers_hormones,
                    }})

                elif mtype == "keydown":      keys_pressed.add(msg["key"])
                elif mtype == "keyup":        keys_pressed.discard(msg["key"])
                elif mtype == "mouse":
                    if mouse_look: on_mouse(msg)
                elif mtype == "mouse_look":   mouse_look = msg.get("enabled", False)
                elif mtype == "camera_speed": cam_speed  = float(msg.get("speed", 1.5))

            except json.JSONDecodeError as e:
                await _status(ws, f"Bad JSON: {e}", "error")
            except Exception as e:
                print(f"WS: {e}")
                import traceback; traceback.print_exc()

    except WebSocketDisconnect:
        ws_connections.discard(ws)


async def _status(ws, msg, t="info"):
    try:
        await ws.send_json({"type": "status_update", "message": msg, "status_type": t})
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    print("🚀  http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
