import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from humanoid import *
from population_display import *
from ppo_agent import PPOAgent
from Render import *
import asyncio
import pygame
import time
import math
import numpy as np
import random as _random

WIDTH, HEIGHT     = 1600, 900
RENDER_FPS        = 60
RENDER_DT         = 1.0 / RENDER_FPS
Population_number = 1

TARGET = np.array([4.0, 1.25, 0.0], dtype=np.float32)

# ── Target management ─────────────────────────────────────────────────
TARGET_REACH_DISTANCE  = 1.5
_target_reached_flag   = False
_targets_reached_count = 0

# ── Trajectory trail ──────────────────────────────────────────────────
MAX_TRAJECTORY_LEN  = 500
_trajectory_points: list = []
_traj_lock = threading.Lock()



def _record_trajectory():
    try:
        cx = float(np.mean([h.points[h.joint_names['torso']]['x']        for h in population.humans if not h.dead]))
        cy = float(np.mean([h.points[h.joint_names['torso']]['y']        for h in population.humans if not h.dead]))
        cz = float(np.mean([h.points[h.joint_names['torso']].get('z',0.) for h in population.humans if not h.dead]))
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
    n = _targets_reached_count

    dmin = 5
    dmax = 5

    distance = _random.uniform(dmin, dmax)
    angle    = _random.uniform(-math.pi, math.pi)

    new_x = spawn_x + distance * math.cos(angle)
    new_z = spawn_z + distance * math.sin(angle)
    new_y = 1.25

    TARGET[:] = [new_x, new_y, new_z]
    population.set_target(new_x, new_y, new_z)
    ppo_agent.set_target(new_x, new_y, new_z)

    deg = math.degrees(angle)
    print(f"🎯 Target #{n}: ({new_x:.2f}, {new_y:.2f}, {new_z:.2f})  "
          f"dist={distance:.2f}m  angle={deg:.1f}°  spawn=({spawn_x:.2f},{spawn_z:.2f})")

    population._reset_all(spawn_x=spawn_x, spawn_z=spawn_z, face_target=True)

    for i, h in enumerate(population.humans):
        ti  = h.joint_names['torso']
        tx  = h.points[ti]['x']
        tz  = h.points[ti].get('z', 0.0)
        bz  = h.points[ti].get('_base_z', 'MISSING')
        dxt = float(TARGET[0]) - tx
        dzt = float(TARGET[2]) - tz
        dist = math.sqrt(dxt**2 + dzt**2)
        print(f"  [REORIENT walker {i}] torso=({tx:.3f},{tz:.3f})  _base_z={bz:.3f}  "
              f"target=({TARGET[0]:.3f},{TARGET[2]:.3f})  "
              f"dX={dxt:.3f}  dZ={dzt:.3f}  dist={dist:.3f}m")


print("Initialising population…")
population = Population(n=Population_number, target=TARGET)

print("Initialising PPO agent…")
ppo_agent    = PPOAgent(target=tuple(TARGET), load_existing=True,
                        device="cpu", number_of_population=Population_number)
agent_active = False


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
    with frame_lock: frame_bytes = d

def read_frame():
    with frame_lock: return frame_bytes


# ── Control ───────────────────────────────────────────────────────────
physics_stop = threading.Event()
render_stop  = threading.Event()
start = False
pause = False

keys_pressed = set()

# ── Camera state ──────────────────────────────────────────────────────
cam_yaw, cam_pitch, cam_roll = -0.5, -0.4, 0.0
cam_x, cam_y, cam_z          = 0.0, 8.0, 18.0
mouse_look  = False
mouse_sens  = 0.005
cam_speed   = 0.5

# ── Camera mode ───────────────────────────────────────────────────────
first_person_mode = False
FP_EYE_HEIGHT = 0.35

ws_connections: set = set()
async_loop: Optional[asyncio.AbstractEventLoop] = None

total_episodes = 0
all_time_best  = float('inf')

EP_MAX_STEPS  = 3000
reached_x, reached_z = 0.0, 0.0


def _update_fp_camera():
    """
    FPV camera rides on the head and looks in the gaze direction.
    During SEARCHING the view sweeps left/right with the head.
    During PURSUING it locks on the target — exactly what the walker sees.
    """
    global cam_x, cam_y, cam_z, cam_yaw, cam_pitch

    try:
        alive = [h for h in population.humans if not h.dead]
        if not alive:
            return

        h  = alive[0]

        # ── Camera position: head joint if available, else above torso ──
        if 'head' in h.joint_names:
            hi   = h.joint_names['head']
            cam_x = float(h.points[hi]['x'])
            cam_y = float(h.points[hi]['y']) + 0.05   # slightly above head centre
            cam_z = float(h.points[hi].get('z', 0.0))
        else:
            ti    = h.joint_names['torso']
            cam_x = float(h.points[ti]['x'])
            cam_y = float(h.points[ti]['y']) + FP_EYE_HEIGHT
            cam_z = float(h.points[ti].get('z', 0.0))

        # ── Camera yaw: follow gaze direction, not body direction ────────
        if hasattr(h, 'gaze'):
            cam_yaw = h.gaze.world_yaw
            # Pitch: look slightly downward toward ground level (target is at ~1.25m)
            # Compute from gaze forward direction how far down to tilt
            gaze_fwd_x =  math.sin(h.gaze.world_yaw)
            gaze_fwd_z = -math.cos(h.gaze.world_yaw)
            # Project target onto gaze ray to get a sensible pitch
            ti  = h.joint_names['torso']
            tx  = float(h.points[ti]['x'])
            tz  = float(h.points[ti].get('z', 0.0))
            dy  = 1.25 - cam_y          # target height minus eye height
            # Horizontal distance along gaze ray toward target
            dx_t = float(TARGET[0]) - cam_x
            dz_t = float(TARGET[2]) - cam_z
            horiz = math.sqrt(dx_t*dx_t + dz_t*dz_t) + 1e-6
            cam_pitch = math.atan2(dy, horiz)
            cam_pitch = max(-math.pi/2 + 0.05, min(math.pi/2 - 0.05, cam_pitch))
        else:
            # Fallback: look at target directly
            dx = float(TARGET[0]) - cam_x
            dz = float(TARGET[2]) - cam_z
            cam_yaw = math.atan2(dx, -dz)
            horiz_dist = math.sqrt(dx*dx + dz*dz) + 1e-6
            cam_pitch = math.atan2(1.25 - cam_y, horiz_dist)
            cam_pitch = max(-math.pi/2 + 0.05, min(math.pi/2 - 0.05, cam_pitch))

    except Exception as e:
        print(f"⚠️  FP camera: {e}")


# ── Physics thread ─────────────────────────────────────────────────────

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
                target_point.update(x=float(TARGET[0]), y=float(TARGET[1]), z=float(TARGET[2]))
                write_snapshot(points, lines, target_point, _get_trajectory_snapshot(),
                               population.n_alive, total_episodes, all_time_best)
                time.sleep(0.008)
                continue

            # ── Always update gaze (head turns even before episode starts) ──
            population.update_gazes()

            # ── Check if all alive walkers have spotted the target ────
            all_detected = all(
                (h.gaze.state != 'searching') if hasattr(h, 'gaze') else True
                for h in population.humans if not h.dead
            )

            if not all_detected:
                # SEARCHING PHASE — render the head turning, nothing else moves
                points, lines = population.get_render_data()
                target_point.update(x=float(TARGET[0]), y=float(TARGET[1]), z=float(TARGET[2]))
                write_snapshot(points, lines, target_point, _get_trajectory_snapshot(),
                               population.n_alive, total_episodes, all_time_best)
                time.sleep(0.002)
                continue

            # ── Target detected — run the episode ─────────────────────
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
                for i, h in enumerate(population.humans):
                    if h.dead: continue
                    ti   = h.joint_names['torso']
                    tx   = h.points[ti]['x']
                    tz   = h.points[ti].get('z', 0.0)
                    bz   = h.points[ti].get('_base_z', '?')
                    vx   = h.velocities[ti]['vx']
                    vz   = h.velocities[ti].get('vz', 0.0)
                    dxt  = float(TARGET[0]) - tx
                    dzt  = float(TARGET[2]) - tz
                    dist = math.sqrt(dxt**2 + dzt**2)
                    print(f"  [STEP {ep_step:4d}] torso=({tx:.3f},{tz:.3f})  "
                          f"_base_z={bz}  vx={vx:.3f}  vz={vz:.3f}  "
                          f"target=({TARGET[0]:.3f},{TARGET[2]:.3f})  "
                          f"dX={dxt:.3f}  dZ={dzt:.3f}  dist={dist:.3f}m  "
                          f"gaze={h.gaze.state}")          # ← gaze state in log

            any_reached = any(
                math.sqrt(
                    (h.points[h.joint_names['torso']]['x']        - TARGET[0])**2 +
                    (h.points[h.joint_names['torso']].get('z',0.) - TARGET[2])**2
                ) < TARGET_REACH_DISTANCE
                for h in population.humans if not h.dead
            )

            if any_reached and not _target_reached_flag:
                _target_reached_flag = True

                reached_x = float(np.mean([h.points[h.joint_names['torso']]['x']        for h in population.humans if not h.dead]))
                reached_z = float(np.mean([h.points[h.joint_names['torso']].get('z',0.) for h in population.humans if not h.dead]))
                print(f"✅ Target reached! Walker at ({reached_x:.3f}, {reached_z:.3f})")

                ep_step = 0
                _spawn_target_and_reorient(reached_x, reached_z)
                target_point.update(x=float(TARGET[0]), y=float(TARGET[1]), z=float(TARGET[2]))
                # New target spawned — reset gaze so walker searches again
                for h in population.humans:
                    if hasattr(h, 'gaze'):
                        h.gaze.reset()
                print("👀 Searching for new target...")

            elif not any_reached:
                _target_reached_flag = False

            if agent_active and ppo_agent.is_training:
                ppo_agent.record_step(obs_batch, actions, log_probs, rewards, values, dones)

            best_dist = min(
                (math.sqrt(
                    (h.points[h.joint_names['torso']]['x']        - TARGET[0])**2 +
                    (h.points[h.joint_names['torso']].get('z',0.) - TARGET[2])**2
                ) for h in population.humans if not h.dead),
                default=float('inf'),
            )

            episode_done = (ep_step >= EP_MAX_STEPS or all(h.dead for h in population.humans))
            if episode_done:
                if best_dist < all_time_best:
                    all_time_best = best_dist
                total_episodes          += 1
                population.episode      += 1
                population.all_time_best = all_time_best
                reason = "⏱ time" if ep_step >= EP_MAX_STEPS else "💀 all dead"
                print(f"🏁 Ep {total_episodes} [{reason}] | steps={ep_step} | "
                      f"best_dist={best_dist:.2f}m | all_time_best={all_time_best:.2f}m | "
                      f"targets_reached={_targets_reached_count} | "
                      f"ppo_steps={ppo_agent.total_steps:,}")
                ep_step = 0
                population._reset_all(spawn_x=reached_x, spawn_z=reached_z, face_target=True)
                # Reset gaze so each new episode starts with a fresh search
                for h in population.humans:
                    if hasattr(h, 'gaze'):
                        h.gaze.reset()
                print("👀 Searching for target...")

            points, lines = population.get_render_data()
            target_point.update(x=float(TARGET[0]), y=float(TARGET[1]), z=float(TARGET[2]))
            write_snapshot(points, lines, target_point, _get_trajectory_snapshot(),
                           population.n_alive, population.episode,
                           population.all_time_best, rot_y=0.0)

        except Exception as e:
            print(f"⚠️  Physics: {e}")
            import traceback; traceback.print_exc()
            time.sleep(0.02)

    print("✅ Physics stopped")


# ── Render thread ──────────────────────────────────────────────────────

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
        print(f"❌ OpenGL: {e}"); return False


def render_loop():
    if not init_gl(): return
    print("Render thread started")
    next_t = time.perf_counter()
    while not render_stop.is_set():
        sleep_t = next_t - time.perf_counter()
        if sleep_t > 0.001: time.sleep(sleep_t - 0.001)
        while time.perf_counter() < next_t: pass
        next_t += RENDER_DT

        if first_person_mode:
            _update_fp_camera()
        else:
            handle_camera(cam_speed)

        points, lines, target_pt, trajectory, n_alive, episode, best, rot_y = read_snapshot()
        if not points:
            continue
        try:
            write_frame(render_frame(cam_yaw, cam_pitch, cam_roll,
                                     cam_x, cam_y, cam_z,
                                     points, lines, target_pt, rot_y,
                                     trajectory=trajectory,
                                     population=population))  # ← ADDED population kwarg
        except Exception as e:
            print(f"⚠️  Render: {e}")
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
    if "z" in keys_pressed: cam_x += fx*step; cam_z += fz*step
    if "s" in keys_pressed: cam_x -= fx*step; cam_z -= fz*step
    if "d" in keys_pressed: cam_x += rx*step; cam_z += rz*step
    if "q" in keys_pressed: cam_x -= rx*step; cam_z -= rz*step
    cam_x = float(np.clip(cam_x, -500., 500.))
    cam_y = float(np.clip(cam_y,    1., 200.))
    cam_z = float(np.clip(cam_z, -200., 200.))
    cam_pitch = float(np.clip(cam_pitch, -math.pi/2+0.01, math.pi/2-0.01))


# ── FastAPI ─────────────────────────────────────────────────────────────

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
        if f: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
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
        return Response(
            content=data,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma":        "no-cache",
                "Expires":       "0",
            },
        )
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
                    for h in population.humans: h.reset()
                    _clear_trajectory()
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
                    mode_name = "👁 First-Person" if first_person_mode else "🌍 Overview"
                    await _status(ws, f"Camera: {mode_name}", "info")
                    await ws.send_json({"type": "camera_mode",
                                        "first_person": first_person_mode})

                elif mtype == "dqn_status":
                    _, _, _, _, n_alive, episode, best, _ = read_snapshot()

                    # collect hormone data per walker
                    walkers_hormones = []
                    for h in population.humans:
                        if hasattr(h, 'isv'):
                            walkers_hormones.append({
                                "alive": not h.dead,
                                "hormones": h.isv.get_all_hormones_grouped()
                            })
                    if walkers_hormones:
                        print(f"[HORMONE DEBUG] Walker 0 G1: {walkers_hormones[0]['hormones'].get('G1', {})}")

                    # collect gaze states
                    gaze_states = {}
                    for h in population.humans:
                        if not h.dead and hasattr(h, 'gaze'):
                            s = h.gaze.state
                            gaze_states[s] = gaze_states.get(s, 0) + 1

                    # send everything in ONE message
                    await ws.send_json({"type": "dqn_status", "data": {
                        'initialized':     True,
                        'training':        ppo_agent.is_training,
                        'agent_active':    agent_active,
                        'deterministic':   ppo_agent._deterministic,
                        'episode':         episode,
                        'total_steps':     ppo_agent.total_steps,
                        'episode_reward':  float(np.mean(ppo_agent.episode_rewards[-5:])) if ppo_agent.episode_rewards else 0.0,
                        'best_distance':   round(all_time_best if all_time_best < 1e9 else 0., 3),
                        'target_position': TARGET.tolist(),
                        'targets_reached': _targets_reached_count,
                        'n_alive':         n_alive,
                        'epsilon':         0.,
                        'buffer_size':     Population_number,
                        'trajectory_len':  len(_trajectory_points),
                        'first_person':    first_person_mode,
                        'gaze_states':     gaze_states,
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
