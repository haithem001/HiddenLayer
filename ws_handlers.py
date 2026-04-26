"""
ws_handlers.py
──────────────
All WebSocket message routing.

Call `register_ws_routes(app, state)` from main.py.

Imports shared I/O helpers directly from shared_state instead of going
through state-callable indirection — simpler and one fewer dict copy per
call.
"""

from __future__ import annotations

import json
import math

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from shared_state import (
    state,
    read_snapshot,
    read_retinal,
    read_eye_data,
)


def register_ws_routes(app, st=None) -> None:
    """Attach the /ws endpoint to *app*.  `st` is ignored (uses module-level state)."""

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        state.ws_connections.add(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg   = json.loads(raw)
                    mtype = msg.get("type")
                    await _dispatch(ws, msg, mtype)
                except json.JSONDecodeError as e:
                    await _status(ws, f"Bad JSON: {e}", "error")
                except Exception as e:
                    print(f"WS dispatch error: {e}")
                    import traceback; traceback.print_exc()
        except WebSocketDisconnect:
            state.ws_connections.discard(ws)


# ── helpers ───────────────────────────────────────────────────────────

async def _status(ws: WebSocket, msg: str, t: str = "info") -> None:
    try:
        await ws.send_json({"type": "status_update", "message": msg, "status_type": t})
    except Exception:
        pass


async def _dispatch(ws: WebSocket, msg: dict, mtype: str) -> None:

    if mtype == "start":
        state.start = True
        state.pause = False
        await _status(ws, f"Started — {state.Population_number} walkers", "success")

    elif mtype == "pause":
        state.pause = not state.pause
        await _status(ws, "Paused" if state.pause else "▶ Resumed", "info")

    elif mtype == "dqn_activate":
        state.agent_active = True
        state.ppo_agent.disable_best_mode()
        await _status(ws, "Agent ON (stochastic)", "success")

    elif mtype == "dqn_deactivate":
        state.agent_active = False
        await _status(ws, "Agent OFF", "info")

    elif mtype == "train_start":
        state.ppo_agent.start_training()
        state.agent_active = True
        await _status(ws, "Training started", "success")

    elif mtype == "train_stop":
        state.ppo_agent.stop_training()
        await _status(ws, "Training stopped", "info")

    elif mtype == "save_model":
        state.ppo_agent.save()
        await _status(ws, "Saved walker_ppo.pt", "success")

    elif mtype == "load_best":
        state.ppo_agent.stop_training()
        ok = state.ppo_agent.load("checkpoints/walker_ppo_best.pt")
        if ok:
            state.ppo_agent.enable_best_mode()
            state.agent_active = True
            await _status(ws, "Best model loaded — deterministic mode ON", "success")
        else:
            await _status(ws, "No best checkpoint found", "error")

    elif mtype == "load_latest":
        ok = state.ppo_agent.load("Checkpoints/walker_ppo.pt")
        if ok:
            state.ppo_agent.disable_best_mode()
            await _status(ws, "Latest checkpoint loaded", "success")
        else:
            await _status(ws, "No checkpoint found", "error")

    elif mtype == "spawn_random_target":
        state.spawn_target_fn(state.reached_x, state.reached_z)
        await _status(
            ws, f"Target → ({state.TARGET[0]:.1f}, {state.TARGET[2]:.1f})", "success"
        )

    elif mtype == "reset":
        for h in state.population.humans:
            h.reset()
        state.clear_trajectory_fn()
        state.light.reset()
        await _status(ws, "All walkers reset", "success")

    elif mtype == "set_target":
        x = float(msg.get("x", 10.0))
        y = float(msg.get("y",  1.25))
        z = float(msg.get("z",  0.0))
        state.TARGET[:] = [x, y, z]
        state.population.set_target(x, y, z)
        state.ppo_agent.set_target(x, y, z)
        await _status(ws, f"Target → ({x:.1f},{y:.1f},{z:.1f})", "success")

    elif mtype == "toggle_camera":
        state.first_person_mode = not state.first_person_mode
        label = "👁 First-Person (retinal)" if state.first_person_mode else "🌍 Overview"
        await _status(ws, f"Camera: {label}", "info")
        await ws.send_json({"type": "camera_mode", "first_person": state.first_person_mode})

    elif mtype == "dqn_status":
        await _send_dqn_status(ws)

    elif mtype == "keydown":
        state.keys_pressed.add(msg["key"])
    elif mtype == "keyup":
        state.keys_pressed.discard(msg["key"])
    elif mtype == "mouse":
        if state.mouse_look:
            state.on_mouse_fn(msg)
    elif mtype == "mouse_look":
        state.mouse_look = msg.get("enabled", False)
    elif mtype == "camera_speed":
        state.cam_speed = float(msg.get("speed", 1.5))


async def _send_dqn_status(ws: WebSocket) -> None:
    """
    Build and send the dqn_status JSON frame — kept as cheap as possible:
    • Uses state.cached_walker_dists (written by physics each step, no sqrt here)
    • One read_snapshot() call
    • One read_retinal() call (single lock)
    • One read_eye_data() call (single lock)
    """
    _, _, _, _, n_alive, episode, best, _ = read_snapshot()
    retinal  = read_retinal()    # fresh dict, one lock
    eye_data = read_eye_data()   # fresh dict, one lock

    cached_dists = state.cached_walker_dists   # {i: float}, written by physics

    walkers_hormones = []
    gaze_states: dict[str, int] = {}

    for i, h in enumerate(state.population.humans):
        dist_tgt = cached_dists.get(i, 0.0)
        gaze_off = 0.0
        in_fov   = False

        if not h.dead and hasattr(h, "gaze"):
            g = h.gaze
            s = g.state
            gaze_states[s] = gaze_states.get(s, 0) + 1
            fwd_yaw  = math.atan2(h.physics.walk_dx, -h.physics.walk_dz)
            gaze_off = float(g.world_yaw - fwd_yaw)
            in_fov   = state.eye.target_in_fov

        walkers_hormones.append({
            "alive":           not h.dead,
            "hormones":        h.isv.get_all_hormones_grouped(),

        })


    await ws.send_json({
        "type": "dqn_status",
        "data": {
            "initialized":      True,
            "training":         state.ppo_agent.is_training,
            "agent_active":     state.agent_active,
            "first_person":     state.first_person_mode,
            "walkers_hormones": walkers_hormones,
        },
    })
