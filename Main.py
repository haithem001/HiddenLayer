"""
main.py
───────
Entry point.  Responsibilities:
  1. Construct all heavyweight objects (Population, PPOAgent, Eye, …)
  2. Populate `state` from shared_state.py
  3. Start physics + render threads via their own modules
  4. Register WebSocket routes (ws_handlers.py)
  5. Serve HTTP endpoints (frame, video feed, reward plot, index)

No simulation logic lives here.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from population_display import Population
from ppo_agent import PPOAgent
from eye import Eye
from Eye_camera import EyeCamera
from Light_model import LightModel

from shared_state import state, read_frame
from physics_loop import run_physics_loop
from render_loop  import run_render_loop
from ws_handlers  import register_ws_routes

# ── Constants ─────────────────────────────────────────────────────────
WIDTH, HEIGHT     = 1600, 900
RENDER_FPS        = 60
RENDER_DT         = 1.0 / RENDER_FPS
Population_number = 1

TARGET = np.array([4.0, 1.25, 0.0], dtype=np.float32)

# ── Construct simulation objects ──────────────────────────────────────
print("Initialising population…")
population = Population(n=Population_number, target=TARGET)

print("Initialising PPO agent…")
ppo_agent = PPOAgent(
    target=tuple(TARGET), load_existing=True,
    device="cpu", number_of_population=Population_number,
)

print("Initialising eye / camera / light…")
eye        = Eye(D_width=float(WIDTH), D_height=float(HEIGHT))
eye_camera = EyeCamera()
light      = LightModel()

# ── Populate shared state ─────────────────────────────────────────────
state.TARGET            = TARGET
state.Population_number = Population_number
state.population        = population
state.ppo_agent         = ppo_agent
state.eye               = eye
state.eye_camera        = eye_camera
state.light             = light
state.EP_MAX_STEPS      = 3000

physics_stop = threading.Event()
render_stop  = threading.Event()
state.physics_stop = physics_stop
state.render_stop  = render_stop

# physics_loop.py wires its own callables (spawn_target_fn, etc.) into
# state at import time — nothing extra to do here.


# ── MJPEG generator ───────────────────────────────────────────────────
def _mjpeg():
    while not render_stop.is_set():
        f = read_frame()
        if f:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(RENDER_DT)


# ── FastAPI lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    state.async_loop = loop

    def _on_plot_ready():
        if state.async_loop is None:
            return
        async def _broadcast():
            dead = set()
            for ws in list(state.ws_connections):
                try:
                    await ws.send_json({"type": "plot_updated"})
                except Exception:
                    dead.add(ws)
            state.ws_connections.difference_update(dead)
        asyncio.run_coroutine_threadsafe(_broadcast(), state.async_loop)

    ppo_agent.on_plot_updated = _on_plot_ready
    physics_stop.clear()
    render_stop.clear()

    pt = threading.Thread(target=run_physics_loop, daemon=True, name="physics")
    rt = threading.Thread(target=run_render_loop,  daemon=True, name="render")
    pt.start()
    rt.start()
    await asyncio.sleep(0.5)

    try:
        yield
    finally:
        ppo_agent.stop_training()
        physics_stop.set()
        render_stop.set()
        pt.join(timeout=5.0)
        rt.join(timeout=5.0)
        population.close()
        ppo_agent.close()


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="templates")

register_ws_routes(app, state)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "version": str(int(time.time()))}
    )


@app.get("/frame")
def get_frame():
    f = read_frame()
    return Response(
        content=f or b"", media_type="image/jpeg",
        status_code=200 if f else 503,
    )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/reward_plot")
def get_reward_plot():
    p = "checkpoints/reward_curve.png"
    if os.path.exists(p):
        with open(p, "rb") as fh:
            data = fh.read()
        return Response(
            content=data, media_type="image/png",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache", "Expires": "0",
            },
        )
    return JSONResponse({"error": "No plot yet"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    print("🚀  http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
