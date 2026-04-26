"""
physics_loop.py
───────────────
All simulation logic: walker steps, PPO agent, gaze FSM, trajectory
recording, target-reach detection, episode management, and eye model
updates for overview mode.

Public API
──────────
    run_physics_loop()  — target for the physics thread

Dependencies on shared_state
─────────────────────────────
    Reads : state.{start, pause, agent_active, first_person_mode,
                   TARGET, population, ppo_agent, eye, light,
                   Population_number, EP_MAX_STEPS, …}
    Writes: state.{cached_walker_dists, trajectory_len, targets_reached_count,
                   reached_x, reached_z, total_episodes, all_time_best}
    Calls : write_snapshot(), write_eye_data(), read_retinal()
"""

from __future__ import annotations

import math
import random as _random
import time
import traceback

import numpy as np

from shared_state import (
    state,
    write_snapshot, read_retinal,
    write_eye_data,
)

# Trajectory constants
MAX_TRAJECTORY_LEN  = 500
TARGET_REACH_DISTANCE = 1.5

# Module-private mutable state
_trajectory_points: list[dict] = []
_target_reached_flag: bool     = False


# ── Trajectory helpers ────────────────────────────────────────────────

def _record_trajectory() -> None:
    """Append current centroid to trajectory.  Fast-path for n=1 walker."""
    try:
        humans = [h for h in state.population.humans if not h.dead]
        if not humans:
            return
        if len(humans) == 1:
            pt = humans[0].points[humans[0].joint_names['torso']]
            cx, cy, cz = float(pt['x']), float(pt['y']), float(pt.get('z', 0.))
        else:
            cx = float(np.mean([h.points[h.joint_names['torso']]['x'] for h in humans]))
            cy = float(np.mean([h.points[h.joint_names['torso']]['y'] for h in humans]))
            cz = float(np.mean([h.points[h.joint_names['torso']].get('z', 0.) for h in humans]))
    except Exception:
        return

    _trajectory_points.append({'x': cx, 'y': cy, 'z': cz})
    if len(_trajectory_points) > MAX_TRAJECTORY_LEN:
        _trajectory_points.pop(0)
    state.trajectory_len = len(_trajectory_points)


def _get_trajectory_snapshot() -> list[dict]:
    return list(_trajectory_points)


def _clear_trajectory() -> None:
    _trajectory_points.clear()
    state.trajectory_len = 0


# ── Target helpers ────────────────────────────────────────────────────

def _spawn_target_and_reorient(spawn_x: float, spawn_z: float) -> None:
    state.targets_reached_count += 1
    n        = state.targets_reached_count
    distance = _random.uniform(5, 5)
    angle    = _random.uniform(-math.pi, math.pi)
    new_x    = spawn_x + distance * math.cos(angle)
    new_z    = spawn_z + distance * math.sin(angle)

    state.TARGET[:] = [new_x, 1.25, new_z]
    state.population.set_target(new_x, 1.25, new_z)
    state.ppo_agent.set_target(new_x, 1.25, new_z)
    print(f"🎯 Target #{n}: ({new_x:.2f}, 1.25, {new_z:.2f})  "
          f"angle={math.degrees(angle):.1f}°")
    state.population.reset_all(spawn_x=spawn_x, spawn_z=spawn_z, face_target=True)

    # Expose so WS handler's "spawn_random_target" message can call this
    state.spawn_target_fn = _spawn_target_and_reorient


# Wire callables into state immediately (called once at module import)
state.spawn_target_fn     = _spawn_target_and_reorient
state.clear_trajectory_fn = _clear_trajectory
state.get_trajectory_fn   = _get_trajectory_snapshot


# ── Eye helpers (overview mode) ───────────────────────────────────────

def _get_sweep_offset(alive: list | None = None) -> float:
    """Pass already-computed alive list to avoid a second filter."""
    try:
        if alive is None:
            alive = [h for h in state.population.humans if not h.dead]
        if alive and hasattr(alive[0], 'gaze'):
            return float(alive[0].gaze._sweep_offset)
    except Exception:
        pass
    return 0.0


def _update_eye_overview() -> None:
    """Angular-math eye update used when NOT in first-person mode."""
    try:
        alive = [h for h in state.population.humans if not h.dead]
        if not alive:
            return
        h  = alive[0]
        ti = h.joint_names['torso']
        if not hasattr(h, 'gaze'):
            return

        TARGET = state.TARGET
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

        state.eye.update_from_walker(
            head_yaw     = h.gaze.world_yaw,
            head_pitch   = 0.0,
            target_yaw   = math.atan2(dx, -dz),
            target_pitch = math.atan2(float(TARGET[1]) - hy, dist),
            gaze_state   = h.gaze.state,
            target_dist  = dist,
            fov_half     = h.gaze.fov_half_angle,
            sweep_offset = _get_sweep_offset(alive),   # reuse alive list
        )
        write_eye_data(state.eye.get_render_data())
    except Exception as e:
        print(f"⚠️  Eye overview: {e}")


# ── Main physics loop ─────────────────────────────────────────────────

def run_physics_loop() -> None:
    global _target_reached_flag

    print("Physics thread started")

    population      = state.population
    ppo_agent       = state.ppo_agent
    Population_number = state.Population_number
    TARGET          = state.TARGET
    EP_MAX_STEPS    = state.EP_MAX_STEPS

    # Reusable target-point dict (mutated in-place — no per-tick allocation)
    target_point = {
        'x': float(TARGET[0]), 'y': float(TARGET[1]),
        'z': float(TARGET[2]), 'mass': 50.0, 'radius': 0.4,
    }

    ep_step      = 0
    walker_dists: dict[int, float] = {}
    state.cached_walker_dists = walker_dists   # expose for ws_handlers
    _paused_rendered = False   # ensures at least one snapshot while paused

    while not state.physics_stop.is_set():
        try:
            # Reset the "already rendered paused frame" flag when running
            if state.start and not state.pause:
                _paused_rendered = False
            # ── Paused / not started ───────────────────────────────────
            if not state.start or state.pause:
                # Always push a snapshot so the render loop can draw the
                # initial/paused pose.  After the first write, only refresh
                # when TARGET changes (e.g. WS set_target).
                tx_now = float(TARGET[0])
                tz_now = float(TARGET[2])
                if tx_now != target_point['x'] or tz_now != target_point['z'] or not _paused_rendered:
                    target_point['x'] = tx_now
                    target_point['y'] = float(TARGET[1])
                    target_point['z'] = tz_now
                    points, lines = population.get_render_data()
                    write_snapshot(
                        points, lines, target_point,
                        list(_trajectory_points),
                        population.n_alive, state.total_episodes, state.all_time_best,
                    )
                    _paused_rendered = True
                if not state.first_person_mode:
                    _update_eye_overview()
                time.sleep(0.016)
                continue

            # ── Gaze FSM ──────────────────────────────────────────────
            population.update_gazes()

            # ── Eye overview (once per tick, not duplicated) ──────────
            if not state.first_person_mode:
                _update_eye_overview()

            # ── Gating: wait for target detection ─────────────────────
            retinal = read_retinal()
            all_detected = all(
                (h.gaze.state != 'searching') if hasattr(h, 'gaze') else True
                for h in population.humans if not h.dead
            ) or (state.first_person_mode and retinal['detected'])

            if not all_detected:
                # Still write snapshot so the render loop keeps drawing the
                # current walker pose while we wait for gaze acquisition.
                points, lines = population.get_render_data()
                target_point['x'] = float(TARGET[0])
                target_point['y'] = float(TARGET[1])
                target_point['z'] = float(TARGET[2])
                write_snapshot(
                    points, lines, target_point,
                    list(_trajectory_points),
                    population.n_alive, state.total_episodes, state.all_time_best,
                )
                time.sleep(0.002)
                continue

            # ── Agent actions ─────────────────────────────────────────
            obs_batch = population.get_observator_batch()

            if state.agent_active:
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

            # ── Walker steps + single-pass distance cache ─────────────
            # One loop → step + distance + reached-check + best_dist.
            # No redundant sqrt() elsewhere (ws_handlers reads walker_dists).
            tx0, tz0 = float(TARGET[0]), float(TARGET[2])
            any_reached = False
            best_dist   = float('inf')

            for i, human in enumerate(population.humans):
                if human.dead:
                    dones[i]        = 1.0
                    walker_dists[i] = float('inf')
                    continue

                _, reward, done, _ = human.step(actions[i])
                rewards[i] = float(reward)
                dones[i]   = float(done)

                ti   = human.joint_names['torso']
                hx   = human.points[ti]['x']
                hz   = human.points[ti].get('z', 0.0)
                dist = math.sqrt((hx - tx0) ** 2 + (hz - tz0) ** 2)
                walker_dists[i] = dist

                if dist < best_dist:
                    best_dist = dist
                if dist < TARGET_REACH_DISTANCE:
                    any_reached = True

            ep_step += 1
            _record_trajectory()

            # ── Periodic console log (every 300 steps) ────────────────
            if ep_step % 300 == 1:
                retinal_snap = read_retinal()
                for i, h in enumerate(population.humans):
                    if h.dead:
                        continue
                    ti = h.joint_names['torso']
                    vx = h.velocities[ti]['vx']
                    vz = h.velocities[ti].get('vz', 0.0)
                    print(
                        f"  [STEP {ep_step:4d}] "
                        f"torso=({h.points[ti]['x']:.3f},{h.points[ti].get('z',0.):.3f})  "
                        f"vx={vx:.3f}  vz={vz:.3f}  dist={walker_dists[i]:.3f}m  "
                        f"gaze={h.gaze.state}  "
                        f"retinal_det={retinal_snap['detected']}  lux={retinal_snap['lux']:.2f}"
                    )

            # ── Target reached ────────────────────────────────────────
            if any_reached and not _target_reached_flag:
                _target_reached_flag = True
                alive_h = [h for h in population.humans if not h.dead]
                rx = float(np.mean([h.points[h.joint_names['torso']]['x']       for h in alive_h]))
                rz = float(np.mean([h.points[h.joint_names['torso']].get('z',0.) for h in alive_h]))
                state.reached_x = rx
                state.reached_z = rz
                print(f"✅ Target reached! Walker at ({rx:.3f}, {rz:.3f})")
                ep_step = 0
                _clear_trajectory()
                _spawn_target_and_reorient(rx, rz)
                for h in population.humans:
                    if hasattr(h, 'gaze'):
                        h.gaze.reset()
                state.light.reset()
                target_point['x'] = float(TARGET[0])
                target_point['y'] = float(TARGET[1])
                target_point['z'] = float(TARGET[2])
                print("👀 Searching for new target (retinal)…")

            elif not any_reached:
                _target_reached_flag = False

            # ── PPO record ────────────────────────────────────────────
            if state.agent_active and ppo_agent.is_training:
                ppo_agent.record_step(obs_batch, actions, log_probs,
                                      rewards, values, dones)

            # ── Episode termination ───────────────────────────────────
            episode_done = (
                    ep_step >= EP_MAX_STEPS or
                    all(h.dead for h in population.humans)
            )
            if episode_done:
                if best_dist < state.all_time_best:
                    state.all_time_best = best_dist
                state.total_episodes    += 1
                population.episode      += 1
                population.all_time_best = state.all_time_best
                reason = "⏱ time" if ep_step >= EP_MAX_STEPS else "💀 all dead"
                print(
                    f"🏁 Ep {state.total_episodes} [{reason}] | steps={ep_step} | "
                    f"best_dist={best_dist:.2f}m | "
                    f"targets_reached={state.targets_reached_count}"
                )
                ep_step = 0
                _clear_trajectory()
                state.light.reset()
                population.reset_all(
                    spawn_x=state.reached_x, spawn_z=state.reached_z,
                    face_target=True,
                )
                for h in population.humans:
                    if hasattr(h, 'gaze'):
                        h.gaze.reset()

            # ── Write snapshot ────────────────────────────────────────
            points, lines = population.get_render_data()
            target_point['x'] = float(TARGET[0])
            target_point['y'] = float(TARGET[1])
            target_point['z'] = float(TARGET[2])
            write_snapshot(
                points, lines, target_point,
                _get_trajectory_snapshot(),
                population.n_alive, population.episode,
                population.all_time_best, rot_y=0.0,
            )

        except Exception:
            print("⚠️  Physics exception:")
            traceback.print_exc()
            time.sleep(0.02)

    print("✅ Physics stopped")
