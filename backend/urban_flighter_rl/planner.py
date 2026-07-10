from __future__ import annotations

import heapq
import numpy as np


def _overflight_fallback(env, margin: float) -> list[np.ndarray]:
    world = env.world
    max_building_h = max((building.height for building in world.buildings), default=0.0)
    safe_z = max(env.pos[2], env.goal[2], max_building_h + margin + 6.0)
    if safe_z >= world.bounds[5] - 3.0:
        return [env.pos.copy()]
    high_start = env.pos.copy()
    high_goal = env.goal.copy()
    high_start[2] = safe_z
    high_goal[2] = safe_z
    path = [env.pos.copy(), high_start, high_goal, env.goal.copy()]
    for index in range(1, len(path)):
        if world.segment_collides(path[index - 1], path[index], margin=margin):
            return [env.pos.copy()]
    return path


def _grid_astar(env, spacing: float = 5.0, z_levels=(15.0, 25.0, 35.0, 45.0)):
    world = env.world
    x0, x1, y0, y1, _, z1 = world.bounds
    xs = np.arange(x0 + spacing, x1, spacing)
    ys = np.arange(y0 + spacing, y1, spacing)
    zs = np.array([z for z in z_levels if z < z1], dtype=float)
    edge_margin = 3.0

    if not world.segment_collides(env.pos, env.goal, margin=edge_margin):
        return [env.pos.copy(), env.goal.copy()]

    def point(idx):
        return np.array([xs[idx[0]], ys[idx[1]], zs[idx[2]]], dtype=float)

    def free(idx):
        return not world.collides(point(idx), margin=2.5)

    def edge_free(a_idx, b_idx) -> bool:
        return not world.segment_collides(point(a_idx), point(b_idx), margin=edge_margin)

    nbrs = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                if abs(dx) + abs(dy) + abs(dz) <= 2:
                    nbrs.append((dx, dy, dz))

    shape = (len(xs), len(ys), len(zs))
    start_candidates = [
        idx
        for idx in np.ndindex(shape)
        if free(idx) and not world.segment_collides(env.pos, point(idx), margin=edge_margin)
    ]
    goal_candidates = {
        idx
        for idx in np.ndindex(shape)
        if free(idx) and not world.segment_collides(point(idx), env.goal, margin=edge_margin)
    }
    if not start_candidates or not goal_candidates:
        return _overflight_fallback(env, edge_margin)

    q = []
    came = {}
    cost = {}
    for idx in start_candidates:
        start_cost = float(np.linalg.norm(point(idx) - env.pos))
        cost[idx] = start_cost
        priority = start_cost + min(float(np.linalg.norm(point(goal_idx) - point(idx))) for goal_idx in goal_candidates)
        heapq.heappush(q, (priority, idx))
    seen = set()
    reached_goal = None

    while q:
        _, cur = heapq.heappop(q)
        if cur in seen:
            continue
        seen.add(cur)
        if cur in goal_candidates:
            reached_goal = cur
            break
        cp = point(cur)
        for d in nbrs:
            nxt = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
            if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1] and 0 <= nxt[2] < shape[2]):
                continue
            if not free(nxt):
                continue
            if not edge_free(cur, nxt):
                continue
            npnt = point(nxt)
            step = float(np.linalg.norm(npnt - cp))
            # Wind-aware edge cost: prefer moving with tailwind, penalize relative airspeed.
            v_ground = (npnt - cp) / max(step / 8.0, 1e-6)
            w = env.wind.at(cp, env.t)
            e = float(np.linalg.norm(v_ground - w) ** 2) * 0.015
            new_cost = cost[cur] + step + e
            if new_cost < cost.get(nxt, 1e18):
                cost[nxt] = new_cost
                priority = new_cost + min(float(np.linalg.norm(point(goal_idx) - npnt)) for goal_idx in goal_candidates)
                came[nxt] = cur
                heapq.heappush(q, (priority, nxt))

    if reached_goal is None:
        return _overflight_fallback(env, edge_margin)
    path = [reached_goal]
    while path[-1] in came:
        path.append(came[path[-1]])
    path.reverse()
    pts = [env.pos.copy()] + [point(i) for i in path] + [env.goal.copy()]
    return pts


def _pd_action_to_target(env, target):
    err = target - env.pos
    d = np.linalg.norm(err) + 1e-9
    desired_speed = min(env.max_speed * 0.8, max(3.0, d * 0.8))
    desired_vel = err / d * desired_speed
    # Let tailwind help a little, but control ground trajectory.
    desired_vel += 0.08 * env.wind.at(env.pos, env.t)
    acc = (desired_vel - env.vel) / max(env.dt * 4.0, 1e-6)
    n = np.linalg.norm(acc)
    return acc / n if n > 1.0 else acc


def run_astar_baseline(env, max_steps=None):
    max_steps = max_steps or env.max_steps
    env.reset()
    waypoints = _grid_astar(env)
    wp_i = 1 if len(waypoints) > 1 else 0
    rewards = []
    for _ in range(max_steps):
        target = waypoints[wp_i]
        if np.linalg.norm(target - env.pos) < 3.0 and wp_i < len(waypoints) - 1:
            wp_i += 1
            target = waypoints[wp_i]
        result = env.step(_pd_action_to_target(env, target))
        rewards.append(result.reward)
        if result.terminated or result.truncated:
            break
    m = env.metrics()
    m["return"] = float(np.sum(rewards))
    m["controller"] = "wind_aware_grid_astar_pd"
    m["waypoint_count"] = len(waypoints)
    return m


def wind_aware_potential_action(env):
    pos = env.pos
    to_goal = env.goal - pos
    d = np.linalg.norm(to_goal) + 1e-9
    desired_dir = to_goal / d
    repel = env.world.obstacle_repulsion(pos, radius=18.0)
    climb_bias = np.zeros(3)
    if pos[2] < 14.0:
        climb_bias[2] = 0.6
    wind = env.wind.at(pos, env.t)
    desired_velocity = desired_dir * min(env.max_speed, 0.18 * d) + 0.20 * wind + 5.0 * repel + climb_bias
    action = (desired_velocity - env.vel) / max(env.max_acc, 1e-6)
    n = np.linalg.norm(action)
    return action / n if n > 1.0 else action


def run_baseline(env, max_steps=None):
    return run_astar_baseline(env, max_steps=max_steps)
