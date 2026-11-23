"""086 C6a: R1's cost line, reported, not judged (C6b's budget judges it): the frame's wall
time with 1, 2 or 4 of aib's Model 3 driving the fixture course at once, under one wheel
query, and our contact's share (`RoadLayer.cost_ms`, summed over the frame's PhysX steps).

The cars start 20 m apart, alternately in the left and the right lane (all driving +s), at
s = 40 m, and hold 10 m/s over the cleats and the pothole (the same pursuit and speed hold as
`fixture_drive`); FRAMES frames are timed once every car is at speed. Writes `cost.json`:

    python -m scensim_isaac_world.course.fixture_cost --map <map dir> --out <dir>
        [--query raycast|sweep] [--cars 1|2|4]
"""
import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from scensim_isaac_world.course.fixture_drive import (FRAME, LANE, LOOKAHEAD, load_road,
                                                      quat_rpy, rate)

FRAMES = 400
WARMUP_S = 6.0
SPEED = 10.0
GAP = 20.0
START = 40.0


def pct(xs, q):
    return float(np.percentile(np.asarray(xs), q)) if len(xs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ego", default=str(Path(os.environ["AIB_SRC"]) / "config/ego.yaml"))
    ap.add_argument("--query", choices=("raycast", "sweep"), default="raycast")
    ap.add_argument("--cars", type=int, default=1)
    ap.add_argument("--no-colliders", action="store_true",
                    help="no detail collision tiles (as before issue 119)")
    ap.add_argument("--hz", type=int, default=60, help="physics rate: 60, 120 or 240 (C6a2)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from scensim_isaac_world import road_layer, surfaces

    cl = np.load(os.path.join(a.map, "course", "centreline.npz"))
    x0, y0 = float(cl["x"][0]), float(cl["y"][0])
    h = float(math.atan2(cl["y"][-1] - y0, cl["x"][-1] - x0))
    d, n = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])

    def xy(s, v):
        return np.array([x0, y0]) + s * d + v * n

    from aib.rig import EgoConfig

    rig = EgoConfig.load(a.ego).load_rig()
    params = dataclasses.replace(road_layer.vehicle_params(rig.vehicle), wheel_query=a.query)
    SUB, substeps = rate(a.hz, params.physx_substeps)
    params = dataclasses.replace(params, physx_substeps=substeps)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
    import omni.physx
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import PhysicsSchemaTools

    from aib.control import GEAR_DRIVE, Command, State
    from aib.isaac.vehicle import Vehicle, ensure_vehicle_scene
    from scensim_isaac_world.surface_layer import SurfaceLayer

    dt = FRAME / SUB
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.realpath(os.path.join(a.map, "isaac", "world.usdc")),
                           prim_path="/World/env")
    ensure_vehicle_scene(stage)
    world.reset()
    world.play()
    physx = omni.physx.get_physx_interface()
    sim = omni.physx.get_physx_simulation_interface()

    def apply_force(path, force, position):
        sim.apply_force_at_pos(omni.usd.get_context().get_stage_id(),
                               PhysicsSchemaTools.sdfPathToInt(path), force, position)

    smap = surfaces.load(os.path.join(a.map, surfaces.FILE))
    layer = SurfaceLayer(stage, smap, apply_force=apply_force)
    load_road(a.map, stage, layer, physx, world, apply_force, set(smap.table.classes),
              colliders=not a.no_colliders)
    cars = []
    for k in range(a.cars):
        lane = LANE if k % 2 == 0 else -LANE
        s0 = START + GAP * (k // 2) + GAP / 2 * (k % 2)
        p = xy(s0, lane)
        pose = ((float(p[0]), float(p[1]), 0.6), (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)))
        veh = Vehicle(stage, f"/World/car{k}", params, rig.make_component(), rig.box_center,
                      rig.box_dims, pose, ground_clearance=rig.ground_clearance,
                      truth_settings=rig.component.get("truth"))
        cars.append({"veh": veh, "state": State(), "lane": lane, "i": 0.0,
                     "cmd": Command(received=True, gear=GEAR_DRIVE)})

    t, frame_ms, contact_ms, flush_ms, tiles = 0.0, [], [], [], []
    total = int(WARMUP_S / FRAME) + FRAMES
    try:
        for f in range(total):
            for c in cars:
                st, cmd = c["state"], c["cmd"]
                pos = np.array(st.position[:2])
                s_now = float((pos - np.array([x0, y0])) @ d)
                tgt = xy(s_now + LOOKAHEAD, c["lane"])
                yaw = quat_rpy(*st.quat)[2]
                dx, dy = tgt - pos
                steer = math.atan2(2 * params.wheel_base * math.sin(math.atan2(dy, dx) - yaw),
                                   math.hypot(dx, dy))
                cmd.tire_angle = max(-params.max_steer_angle, min(params.max_steer_angle, steer))
                vel = st.velocity[0]
                vref = min(SPEED, 2.5 * t)
                if vref >= SPEED and abs(SPEED - vel) < 0.5:
                    c["i"] = max(-5.0, min(5.0, c["i"] + (SPEED - vel) * FRAME))
                cmd.velocity = vref
                cmd.acceleration = max(-3.0, min(3.0, (2.5 if vref < SPEED else 0.0)
                                                 + 1.5 * (vref - vel) + 0.5 * c["i"]))
            n0 = len(layer.road.cost_ms)
            t0 = time.perf_counter()
            for _ in range(SUB):
                for c in cars:
                    c["veh"].apply(c["cmd"], dt, c["state"])
                layer.place([c["veh"] for c in cars], dt)
                world.step(render=False)
                layer.begin_step()
                for c in cars:
                    c["veh"].sample(physx, dt, c["state"])
                    layer.update(c["veh"], c["state"].velocity[0], dt)
            t1 = time.perf_counter()
            layer.flush()  # the world server's frame end: the tiles (issue 119), the rocks
            t += FRAME
            if f >= total - FRAMES:
                frame_ms.append((time.perf_counter() - t0) * 1e3)
                contact_ms.append(sum(layer.road.cost_ms[n0:]))
                flush_ms.append((time.perf_counter() - t1) * 1e3)
                tiles.append(len(layer.road_colliders.coll) if layer.road_colliders else 0)
        out = {"query": a.query, "cars": a.cars, "physics_hz": a.hz,
               "physx_substeps": params.physx_substeps, "frames": len(frame_ms),
               "aib": os.environ.get("AIB_COMMIT", ""),
               "frame_ms_p50": round(pct(frame_ms, 50), 3),
               "frame_ms_p95": round(pct(frame_ms, 95), 3),
               "contact_ms_p50": round(pct(contact_ms, 50), 3),
               "contact_ms_p95": round(pct(contact_ms, 95), 3),
               "colliders": layer.road_colliders is not None,
               "flush_ms_p50": round(pct(flush_ms, 50), 3),
               "flush_ms_p95": round(pct(flush_ms, 95), 3),
               "tiles_mean": round(float(np.mean(tiles)), 1),
               "speeds": [round(c["state"].velocity[0], 2) for c in cars],
               "s_end": [round(float((np.array(c["state"].position[:2])
                                      - np.array([x0, y0])) @ d), 1) for c in cars]}
        print(f"fixture_cost: {json.dumps(out)}", flush=True)
        json.dump(out, open(os.path.join(a.out, "cost.json"), "w"), indent=1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
