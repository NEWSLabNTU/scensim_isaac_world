"""086 C6a: aib's PhysX Model 3 over R0's fixtures on the C6a fixture course
(`courses/fixture_c6a.yaml`), the tyre contact the road layer's (066 R5), built as the world
server builds it: `road_layer.load` on the course's `road_detail.yaml`, a `RoadLayer` on the
surface layer, its `place` before every PhysX step, `update` after. No Autoware: a straight
pursuit along the left lane's middle and a speed hold (as `lip_drive`). Isaac Python, with aib.

Rates: PhysX at 60 Hz, three steps a 0.05 s frame; the vehicle integrates aib's
`physx_substeps` a step; our contact runs once a PhysX step (its own sub-samples inside it) and
its force holds for the step. Logged at every PhysX step: each wheel's load (our contact's, the
mean over its sub-samples) and airborne share, the body's height, pitch and vertical
acceleration.

Each case starts at rest RUNUP m before its fixture on the left lane's middle, drives up to its
speed, is checked at speed CHECK_AT m before the fixture (speed within START_SPEED_TOL, lateral
offset within START_OFFSET_TOL; else refused, not judged), and ends AFTER m past the fixture.
Writes `fixture.csv` (per PhysX step, `case` its name) and `cases.json`, which `fixture_check`
reads:

    python -m scensim_isaac_world.course.fixture_drive --map <map dir> --out <dir>
        [--query raycast|sweep] [--cases cleat_10@10,...]
"""
import argparse
import csv
import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np

FRAME = 0.05
SUB = 3
LOOKAHEAD = 8.0
LANE = 1.625  # m left of the centreline: the left lane's middle (driving on the left)
CHECK_AT = 10.0
AFTER = 20.0
START_SPEED_TOL = 0.1
START_OFFSET_TOL = 0.05
SETTLE_FRAMES = 40
RAMP, RAMP_FAST = 2.5, 3.0  # m/s^2, the speed hold's target ramp to 10 and to 20 m/s
OFFSET_STEP = 0.07  # m between cleat_10 at 20 m/s's start offsets (R0's question 1)
# fixture -> (s where it starts, its length along the road), the course's `features:`
FIXTURES = {"cleat_05": (100.0, 0.2), "cleat_10": (200.0, 0.2), "cleat_15": (300.0, 0.2),
            "pothole": (400.0, 0.5), "washboard": (500.0, 30.0)}
CASES = ["cleat_05@10", "cleat_10@10", "cleat_15@10", "pothole@10", "washboard@10",
         "cleat_10@20", "washboard@20"] + [f"cleat_10@20#{k}" for k in range(5)]


def runup(speed):
    """m before a fixture a case starts at rest: the spec's 60 m; at 20 m/s 95 m, since 60 m
    cannot reach 20 m/s at the speed hold's 3 m/s^2 (66.7 m) before the start check, CHECK_AT m
    before the fixture."""
    return 60.0 if speed <= 10.0 else 95.0


def parse_case(c):
    name, rest = c.split("@")
    speed, _, k = rest.partition("#")
    return name, float(speed), int(k) if k else None


def quat_rpy(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ego", default=str(Path(os.environ["AIB_SRC"]) / "config/ego.yaml"))
    ap.add_argument("--query", choices=("raycast", "sweep"), default="raycast")
    ap.add_argument("--cases", default="", help="comma separated (default: all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cases = a.cases.split(",") if a.cases else CASES

    from scensim_isaac_world import road_detail as rd
    from scensim_isaac_world import road_layer, surfaces
    from scensim_isaac_world.course import record

    cl = np.load(os.path.join(a.map, "course", "centreline.npz"))
    st = record.load(os.path.join(a.map, "course"))
    corridor = rd.Corridor(np.stack([cl["x"], cl["y"], cl["z"]], 1),
                           section=rd.StationSection(st), stations=st)
    s0 = float(cl["s"][0])

    def xy_at(s, v):
        u = np.clip(s - s0, 0.0, corridor.length)
        seg = int(np.clip(np.searchsorted(corridor.seg_u, u, side="right") - 1, 0,
                          len(corridor.seg_len) - 1))
        d = corridor.seg_dir[seg]
        p = corridor.centreline[seg] + (u - corridor.seg_u[seg]) * d
        return p + v * np.array([-d[1], d[0]]), math.atan2(d[1], d[0])

    # the course is one straight line: (s, v) by projection, not Corridor.uv's segment search
    c0, d0 = corridor.centreline[0], corridor.seg_dir[0]
    if np.abs(corridor.seg_dir - d0).max() > 1e-9:
        raise SystemExit("fixture_drive: the course is not straight")

    def uv(x, y):
        r = np.array([x, y], dtype=float) - c0
        return float(r @ d0) + s0, float(d0[0] * r[1] - d0[1] * r[0])

    from aib.rig import EgoConfig

    rig = EgoConfig.load(a.ego).load_rig()
    # as the world server makes an ego on a world with a road layer (`world.py`)
    params = dataclasses.replace(road_layer.vehicle_params(rig.vehicle), wheel_query=a.query)

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

    def apply_force(path, force, position):  # as world.py's _apply_force
        sim.apply_force_at_pos(omni.usd.get_context().get_stage_id(),
                               PhysicsSchemaTools.sdfPathToInt(path), force, position)

    smap = surfaces.load(os.path.join(a.map, surfaces.FILE))
    layer = SurfaceLayer(stage, smap, apply_force=apply_force)
    corridors = road_layer.load(os.path.join(a.map, "course", road_layer.FILE),
                                classes=set(smap.table.classes))
    layer.road = road_layer.RoadLayer(corridors, (0.0, 0.0, 0.0), apply_force)

    (x0, y0), h0 = xy_at(10.0, LANE)
    pose = ((float(x0), float(y0), 1.0), (math.cos(h0 / 2), 0.0, 0.0, math.sin(h0 / 2)))
    veh = Vehicle(stage, "/World/ego", params, rig.make_component(), rig.box_center,
                  rig.box_dims, pose, truth_settings=rig.component.get("truth"))
    state = State()

    def physx_step(cmd):
        veh.apply(cmd, dt, state)
        layer.place([veh], dt)
        world.step(render=False)
        layer.begin_step()
        veh.sample(physx, dt, state)
        layer.update(veh, state.velocity[0], dt)

    def frame(cmd):
        for _ in range(SUB):
            physx_step(cmd)

    # fl_s: the front-left hub's s (the reference's x); fl_z: its attachment's height (the
    # corner's body, the reference's zs)
    header = ["case", "t", "step", "s", "v", "speed", "z", "pitch", "az", "fl_s", "fl_z"]
    header += [f"load{i}" for i in range(4)] + [f"air{i}" for i in range(4)]
    rows, report = [], {"query": a.query, "physics_hz": round(1.0 / dt, 3),
                        "physx_substeps": params.physx_substeps, "frame_s": FRAME,
                        "aib": os.environ.get("AIB_COMMIT", ""), "cases": []}
    idle = Command(received=True, gear=GEAR_DRIVE)
    try:
        for case in cases:
            name, speed, k = parse_case(case)
            fs, flen = FIXTURES[name]
            start = fs - runup(speed) - (k or 0) * OFFSET_STEP
            (x, y), h = xy_at(start, LANE)
            z = float(np.interp(start, cl["s"], cl["z"])) + 0.05
            veh.teleport(((float(x), float(y), z), (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2))),
                         physx)
            layer.reset(veh.path)
            layer.road.reset(veh.path)
            state = State()
            for _ in range(SETTLE_FRAMES):
                frame(idle)
            info = {"case": case, "fixture": name, "speed": speed, "offset": k,
                    "fixture_s": fs, "fixture_len": flen, "start_s": start}
            ss = np.arange(start - 10.0, fs + flen + AFTER + 20.0, 0.5)
            line = np.array([xy_at(s, LANE)[0] for s in ss])
            cmd = Command(received=True, gear=GEAR_DRIVE)
            t, n, err_sum, checked = 0.0, 0, 0.0, False
            while t < 60.0:
                pos = np.array(state.position[:2])
                s_now, v_now = uv(*pos)
                if s_now >= fs + flen + AFTER:
                    break
                j = int(np.argmin(np.hypot(*(line - pos).T)))
                tgt = line[min(j + int(LOOKAHEAD / 0.5), len(line) - 1)]
                yaw = quat_rpy(*state.quat)[2]
                dx, dy = tgt - pos
                steer = math.atan2(2 * params.wheel_base
                                   * math.sin(math.atan2(dy, dx) - yaw), math.hypot(dx, dy))
                vel = state.velocity[0]
                cmd.tire_angle = max(-params.max_steer_angle, min(params.max_steer_angle, steer))
                # a ramped target with its rate fed forward, the integral only once on speed:
                # a step to `speed` overshot by 0.1 m/s and was still settling at the check
                ramp = RAMP if speed <= 10.0 else RAMP_FAST
                vref = min(speed, ramp * t)
                ff = ramp if vref < speed else 0.0
                cmd.velocity = vref
                if vref >= speed and abs(speed - vel) < 0.5:
                    err_sum = max(-5.0, min(5.0, err_sum + (speed - vel) * FRAME))
                cmd.acceleration = max(-3.0, min(3.0, ff + 1.5 * (vref - vel) + 0.5 * err_sum))
                for _ in range(SUB):
                    physx_step(cmd)
                    t += dt
                    n += 1
                    car = layer.road.cars.get(veh.path)
                    o = car.last_out if car is not None else None
                    P = state.position
                    s_now, v_now = uv(P[0], P[1])
                    _, pitch, _ = quat_rpy(*state.quat)
                    loads = [round(float(x), 1) for x in o["load"]] if o else [""] * 4
                    air = [round(float(x), 3) for x in o["airborne"]] if o else [""] * 4
                    if o is not None:
                        fl_s = round(uv(*car.last_xy[0])[0], 4)
                        fl_z = round(float(car.z_att_prev[0]), 5)
                    else:
                        fl_s = fl_z = ""
                    rows.append([case, round(t, 5), n, round(s_now, 4), round(v_now, 4),
                                 round(state.velocity[0], 4), round(float(P[2]), 5),
                                 round(pitch, 6), round(state.acceleration[2], 4), fl_s, fl_z,
                                 *loads, *air])
                if not checked and s_now >= fs - CHECK_AT:
                    checked = True
                    info.update(start_t=round(t, 4), start_speed=round(state.velocity[0], 4),
                                start_offset=round(v_now - LANE, 4))
                    info["start_ok"] = bool(abs(state.velocity[0] - speed) <= START_SPEED_TOL
                                            and abs(v_now - LANE) <= START_OFFSET_TOL)
            info["steps"] = n
            report["cases"].append(info)
            print(f"fixture_drive: {json.dumps(info)}", flush=True)
    finally:
        with open(os.path.join(a.out, "fixture.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(header)
            wr.writerows(rows)
        json.dump(report, open(os.path.join(a.out, "cases.json"), "w"), indent=1)
        print(f"fixture_drive: {len(rows)} steps, {len(report['cases'])} case(s) to {a.out}",
              flush=True)
        app.close()


if __name__ == "__main__":
    main()
