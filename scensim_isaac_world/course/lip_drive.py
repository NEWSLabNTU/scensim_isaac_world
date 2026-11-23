"""086 C5c: aib's PhysX Model 3 re-entering the road over each segment's lip of the lip course
(`courses/fixture_c5c.yaml`), with no Autoware, as C5b's `arc_drive` drives (the ego config's rig,
0.05 s frames of 3 physics steps, a pure pursuit, a speed hold). Isaac Python, with aib.

Each case is one segment of the course. The car is put at rest 5 m into the segment with its
right wheels' centres OFFSET m past the carriageway's edge (on the shoulder, the left wheels on
the pavement), drives up to --speed along that line (ACCEL_LEN m), holds it parallel to the edge
for PARALLEL_LEN m, then follows a line at --angle deg back toward the road to the right lane's
middle, and on for AFTER_LEN m. Where the parallel leg starts each case checks its start: the
speed within START_SPEED_TOL of --speed and the lateral offset within START_OFFSET_TOL of the
line (else the case is refused, not classified), and whether the right wheels stand on the
shoulder (in contact, the contact on the shoulder's surface at its height).

The wheels' friction comes from the map's surface map with the road record's strips under its
zones (`road_layer.strip_zones`, `SurfaceLayer.add_record_zones`), as the world server's surface
layer gives it; each frame logs each wheel's class and zone.

Writes `lip.csv` (one row per frame, `case` its segment) and `cases.json` (each case's start
check and its turn-in time), which `lip_check` reads:

    python -m scensim_isaac_world.course.lip_drive --map <map dir> --course <course.yaml>
        --out <dir> [--query raycast|sweep] [--speed 15.6] [--angle 4] [--cases 0,4,7]
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
LANE = 1.625  # m right of the centreline: the right lane's middle
OFFSET = 0.5  # m: the right wheels' centres past the carriageway's edge at the start
START_AT = 5.0  # m into the segment
ACCEL_LEN = 60.0  # m from the start to the parallel leg
PARALLEL_LEN = 40.0  # m
AFTER_LEN = 80.0  # m past the turn-in: over lip_check's 4 s held window at 15.6 m/s
START_SPEED_TOL = 0.1  # m/s
START_OFFSET_TOL = 0.05  # m
SHOULDER_Z_TOL = 0.02  # m: a right wheel's contact this near the shoulder's height is on it
SETTLE_FRAMES = 40


def quat_rpy(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def path_v(s, s_turn, v0, v1, angle):
    """The drive's line, m left of the centreline at s: v0 to s_turn, then at `angle` (rad)
    toward v1, then v1."""
    return np.where(s <= s_turn, v0, np.minimum(v0 + np.tan(angle) * (s - s_turn), v1))


def plan(case, half, tread, angle):
    """A case's stations: start, parallel leg, turn-in, end (m along the course) and its line's
    v at the start and at the end (base_link's, the axle's middle)."""
    s0 = float(case["from"]) + START_AT
    v0 = -(half + OFFSET) + tread / 2
    return {"s_start": s0, "s_par": s0 + ACCEL_LEN, "s_turn": s0 + ACCEL_LEN + PARALLEL_LEN,
            "s_end": s0 + ACCEL_LEN + PARALLEL_LEN + AFTER_LEN, "v0": v0, "v1": -LANE,
            "angle": angle}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--course", required=True, help="the course's YAML, its `cases:`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ego", default=str(Path(os.environ["AIB_SRC"]) / "config/ego.yaml"))
    ap.add_argument("--query", choices=("raycast", "sweep"), default="raycast")
    ap.add_argument("--speed", type=float, default=15.6)
    ap.add_argument("--angle", type=float, default=4.0, help="deg")
    ap.add_argument("--cases", default="", help="segments, comma separated (default: all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import yaml

    course = yaml.safe_load(open(a.course))
    cases = course["cases"]
    if a.cases:
        want = {int(k) for k in a.cases.split(",")}
        cases = [c for c in cases if int(c["segment"]) in want]

    from scensim_isaac_world import road_detail as rd
    from scensim_isaac_world import road_layer, surfaces
    from scensim_isaac_world.course import record

    cl = np.load(os.path.join(a.map, "course", "centreline.npz"))
    st = record.load(os.path.join(a.map, "course"))
    sec = rd.StationSection(st)
    corridor = rd.Corridor(np.stack([cl["x"], cl["y"], cl["z"]], 1), section=sec, stations=st)
    s_cl = np.asarray(cl["s"], float)
    half = float(st.common["half_width"][0])

    def xy_at(s, v):
        u = np.clip(s - s_cl[0], 0.0, corridor.length)
        seg = np.clip(np.searchsorted(corridor.seg_u, u, side="right") - 1, 0,
                      len(corridor.seg_len) - 1)
        d = corridor.seg_dir[seg]
        p = corridor.centreline[seg] + (u - corridor.seg_u[seg])[..., None] * d
        n = np.stack([-d[..., 1], d[..., 0]], -1)
        return p + np.asarray(v)[..., None] * n, np.arctan2(d[..., 1], d[..., 0])

    def uv(x, y):
        u, v, _ = corridor.uv(np.atleast_1d(x), np.atleast_1d(y))
        return float(u[0]) + float(s_cl[0]), float(v[0])

    from aib.rig import EgoConfig

    rig = EgoConfig.load(a.ego).load_rig()
    params = dataclasses.replace(rig.vehicle, wheel_query=a.query)
    tread, base, radius = params.wheel_tread, params.wheel_base, params.wheel_radius

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
    import omni.physx
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage

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

    # the wheels' ground, as the world server's surface layer gives it (no road layer: the
    # course's record is in course/, the base road carries its section)
    smap = surfaces.load(os.path.join(a.map, surfaces.FILE))
    layer = SurfaceLayer(stage, smap)
    road = os.path.join(a.map, "course", road_layer.FILE)
    zones = road_layer.strip_zones(road_layer.load(road, classes=set(smap.table.classes)))
    layer.add_record_zones(zones)

    first = plan(cases[0], half, tread, math.radians(a.angle))
    (x0, y0), h0 = xy_at(first["s_start"], first["v0"])
    pose = ((float(x0), float(y0), 1.0), (math.cos(h0 / 2), 0.0, 0.0, math.sin(h0 / 2)))
    veh = Vehicle(stage, "/World/ego", params, rig.make_component(), rig.box_center,
                  rig.box_dims, pose, truth_settings=rig.component.get("truth"))
    state = State()

    def substeps(cmd):
        for _ in range(SUB):
            veh.apply(cmd, dt, state)
            world.step(render=False)
            veh.sample(physx, dt, state)
            layer.update(veh, state.velocity[0], dt)

    header = ["case", "t", "frame", "s", "v", "x", "y", "z", "roll", "pitch", "yaw", "speed",
              "yaw_rate", "ay", "fr_s", "fr_v", "fr_z", "rr_v", "steer"]
    header += [f"w{i}_{k}" for i in range(4) for k in ("on", "cz", "cv", "cls", "zone")]
    header += ["w1_shape", "w3_shape"]  # the colliders the right wheels stand on
    rows, report = [], {"query": a.query, "speed": a.speed, "angle_deg": a.angle,
                        "wheel_tread": tread, "wheel_base": base,
                        "record_zones": [z.id for z in zones], "cases": []}
    idle = Command(received=True, gear=GEAR_DRIVE)
    try:
        for case in cases:
            p = plan(case, half, tread, math.radians(a.angle))
            k = int(case["segment"])
            (x, y), h = xy_at(p["s_start"], p["v0"])
            z = float(np.interp(p["s_start"], s_cl, cl["z"])
                      + sec.height_uv(np.array([p["s_start"] - s_cl[0]]),
                                      np.array([p["v0"]]))[0]) + 0.05
            pose = ((float(x), float(y), z), (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)))
            veh.teleport(pose, physx)
            layer.reset(veh.path)
            state = State()
            for _ in range(SETTLE_FRAMES):
                substeps(idle)
            s_at, v_at = uv(*state.position[:2])
            info = {"segment": k, "expect": case["expect"], "lip_h": case["lip_h"],
                    "lip_angle": case["lip_angle"], **{n: round(float(q), 4)
                                                       for n, q in p.items()},
                    "settled_v_err": round(v_at - p["v0"], 4)}
            ss = np.arange(p["s_start"] - 10.0, p["s_end"] + 20.0, 0.5)
            line = xy_at(ss, path_v(ss, p["s_turn"], p["v0"], p["v1"], p["angle"]))[0]
            cmd = Command(received=True, gear=GEAR_DRIVE)
            t, frame, checked, err_sum = 0.0, 0, False, 0.0
            while t < 60.0:
                pos = np.array(state.position[:2])
                s_now, v_now = uv(*pos)
                if s_now >= p["s_end"]:
                    break
                j = int(np.argmin(np.hypot(*(line - pos).T)))
                tgt = line[min(j + int(LOOKAHEAD / 0.5), len(line) - 1)]
                yaw = quat_rpy(*state.quat)[2]
                dx, dy = tgt - pos
                alpha = math.atan2(dy, dx) - yaw
                steer = math.atan2(2 * base * math.sin(alpha), math.hypot(dx, dy))
                v = state.velocity[0]
                cmd.tire_angle = max(-params.max_steer_angle, min(params.max_steer_angle, steer))
                cmd.velocity = a.speed
                # a speed hold with an integral, so it holds --speed, not a drag's offset below
                # (only near it: no wind-up while it speeds up from rest)
                if abs(a.speed - v) < 1.0:
                    err_sum = max(-5.0, min(5.0, err_sum + (a.speed - v) * FRAME))
                cmd.acceleration = max(-3.0, min(3.0, 1.5 * (a.speed - v) + 0.5 * err_sum))
                substeps(cmd)
                t += FRAME
                frame += 1
                q = state.quat
                R = rot(*q)
                r, pi, ya = quat_rpy(*q)
                P = np.array(state.position)
                fr = P + R @ np.array([base, -tread / 2, radius])
                rr_ = P + R @ np.array([0.0, -tread / 2, radius])
                fr_s, fr_v = uv(*fr[:2])
                s_now, v_now = uv(*P[:2])
                under = layer.surfaces_under(veh)
                wheels = []
                for i, c in enumerate(veh.contacts or []):
                    srf = under[i] if i < len(under) else None
                    cv = uv(*c.position[:2])[1] if c.on_ground else float("nan")
                    wheels += [int(c.on_ground), round(float(c.position[2]), 4), round(cv, 4),
                               getattr(srf, "cls", "") or "", getattr(srf, "zone", "") or ""]
                rows.append([k, round(t, 4), frame, round(s_now, 4), round(v_now, 4),
                             *(round(float(c), 4) for c in P), round(r, 5), round(pi, 5),
                             round(ya, 5), round(v, 4), round(state.angular_velocity[2], 5),
                             round(state.acceleration[1], 4), round(fr_s, 4), round(fr_v, 4),
                             round(float(fr[2]), 4), round(uv(*rr_[:2])[1], 4),
                             round(cmd.tire_angle, 5), *wheels,
                             *(veh.contacts[i].shape if veh.contacts else "" for i in (1, 3))])
                if not checked and s_now >= p["s_par"]:
                    checked = True
                    info["start_t"] = round(t, 3)
                    info["start_speed"] = round(v, 4)
                    info["start_offset"] = round(v_now - p["v0"], 4)
                    on = []
                    for i in (1, 3):  # front right, rear right
                        c = veh.contacts[i]
                        if not c.on_ground:
                            on.append(False)
                            continue
                        cs, cv = uv(*c.position[:2])
                        zs = float(np.interp(cs, s_cl, cl["z"])
                                   + sec.height_uv(np.array([cs - s_cl[0]]), np.array([cv]))[0])
                        on.append(bool(cv < -half and abs(c.position[2] - zs) <= SHOULDER_Z_TOL))
                    info["right_on_shoulder"] = on
                    info["start_ok"] = bool(abs(v - a.speed) <= START_SPEED_TOL
                                            and abs(v_now - p["v0"]) <= START_OFFSET_TOL)
                if "turn_t" not in info and s_now >= p["s_turn"]:
                    info["turn_t"] = round(t, 3)
            info["frames"] = frame
            report["cases"].append(info)
            print(f"lip_drive: case {k}: {json.dumps(info)}", flush=True)
    finally:
        with open(os.path.join(a.out, "lip.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(header)
            wr.writerows(rows)
        json.dump(report, open(os.path.join(a.out, "cases.json"), "w"), indent=1)
        print(f"lip_drive: {len(rows)} frames, {len(report['cases'])} case(s) to {a.out}",
              flush=True)
        app.close()


if __name__ == "__main__":
    main()
