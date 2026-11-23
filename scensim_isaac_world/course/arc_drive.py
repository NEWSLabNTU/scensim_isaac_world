"""086 C5b: aib's PhysX Model 3 driven through the fixture's banked bend at a set speed, with
no Autoware (whose velocity smoother holds the lateral acceleration to 1.0 m/s^2, so it takes
the arc at about 4 m/s). Built as aib.steptest builds it: the ego config's rig, 0.05 s frames
of 3 physics steps. A pure pursuit on the left lane's middle, a speed hold to --speed.

Writes ego.csv in the format of the ego_trace attachment (`arc_check` reads it):
t, frame, x, y, z, roll, pitch, yaw (map, rad), world velocity, world angular velocity.
Under Isaac's Python with aib on the path (aib's `isaac_env.sh`, as `just isaac-steptest`):

    python -m scensim_isaac_world.course.arc_drive --map <map dir> --out <dir> \
        [--speed 11.1] [--from 40] [--to 330] [--ego <aib ego config>]
"""
import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np

FRAME = 0.05
SUB = 3
LOOKAHEAD = 8.0
LANE = 1.625  # m left of the centreline: the left lane's middle


def quat_rpy(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ego", default="", help="aib's ego config (default: aib's own)")
    ap.add_argument("--speed", type=float, default=11.1)
    ap.add_argument("--from", dest="s0", type=float, default=100.0)
    ap.add_argument("--to", dest="s1", type=float, default=330.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = np.load(os.path.join(a.map, "course", "centreline.npz"))
    from .. import road_detail as rd
    from . import record
    from .geometry import _normals

    sec = rd.StationSection(record.load(os.path.join(a.map, "course")))
    nrm = _normals(c["x"], c["y"])
    lane = np.stack([c["x"] + LANE * nrm[:, 0], c["y"] + LANE * nrm[:, 1]], 1)
    s = np.asarray(c["s"], float)
    lane_z = c["z"] + sec.height_uv(s - s[0], np.full(len(s), LANE))

    import aib
    from aib.rig import EgoConfig
    ego = a.ego or str(Path(aib.__file__).resolve().parents[1] / "config/ego.yaml")
    rig = EgoConfig.load(ego).load_rig()
    params = rig.vehicle

    from aib import app_exit
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
    code = 0
    try:
        import omni.physx
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage

        from aib.control import GEAR_DRIVE, Command, State
        from aib.isaac.vehicle import Vehicle, ensure_vehicle_scene

        dt = FRAME / SUB
        world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
        stage = omni.usd.get_context().get_stage()
        add_reference_to_stage(
            usd_path=os.path.realpath(os.path.join(a.map, "isaac", "world.usdc")),
            prim_path="/World/env")
        ensure_vehicle_scene(stage)
        world.reset()
        world.play()
        physx = omni.physx.get_physx_interface()
        i0 = int(np.searchsorted(s, a.s0))
        h0 = float(c["heading"][i0])
        pose = ((float(lane[i0, 0]), float(lane[i0, 1]), float(lane_z[i0]) + 0.05),
                (math.cos(h0 / 2), 0.0, 0.0, math.sin(h0 / 2)))
        veh = Vehicle(stage, "/World/ego", params, rig.make_component(), rig.box_center,
                      rig.box_dims, pose, truth_settings=rig.component.get("truth"))
        veh.teleport(pose, physx)
        state = State()
        idle = Command(received=True, gear=GEAR_DRIVE)
        for _ in range(40):
            for _ in range(SUB):
                veh.apply(idle, dt, state)
                world.step(render=False)
                veh.sample(physx, dt, state)
        cmd = Command(received=True, gear=GEAR_DRIVE)
        rows, t, frame = [], 0.0, 0
        wb = params.wheel_base
        try:
            while t < 120.0:
                p = np.array(state.position[:2])
                j = int(np.argmin(np.hypot(*(lane - p).T)))
                if s[j] >= a.s1:
                    break
                k = int(np.searchsorted(s, s[j] + LOOKAHEAD))
                k = min(k, len(s) - 1)
                q = state.quat
                yaw = quat_rpy(*q)[2]
                dx, dy = lane[k] - p
                alpha = math.atan2(dy, dx) - yaw
                ld = math.hypot(dx, dy)
                steer = math.atan2(2 * wb * math.sin(alpha), ld)
                v = state.velocity[0]
                cmd.tire_angle = max(-params.max_steer_angle, min(params.max_steer_angle, steer))
                cmd.velocity = a.speed
                cmd.acceleration = max(-3.0, min(2.0, 1.0 * (a.speed - v)))
                for _ in range(SUB):
                    veh.apply(cmd, dt, state)
                    world.step(render=False)
                    veh.sample(physx, dt, state)
                t += FRAME
                frame += 1
                q = state.quat
                R = rot(*q)
                vw = R @ np.array(state.velocity)
                ww = R @ np.array(state.angular_velocity)
                r, pi, ya = quat_rpy(*q)
                rows.append([round(t, 4), frame, *state.position, r, pi, ya, *vw, *ww, 1])
        finally:
            with open(os.path.join(a.out, "ego.csv"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["t", "frame", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
                             "wx", "wy", "wz", "measured"])
                wr.writerows(rows)
            print(f"arc_drive: {len(rows)} frames to {a.out}/ego.csv", flush=True)
    except BaseException as e:  # app.close() would end the process with 0 (scensim issue 118)
        code = app_exit.code_of(e)
    finally:
        app_exit.close(app, code)


if __name__ == "__main__":
    main()
