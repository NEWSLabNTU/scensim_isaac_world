"""086 C6a: R1's cost line, reported, not judged (C6b's budget judges it): the frame's wall
time with 1, 2 or 4 of aib's Model 3 driving the fixture course at once, under one wheel
query, and our contact's share (`RoadLayer.cost_ms`, summed over the frame's PhysX steps).

The cars start 20 m apart, alternately in the left and the right lane (all driving +s), at
s = 40 m, and hold 10 m/s over the cleats and the pothole (the same pursuit and speed hold as
`fixture_drive`); FRAMES frames are timed once every car is at speed. `--follow` drives along
the course's centreline (`Line`) instead of the straight line from its first point to its last:
a winding course (086 C6b's budget on Haruna). `--guard-ms` (issue 128): exit 1 when a frame of
the first GUARD_FRAMES (the first tiles around the cars) takes longer, or a rock touches the
base road (issue 123's self-check). The guard drives the whole course: at least FRAMES frames
and until the lead car is END_MARGIN from the course's end, and fails when it never gets there
(issue 145: 400 frames ended at 280 m of Haruna's 1527, before the first base contact, at
about 282 m). Writes `cost.json`:

    python -m scensim_isaac_world.course.fixture_cost --map <map dir> --out <dir>
        [--query raycast|sweep] [--cars 1|2|4] [--follow] [--road <road_detail.yaml>] [--guard-ms N]
"""
import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from scensim_isaac_world.course.fixture_drive import (FRAME, LANE, LOOKAHEAD,
                                                      groups_before_play, load_road, quat_rpy,
                                                      rate)

FRAMES = 400
GUARD_FRAMES = 120  # issue 128: the frames from the load, the first tiles around the cars
WARMUP_S = 6.0
SPEED = 10.0
GAP = 20.0
START = 40.0
END_MARGIN = 2 * LOOKAHEAD  # m before the course's end, where the guard's drive stops (issue 145)
GUARD_SLACK_S = 10.0  # s beyond the course's planned drive before the guard gives up on its end


def warm_distance(speed):
    """The way a car goes in WARMUP_S from rest: 2.5 m/s^2 up to `speed`, then holding it."""
    ramp = speed / 2.5
    if ramp >= WARMUP_S:
        return 1.25 * WARMUP_S ** 2
    return speed * speed / 5.0 + speed * (WARMUP_S - ramp)


def guard_frames(length, s0, speed):
    """The frames after the warm-up that take a car from `s0` to END_MARGIN before the end of
    a course `length` long at `speed`, GUARD_SLACK_S more, and at least FRAMES (issue 145)."""
    left = length - END_MARGIN - s0 - warm_distance(speed)
    return max(FRAMES, math.ceil(max(left, 0.0) / (speed * FRAME) + GUARD_SLACK_S / FRAME))


class Line:
    """A course's centreline (`centreline.npz`'s s, x, y, z) as the cars follow it: the point at
    s and lateral offset v (left positive), its heading, and a position's s."""

    def __init__(self, s, x, y, z):
        self.s, self.x, self.y, self.z = (np.asarray(a, float) for a in (s, x, y, z))
        self.length = float(self.s[-1])
        h = np.arctan2(np.gradient(self.y), np.gradient(self.x))
        self.c, self.sn = np.cos(h), np.sin(h)

    def heading(self, s):
        return math.atan2(np.interp(s, self.s, self.sn), np.interp(s, self.s, self.c))

    def xy(self, s, v):
        h = self.heading(s)
        return np.array([np.interp(s, self.s, self.x) - v * math.sin(h),
                         np.interp(s, self.s, self.y) + v * math.cos(h)])

    def z_at(self, s):
        return float(np.interp(s, self.s, self.z))

    def s_of(self, p):
        """The s of the centreline's point nearest p, refined along its heading."""
        i = int(np.argmin((self.x - p[0]) ** 2 + (self.y - p[1]) ** 2))
        return float(self.s[i] + (p[0] - self.x[i]) * self.c[i] + (p[1] - self.y[i]) * self.sn[i])


class Straight:
    """The straight line from a course's first point to its last (the fixture courses)."""

    def __init__(self, x, y):
        self.o = np.array([x[0], y[0]], float)
        self.h = float(math.atan2(y[-1] - y[0], x[-1] - x[0]))
        self.d = np.array([math.cos(self.h), math.sin(self.h)])
        self.n = np.array([-math.sin(self.h), math.cos(self.h)])
        self.length = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))

    def heading(self, s):
        return self.h

    def xy(self, s, v):
        return self.o + s * self.d + v * self.n

    def z_at(self, s):
        return 0.0

    def s_of(self, p):
        return float((np.asarray(p[:2]) - self.o) @ self.d)


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
    ap.add_argument("--guard-ms", type=float,
                    help="issue 128's guard: exit 1 if a frame of the first GUARD_FRAMES takes "
                    "longer, or a rock touches the base road, on the whole course (issue 145)")
    ap.add_argument("--road", help="the road record to load (a map's own "
                    "<map dir>/road_detail.yaml); default: the course's course/road_detail.yaml")
    ap.add_argument("--follow", action="store_true",
                    help="drive along the centreline, not the straight first-to-last line")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from scensim_isaac_world import road_layer, surfaces

    cl = np.load(os.path.join(a.map, "course", "centreline.npz"))
    line = Line(cl["s"], cl["x"], cl["y"], cl["z"]) if a.follow else Straight(cl["x"], cl["y"])

    from aib.rig import EgoConfig

    rig = EgoConfig.load(a.ego).load_rig()
    params = dataclasses.replace(road_layer.vehicle_params(rig.vehicle), wheel_query=a.query)
    SUB, substeps = rate(a.hz, params.physx_substeps)
    params = dataclasses.replace(params, physx_substeps=substeps)

    from aib import app_exit
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
    code = 0
    try:
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
        add_reference_to_stage(
            usd_path=os.path.realpath(os.path.join(a.map, "isaac", "world.usdc")),
            prim_path="/World/env")
        ensure_vehicle_scene(stage)
        groups_before_play(stage, a.map, a.road)  # issue 123: the world's USD is in before play
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
                  colliders=not a.no_colliders, road=a.road)
        cars = []
        for k in range(a.cars):
            lane = LANE if k % 2 == 0 else -LANE
            s0 = START + GAP * (k // 2) + GAP / 2 * (k % 2)
            p, h = line.xy(s0, lane), line.heading(s0)
            pose = ((float(p[0]), float(p[1]), line.z_at(s0) + 0.6),
                    (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)))
            veh = Vehicle(stage, f"/World/car{k}", params, rig.make_component(), rig.box_center,
                          rig.box_dims, pose, ground_clearance=rig.ground_clearance,
                          truth_settings=rig.component.get("truth"))
            cars.append({"veh": veh, "state": State(), "lane": lane, "i": 0.0,
                         "cmd": Command(received=True, gear=GEAR_DRIVE)})
            layer.placed(veh.path, pose)  # its tiles and rocks before the first step (issue 142)

        t, frame_ms, contact_ms, flush_ms, tiles, first_ms = 0.0, [], [], [], [], []
        lead = max(START + GAP * (k // 2) + GAP / 2 * (k % 2) for k in range(a.cars))
        end = line.length - END_MARGIN
        timed = FRAMES if a.guard_ms is None else guard_frames(line.length, lead, SPEED)
        total = int(WARMUP_S / FRAME) + timed
        reached = False
        for f in range(total):
            if a.guard_ms is not None and \
                    max(line.s_of(c["state"].position[:2]) for c in cars) >= end:
                reached = True  # the whole course (issue 145)
                break
            for c in cars:
                st, cmd = c["state"], c["cmd"]
                pos = np.array(st.position[:2])
                s_now = line.s_of(pos)
                tgt = line.xy(s_now + LOOKAHEAD, c["lane"])
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
                if layer.road_rocks is not None:  # issue 123's self-check
                    layer.road_rocks.check_base(sim.get_contact_report()[0], dt)
                layer.begin_step()
                for c in cars:
                    c["veh"].sample(physx, dt, c["state"])
                    layer.update(c["veh"], c["state"].velocity[0], dt)
            t1 = time.perf_counter()
            layer.flush()  # the world server's frame end: the tiles (issue 119), the rocks
            t += FRAME
            if f < GUARD_FRAMES:
                first_ms.append((time.perf_counter() - t0) * 1e3)
            if f >= total - timed:
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
               "follow": a.follow, "road": a.road or "",
               "s_end": [round(line.s_of(c["state"].position[:2]), 1) for c in cars],
               "course_m": round(line.length, 1), "course_end": reached,
               "first_frames_ms_max": round(max(first_ms), 1),
               "base_contacts": len(layer.road_rocks.base_hits) if layer.road_rocks else 0}
        print(f"fixture_cost: {json.dumps(out)}", flush=True)
        json.dump(out, open(os.path.join(a.out, "cost.json"), "w"), indent=1)
        if a.guard_ms is not None:  # issue 128's guard
            bad = []
            if out["first_frames_ms_max"] > a.guard_ms:
                bad.append(f"a frame of the first {GUARD_FRAMES} took "
                           f"{out['first_frames_ms_max']} ms (> {a.guard_ms})")
            if out["base_contacts"]:
                bad.append(f"{out['base_contacts']} rock(s) on the base road (issue 123)")
            if not reached:
                bad.append(f"the cars stopped at s {max(out['s_end'])} m, short of the course's "
                           f"{round(end, 1)} (issue 145: the guard drives the whole course)")
            print(f"fixture_cost guard: {'; '.join(bad) or 'ok'}", flush=True)
            if bad:
                code = 1
    except BaseException as e:  # app.close() would end the process with 0 (scensim issue 118)
        code = app_exit.code_of(e)
    finally:
        app_exit.close(app, code)


if __name__ == "__main__":
    main()
