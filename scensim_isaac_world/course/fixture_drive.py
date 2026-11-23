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
SUB = 3  # PhysX steps a frame at the default 60 Hz


def rate(hz, vehicle_substeps):
    """(PhysX steps a frame, aib's substeps a step) for a physics rate of `hz`, given aib's
    substeps at 60 Hz: the frame stays FRAME s and the vehicle's own integration its 60 Hz
    rate (aib's 8 substeps: 480 Hz), so only our contact's rate (once a PhysX step) and the
    logging change (086 C6a2)."""
    sub = round(hz * FRAME)
    own = vehicle_substeps * 60
    if hz <= 0 or abs(sub - hz * FRAME) > 1e-9 or own % hz:
        raise SystemExit(f"--hz {hz}: a whole number of steps a {FRAME} s frame, "
                         f"dividing the vehicle's {own} Hz")
    return sub, own // hz


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
# 086 C6b's fixture course (`courses/fixture_c6b.yaml`): R4's rocks, each at s with the length
# its check looks over; 40 km/h is 11.11 m/s, 10 km/h 2.78
FIXTURES_C6B = {"embedded_15": (100.0, 0.3), "loose_15": (200.0, 0.3), "loose_8": (300.0, 0.2),
                "boulder_35": (400.0, 0.8)}
CASES_C6B = ["embedded_15@11.11", "loose_15@11.11", "loose_8@11.11", "boulder_35@2.78"]
STUCK_S = 3.0  # s under 0.1 m/s past a fixture's face less 3 m: the case ends (high-centred)


def runup(speed):
    """m before a fixture a case starts at rest: the spec's 60 m; at 20 m/s 95 m, since 60 m
    cannot reach 20 m/s at the speed hold's 3 m/s^2 (66.7 m) before the start check, CHECK_AT m
    before the fixture."""
    return 60.0 if speed <= 12.0 else 95.0


def load_road(map_dir, stage, layer, physx, world, apply_force, classes, rocks="",
              colliders=True, rocks_on_base=False, pool="on"):
    """The course's road on `layer` as world.py's _load_road loads a map's: the record's strips
    as zones, the road layer, the detail's collision tiles (issue 119; `colliders` False: none,
    as before it), and with rocks their stream on those tiles, then one physics step (issue
    039). `rocks`: "shoulder,runoff,ditch" cover put on every corridor, the carriageway's 0
    (086 C6b; empty: the course's own). `rocks_on_base` (issue 122's test): no collision tiles
    and the world's ground not filtered out for the rocks, so they rest on the base mesh as R4's
    first slice had them."""
    from scensim_isaac_world import road_layer, rock_layer

    road_file = os.path.join(map_dir, "course", road_layer.FILE)
    corridors = road_layer.load(road_file, classes=classes)
    if rocks:
        from scensim_isaac_world import rocks as rk

        sh, ro, di = (float(x) for x in rocks.split(","))
        corridors = [dataclasses.replace(c, rocks=rk.Zones(carriageway=0.0, shoulder=sh,
                                                           runoff=ro, ditch=di))
                     for c in corridors]
    layer.add_record_zones(road_layer.strip_zones(corridors))
    layer.road = road_layer.RoadLayer(corridors, (0.0, 0.0, 0.0), apply_force)
    if rocks_on_base:
        class NoTiles(rock_layer.DetailColliders):
            """The tiles' keys, which the rock stream streams by, with no collider made."""

            def update(self, points):
                d, ox, oy = self.layer.detail, self.layer.ox, self.layer.oy
                want = {k for k in d.tiles_near([p[0] + ox for p in points],
                                                [p[1] + oy for p in points],
                                                rock_layer.COLLISION_REACH)
                        if d.touches(*k) and d.ready(k)} if points else set()
                self.coll = {k: None for k in want}

            def remove(self):
                self.coll = {}

        layer.road_colliders = NoTiles(stage, layer.road)
    elif colliders or layer.road.rocks is not None:
        layer.road_colliders = rock_layer.DetailColliders(stage, layer.road)
    if layer.road.rocks is not None:
        import yaml

        with open(os.path.join(map_dir, "isaac", "world.yaml")) as f:
            ground = list(yaml.safe_load(f).get("ground_paths") or [])
        # issue 123's diagnosis: the stream's pool of parked rock prims as it is ("on"), none
        # (every rock made live), or made with its group membership a physics step later
        layer.road_rocks = layer.road.rock_stream = rock_layer.RockStream(
            stage, layer.road, physx,
            ground=() if rocks_on_base else (road_layer.load_ground(road_file) or ground),
            colliders=layer.road_colliders, prewarm=pool == "on")
        if pool == "late":
            stream = layer.road_rocks
            flush, stream._flush_groups = stream._flush_groups, (lambda: None)
            stream.prewarm()
            world.step(render=False)
            stream._flush_groups = flush
            stream._flush_groups()
    world.step(render=False)


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
    ap.add_argument("--hz", type=int, default=60, help="physics rate: 60, 120 or 240 (C6a2)")
    ap.add_argument("--no-colliders", action="store_true",
                    help="no detail collision tiles without rocks (as before issue 119)")
    ap.add_argument("--trace-push", action="store_true",
                    help="issue 122: every rock push (reaction, limit, applied) and the case's "
                    "rock's own contacts, to push.csv")
    ap.add_argument("--server-order", action="store_true",
                    help="issue 123: the simulation started before the world's USD and the road, "
                    "as the world server loads a map")
    ap.add_argument("--groups-early", action="store_true",
                    help="issue 123: roadBase's members set before the simulation starts too")
    ap.add_argument("--groups-late", action="store_true",
                    help="issue 123, as before its fix: the road's groups made at the road's load")
    ap.add_argument("--rock-pool", choices=("on", "off", "late"), default="on",
                    help="issue 123: the parked-rock pool as it is, none, or its group a step late")
    ap.add_argument("--rocks-on-base", action="store_true",
                    help="issue 122's test: no collision tiles, the rocks on the base mesh")
    ap.add_argument("--fixtures", choices=("c6a", "c6b"), default="c6a",
                    help="the course's fixtures: C6a's cleats or C6b's rocks")
    ap.add_argument("--rocks", default="", help="shoulder,runoff,ditch rock cover put on every "
                    "corridor, the carriageway's 0 (086 C6b; default: the course's own)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fixtures = FIXTURES_C6B if a.fixtures == "c6b" else FIXTURES
    cases = a.cases.split(",") if a.cases else (CASES_C6B if a.fixtures == "c6b" else CASES)

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
    from scensim_isaac_world import rock_layer
    from scensim_isaac_world.surface_layer import SurfaceLayer

    dt = FRAME / SUB
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    stage = omni.usd.get_context().get_stage()

    def add_world():
        add_reference_to_stage(
            usd_path=os.path.realpath(os.path.join(a.map, "isaac", "world.usdc")),
            prim_path="/World/env")

    if not a.server_order:
        add_world()
    ensure_vehicle_scene(stage)
    if not a.groups_late:  # as world.py: the road's groups before the simulation starts
        rock_layer.define_groups(stage)
    if a.groups_early:
        # issue 123's test (iv): the world's ground in roadBase before the simulation starts too
        import yaml
        from pxr import Sdf, UsdPhysics

        from aib.isaac.vehicle import GROUPS

        with open(os.path.join(a.map, "isaac", "world.yaml")) as f:
            gp = list(yaml.safe_load(f).get("ground_paths") or [])
        UsdPhysics.CollisionGroup.Get(stage, f"{GROUPS}/roadBase").GetCollidersCollectionAPI() \
            .GetIncludesRel().SetTargets([Sdf.Path(q) for q in gp])
    world.reset()
    world.play()
    if a.server_order:  # issue 123: as world.py, the simulation running before the map's USD
        add_world()
    physx = omni.physx.get_physx_interface()
    sim = omni.physx.get_physx_simulation_interface()

    def apply_force(path, force, position):  # as world.py's _apply_force
        sim.apply_force_at_pos(omni.usd.get_context().get_stage_id(),
                               PhysicsSchemaTools.sdfPathToInt(path), force, position)

    smap = surfaces.load(os.path.join(a.map, surfaces.FILE))
    layer = SurfaceLayer(stage, smap, apply_force=apply_force)
    load_road(a.map, stage, layer, physx, world, apply_force, set(smap.table.classes),
              rocks=a.rocks, colliders=not a.no_colliders, rocks_on_base=a.rocks_on_base,
              pool=a.rock_pool)

    (x0, y0), h0 = xy_at(10.0, LANE)
    pose = ((float(x0), float(y0), 1.0), (math.cos(h0 / 2), 0.0, 0.0, math.sin(h0 / 2)))
    veh = Vehicle(stage, "/World/ego", params, rig.make_component(), rig.box_center,
                  rig.box_dims, pose, ground_clearance=rig.ground_clearance,
                  truth_settings=rig.component.get("truth"))
    if layer.road_rocks is not None:
        if a.rock_pool == "on":
            layer.road_rocks.reserve(1)
    state = State()
    # the skid plate's bottom (aib's: between the axles, clear of the wheels), sampled 5 x 3
    from aib.frames import qrot

    r_w = params.wheel_radius
    px = np.linspace(r_w, params.wheel_base - r_w, 5)
    py = np.linspace(-1, 1, 3) * (params.wheel_tread - params.wheel_width - 0.1) / 2
    plate = [(x, y, rig.ground_clearance) for x in px for y in py]

    def road_z(x, y):
        """The true surface at world (x, y): the base road (the section in it) and the
        detail."""
        u, v = uv(x, y)
        u0 = u - s0
        return float(corridor.base_z(u0) + corridor.section.height_uv(u0, v)
                     + layer.road.height(np.array([x]), np.array([y]))[0])

    def plate_clearance():
        if rig.ground_clearance <= 0:
            return ""
        P, q = state.position, state.quat
        return round(min(P[2] + qrot(q, p)[2] - road_z(P[0] + qrot(q, p)[0],
                                                       P[1] + qrot(q, p)[1]) for p in plate), 4)

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
        layer.flush()

    # fl_s: the front-left hub's s (the reference's x); fl_z: its attachment's height (the
    # corner's body, the reference's zs)
    header = ["case", "t", "step", "s", "v", "speed", "z", "pitch", "az", "fl_s", "fl_z"]
    header += [f"load{i}" for i in range(4)] + [f"air{i}" for i in range(4)]
    # C6b: PhysX's jounce per wheel (its rigid limit: px_travel), the car's longitudinal
    # acceleration (a plate strike's spike), the plate's lowest clearance over the true surface
    header += [f"jounce{i}" for i in range(4)] + ["ax", "plate"]
    # C6b (133): the front-left wheel's centre height (our contact's), the rocks in the data
    # and in the stage and awake, and the case's own rock (the one nearest its fixture)
    header += ["zu0", "rocks_loaded", "rock_prims", "rocks_awake", "rock_x", "rock_y", "rock_z",
               "rock_speed"]

    # issue 122: each push on a rock as rocks.limit sees it, and the case's rock's contacts
    push_rows, cur = [], {"case": "", "n": 0}
    if a.trace_push and layer.road_rocks is not None:
        from scensim_isaac_world import rock_layer

        limit0 = rock_layer.rk.limit

        def traced_limit(forces, mass, v_rock, v_push, dt_):
            out, sc = limit0(forces, mass, v_rock, v_push, dt_)
            fh = sum(f[:2] for f, _ in forces)
            fz = sum(f[2] for f, _ in forces)
            push_rows.append([cur["case"], cur["n"], "push", round(mass, 2),
                              round(float(np.linalg.norm(fh)), 1), round(float(fz), 1),
                              round(float(v_push), 4), round(float(sc), 4),
                              round(float(np.linalg.norm(fh)) * sc, 1),
                              round(float(np.linalg.norm(np.asarray(v_rock)[:2])), 4),
                              "", "", ""])
            return out, sc

        rock_layer.rk.limit = traced_limit
    reported = set()

    def rock_contacts(found):
        """The case's rock's contacts this step (PhysX's contact report on its prim): the
        normal and tangential force summed over its contact points, N."""
        if not a.trace_push or found is None or layer.road_rocks is None:
            return
        path = layer.road_rocks.paths[found] if found < len(layer.road_rocks.paths) else None
        if not path:
            return
        if path not in reported:
            from pxr import PhysxSchema

            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                reported.add(path)
                # issue 123: every collision group that holds the rock or the base road, and
                # what each filters
                from pxr import Sdf, UsdPhysics

                for g in stage.Traverse():
                    if not g.IsA(UsdPhysics.CollisionGroup):
                        continue
                    grp = UsdPhysics.CollisionGroup(g)
                    q = grp.GetCollidersCollectionAPI().ComputeMembershipQuery()
                    has = [x for x in (path, "/World/env/Road", "/World/env/Terrain")
                           if q.IsPathIncluded(Sdf.Path(x))]
                    filt = [str(t) for t in grp.GetFilteredGroupsRel().GetTargets()]
                    print(f"groups: {g.GetPath()} holds {has} filters {filt}", flush=True)
            return
        headers, data = sim.get_contact_report()
        fn = ft = 0.0
        per = {}  # the other collider -> its normal force, N
        for h in headers:
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            if path not in (a0, a1):
                continue
            other = str(PhysicsSchemaTools.intToSdfPath(h.collider1 if a0 == path
                                                         else h.collider0))
            for c in data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]:
                imp, nrm = np.array(c.impulse), np.array(c.normal)
                jn = float(np.dot(imp, nrm))
                fn += abs(jn) / dt
                ft += float(np.linalg.norm(imp - jn * nrm)) / dt
                per[other] = per.get(other, 0.0) + abs(jn) / dt
        push_rows.append([cur["case"], cur["n"], "contact", "", "", "", "", "", "", "",
                          round(fn, 1), round(ft, 1),
                          ";".join(f"{k}={v:.0f}" for k, v in sorted(per.items()))])

    def rock_state(fixture_xy, found):
        rs, stream = layer.road.rocks, layer.road_rocks
        if rs is None or stream is None:
            return ["", "", "", "", "", "", ""], found
        if found is None and len(rs.pos):
            dd = np.hypot(rs.pos[:, 0] - fixture_xy[0], rs.pos[:, 1] - fixture_xy[1])
            if dd.min() <= 1.5:
                found = int(np.argmin(dd))
        prims = sum(1 for q in stream.paths if q) + sum(1 for q in stream.win.values() if q)
        rock = ["", "", "", ""]
        if found is not None:
            v = stream.vel.get(found)
            rock = [round(float(c), 4) for c in rs.pos[found]] + \
                [round(float(np.linalg.norm(v)), 3) if v is not None else 0.0]
        return [len(rs.pos), prims, stream.awake()] + rock, found
    rows, report = [], {"fixtures": a.fixtures, "rocks": a.rocks,
                        "rocks_on_base": a.rocks_on_base, "rock_pool": a.rock_pool,
                        "colliders": layer.road_colliders is not None,
                        "ground_clearance": rig.ground_clearance,
                        "px_travel": params.travels()[0], "query": a.query,
                        "physics_hz": round(1.0 / dt, 3),
                        "physx_substeps": params.physx_substeps, "frame_s": FRAME,
                        "aib": os.environ.get("AIB_COMMIT", ""), "cases": []}
    idle = Command(received=True, gear=GEAR_DRIVE)
    try:
        for case in cases:
            name, speed, k = parse_case(case)
            fs, flen = fixtures[name]
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
            fixture_xy, found, still = xy_at(fs + flen / 2, LANE)[0], None, 0.0
            while t < 60.0:
                pos = np.array(state.position[:2])
                s_now, v_now = uv(*pos)
                if s_now >= fs + flen + AFTER:
                    break
                if s_now > fs - 3.0 and abs(state.velocity[0]) < 0.1:
                    still += FRAME
                    if still >= STUCK_S:
                        info["stuck_at_s"] = round(s_now, 3)
                        break
                else:
                    still = 0.0
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
                    rock_cols, found = rock_state(fixture_xy, found)
                    cur.update(case=case, n=n)
                    rock_contacts(found)
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
                                 *loads, *air,
                                 *[round(float(j), 4) for j in (veh.jounce or [""] * 4)],
                                 round(state.acceleration[0], 4), plate_clearance(),
                                 round(float(car.contact.zu[0]), 5) if car is not None else "",
                                 *rock_cols])
                layer.flush()
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
        if push_rows:
            with open(os.path.join(a.out, "push.csv"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["case", "step", "kind", "mass", "fh_in", "fz_in", "v_push", "scale",
                             "fh_out", "v_rock", "contact_fn", "contact_ft", "contact_with"])
                wr.writerows(push_rows)
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
