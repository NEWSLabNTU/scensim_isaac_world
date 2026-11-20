"""An authored floor's friction reaches the wheels (roadmap 091 W13's exit, design
roboracer-indoor-twin.md 3.3).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/scensim_isaac_world/spikes/authored_floor.py \\
        <out dir>
    # MU="0.5,1.0" (the floor's dynamicFriction, one field each); prints one SPIKE-RESULT json

For each friction it writes a small field as a user would author it in Isaac's editor (a 200 m
by 20 m floor collider in metres, Z up, a physics material bound with purpose physics, the
field's own physics scene at 60 steps/s), and its map directory with `authored: true`. One aib
ego without Autoware speeds up to 10.4 m/s at 3 m/s^2 and brakes asking 10 m/s^2, past what
either floor holds, so the stop is friction-limited: the distance should go about as 1/mu. It
also records what decided each collider's ground (`SurfaceLayer.decided`), the replaced step
count (`stage_overrides`), and that the field's file is untouched (mtime and bytes).
"""
import hashlib
import json
import math
import os
import sys

os.environ.pop("DISPLAY", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import logging  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from scensim_isaac_world.surface_layer import register  # noqa: E402
from scensim_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "authored_floor")
MUS = [float(m) for m in os.environ.get("MU", "0.5,1.0").split(",")]
REPO = os.environ.get("SCENSIM_REPO") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
START = ((90.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))  # heading -x
FLOOR = "/World/env/floor"  # the field's floor where the server references it


def write_field(d, mu):
    """A field as authored in Isaac's editor, and its map directory (authored: true)."""
    os.makedirs(os.path.join(d, "isaac"), exist_ok=True)
    path = os.path.join(d, "field.usda")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/field")
    stage.SetDefaultPrim(root.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/field/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    scene.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond",
                                    Sdf.ValueTypeNames.Int).Set(60)
    floor = UsdGeom.Cube.Define(stage, "/field/floor")
    floor.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(floor)
    xf.SetTranslate((0.0, 0.0, -0.5))
    xf.SetScale((200.0, 20.0, 1.0))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    mat = UsdShade.Material.Define(stage, "/field/Looks/floor")
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateDynamicFrictionAttr(mu)
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    wy = os.path.join(d, "isaac", "world.yaml")
    with open(wy, "w") as f:
        f.write("usd: field.usda\nauthored: true\nmap_origin: [0.0, 0.0, 0.0]\n"
                f"ground_paths:\n  - \"{FLOOR}\"\n")
    return path, wy


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest(), os.stat(path).st_mtime_ns


fields = []
for mu in MUS:
    usd, wy = write_field(os.path.join(OUT, f"mu_{mu:g}"), mu)
    fields.append((mu, os.path.realpath(usd), wy, digest(usd)))
maps = register([wy for _, _, wy, _ in fields])
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps=maps)
for s in world.ego_rig.sensors.values():
    s.enabled = False
from aib.control import GEAR_DRIVE  # noqa: E402

handle = [100]


def frame(ego, trace, t):
    world.step({}, SUBSTEPS, DT, False)
    s = ego.state
    trace.append((t, s.velocity[0], s.position[0], s.position[1]))


def command(ego, accel, velocity):
    c = ego.ros.cmd
    c.acceleration, c.velocity, c.tire_angle = accel, velocity, 0.0
    c.gear, c.received = GEAR_DRIVE, True


def brake(d):
    handle[0] += 1
    world.spawn(handle[0], "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5), START)
    ego = world.egos[handle[0]]
    for _ in range(20):
        world.step({}, SUBSTEPS, DT, False)
    trace, t = [], 0.0
    command(ego, 3.0, 40.0)
    while (not trace or trace[-1][1] < 10.4) and t < 15.0:
        t += FRAME
        frame(ego, trace, t)
    v0, t0, x0, y0 = trace[-1][1], t, trace[-1][2], trace[-1][3]
    command(ego, -d, 0.0)
    while trace[-1][1] > 0.05 and t - t0 < 20.0:
        t += FRAME
        frame(ego, trace, t)
    world.despawn(handle[0])
    stop = t - t0
    dist = math.hypot(trace[-1][2] - x0, trace[-1][3] - y0)
    return {"from": round(v0, 2), "stop_s": round(stop, 2), "distance_m": round(dist, 2),
            "mean_decel": round(v0 / stop, 3) if stop > 0 else None}


out = {"fields": []}
for mu, usd, wy, before in fields:
    world.load(usd, [FLOOR])
    r = brake(10.0)
    r.update({"mu": mu, "decided": world.surface.decided(),
              "overrides": world.stage_overrides,
              "field_untouched": digest(usd) == before})
    print("field", r, flush=True)
    out["fields"].append(r)
if len(out["fields"]) == 2:
    a, b = out["fields"]
    out["distance_ratio"] = round(a["distance_m"] / b["distance_m"], 3)
    out["mu_ratio_inverse"] = round(b["mu"] / a["mu"], 3)
print("SPIKE-RESULT", json.dumps(out), flush=True)
app.close()
