"""A layered field: the car stands on the physics layer, never on the visual one, and the run's
record names every layer and the evidence (roadmap 091 W14's exit, design
roboracer-indoor-twin.md 3.1, 3.4).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/scensim_isaac_world/spikes/layered_field.py \\
        <out dir>
    # prints one SPIKE-RESULT json; NO_VISUAL=1 leaves the visual layer out (the baseline height)

The field as a scan-based twin is made: a root stage sublayering `visual.usda` (a scan mesh,
no collider, 5 cm above the floor so a wheel resting on it would show) and `physics.usda` (the
floor collider at z 0 with its physics material, and its friction's `scensim:source`), and a
raw scan file listed as `evidence:`. One aib ego settles on it and drives 5 m. It records the
car's height against both layers, what decided the ground, and the stage.json the server wrote
(layers, evidence, measured).
"""
import json
import os
import sys

os.environ.pop("DISPLAY", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import logging  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from scensim_isaac_world import stage_record  # noqa: E402
from scensim_isaac_world.surface_layer import register  # noqa: E402
from scensim_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "layered_field")
REPO = os.environ.get("SCENSIM_REPO") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
VISUAL_Z = 0.05  # the scan mesh's height above the collider
START = ((0.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))
FLOOR = "/World/env/floor"


def stage(path):
    s = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(s, 1.0)
    UsdGeom.SetStageUpAxis(s, UsdGeom.Tokens.z)
    return s


def write_field(d):
    os.makedirs(os.path.join(d, "isaac"), exist_ok=True)
    os.makedirs(os.path.join(d, "scan"), exist_ok=True)
    with open(os.path.join(d, "scan", "field.ply"), "w") as f:  # the raw scan, as evidence
        f.write("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                "property float z\nend_header\n0 0 0\n")
    with open(os.path.join(d, "scan", "tilt.csv"), "w") as f:
        f.write("angle_deg,slid\n41.9,1\n")
    vis = stage(os.path.join(d, "visual.usda"))
    UsdGeom.Xform.Define(vis, "/field")
    mesh = UsdGeom.Mesh.Define(vis, "/field/scan_mesh")  # visual only: no CollisionAPI
    h = VISUAL_Z
    mesh.CreatePointsAttr([(-50, -10, h), (50, -10, h), (50, 10, h), (-50, 10, h)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    vis.GetRootLayer().Save()
    phy = stage(os.path.join(d, "physics.usda"))
    UsdGeom.Xform.Define(phy, "/field")
    floor = UsdGeom.Cube.Define(phy, "/field/floor")
    floor.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(floor)
    xf.SetTranslate((0.0, 0.0, -0.5))
    xf.SetScale((100.0, 20.0, 1.0))
    UsdGeom.Imageable(floor).CreateVisibilityAttr("invisible")  # the scan shows, the collider not
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    mat = UsdShade.Material.Define(phy, "/field/Looks/floor")
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateDynamicFrictionAttr(0.9)
    mat.GetPrim().CreateAttribute(stage_record.SOURCE_ATTR, Sdf.ValueTypeNames.Asset).Set(
        "scan/tilt.csv")
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    phy.GetRootLayer().Save()
    root = stage(os.path.join(d, "field.usda"))
    if not os.environ.get("NO_VISUAL"):  # the baseline: the physics layer alone
        root.GetRootLayer().subLayerPaths.append("visual.usda")
    root.GetRootLayer().subLayerPaths.append("physics.usda")
    root.SetDefaultPrim(root.DefinePrim("/field", "Xform"))
    root.GetRootLayer().Save()
    wy = os.path.join(d, "isaac", "world.yaml")
    with open(wy, "w") as f:
        f.write("usd: field.usda\nauthored: true\nmap_origin: [0.0, 0.0, 0.0]\n"
                f"ground_paths:\n  - \"{FLOOR}\"\nevidence: [scan/field.ply]\n")
    return os.path.realpath(os.path.join(d, "field.usda")), wy


d = os.path.join(OUT, "field")
usd, wy = write_field(d)
check = stage_record.check_map(wy)
maps = register([wy])
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps=maps)
for s in world.ego_rig.sensors.values():
    s.enabled = False
from aib.control import GEAR_DRIVE  # noqa: E402

world.load(usd, [FLOOR])
world.spawn(100, "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5), START)
ego = world.egos[100]
for _ in range(60):
    world.step({}, SUBSTEPS, DT, False)
z_rest = ego.state.position[2]
c = ego.ros.cmd
c.acceleration, c.velocity, c.tire_angle, c.gear, c.received = 1.0, 2.0, 0.0, GEAR_DRIVE, True
x0 = ego.state.position[0]
for _ in range(240):
    world.step({}, SUBSTEPS, DT, False)
    if abs(ego.state.position[0] - x0) > 5.0:  # it faces -x
        break
moved = abs(ego.state.position[0] - x0)
z_drive = ego.state.position[2]
rec = world.stage_provenance or {}
out = {"check": check, "z_rest": round(z_rest, 4), "z_drive": round(z_drive, 4),
       "moved_m": round(moved, 2), "decided": world.surface.decided(),
       "layers": sorted(rec.get("layers") or {}), "evidence": rec.get("evidence"),
       "measured": rec.get("measured")}
print("SPIKE-RESULT", json.dumps(out), flush=True)
app.close()
