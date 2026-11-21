"""The stage's record and the map check (roadmap 091 W15, W16): pxr alone."""
import json
import os
import tempfile
import time
import unittest

try:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:  # outside a Python with pxr
    Usd = None

if Usd is not None:
    from scensim_isaac_world import stage_record

CLASSES = ("asphalt", "concrete")


def field(d, name="field.usda", mpu=1.0, up="Z", sub=True, texture=True, vehicle=False,
          token=None, mu=None):
    """A small authored field: a root layer with a floor, a sublayer, a referenced prop, a
    texture asset."""
    root = os.path.join(d, name)
    stage = Usd.Stage.CreateNew(root)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/field")
    floor = UsdGeom.Cube.Define(stage, "/field/floor")
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    if sub:
        s = Sdf.Layer.CreateNew(os.path.join(d, "physics.usda"))
        s.Save()
        stage.GetRootLayer().subLayerPaths.append("physics.usda")
        p = Usd.Stage.CreateNew(os.path.join(d, "props", "cone.usda"))
        UsdGeom.Xform.Define(p, "/cone")
        p.SetDefaultPrim(p.GetPrimAtPath("/cone"))
        p.GetRootLayer().Save()
        stage.DefinePrim("/field/cone").GetReferences().AddReference("props/cone.usda")
    if texture:
        os.makedirs(os.path.join(d, "textures"), exist_ok=True)
        with open(os.path.join(d, "textures", "floor.png"), "wb") as f:
            f.write(b"png")
        shader = UsdShade.Shader.Define(stage, "/field/Looks/floor/tex")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("textures/floor.png")
    if vehicle:
        prim = stage.DefinePrim("/field/car", "Xform")
        prim.AddAppliedSchema("PhysxVehicleAPI")
    if token is not None:
        floor.GetPrim().CreateAttribute("scensim:surface", Sdf.ValueTypeNames.Token).Set(token)
    if mu is not None:
        mat = UsdShade.Material.Define(stage, "/field/Looks/phys")
        UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateDynamicFrictionAttr(mu)
    stage.GetRootLayer().Save()
    return root


@unittest.skipIf(Usd is None, "no pxr")
class Record(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_layer_and_asset_by_content(self):
        root = field(self.d)
        rec = stage_record.record(root, {"/x.a": {"authored": 60, "used": 600}},
                                  {"surfaces.yaml": os.path.join(self.d, "surfaces.yaml")})
        self.assertEqual(rec["root"], "field.usda")
        self.assertEqual(sorted(rec["layers"]), ["field.usda", "physics.usda",
                                                 os.path.join("props", "cone.usda")])
        self.assertEqual(sorted(rec["assets"]), [os.path.join("textures", "floor.png")])
        self.assertEqual(rec["overrides"], {"/x.a": {"authored": 60, "used": 600}})
        self.assertEqual(rec["records"], {"surfaces.yaml": None})
        self.assertEqual(rec["sha256"], stage_record.digest(rec["layers"], rec["assets"]))
        self.assertEqual(rec["hashing"]["files"], 4)

    def test_same_content_same_hash_an_edit_changes_it(self):
        root = field(self.d)
        a = stage_record.record(root)["sha256"]
        os.utime(os.path.join(self.d, "physics.usda"))  # touched, not changed
        self.assertEqual(stage_record.record(root)["sha256"], a)
        with open(os.path.join(self.d, "textures", "floor.png"), "ab") as f:
            f.write(b"!")
        b = stage_record.record(root)
        self.assertNotEqual(b["sha256"], a)

    def test_unresolved_refuses_naming_it(self):
        root = field(self.d)
        os.remove(os.path.join(self.d, "props", "cone.usda"))
        with self.assertRaises(stage_record.Unresolved) as e:
            stage_record.record(root)
        self.assertIn("cone.usda", str(e.exception))

    def test_cache_rehashes_only_changed_files(self):
        root = field(self.d)
        path = os.path.join(self.d, "state", stage_record.CACHE)
        first = stage_record.record(root, cache=stage_record.HashCache(path))
        self.assertEqual(first["hashing"]["files_hashed"], 4)
        again = stage_record.record(root, cache=stage_record.HashCache(path))
        self.assertEqual(again["hashing"]["files_hashed"], 0)
        self.assertEqual(again["sha256"], first["sha256"])
        time.sleep(0.01)
        with open(os.path.join(self.d, "textures", "floor.png"), "ab") as f:
            f.write(b"!")
        third = stage_record.record(root, cache=stage_record.HashCache(path))
        self.assertEqual(third["hashing"]["files_hashed"], 1)
        self.assertNotEqual(third["sha256"], first["sha256"])

    def test_torn_cache_hashes_everything(self):
        root = field(self.d)
        path = os.path.join(self.d, stage_record.CACHE)
        with open(path, "w") as f:
            f.write("{not json")
        rec = stage_record.record(root, cache=stage_record.HashCache(path))
        self.assertEqual(rec["hashing"]["files_hashed"], 4)

    def test_write_is_whole(self):
        out = stage_record.write(os.path.join(self.d, "state"), {"sha256": "x"})
        with open(out) as f:
            self.assertEqual(json.load(f), {"sha256": "x"})
        self.assertEqual(os.listdir(os.path.join(self.d, "state")), [stage_record.FILE])
        m = os.umask(0)
        os.umask(m)
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o666 & ~m)  # not mkstemp's 0600


@unittest.skipIf(Usd is None, "no pxr")
class Check(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_good_field_passes(self):
        self.assertEqual(stage_record.check(field(self.d), CLASSES), [])

    def test_units_and_axis(self):
        p = stage_record.check(field(self.d, mpu=0.01, up="Y"), CLASSES)
        self.assertTrue(any("metersPerUnit is 0.01" in x for x in p), p)
        self.assertTrue(any("upAxis is Y" in x for x in p), p)

    def test_a_vehicle_prim(self):
        p = stage_record.check(field(self.d, vehicle=True), CLASSES)
        self.assertTrue(any(x.startswith("/field/car: PhysxVehicleAPI") for x in p), p)

    def test_unresolved(self):
        root = field(self.d)
        os.remove(os.path.join(self.d, "textures", "floor.png"))
        p = stage_record.check(root, CLASSES)
        self.assertTrue(any("floor.png" in x for x in p), p)

    def test_authored_problems(self):
        p = stage_record.check(field(self.d, token="ice", mu=7.0), CLASSES)
        self.assertTrue(any("'ice' is not a surface class" in x for x in p), p)
        self.assertTrue(any("dynamicFriction 7 out of range" in x for x in p), p)

    def test_cli_check(self):
        root = field(self.d, mpu=0.01)
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(stage_record.main(["check", root]), 0)
        out = json.loads(buf.getvalue())
        self.assertTrue(any("metersPerUnit" in x for x in out["problems"]))

    def _map(self, authored, **kw):
        d = os.path.join(self.d, "assets", "maps", "lab")
        os.makedirs(os.path.join(d, "isaac"))
        field(d, **kw)
        wy = os.path.join(d, "isaac", "world.yaml")
        with open(wy, "w") as f:
            f.write("usd: field.usda\n" + ("authored: true\n" if authored else ""))
        return wy

    def test_map_authored(self):
        out = stage_record.check_map(self._map(True, mpu=0.01))
        self.assertEqual((out["map"], out["authored"]), ("lab", True))
        self.assertTrue(out["stage"].endswith(os.path.join("lab", "field.usda")))
        self.assertTrue(any("metersPerUnit" in x for x in out["problems"]), out)

    def test_map_without_a_lanelet_map(self):
        out = stage_record.check_map(self._map(True))
        self.assertTrue(any("no lanelet2_map.osm" in x and "scensim_isaac_world.loop" in x
                            for x in out["problems"]), out)
        with open(os.path.join(self.d, "assets", "maps", "lab", "lanelet2_map.osm"), "w") as f:
            f.write("<osm/>\n")
        self.assertEqual(stage_record.check_map(os.path.join(
            self.d, "assets", "maps", "lab", "isaac", "world.yaml"))["problems"], [])

    def test_map_not_authored_is_not_checked(self):
        out = stage_record.check_map(self._map(False, mpu=0.01))
        self.assertEqual((out["authored"], out["problems"]), (False, []))

    def test_map_missing_stage(self):
        wy = self._map(True)
        os.remove(os.path.join(os.path.dirname(os.path.dirname(wy)), "field.usda"))
        out = stage_record.check_map(wy)
        self.assertTrue(any("no such file" in x for x in out["problems"]), out)



@unittest.skipIf(Usd is None, "no pxr")
class Layered(unittest.TestCase):
    """The layered field (roadmap 091 W14): physics required, evidence and sources recorded."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.join(self.tmp.name, "lab")
        os.makedirs(os.path.join(self.d, "isaac"))

    def tearDown(self):
        self.tmp.cleanup()

    def world(self, evidence=None):
        wy = os.path.join(self.d, "isaac", "world.yaml")
        with open(wy, "w") as f:
            f.write("usd: field.usda\nauthored: true\n")
            if evidence is not None:
                f.write("evidence: [" + ", ".join(evidence) + "]\n")
        with open(os.path.join(self.d, "lanelet2_map.osm"), "w") as f:
            f.write("<osm/>\n")
        return wy

    def test_a_visual_layer_alone_is_refused(self):
        root = os.path.join(self.d, "field.usda")
        stage = Usd.Stage.CreateNew(root)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Mesh.Define(stage, "/field/scan_mesh")  # visual only, no collider
        stage.GetRootLayer().Save()
        p = stage_record.check(root, CLASSES)
        self.assertTrue(any("no collider" in x and "physics layer" in x for x in p), p)

    def test_evidence_is_hashed_and_a_missing_one_refused(self):
        field(self.d, sub=False, texture=False)
        os.makedirs(os.path.join(self.d, "scan"))
        with open(os.path.join(self.d, "scan", "field.e57"), "wb") as f:
            f.write(b"points")
        wy = self.world(["scan/field.e57"])
        self.assertEqual(stage_record.check_map(wy)["problems"], [])
        rec = stage_record.record(os.path.join(self.d, "field.usda"), map_dir=self.d)
        self.assertEqual(list(rec["evidence"]), [os.path.join("scan", "field.e57")])
        self.assertEqual(rec["evidence"][os.path.join("scan", "field.e57")],
                         stage_record.file_sha256(os.path.join(self.d, "scan", "field.e57")))
        wy = self.world(["scan/gone.e57", "../outside.e57"])
        p = stage_record.check_map(wy)["problems"]
        self.assertTrue(any("gone.e57: no such file" in x for x in p), p)
        self.assertTrue(any("outside the map directory" in x for x in p), p)

    def test_measured_attributes_and_their_sources(self):
        root = field(self.d, sub=False, texture=False, mu=0.62)
        stage = Usd.Stage.Open(root)
        mat = stage.GetPrimAtPath("/field/Looks/phys")
        with open(os.path.join(self.d, "tilt.csv"), "w") as f:
            f.write("angle,slid\n31.8,1\n")
        mat.CreateAttribute(stage_record.SOURCE_ATTR, Sdf.ValueTypeNames.Asset).Set("tilt.csv")
        cone = UsdGeom.Xform.Define(stage, "/field/cone").GetPrim()
        UsdPhysics.MassAPI.Apply(cone).CreateMassAttr(0.42)
        cone.CreateAttribute(stage_record.SOURCE_ATTR, Sdf.ValueTypeNames.String).Set(
            "kitchen scale, 2026-11-21")
        stage.GetRootLayer().Save()
        meas, notes = stage_record.measured(Usd.Stage.Open(root))
        f = meas["/field/Looks/phys.dynamicFriction"]
        self.assertAlmostEqual(f["value"], 0.62, places=6)
        self.assertEqual(f["source"]["path"], "tilt.csv")
        self.assertEqual(f["source"]["sha256"],
                         stage_record.file_sha256(os.path.join(self.d, "tilt.csv")))
        self.assertEqual(meas["/field/cone.mass"],
                         {"value": 0.41999998688697815, "source": "kitchen scale, 2026-11-21"})
        self.assertEqual(notes, [])
        self.assertEqual(stage_record.record(root)["measured"], meas)

    def test_an_unsourced_friction_is_a_note_not_a_refusal(self):
        field(self.d, sub=False, texture=False, mu=0.7)
        out = stage_record.check_map(self.world())
        self.assertEqual(out["problems"], [])
        self.assertTrue(any("/field/Looks/phys: dynamicFriction 0.7 has no scensim:source" in x
                            for x in out["notes"]), out)

if __name__ == "__main__":
    unittest.main()
