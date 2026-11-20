"""An authored stage's ground (roadmap 091 W13, design roboracer-indoor-twin.md 3.3).

The map and wheel side needs nothing; the stage reader needs pxr, which Isaac's venv has and
the system Python does not (those tests are skipped there):

    source src/scensim_isaac_world/scripts/isaac_env.sh && python -m unittest discover -s src/scensim_isaac_world/test
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scensim_isaac_world import surface_layer, surfaces  # noqa: E402
from scensim_isaac_world.surfaces import RoadCondition, SurfaceError  # noqa: E402

try:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from scensim_isaac_world import authored
except ImportError:
    Usd = None

from types import SimpleNamespace  # noqa: E402

TABLE = surfaces.load_classes()
SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
VEHICLE_GRAPH = ((0.0, 1.0), (0.1, 1.0), (1.0, 0.8))
RIG = SimpleNamespace(mass=1760.0, lat_stiff_value=17.0, long_stiff_value=4612.5)
RIG_PEAKS = surface_layer.rig_peaks(RIG)


def ground(source, cls=None, mu=None):
    """An `authored.Ground` without pxr (the map side takes any object with these fields)."""
    return SimpleNamespace(source=source, cls=cls, mu=mu)


class MapPrecedence(unittest.TestCase):
    """3.3's order at a point: a zone, then the token, then the material, then the label's."""

    def setUp(self):
        self.map = surfaces.parse({"defaults": {"Road": "asphalt"},
                                   "zones": [{"id": "wet_patch", "class": "ice",
                                              "polygon": SQUARE}]}, table=TABLE)

    def test_a_zone_wins_over_everything_authored(self):
        s = self.map.at(5.0, 5.0, "Road", ground("token", cls="sand"))
        self.assertEqual((s.cls, s.source, s.zone, s.mu), ("ice", "zone", "wet_patch", None))
        s = self.map.at(5.0, 5.0, None, ground("material", mu=0.6))
        self.assertEqual((s.cls, s.source, s.mu), ("ice", "zone", None))

    def test_outside_a_zone_the_token_names_the_class(self):
        s = self.map.at(50.0, 5.0, "Road", ground("token", cls="concrete"))
        self.assertEqual((s.cls, s.source, s.mu), ("concrete", "token", None))

    def test_a_material_is_a_peak_on_the_reference_class(self):
        s = self.map.at(50.0, 5.0, "Road", ground("material", mu=0.6))
        self.assertEqual((s.cls, s.source, s.mu), (TABLE.reference, "material", 0.6))

    def test_nothing_authored_is_the_labels_default(self):
        s = self.map.at(50.0, 5.0, None, ground("no_map"))
        self.assertEqual((s.cls, s.source), (TABLE.reference, "no_map"))

    def test_a_map_that_is_not_authored_says_label(self):
        s = self.map.at(50.0, 5.0, "Road")
        self.assertEqual((s.cls, s.source), ("asphalt", "label"))


class MaterialFriction(unittest.TestCase):
    """A material's dynamicFriction is realised as the tire's peak, as a class's mu is."""

    def wheel(self, tire_friction=1.0):
        return surfaces.WheelSurfaces(surfaces.parse({}, table=TABLE), tire_friction,
                                      VEHICLE_GRAPH, RIG_PEAKS, wheels=1)

    def test_the_peak_is_the_material_over_the_reference(self):
        ref = TABLE[TABLE.reference].mu
        w = self.wheel(tire_friction=1.0)
        s = surfaces.Surface(TABLE.reference, mu=0.56, source="material")
        mu = w.update(0, s, RoadCondition(), 5.0, 0.0, 0.01)
        self.assertAlmostEqual(w.mu[0], 0.56 / ref)
        self.assertAlmostEqual(mu, 1.0 * 0.56 / ref)  # at zero slip: the peak itself

    def test_the_same_number_as_a_class_drives_the_same(self):
        # a material at asphalt's own mu is asphalt: one meaning of mu for both
        a, b = self.wheel(), self.wheel()
        mat = surfaces.Surface(TABLE.reference, mu=TABLE["asphalt"].mu, source="material")
        cls = surfaces.Surface("asphalt", source="label")
        for slip in (0.0, 0.05, 0.3):
            self.assertAlmostEqual(a.update(0, mat, RoadCondition(), 5.0, slip, 0.01),
                                   b.update(0, cls, RoadCondition(), 5.0, slip, 0.01))

    def test_the_scenarios_factor_scales_it_and_wetness_does_not(self):
        ref = TABLE[TABLE.reference].mu
        w = self.wheel()
        s = surfaces.Surface(TABLE.reference, mu=0.8, source="material")
        w.update(0, s, RoadCondition(friction_scale=0.5, wetness="wetWithPuddles"), 5.0, 0.0,
                 0.01)
        self.assertAlmostEqual(w.mu[0], 0.5 * 0.8 / ref)


class Register(unittest.TestCase):
    def world(self, d, text):
        (Path(d) / "isaac").mkdir(exist_ok=True)
        (Path(d) / "isaac" / "world.yaml").write_text(text)
        return str(Path(d) / "isaac" / "world.yaml")

    def test_authored_is_read_and_defaults_to_false(self):
        with tempfile.TemporaryDirectory() as d:
            got = surface_layer.register([self.world(d, "usd: field.usd\n")])
            self.assertFalse(next(iter(got.values())).authored)
            got = surface_layer.register([self.world(d, "usd: field.usd\nauthored: true\n")])
            self.assertTrue(next(iter(got.values())).authored)

    def test_authored_must_be_true_or_false(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SurfaceError, "authored: 'yes please' is not true"):
                surface_layer.register([self.world(d, "usd: f.usd\nauthored: yes please\n")])


@unittest.skipIf(Usd is None, "no pxr (run in Isaac's venv)")
class Reader(unittest.TestCase):
    """The reader on a small in-memory field: pxr only, no Isaac app."""

    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.stage, "/World/env")
        for p in ("/World/env/floor", "/World/env/floor/tile", "/World/env/mat_floor",
                  "/World/env/painted", "/World/env/bare"):
            UsdGeom.Cube.Define(self.stage, p)
        self.read = authored.GroundReader(self.stage, TABLE.classes)

    def physics_material(self, path, mu):
        mat = UsdShade.Material.Define(self.stage, path)
        UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateDynamicFrictionAttr(mu)
        return mat

    def bind(self, prim_path, mat, purpose=authored.PHYSICS_PURPOSE):
        prim = self.stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, purpose)

    def token(self, prim_path, name):
        self.stage.GetPrimAtPath(prim_path).CreateAttribute(
            authored.SURFACE_ATTR, Sdf.ValueTypeNames.Token).Set(name)

    def test_a_token_on_an_ancestor_names_the_class(self):
        self.token("/World/env/floor", "concrete")
        g = self.read("/World/env/floor/tile")
        self.assertEqual((g.source, g.cls, g.prim), ("token", "concrete", "/World/env/floor"))

    def test_a_bound_physics_material_gives_its_friction(self):
        self.bind("/World/env/mat_floor", self.physics_material("/World/Looks/vinyl", 0.62))
        g = self.read("/World/env/mat_floor")
        self.assertEqual((g.source, g.mu, g.prim), ("material", 0.62, "/World/Looks/vinyl"))

    def test_a_token_wins_over_a_material(self):
        self.bind("/World/env/floor/tile", self.physics_material("/World/Looks/vinyl", 0.62))
        self.token("/World/env/floor", "concrete")
        self.assertEqual(self.read("/World/env/floor/tile").source, "token")

    def test_a_render_material_is_not_ground(self):
        # an all-purpose binding to a look without the physics API says nothing of friction
        look = UsdShade.Material.Define(self.stage, "/World/Looks/paint")
        self.bind("/World/env/painted", look, UsdShade.Tokens.allPurpose)
        self.assertEqual(self.read("/World/env/painted").source, "no_map")

    def test_nothing_authored_and_no_prim_are_no_map(self):
        self.assertEqual(self.read("/World/env/bare"), authored.NOTHING)
        self.assertEqual(self.read("/World/env/gone"), authored.NOTHING)
        self.assertEqual(self.read(None), authored.NOTHING)

    def test_an_unknown_token_and_a_bad_friction_are_refused_naming_the_prim(self):
        self.token("/World/env/floor", "lino")
        self.bind("/World/env/mat_floor", self.physics_material("/World/Looks/oops", 62.0))
        with self.assertRaisesRegex(ValueError, "/World/env/floor: scensim:surface 'lino'"):
            self.read("/World/env/floor/tile")
        with self.assertRaisesRegex(ValueError, "/World/Looks/oops: dynamicFriction 62"):
            self.read("/World/env/mat_floor")
        got = authored.problems(self.stage, TABLE.classes)
        self.assertEqual(len(got), 2, got)
        self.assertTrue(got[0].startswith("/World/Looks/oops") or
                        got[1].startswith("/World/Looks/oops"))

    def test_a_frictionless_wall_is_no_problem(self):
        self.physics_material("/World/Looks/glass", 0.0)
        self.assertEqual(authored.problems(self.stage, TABLE.classes), [])

    def test_problems_under_a_root_only(self):
        self.token("/World/env/floor", "lino")
        self.assertEqual(authored.problems(self.stage, TABLE.classes, "/World/Looks"),
                         ["/World/Looks: no such prim"])
        self.assertEqual(len(authored.problems(self.stage, TABLE.classes, "/World/env")), 1)

    def test_the_cache_holds_until_a_new_reader(self):
        self.read("/World/env/bare")
        self.token("/World/env/bare", "sand")
        self.assertEqual(self.read("/World/env/bare").source, "no_map")
        self.assertEqual(authored.GroundReader(self.stage, TABLE.classes)(
            "/World/env/bare").cls, "sand")


@unittest.skipIf(Usd is None, "no pxr (run in Isaac's venv)")
class UserLayers(unittest.TestCase):
    """The user's stage referenced as the world server does: its scene found, its layers
    never edited by an opinion in the server's own layer."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.field = os.path.join(self.dir.name, "field.usda")
        user = Usd.Stage.CreateNew(self.field)
        UsdGeom.SetStageMetersPerUnit(user, 1.0)
        UsdGeom.SetStageUpAxis(user, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(user, "/field")
        user.SetDefaultPrim(root.GetPrim())
        scene = UsdPhysics.Scene.Define(user, "/field/physicsScene")
        scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)
        scene.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond",
                                        Sdf.ValueTypeNames.Int).Set(60)
        UsdGeom.Cube.Define(user, "/field/floor")
        user.GetRootLayer().Save()
        self.stage = Usd.Stage.CreateInMemory()
        self.stage.DefinePrim("/World/env").GetReferences().AddReference(self.field)

    def tearDown(self):
        self.dir.cleanup()

    def test_the_authored_scene_is_found_with_its_values(self):
        self.assertEqual(authored.authored_scenes(self.stage, "/World/env"),
                         [("/World/env/physicsScene", (0.0, 0.0, -1.0), 9.81, 60)])
        self.assertEqual(authored.authored_scenes(self.stage, "/World/none"), [])

    def test_the_servers_opinions_leave_the_users_layer_clean(self):
        self.stage.GetPrimAtPath("/World/env/physicsScene").SetActive(False)
        self.stage.GetPrimAtPath("/World/env/floor").CreateAttribute(
            authored.SURFACE_ATTR, Sdf.ValueTypeNames.Token).Set("concrete")
        self.assertEqual(authored.edited_layers(self.stage), [])

    def test_an_edit_in_the_users_layer_is_named(self):
        Sdf.Layer.Find(self.field).GetPrimAtPath("/field/floor").active = False
        got = authored.edited_layers(self.stage)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].endswith("field.usda"), got)


if __name__ == "__main__":
    unittest.main()
