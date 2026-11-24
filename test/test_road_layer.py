"""The road detail in the world server (roadmap 066 R3), with a stand-in for aib's vehicle."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deps/autoware_isaac_bridge/src/aib"))

from scensim_isaac_world import road_layer  # noqa: E402

DT = 1.0 / 60


def params(travel=0.2):
    def suspension():
        m = 1760 * 0.47 / 2, 1760 * 0.53 / 2
        w = 2 * math.pi * 1.6
        return tuple((s, s * w * w, 2 * 0.45 * math.sqrt(s * w * w * s)) for s in m), 9.81 / w / w
    return SimpleNamespace(suspension_travel=0.2, wheel_radius=0.37, wheel_width=0.235,
                           wheel_base=3.005, wheel_tread=1.667, suspension=suspension,
                           travels=lambda: (travel, 0.2))


class Parse(unittest.TestCase):
    def test_a_corridor(self):
        c, = road_layer.parse({"corridors": [{
            "centreline": [[0, 0, 10], [100, 0, 12]], "seed": 4, "roughness": {"cls": "D"},
            "section": {"crown": 0.0, "half_width": 1.5}, "ruts": {},
            "washboard": [{"u0": 10, "u1": 20}], "water_bars": {"at": [50]}}]})
        self.assertEqual(c.roughness.cls, "D")
        self.assertEqual(c.section.crown, 0.0)
        self.assertAlmostEqual(float(c.base_z(50.0)), 11.0)
        self.assertEqual(len(c.washboard), 1)

    def test_problems_are_listed(self):
        with self.assertRaises(road_layer.RoadDetailError) as e:
            road_layer.parse({"corridors": [{"centreline": [[0, 0]]},
                                            {"centreline": [[0, 0], [1, 0]], "ruts": {"x": 1},
                                             "roughness": {"cls": "Z"}, "bumps": 1}],
                              "other": 1})
        msg = str(e.exception)
        for part in ("unknown key(s) ['other']", "centreline must be", "ruts: unknown ['x']",
                     "class Z", "['bumps']"):
            self.assertIn(part, msg)


class Schema(unittest.TestCase):
    """road_detail.yaml's keys, documented and refused by name (roadmap 076 A9)."""

    def test_every_key_is_listed(self):
        keys = road_layer.keys()
        for k in ("corridors", "ground", "corridors[].centreline", "corridors[].taper",
                  "corridors[].washboard[].*", "corridors[].rocks.*"):
            self.assertIn(k, keys)
        self.assertEqual(keys["corridors[].washboard[].*"]["u0"], road_layer.REQUIRED)
        self.assertEqual(keys["corridors[].section.*"]["half_width"], 2.5)

    def test_washboard_and_shapes_are_refused_by_name(self):
        line = [[0, 0], [100, 0]]
        with self.assertRaises(road_layer.RoadDetailError) as e:
            road_layer.parse({"corridors": [
                {"centreline": line, "washboard": [{"u0": 1, "u1": 2, "wave": 3}, {"u0": 1}]},
                {"centreline": line, "ruts": [1, 2]},
                "nope"]})
        msg = str(e.exception)
        for part in ("washboard[0]: unknown ['wave']", "washboard[1]: needs u1",
                     "corridors[1].ruts: a mapping", "corridors[2]: not a mapping"):
            self.assertIn(part, msg)
        with self.assertRaises(road_layer.RoadDetailError):
            road_layer.parse([1, 2])

    def test_the_examples_parse(self):
        ex = ROOT / "scensim_isaac_world" / "examples" / "road"
        files = sorted(ex.glob("*/road_detail.yaml"))
        self.assertGreaterEqual(len(files), 2)
        for f in files:
            self.assertTrue(road_layer.load(f), f)
        town, = road_layer.load(ex / "town01" / "road_detail.yaml")
        self.assertIsNotNone(town.rocks)
        self.assertEqual(road_layer.find(str(ex / "town01")),
                         str(ex / "town01" / "road_detail.yaml"))

    def test_check_and_profile(self):
        import contextlib
        import io

        ex = ROOT / "scensim_isaac_world" / "examples" / "road" / "town01"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(road_layer.main(["check", str(ex)]), 0)
        self.assertIn("1 corridor(s)", out.getvalue())
        self.assertIn("washboard 1 zone(s)", out.getvalue())
        c, = road_layer.load(ex / "road_detail.yaml")
        u, v, h = road_layer.profile(c, length=5.0)
        self.assertEqual(h.shape, (len(v), len(u)))
        self.assertTrue(np.isfinite(h).all())
        self.assertLess(h.min(), -0.2)  # the ditches


class Travel(unittest.TestCase):
    """PhysX's travel on a road detail, with aib's own (scensim issue 077)."""

    def test_physx_travel_is_longer_and_the_real_one_stays(self):
        from aib.params import VehicleParams

        p = road_layer.vehicle_params(VehicleParams(suspension_travel=0.2))
        self.assertEqual(p.travels(), (0.2 + road_layer.PX_EXTRA, 0.2))
        # a car whose PhysX travel is already longer (the sweep's 0.50 m) keeps it
        p = road_layer.vehicle_params(VehicleParams(suspension_travel=0.2, physx_travel=0.5))
        self.assertEqual(p.travels(), (0.5, 0.2))

    def test_the_layer_turns_aibs_bump_stop_off(self):
        v = SimpleNamespace(p=params(0.5), bump_enabled=True, path="/ego")
        car = road_layer._Car(v)
        self.assertFalse(v.bump_enabled)
        self.assertAlmostEqual(car.frame_z, 0.37 + 0.5 - v.p.suspension()[1])


class Layer(unittest.TestCase):
    def car(self, z0=0.0, grade=0.0, x=10.0, radius=None):
        """A vehicle standing at rest on a base of `grade`, PhysX's frame raised (R2). With
        `radius` (m; > 0 a crest, < 0 a sag) the base is a vertical curve, `grade` its slope at
        the car's middle, the hits on the curve (issue 158)."""
        p = params(0.2 + road_layer.PX_EXTRA)
        (_, _), sag = p.suspension()
        frame_z = p.wheel_radius + p.travels()[0] - sag
        pitch = -math.atan(grade)
        q = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
        xm = x + p.wheel_base / 2 * math.cos(pitch)

        def z_at(hx):
            dz = 0.0 if radius is None else -(hx - xm) ** 2 / (2 * radius)
            return z0 + grade * hx + dz
        hits = []
        for f, side in road_layer.ORDER:
            hx = x + (p.wheel_base if f else 0.0) * math.cos(pitch)
            hits.append(SimpleNamespace(on_ground=True, position=(hx, side * 0.83, z_at(hx))))
        pos = (x, 0.0, z0 + grade * x + frame_z / math.cos(pitch) - frame_z)
        return SimpleNamespace(path="/ego", p=p, last=(pos, q, (0.0, 0.0, 0.0)), contacts=hits,
                               wheel_paths=[0] * 4, tire_angle=0.0, bump_enabled=True)

    def test_no_detail_no_force_on_a_grade(self):
        """A flat corridor with no roughness on a 10 % base: the plane base, no push."""
        c = road_layer.parse({"corridors": [{"centreline": [[0, 0], [200, 0]],
                                             "roughness": {"gd0": 1e-14},
                                             "section": {"crown": 0.0}}]})
        forces = []
        layer = road_layer.RoadLayer(c, apply_force=lambda *a: forces.append(a), workers=0)
        v = self.car(z0=5.0, grade=0.1)
        for _ in range(30):
            forces.clear()
            layer.before_substep([v], DT)
        fz = [f[1][2] for f in forces]
        self.assertEqual(len(fz), 4)
        # what is left is the stand-in pose's own sag error through the tire's series spring
        # (k / k_tire of it): under 3 % of a corner's weight, and the same left and right
        self.assertLess(max(abs(f) for f in fz), 150.0, fz)
        self.assertAlmostEqual(fz[0], fz[1], delta=1.0)
        for i in range(4):
            self.assertAlmostEqual(layer.scale("/ego", i), 1.0, delta=0.05)
        layer.close()

    def settled(self, features, **car):
        """The forces after 30 steps of a stand-in car on a corridor with `features`."""
        c = road_layer.parse({"corridors": [dict({"centreline": [[0, 0], [200, 0]],
                                                  "roughness": {"gd0": 1e-14},
                                                  "section": {"crown": 0.0}}, **features)]})
        forces = []
        layer = road_layer.RoadLayer(c, apply_force=lambda *a: forces.append(a), workers=0)
        v = self.car(**car)
        for _ in range(30):
            forces.clear()
            layer.before_substep([v], DT)
        car_ = layer.cars["/ego"]
        load, xy = car_.last_out["load"].copy(), car_.last_xy.copy()
        layer.close()
        return [f[1] for f in forces], load, xy

    def test_a_grade_alone_pushes_nothing_along_the_road(self):
        """Issue 158: on a bare corridor the push along the road is 0 at any grade, on a crest
        and in a sag; PhysX carries the base road's grade, the layer only the detail's."""
        cases = [dict(grade=g) for g in (0.0, 0.06, 0.1, -0.06)]
        cases += [dict(grade=g, radius=r) for g in (0.0, 0.06) for r in (100.0, -100.0)]
        for case in cases:
            forces, _, _ = self.settled({}, z0=5.0, **case)
            self.assertEqual(len(forces), 4)
            for f in forces:
                self.assertLess(math.hypot(f[0], f[1]), 1.0, (case, forces))

    def test_a_bars_push_along_the_road_is_its_own_on_a_grade(self):
        """Issue 158: the slope the push acts through (the cams' less the base's) over a water
        bar's flank on a 6 % grade is the level road's: its largest within 1 %, at the place
        shifted uphill by r sin(grade), where a disc on a grade meets the road, within 1 cm."""
        from scensim_isaac_world import road_contact as rc

        c = road_layer.parse({"corridors": [{"centreline": [[0, 0], [200, 0]],
                                             "roughness": {"gd0": 1e-14},
                                             "section": {"crown": 0.0},
                                             "water_bars": {"at": [10.0], "angle": 0.0,
                                                            "height": 0.05}}]})
        layer = road_layer.RoadLayer(c, workers=0)
        r = 0.37
        tc = rc.TireContact([rc.Corner(413.6, 41.8e3, 3.74e3, 0.2, r, 0.235)])
        xs = np.arange(9.70, 10.30, 0.002)
        hd = np.array([[1.0, 0.0]])
        peaks = []
        for g in (0.0, 0.06):
            def height(x, y, g=g):
                return g * np.asarray(x) + layer.height(x, y)

            def base(x, y, g=g):
                return g * np.asarray(x) + 0.0 * np.asarray(y)
            sl = []
            for x in xs:
                xy = np.array([[x, -0.83]])
                s, _, slope = tc.envelope(height, xy, hd, 0.0)
                sl.append(float(slope[0, 0] - tc.base_slope(base, xy, hd, s)[0, 0]))
            k = int(np.argmax(sl))
            peaks.append((sl[k], xs[k]))
        layer.close()
        (p0, x0), (p6, x6) = peaks
        self.assertGreater(p0, 0.2)  # the bar's rising flank
        self.assertAlmostEqual(p6 / p0, 1.0, delta=0.01, msg=peaks)
        self.assertAlmostEqual(x0 - x6, r * math.sin(math.atan(0.06)), delta=0.01, msg=peaks)

    def test_a_bar_under_a_wheel_pushes_it_up(self):
        c = road_layer.parse({"corridors": [{"centreline": [[0, 0], [200, 0]],
                                             "roughness": {"gd0": 1e-14},
                                             "section": {"crown": 0.0},
                                             "water_bars": {"at": [10.0], "angle": 0.0,
                                                            "height": 0.05}}]})
        forces = []
        layer = road_layer.RoadLayer(c, apply_force=lambda *a: forces.append(a), workers=0)
        v = self.car(x=10.0 - 0.0)  # the rear wheels on the bar
        for _ in range(30):
            forces.clear()
            layer.before_substep([v], DT)
        fz = {i: f[1][2] for i, f in enumerate(forces)}
        self.assertGreater(min(fz[2], fz[3]), 1000.0)  # rear: tens of mm of extra compression
        self.assertLess(max(abs(fz[0]), abs(fz[1])), 100.0)
        layer.close()

    def test_a_rock_under_a_wheel_lifts_it_and_is_pushed(self):
        """The rocks with the layer as the world runs it: a stand-in stream (no USD) and one
        loose rock under the rear left wheel."""
        from scensim_isaac_world import rocks as rk

        c = road_layer.parse({"corridors": [{"centreline": [[0, 0], [200, 0]],
                                             "roughness": {"gd0": 1e-14},
                                             "section": {"crown": 0.0},
                                             "rocks": {"carriageway": 0.0, "shoulder": 0.0,
                                                       "ditch": 0.0}}]})
        self.assertIsInstance(c[0].rocks, rk.Zones)
        forces = []
        layer = road_layer.RoadLayer(c, apply_force=lambda *a: forces.append(a), workers=0)
        self.assertIsNotNone(layer.rocks)
        rock = rk.make_rock(np.random.default_rng(1), 0.12, rough=0.0)
        layer.rocks.add([rock], [(10.0, 0.83, -rock.verts[:, 2].min())], [(1, 0, 0, 0)])
        pushed = []

        class Stream:  # the part of rock_layer.RockStream the car step uses
            def read_near(self, xy, road, dt):
                layer.rocks.seen[:] = True

            def push(self, forces_, speed, radius, road, dt, apply_force):
                pushed.append(forces_)
                return 1.0

        layer.rock_stream = Stream()
        v = self.car(x=10.0)  # the rear left wheel on the rock
        for _ in range(30):
            forces.clear()
            layer.before_substep([v], DT)
        fz = [f[1][2] for f in forces[:4]]
        self.assertGreater(fz[2], 500.0)  # rear left lifted by the 6 cm rock
        self.assertLess(abs(fz[0]), 150.0)
        self.assertTrue(pushed and 0 in pushed[-1])  # and the rock takes its reaction
        f = sum(ff for ff, _ in pushed[-1][0])
        self.assertLess(f[2], -1000.0)  # pressed down by the tire
        layer.close()


class MeshFaces(unittest.TestCase):
    """The render mesh's faces: only the corridors', road and verge apart (no Isaac)."""

    def test_faces_cover_the_corridor_only(self):
        from scensim_isaac_world import road_detail as rd

        c = rd.Corridor(np.array([[0.0, 0.0], [400.0, 0.0]]), seed=1)
        d = rd.Detail([c])
        ms = road_layer.MeshStream(None, SimpleNamespace(detail=d, ox=0.0, oy=0.0))
        quads, road = ms.tile_faces(d.zones_tile(2, -1))
        n, s = d.n, d.spacing
        ys = -16.0 + (np.arange(n + 1) * s)[quads // (n + 1)].mean(axis=1)
        self.assertLess(ys.min(), -c.reach + 0.2)  # out to the reach
        self.assertGreater(ys.min(), -c.reach - 0.2)  # and no further
        edge = c.section.half_width + c.section.shoulder
        self.assertTrue((np.abs(ys[road]) <= edge + s).all())
        self.assertTrue((np.abs(ys[~road]) >= edge - s).all())
        self.assertGreater(road.sum(), 0)
        self.assertGreater((~road).sum(), 0)


class PlaneFit(unittest.TestCase):
    def test_the_closed_form_is_lstsq(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            n = int(rng.integers(3, 5))
            x = 300.0 + rng.uniform(-2, 2, n)  # world coordinates, far from the origin
            y = -150.0 + rng.uniform(-1, 1, n)
            z = 0.02 * x - 0.03 * y + rng.normal(0, 0.01, n)
            a = np.stack([x, y, np.ones(n)], axis=1)
            (ga, gb, _), *_ = np.linalg.lstsq(a, z, rcond=None)
            fa, fb = road_layer._fit_slope(x, y, z)
            self.assertAlmostEqual(fa, ga, delta=1e-7)
            self.assertAlmostEqual(fb, gb, delta=1e-7)

    def test_collinear_points_fall_back_to_lstsq(self):
        x = np.array([0.0, 1.0, 2.0])
        y = 2.0 * x
        z = np.array([0.0, 0.1, 0.2])
        a = np.stack([x, y, np.ones(3)], axis=1)
        (ga, gb, _), *_ = np.linalg.lstsq(a, z, rcond=None)
        fa, fb = road_layer._fit_slope(x, y, z)
        self.assertAlmostEqual(fa, ga, places=9)
        self.assertAlmostEqual(fb, gb, places=9)
