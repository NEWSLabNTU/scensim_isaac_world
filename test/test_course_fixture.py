"""086 C5b: the fixture course, and the road mesh carrying the record's section; no Isaac."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_record as rr  # noqa: E402
from scensim_isaac_world.course import check  # noqa: E402
from scensim_isaac_world.course import fixture  # noqa: E402
from scensim_isaac_world.course import geometry  # noqa: E402

COURSE = os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world", "courses",
                      "fixture_c5.yaml")


def course():
    with open(COURSE) as f:
        return yaml.safe_load(f)


def short(c, s0, s1):
    """The centreline between s0 and s1 (a part of the course, for a quick mesh; its s the
    course's, its record's u from 0)."""
    k = (c["s"] >= s0) & (c["s"] <= s1)
    return {n: np.asarray(v)[k] for n, v in c.items() if n != "hairpins"}


class FixtureCourse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = course()
        cls.c = fixture.centreline(cls.course)
        cls.st = fixture.stations(cls.course, cls.c)

    def test_the_plan_is_the_spec(self):
        c = self.c
        self.assertAlmostEqual(float(c["s"][-1]), 510.0)
        # two 60 m clothoids to and from 1/60 (half a radian each) and a 90 m arc at 1/60
        turn = math.degrees(float(c["heading"][-1] - c["heading"][0]))
        self.assertAlmostEqual(turn, math.degrees(0.5 + 1.5 + 0.5), delta=0.05)
        i = np.searchsorted(c["s"], 255.0)
        self.assertAlmostEqual(1.0 / float(c["curvature"][i]), 60.0, delta=0.01)
        self.assertAlmostEqual(float(c["z"][-1]), -0.03 * 510.0, places=6)
        step = np.hypot(np.diff(c["x"]), np.diff(c["y"]))
        self.assertLess(float(np.abs(step - 0.5).max()), 1e-3)

    def test_the_record_is_the_table_linear_between_rows(self):
        st = self.st
        at = lambda a, s: float(np.interp(s, st.u, a))  # noqa: E731
        self.assertAlmostEqual(math.tan(at(st.common["bank"], 255.0)), 0.06, places=9)
        self.assertAlmostEqual(math.tan(at(st.common["bank"], 180.0)), 0.03, places=9)
        self.assertAlmostEqual(at(st.side["crown"][0], 75.0), 0.025)
        self.assertAlmostEqual(at(st.side["crown"][1], 330.0), 0.015)
        self.assertAlmostEqual(at(st.common["crown_offset"], 400.0), 0.5)
        self.assertAlmostEqual(at(st.side["lip_h"][1], 450.0), 0.10)
        self.assertAlmostEqual(at(st.side["lip_h"][1], 419.0), 0.05)
        self.assertEqual(at(st.side["lip_h"][0], 450.0), 0.0)
        self.assertEqual(float(st.common["half_width"][0]), 3.25)
        # what the reader accepts, written and read back
        d = tempfile.mkdtemp()
        rr.save_stations(os.path.join(d, "r.npz"), st)
        back, problems = rr.load_stations(os.path.join(d, "r.npz"))
        self.assertEqual(problems, [])
        self.assertEqual(rr.detail_section(back, rd.Section)[0], None)  # per station

    def _mesh(self, s0, s1):
        c = short(self.c, s0, s1)
        st = fixture.stations(self.course, c)
        sec = rd.StationSection(st)
        rd_ = geometry.road(c, self.course, sec)
        m = {"half_width": rd_["half_width"], "road_points": rd_["road"][0],
             "road_faces": rd_["road"][1]}
        return c, sec, m

    def test_the_mesh_carries_the_record_on_the_banked_bend(self):
        c, sec, m = self._mesh(140.0, 240.0)  # the runoff into the arc
        out = check.check_mesh(c, m, sec)
        self.assertTrue(out["pass"], out)

    def test_the_mesh_carries_the_offset_crown_and_the_lip(self):
        c, sec, m = self._mesh(400.0, 500.0)
        out = check.check_mesh(c, m, sec)
        self.assertTrue(out["pass"], out)
        self.assertGreater(out["skipped_on_a_face"], 0)
        # the shoulder past the lip lies its height below the carriageway's edge
        u = np.array([50.0])
        w = 3.25
        top = check.mesh_height(c, m["road_points"], m["road_faces"], u, np.array([-w + 0.05]), 4.0)
        foot = check.mesh_height(c, m["road_points"], m["road_faces"], u, np.array([-w - 0.05]), 4.0)
        self.assertAlmostEqual(float(top - foot), 0.10 + 0.03 * 0.10, delta=1e-3)

    def test_a_course_without_a_record_gets_todays_mesh(self):
        # Haruna and Tsukuba (level records, C3) are built without one: 17 rows, 0.5 m apart
        c = short(self.c, 0.0, 100.0)
        p, f = geometry.road(c, self.course)["road"]
        n = len(c["s"])
        nrm = geometry._normals(c["x"], c["y"])
        want = np.concatenate([np.stack([c["x"] + o * nrm[:, 0], c["y"] + o * nrm[:, 1], c["z"]],
                                        axis=1) for o in np.linspace(4.0, -4.0, 17)])
        np.testing.assert_array_equal(p, want)
        self.assertEqual(f.shape, (16 * (n - 1), 4))

    def test_a_level_record_lies_on_the_level_mesh(self):
        c = short(self.c, 0.0, 100.0)
        st, problems = rr.from_constants({"half_width": 3.25, "shoulder": 0.75, "crown": 0.0})
        self.assertEqual(problems, [])
        n = len(c["s"])
        st = rr.Stations(c["s"].copy(), {k: np.repeat(v, n) for k, v in st.common.items()},
                         {k: np.repeat(v, n, axis=1) for k, v in st.side.items()},
                         {k: np.repeat(v, n, axis=1) for k, v in st.text.items()}, None)
        p = geometry.road(c, self.course, rd.StationSection(st))["road"][0]
        np.testing.assert_allclose(p[:, 2], np.tile(c["z"], len(p) // n), atol=1e-12)

    def test_the_record_says_the_mesh_carries_it_and_the_layer_adds_none_of_it(self):
        from scensim_isaac_world import road_layer as rl
        from scensim_isaac_world.course import record

        d = tempfile.mkdtemp()
        c = short(self.c, 140.0, 240.0)
        record.write(d, fixture.stations(self.course, c), in_mesh=True)
        on, = rl.load(os.path.join(d, record.YAML))
        self.assertTrue(on.section_in_base)
        off, = rl.parse({"corridors": [{"stations": record.NPZ}]}, base=d)
        self.assertFalse(off.section_in_base)
        u, v = np.array([60.0, 60.0, 10.0]), np.array([-3.0, 2.0, 0.5])
        np.testing.assert_array_equal(on.features(u, v), 0.0)
        self.assertGreater(float(np.abs(off.features(u, v)).max()), 0.05)
        with self.assertRaises(rl.RoadDetailError):
            rl.parse({"corridors": [{"stations": record.NPZ, "section_in_base": 1}]}, base=d)


class ArcCheck(unittest.TestCase):
    """course.arc_check on a made-up trace that rides the road exactly, and one that does not."""

    @classmethod
    def setUpClass(cls):
        from scensim_isaac_world.course import record

        cls.map = tempfile.mkdtemp()
        work = os.path.join(cls.map, "course")
        os.makedirs(work)
        co = course()
        c = fixture.centreline(co)
        np.savez(os.path.join(work, "centreline.npz"), **{k: v for k, v in c.items() if k != "hairpins"})
        record.write(work, fixture.stations(co, c), in_mesh=True)
        cls.c = c

    def trace(self, flat=False, v=11.0):
        c = self.c
        s = np.arange(40.0, 330.0, v * 0.05)
        h = np.interp(s, c["s"], c["heading"])
        k = np.interp(s, c["s"], c["curvature"])
        nrm = np.stack([-np.sin(h), np.cos(h)], 1)
        x = np.interp(s, c["s"], c["x"]) + 1.625 * nrm[:, 0]
        y = np.interp(s, c["s"], c["y"]) + 1.625 * nrm[:, 1]
        st = fixture.stations(course(), c)
        bank = np.interp(s, st.u, st.common["bank"])
        crown = np.interp(s, st.u, st.side["crown"][0])
        roll = np.zeros(len(s)) if flat else -bank - np.arctan(crown)
        path = os.path.join(tempfile.mkdtemp(), "ego.csv")
        t = np.arange(len(s)) * 0.05
        rows = np.stack([t, np.arange(len(s)), x, y, np.zeros(len(s)), roll, np.zeros(len(s)), h,
                         v * np.cos(h), v * np.sin(h), np.zeros(len(s)), np.zeros(len(s)),
                         np.zeros(len(s)), v * k / (1.0 - 1.625 * k), np.ones(len(s))], 1)
        np.savetxt(path, rows, delimiter=",", comments="",
                   header="t,frame,x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,measured")
        return path

    def test_a_car_riding_the_record_passes(self):
        from scensim_isaac_world.course import arc_check

        out = arc_check.check(self.trace(), self.map)
        self.assertTrue(out["pass"], out)
        self.assertAlmostEqual(out["road_roll_arc_deg"], -math.degrees(math.atan(0.06)), places=2)
        self.assertGreater(out["fy_if_level"] - out["fy_want"], 0.55)

    def test_a_car_on_a_level_road_fails(self):
        from scensim_isaac_world.course import arc_check

        out = arc_check.check(self.trace(flat=True), self.map)
        self.assertFalse(out["checks"]["arc_roll"])
        self.assertFalse(out["checks"]["lateral"])

if __name__ == "__main__":
    unittest.main()
