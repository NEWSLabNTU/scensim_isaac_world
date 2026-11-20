"""The course's road record (scensim roadmap 086 C3): the section per station, the hazards from
the rail rule, and the rails read back from it."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_layer as rl  # noqa: E402
from scensim_isaac_world import road_record as rr  # noqa: E402
from scensim_isaac_world.course import props, record  # noqa: E402
from scensim_isaac_world.course.geometry import _normals, is_closed  # noqa: E402

COURSE = {"road": {"lanes": 2, "lane_width": 3.0, "shoulder": 0.5}}
HALF = 3.5


def old_guardrails(c, m, half, seg_len):
    """props.guardrails before 086 C3, frozen: the rule the record must reproduce."""
    tz = props._terrain_z(m)
    x, y, z, k = c["x"], c["y"], c["z"], c["curvature"]
    nrm = _normals(x, y, is_closed(c))
    step = int(round(seg_len / 0.5))
    out = []
    for side in (1, -1):
        off = side * (half + props.RAIL_OFFSET)
        probe = side * (half + props.DROP_AT)
        for i in range(0, len(x) - step, step):
            gz = float(tz(x[i] + probe * nrm[i, 0], y[i] + probe * nrm[i, 1]))
            outer = abs(k[i]) > 1.0 / props.RAIL_RADIUS and np.sign(k[i]) == -side
            if gz < z[i] - props.DROP or outer:
                px, py = x[i] + off * nrm[i, 0], y[i] + off * nrm[i, 1]
                qx, qy = (x[i + step] + off * nrm[i + step, 0],
                          y[i + step] + off * nrm[i + step, 1])
                yaw = math.atan2(qy - py, qx - px)
                out.append((px, py, float(z[i]), yaw, math.hypot(qx - px, qy - py)))
    return out


def straight(n=401, z=None):
    """A 200 m straight along x at y 0, 0.5 m stations; a gentle sag in z by default."""
    s = np.arange(n) * 0.5
    zz = 0.0005 * (s - 100.0) ** 2 if z is None else z
    return {"s": s, "x": s.copy(), "y": np.zeros(n), "z": zz, "curvature": np.zeros(n)}


def terrain(ground):
    """A 2 m grid over x -10..210, y -30..30, its height ground(x, y)."""
    xs, ys = np.arange(-10.0, 210.01, 2.0), np.arange(-30.0, 30.01, 2.0)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), ground(gx, gy).ravel()], axis=1)
    return {"terrain_points": pts, "terrain_shape": np.array([len(ys), len(xs)]),
            "half_width": np.array(HALF)}


class Classify(unittest.TestCase):
    def setUp(self):
        self.c = straight()

        def ground(x, y):  # left: a 3 m drop from x 100; right: a cut face rising 2 m
            road = 0.0005 * (x - 100.0) ** 2
            left = np.where((y > 0) & (x >= 100.0), road - 3.0, road)
            return np.where((y < 0) & (x >= 150.0), road + 2.0, left)
        self.m = terrain(ground)

    def test_the_rail_rule_becomes_hazard_classes(self):
        st = record.build(self.c, COURSE, self.m)
        left, right = st.text["hazard"]
        self.assertEqual(set(left[:180]), {"runoff"})
        self.assertEqual(set(left[220:]), {"barrier"})
        self.assertEqual(set(right[:280]), {"runoff"})
        self.assertEqual(set(right[320:]), {"ditch"})  # rising 2 m in 3 m: steeper than 1V:4H
        np.testing.assert_array_equal(np.isnan(st.side["barrier_offset"][0]), left != "barrier")
        self.assertEqual(st.side["barrier_offset"][0, 300], HALF + props.RAIL_OFFSET)

    def test_the_section_is_the_builders_and_kappa_v_the_profiles(self):
        st = record.build(self.c, COURSE, self.m)
        self.assertEqual(float(st.common["half_width"][0]), 3.0)
        self.assertEqual(float(st.side["shoulder"][0, 0]), 0.5)
        self.assertTrue(np.all(st.side["crown"] == 0) and np.all(st.common["bank"] == 0))
        # z = 0.0005 (s - 100)^2: z'' 0.001, a sag; at the bottom z' is 0
        self.assertAlmostEqual(float(st.common["kappa_v"][200]), 0.001, places=6)
        self.assertTrue(np.all(st.common["kappa_v"][2:-2] > 0))

    def test_the_rails_from_the_record_are_the_old_rules(self):
        st = record.build(self.c, COURSE, self.m)
        for seg in (1.0, 2.5, 4.0):
            new = props.guardrails(self.c, st, seg)
            self.assertTrue(new)
            self.assertEqual(new, old_guardrails(self.c, self.m, HALF, seg))

    def test_the_outside_of_a_tight_bend_gets_a_rail(self):
        r, n = 25.0, 200
        a = np.arange(n) * 0.5 / r
        c = {"s": np.arange(n) * 0.5, "x": r * np.sin(a), "y": r * (1 - np.cos(a)),
             "z": np.zeros(n), "curvature": np.full(n, 1.0 / r)}  # a left bend, R 25
        m = terrain(lambda x, y: np.zeros_like(x))
        st = record.build(c, COURSE, m)
        self.assertEqual(set(st.text["hazard"][1]), {"barrier"})  # right: the outside
        self.assertEqual(set(st.text["hazard"][0]), {"runoff"})
        self.assertEqual(props.guardrails(c, st, 2.0), old_guardrails(c, m, HALF, 2.0))


class Written(unittest.TestCase):
    def test_the_record_is_a_road_record_with_its_centreline(self):
        c, m = straight(), terrain(lambda x, y: 0.0005 * (x - 100.0) ** 2)
        work = tempfile.mkdtemp()
        out = record.write(work, record.build(c, COURSE, m))
        self.assertEqual(out["stations"], 401)
        corridor, = rl.load(os.path.join(work, record.YAML), classes=rl.class_names())
        self.assertAlmostEqual(corridor.length, 200.0)
        np.testing.assert_allclose(corridor.centreline[:, 0], c["x"])
        st = record.load(work)
        np.testing.assert_allclose(st.xyz[:, 2], c["z"])

    def test_a_closed_course_repeats_its_first_station(self):
        r, n = 50.0, 628
        a = np.arange(n) * 2 * math.pi / n
        c = {"s": np.arange(n) * (2 * math.pi * r / n), "x": r * np.cos(a), "y": r * np.sin(a),
             "z": np.zeros(n), "curvature": np.full(n, 1 / r), "closed": np.array(True)}
        xs = np.arange(-70.0, 70.01, 2.0)
        gx, gy = np.meshgrid(xs, xs)
        m = {"terrain_points": np.stack([gx.ravel(), gy.ravel(), 0 * gx.ravel()], axis=1),
             "terrain_shape": np.array([len(xs), len(xs)]), "half_width": np.array(HALF)}
        st = record.build(c, COURSE, m)
        self.assertEqual(len(st), n + 1)
        np.testing.assert_allclose(st.xyz[-1], st.xyz[0])
        chord = math.hypot(c["x"][0] - c["x"][-1], c["y"][0] - c["y"][-1])
        self.assertAlmostEqual(float(st.u[-1]), float(c["s"][-1]) + chord, places=9)

    def test_no_record_names_the_step_that_writes_it(self):
        with self.assertRaises(FileNotFoundError) as e:
            record.load(tempfile.mkdtemp())
        self.assertIn("record step", str(e.exception))

    def test_a_record_whose_u_is_not_its_centreline_is_refused(self):
        d = tempfile.mkdtemp()
        np.savez(os.path.join(d, "s.npz"), u=np.arange(5.0) * 2, x=np.arange(5.0),
                 y=np.zeros(5))
        _, problems = rr.load_stations(os.path.join(d, "s.npz"))
        self.assertIn("u is not the distance along x, y", problems[0])
        np.savez(os.path.join(d, "z.npz"), u=np.arange(5.0), z=np.zeros(5))
        _, problems = rr.load_stations(os.path.join(d, "z.npz"))
        self.assertIn("z without x and y", problems[0])


if __name__ == "__main__":
    unittest.main()
