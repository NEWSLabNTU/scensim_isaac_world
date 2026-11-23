"""086 C6a: the fixtures as road-layer features (cleats, fixed potholes), the course that holds
them, and fixture_check on made-up traces; no Isaac."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_layer  # noqa: E402
from scensim_isaac_world.course import fixture_check as fc  # noqa: E402
from scensim_isaac_world.course import fixture_drive as fd  # noqa: E402

COURSES = os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world", "courses")


def corridor(features):
    """A straight corridor whose base carries the section (as a built course's): features only."""
    c, = road_layer.parse({"corridors": [dict({"centreline": [[0, 0], [640, 0]],
                                               "section_in_base": True}, **features)]})
    return c


class Features(unittest.TestCase):
    def test_cleats_are_boxes_across_the_carriageway(self):
        c = corridor({"cleats": {"at": [[100, 0.10, 0.2]]}, "roughness": {"gd0": 1e-12}})
        u = np.array([99.99, 100.0, 100.1, 100.2, 100.21])
        for v in (-2.0, 0.0, 2.0):
            np.testing.assert_allclose(c.features(u, np.full(5, v)), [0, 0.1, 0.1, 0.1, 0],
                                       atol=1e-9)
        far = c.section.carriageway(100.1) + 0.5  # off the carriageway: none
        self.assertAlmostEqual(float(c.features(np.array([100.1]), np.array([far]))[0]), 0.0)

    def test_a_fixed_pothole_is_a_rectangle_and_no_random_ones(self):
        c = corridor({"potholes": {"at": [[400, 2.425, 0.08, 0.5, 0.8]]}})
        self.assertEqual(len(c._holes), 0)
        u = np.array([399.99, 400.0, 400.5, 400.51, 400.25, 400.25, 400.25])
        v = np.array([2.425, 2.425, 2.425, 2.425, 2.03, 2.82, 1.0])
        np.testing.assert_allclose(c.features(u, v), [0, -0.08, -0.08, 0, -0.08, -0.08, 0],
                                   atol=1e-9)

    def test_rows_of_the_wrong_shape_are_refused(self):
        for bad in ({"cleats": {"at": [[100, 0.1]]}},
                    {"potholes": {"at": [[400, 0.0, 0.08, 0.5]]}},
                    {"cleats": {"at": [["x", 0.1, 0.2]]}}):
            with self.assertRaises(road_layer.RoadDetailError, msg=str(bad)):
                corridor(bad)

    def test_the_course_holds_r0s_fixtures(self):
        with open(os.path.join(COURSES, "fixture_c6a.yaml")) as f:
            course = yaml.safe_load(f)
        c = corridor(course["fixture"]["features"])
        lane = fd.LANE
        left, right = lane + 0.8, lane - 0.8  # the left and right tracks
        for name, (s, length) in fd.FIXTURES.items():
            if name.startswith("cleat"):
                h = {"cleat_05": 0.05, "cleat_10": 0.10, "cleat_15": 0.15}[name]
                for v in (left, right):
                    got = c.features(np.array([s - 0.01, s + length / 2, s + length + 0.01]),
                                     np.full(3, v))
                    np.testing.assert_allclose(got - got[0], [0, h, 0], atol=1e-6)
        mid = np.array([400.25])
        flat = float(c.features(np.array([395.0]), np.array([left]))[0])
        self.assertAlmostEqual(float(c.features(mid, np.array([left]))[0]) - flat, -0.08, 6)
        flat_r = float(c.features(np.array([395.0]), np.array([right]))[0])
        self.assertAlmostEqual(float(c.features(mid, np.array([right]))[0]), flat_r, 6)
        w = c.features(np.arange(500.0, 530.0, 0.01), np.full(3000, left))
        self.assertAlmostEqual(float(w.max() - w.min()), 0.02, 3)


def trace(case, fs, flen, speed, load_fn, z_fn=lambda s: 0.0, air_fn=lambda s: 0.0, hz=60.0):
    """A made-up run of one case: the hub moving at `speed`, the load and height by its s."""
    dt = 1 / hz
    rows = []
    for k, s in enumerate(np.arange(fs - 20.0, fs + flen + 20.0, speed * dt)):
        loads = [load_fn(s)] + [4500.0] * 3
        rows.append({"case": case, "t": (k + 1) * dt, "fl_s": s, "fl_z": 0.5 + z_fn(s),
                     "pitch": 0.0, **{f"load{i}": v for i, v in enumerate(loads)},
                     **{f"air{i}": (air_fn(s) if i == 0 else 0.0) for i in range(4)}})
    return rows


class Check(unittest.TestCase):
    def run_check(self, cases):
        """cases: (info, rows) each; the check's result."""
        with tempfile.TemporaryDirectory() as d:
            keys = ["case", "t", "fl_s", "fl_z", "pitch"] + [f"load{i}" for i in range(4)] \
                + [f"air{i}" for i in range(4)]
            with open(f"{d}/fixture.csv", "w") as f:
                f.write(",".join(keys) + "\n")
                for _, rows in cases:
                    for r in rows:
                        f.write(",".join(str(r[k]) for k in keys) + "\n")
            json.dump({"query": "raycast", "physics_hz": 60.0, "aib": "x",
                       "cases": [i for i, _ in cases]}, open(f"{d}/cases.json", "w"))
            return fc.judge(*fc.read(d))

    def info(self, case, ok=True):
        name, speed, k = fd.parse_case(case)
        fs, flen = fd.FIXTURES[name]
        return {"case": case, "fixture": name, "speed": speed, "offset": k, "fixture_s": fs,
                "fixture_len": flen, "start_ok": ok}

    def test_a_cleat_like_the_reference_passes_and_one_behind_the_hub_fails(self):
        fs, flen = fd.FIXTURES["cleat_10"]

        def load(s, lead=0.3):
            return 4500.0 + 15700.0 * max(0.0, 1 - abs(s - (fs + 0.1 - lead / 2)) / lead)

        def z(s):
            return 0.078 * max(0.0, 1 - abs(s - fs - 2.0) / 2.0)

        good = self.run_check([(self.info("cleat_10@10"), trace("cleat_10@10", fs, flen, 10.0,
                                                                load, z))])
        r = good["cases"][0]
        self.assertEqual(r["verdict"], "passed", r)
        self.assertLess(r["first_rise_m"], 0.0)
        self.assertGreaterEqual(r["samples_over_2x"], 1)

        def behind(s):  # the load rises only once the hub is past the face
            return 4500.0 + (15700.0 if fs + 0.05 <= s <= fs + 0.4 else 0.0)

        late = self.run_check([(self.info("cleat_10@10"),
                                trace("cleat_10@10", fs, flen, 10.0, behind, z))])
        self.assertFalse(late["cases"][0]["items"]["ahead_of_hub"])
        self.assertEqual(late["verdict"], "failed")

    def test_a_refused_start_fails_the_run(self):
        fs, flen = fd.FIXTURES["cleat_05"]
        out = self.run_check([(self.info("cleat_05@10", ok=False),
                               trace("cleat_05@10", fs, flen, 10.0, lambda s: 4500.0))])
        self.assertEqual(out["cases"][0]["verdict"], "refused")
        self.assertEqual(out["verdict"], "failed")

    def test_the_pothole_unloads_then_peaks(self):
        fs, flen = fd.FIXTURES["pothole"]

        def load(s):
            if fs <= s <= fs + 0.4:
                return 500.0
            if fs + 0.4 < s <= fs + 0.9:
                return 10000.0
            return 4500.0
        out = self.run_check([(self.info("pothole@10"),
                               trace("pothole@10", fs, flen, 10.0, load))])
        r = out["cases"][0]
        self.assertTrue(r["items"]["unloads"] and r["items"]["far_edge"], r)

    def test_a_landing_is_timed_from_the_loads_return(self):
        # wheel 0 off the ground for 3 steps, back half way through the 4th, peak in the 5th
        c = {"t": np.arange(1, 9) / 60.0, "load0": np.array([4500, 0, 0, 0, 2000, 9000, 6000,
                                                             4500], float),
             "air0": np.array([0, 1, 1, 1, 0.5, 0, 0, 0], float)}
        for k in (1, 2, 3):
            c[f"load{k}"] = np.full(8, 4500.0)
            c[f"air{k}"] = np.zeros(8)
        (wheel, t, spread), = fc.landings(c, 1 / 60.0)
        self.assertEqual(wheel, 0)
        # the 5th step (4/60 to 5/60) loaded after half of it: back at 4.5/60; the peak in the
        # 6th, its middle 5.5/60: 1/60 s
        self.assertAlmostEqual(spread, 1 / 60.0, places=4)

    def test_washboard_range(self):
        fs, flen = fd.FIXTURES["washboard"]
        out = self.run_check([(self.info("washboard@10"), trace(
            "washboard@10", fs, flen, 10.0,
            lambda s: 4260.0 + 3100.0 * np.cos(2 * np.pi * (s - fs) / 0.7)
            if fs <= s <= fs + flen else 4500.0))])
        self.assertEqual(out["cases"][0]["verdict"], "passed", out["cases"][0])

    def test_the_rate_keeps_the_frame_and_the_vehicles_own_rate(self):
        self.assertEqual(fd.rate(60, 8), (3, 8))
        self.assertEqual(fd.rate(120, 8), (6, 4))
        self.assertEqual(fd.rate(240, 8), (12, 2))
        for bad in (100, 960, 0):
            with self.assertRaises(SystemExit):
                fd.rate(bad, 8)

    def test_cases(self):
        self.assertEqual(len(fd.CASES), 12)
        self.assertEqual(fd.parse_case("cleat_10@20#3"), ("cleat_10", 20.0, 3))
        for c in fd.CASES:
            self.assertIn(c.split("#")[0], fc.REF)


if __name__ == "__main__":
    unittest.main()
