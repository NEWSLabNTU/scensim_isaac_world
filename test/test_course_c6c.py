"""086 C6c: the ISO 8608 profile as a road-layer feature, the course that holds it with C6a's
washboard, and c6c_check's spectrum and amplitude on made-up traces; no Isaac."""
import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_layer  # noqa: E402
from scensim_isaac_world.course import c6c_check as cc  # noqa: E402
from scensim_isaac_world.course import fixture_drive as fd  # noqa: E402

COURSES = os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world", "courses")


def corridor(features):
    c, = road_layer.parse({"corridors": [dict({"centreline": [[0, 0], [800, 0]],
                                               "section_in_base": True}, **features)]})
    return c


class Profile(unittest.TestCase):
    def test_the_profile_is_class_c_seeded_and_the_same_each_time(self):
        a = rd.iso8608_profile(1, 400.0)
        self.assertEqual(len(a), 40000)
        self.assertAlmostEqual(float(a.std()), 0.0143265, places=6)
        np.testing.assert_array_equal(a, rd.iso8608_profile(1, 400.0))
        self.assertFalse(np.array_equal(a, rd.iso8608_profile(2, 400.0)))
        # band-limited: nothing over 2.83 cycle/m
        H = np.abs(np.fft.rfft(a))
        f = np.fft.rfftfreq(len(a), 0.01)
        self.assertLess(float(H[f > 2.9].max()), 1e-9 * float(H.max()))

    def test_across_the_carriageway_ramped_and_nothing_outside(self):
        c = corridor({"profiles": [{"u0": 300.0, "length": 400.0}], "roughness": {"gd0": 1e-12}})
        p = c.profiles[0]
        u = np.linspace(310.0, 690.0, 50)
        left, right = c.features(u, np.full(50, -1.6)), c.features(u, np.full(50, 1.6))
        np.testing.assert_allclose(left, right, atol=1e-12)
        np.testing.assert_allclose(c.features(u, np.zeros(50)), p.h(u), atol=1e-12)
        self.assertEqual(float(p.h(np.array([299.0]))[0]), 0.0)
        self.assertEqual(float(p.h(np.array([701.0]))[0]), 0.0)
        self.assertEqual(float(p.h(np.array([300.0]))[0]), 0.0)  # ramps in
        far = c.section.carriageway(500.0) + 0.5
        self.assertEqual(float(c.features(np.array([500.0]), np.array([far]))[0]), 0.0)

    def test_bad_profiles_are_refused(self):
        for bad in ({"profiles": {"u0": 1.0}}, {"profiles": [{"length": 10.0}]},
                    {"profiles": [{"u0": 1.0, "colour": 2}]}, {"profiles": [{"u0": 1.0,
                                                                            "length": -1.0}]},
                    {"profiles": [{"u0": "x"}]}):
            with self.assertRaises(road_layer.RoadDetailError, msg=str(bad)):
                corridor(bad)

    def test_the_course_holds_the_washboard_and_the_profile_where_the_drive_looks(self):
        with open(os.path.join(COURSES, "fixture_c6c.yaml")) as f:
            course = yaml.safe_load(f)
        c = corridor(course["fixture"]["features"])
        s, n = fd.FIXTURES_C6C["washboard"]
        self.assertEqual((c.washboard[0].u0, c.washboard[0].u1), (s, s + n))
        s, n = fd.FIXTURES_C6C["class_c"]
        self.assertEqual((c.profiles[0].u0, c.profiles[0].length, c.profiles[0].seed),
                         (s, n, 1))
        length = course["fixture"]["plan"][0]["straight"]
        for case in fd.CASES_C6C:
            name, speed, _ = fd.parse_case(case)
            fs, flen = fd.FIXTURES_C6C[name]
            self.assertGreater(fs - fd.runup(speed), 0.0, case)
            self.assertLess(fs + flen + fd.AFTER, length, case)
            self.assertIn(speed, cc.REF_WB if name == "washboard" else cc.REF_CC, case)


# A corridor with every feature but a profile: its heights' sha256 at siw 7813843 (before
# 086 C6c's profiles), computed with numpy 1.21 (the gate host's, Jammy's). Another numpy may
# round the roughness's FFT differently, so the test runs on that numpy only.
GOLDEN = "f76f2041b871e556b5428d5cd322650faf44ddf0ab1efafdb54a193de0754b8b"
GOLDEN_NUMPY = "1.21"


class Unchanged(unittest.TestCase):
    @unittest.skipUnless(np.__version__.startswith(GOLDEN_NUMPY + "."),
                         f"the golden hash is numpy {GOLDEN_NUMPY}'s")
    def test_a_corridor_without_profiles_is_byte_identical_to_before(self):
        import hashlib
        c, = road_layer.parse({"corridors": [{
            "centreline": [[0, 0], [120, 0], [200, 40]], "seed": 7, "ruts": {}, "potholes": {},
            "water_bars": {}, "cleats": {"at": [[30.0, 0.05, 0.2]]},
            "washboard": [{"u0": 60.0, "u1": 90.0}]}]})
        self.assertEqual(c.profiles, ())
        d = rd.Detail([c], workers=0)
        x, y = np.meshgrid(np.arange(-10, 210, 0.37), np.arange(-15, 60, 0.41))
        h = np.asarray(d.height(x, y), dtype=np.float64)
        self.assertEqual(hashlib.sha256(h.tobytes()).hexdigest(), GOLDEN)


class Check(unittest.TestCase):
    def test_welch_finds_a_tone_and_its_height_over_the_low_band(self):
        rate = 480.0
        t = np.arange(int(30 * rate)) / rate
        rng = np.random.default_rng(0)
        y = 4000 + 1000 * np.sin(2 * np.pi * 11.0 * t) + 100 * rng.standard_normal(len(t))
        hz, height, f, _ = cc.welch_peak(y, rate)
        self.assertLess(abs(hz - 11.0), f[1] - f[0])
        self.assertGreater(height, 100.0)
        with self.assertRaises(ValueError):
            cc.welch_peak(y[:100], rate)

    def run_dir(self, d, cases, loads):
        json.dump({"cases": cases}, open(os.path.join(d, "cases.json"), "w"))
        with open(os.path.join(d, "inner.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "step", "fl_s"] + [f"load_in{i}" for i in range(8)])
            for case, rows in loads.items():
                for n, (s, ls) in enumerate(rows):
                    w.writerow([case, n, s, *ls])

    def test_washboard_amplitude_in_its_window_and_a_refused_start(self):
        with tempfile.TemporaryDirectory() as d:
            base = {"fixture": "washboard", "fixture_s": 150.0, "fixture_len": 30.0}
            cases = [dict(base, case="washboard@10", speed=10.0, start_ok=True),
                     dict(base, case="washboard@20", speed=20.0, start_ok=False)]
            ref = cc.REF_WB[10.0]
            rows = [(150.0 + 0.1 * k, [4470 + (ref if k % 2 else -ref)] * 8) for k in range(300)]
            rows.append((149.0, [99999.0] * 8))  # outside the window: ignored
            self.run_dir(d, cases, {"washboard@10": rows})
            res = cc.judge(d)
        self.assertTrue(res["cases"]["washboard@10"]["pass"])
        self.assertAlmostEqual(res["cases"]["washboard@10"]["amplitude_N"], ref, places=0)
        self.assertEqual(res["cases"]["washboard@20"]["verdict"], "refused")
        self.assertEqual(res["verdict"], "fail")
        self.assertEqual(res["rate_hz"], 480.0)

    def test_class_c_peak_against_the_reference(self):
        with tempfile.TemporaryDirectory() as d:
            rate, v = 480.0, 10.0
            hz_ref, h_ref = cc.REF_CC[v]
            t = np.arange(int(40 * rate)) / rate
            rng = np.random.default_rng(3)
            y = 4470 + 300 * rng.standard_normal(len(t)) + 400 * np.sin(2 * np.pi * hz_ref * t)
            steps = [(300.0 + v * t[i], list(y[i:i + 8])) for i in range(0, len(t) - 7, 8)]
            self.run_dir(d, [{"case": "class_c@10", "fixture": "class_c", "speed": v,
                              "fixture_s": 300.0, "fixture_len": 400.0, "start_ok": True}],
                         {"class_c@10": steps})
            res = cc.judge(d)
            got = res["cases"]["class_c@10"]
        self.assertLess(abs(got["peak_hz"] - hz_ref), cc.PEAK_HZ)


if __name__ == "__main__":
    unittest.main()
