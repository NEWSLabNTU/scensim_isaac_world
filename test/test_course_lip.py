"""086 C5c: the lip course, the record's strips as surface zones, and lip_check's classes; no
Isaac."""
import csv
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_layer, surface_layer, surfaces  # noqa: E402
from scensim_isaac_world.course import fixture, lip_check, lip_drive  # noqa: E402

COURSE = os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world", "courses",
                      "fixture_c5c.yaml")


def course():
    with open(COURSE) as f:
        return yaml.safe_load(f)


class LipCourse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = course()
        cls.c = fixture.centreline(cls.course)
        cls.st = fixture.stations(cls.course, cls.c)
        cls.sec = rd.StationSection(cls.st)
        cls.corridor = rd.Corridor(np.stack([cls.c["x"], cls.c["y"], cls.c["z"]], 1),
                                   section=cls.sec, stations=cls.st)

    def drop(self, s):
        """The lip's drop at s: the right carriageway edge's height less the shoulder's
        just past it."""
        u = np.array([s, s])
        h = self.sec.height_uv(u, np.array([-3.25, -3.25 - 0.3]))
        crown = 0.02 * 0.3
        return float(h[0] - h[1] - crown)

    def test_each_case_is_its_segment(self):
        cases = self.course["cases"]
        self.assertEqual([c["segment"] for c in cases], list(range(11)))
        for c in cases:
            mid = c["from"] + 100.0
            self.assertAlmostEqual(self.drop(mid), c["lip_h"], places=6, msg=c)
            i = int(round(mid / 0.5))
            if c["lip_h"] > 0:
                self.assertAlmostEqual(float(self.st.side["lip_angle"][1][i]), c["lip_angle"])

    def test_the_expected_outcomes_are_the_rule(self):
        """069's `lip`: over about 5 cm and a face steeper than 45 deg holds; on either edge of
        the rule, boundary; any other re-enters."""
        for c in self.course["cases"]:
            h, a = c["lip_h"], c["lip_angle"] or 0
            edge = math.isclose(h, 0.05) and a > 45 or (h > 0.05 and math.isclose(a, 45))
            rule = "boundary" if edge else ("held" if h > 0.05 and a > 45 else "re-enters")
            self.assertEqual(c["expect"], rule, c)

    def test_the_strips_are_zones_under_the_maps(self):
        zones = road_layer.strip_zones([self.corridor])
        self.assertEqual([(z.id, z.cls) for z in zones],
                         [("record_right_shoulder_1", "gravel"), ("record_right_runoff_1", "grass")])
        table = surfaces.load_classes()
        own = surfaces.SurfaceMap("t", table, {surfaces.UNLABELLED: "asphalt"},
                                  [surfaces.Zone("pad", np.array([[2090.0, -10.0], [2110.0, -10.0],
                                                                  [2110.0, 10.0], [2090.0, 10.0]]),
                                                 cls="ice")])
        layer = surface_layer.SurfaceLayer(None, own)
        layer.add_record_zones(zones)
        at = layer.map.at
        self.assertEqual(at(2050.0, -4.0).cls, "gravel")  # the shoulder
        self.assertEqual(at(2050.0, -4.0).zone, "record_right_shoulder_1")
        self.assertEqual(at(2050.0, -6.0).cls, "grass")  # the run-off
        self.assertEqual(at(2050.0, -2.0).cls, "asphalt")  # the carriageway: the map's
        self.assertEqual(at(2050.0, 4.0).cls, "asphalt")  # the left shoulder has no class
        self.assertEqual(at(1500.0, -4.0).cls, "asphalt")  # another segment's shoulder
        self.assertEqual(at(2100.0, -4.0).cls, "ice")  # the map's zone wins over the strip
        self.assertIs(layer.own, layer.map)

    def test_a_constant_section_strip_runs_the_corridor(self):
        st, problems = road_record_constants({"shoulder_surface": {"left": "", "right": "gravel"}})
        self.assertFalse(problems)
        cor = rd.Corridor(np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]), stations=st)
        (z,) = road_layer.strip_zones([cor])
        self.assertEqual((z.id, z.cls), ("record_right_shoulder_1", "gravel"))
        self.assertTrue(z.contains(25.0, -2.7))
        self.assertFalse(z.contains(25.0, -2.4))


def road_record_constants(section):
    from scensim_isaac_world import road_record as rr

    return rr.from_constants(section)


def fake_run(d, delays, speed=15.6, angle=4.0, strips_right=("gravel", "record_right_shoulder_1"),
             tread=1.6, base=3.0, half=3.25):
    """A lip_drive run whose right front wheel crosses the edge `delays[k]` s after case k's
    turn-in (None: never)."""
    header = ["case", "t", "fr_v", "yaw_rate", "ay", "roll"]
    header += [f"w{i}_{k}" for i in range(4) for k in ("cls", "zone")]
    rows, cases = [], []
    for k, dl in delays.items():
        p = lip_drive.plan({"from": 200.0 * k}, half, tread, math.radians(angle))
        cases.append({"segment": k, "expect": "re-enters", "lip_h": 0.1, "lip_angle": 90,
                      "start_ok": True, "start_speed": speed, "start_offset": 0.0,
                      "right_on_shoulder": [True, True], "start_t": 5.0, "turn_t": 8.0,
                      "v0": p["v0"], "v1": p["v1"], "angle": p["angle"]})
        for t in np.arange(0.0, 14.0, 0.05):
            fv = -half - 0.5 if dl is None or t < 8.0 + dl - 0.5 else \
                -half - 0.5 + (t - (8.0 + dl - 0.5))
            sr = strips_right if k == 10 else ("asphalt", "")
            rows.append([k, round(t, 3), fv, 0.1, 0.5, 0.0, "asphalt", "", *sr, "asphalt", "",
                         *sr])
    with open(os.path.join(d, "lip.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    json.dump({"query": "raycast", "speed": speed, "angle_deg": angle, "wheel_tread": tread,
               "wheel_base": base, "cases": cases}, open(os.path.join(d, "cases.json"), "w"))


class LipCheck(unittest.TestCase):
    def run_check(self, delays, **kw):
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            fake_run(d, delays, **kw)
            st = mock.Mock()
            st.common = {"half_width": np.array([3.25])}
            with mock.patch("scensim_isaac_world.course.record.load", return_value=st):
                return lip_check.check(d, "unused")

    def line_says(self, tread=1.6, base=3.0):
        p = lip_drive.plan({"from": 0.0}, 3.25, tread, math.radians(4.0))
        return lip_check.path_crossing(p, 3.25, tread, base, 15.6)

    def test_classes_against_the_control(self):
        d0 = self.line_says()
        res = self.run_check({0: d0, 4: d0 + 0.2, 7: d0 + 1.0, 9: None, 10: d0})
        got = {c["segment"]: c["got"] for c in res["cases"]}
        self.assertEqual(got, {0: "re-enters", 4: "re-enters", 7: "held", 9: "held",
                               10: "re-enters"})
        self.assertTrue(res["ok"], res["problems"])

    def test_the_control_must_cross_where_its_line_says(self):
        res = self.run_check({0: self.line_says() + 0.6})
        self.assertFalse(res["ok"])
        self.assertIn("control", res["problems"][0])

    def test_the_strips_segment_needs_gravel_on_the_right(self):
        d0 = self.line_says()
        res = self.run_check({0: d0, 10: d0}, strips_right=("asphalt", ""))
        self.assertFalse(res["ok"])
        self.assertTrue(any(p.startswith("strips") for p in res["problems"]))

    def test_the_line_crosses_about_046_s_after_the_turn_in(self):
        # 133's estimate: 0.5 m to the edge at 15.6 m/s and 4 deg (no front axle lead)
        want = 0.5 / (15.6 * math.tan(math.radians(4.0)))
        self.assertAlmostEqual(want, 0.458, places=3)
        self.assertAlmostEqual(self.line_says(base=0.0), want, delta=0.01)
        # the front axle 3 m ahead turns with the car: it reaches the edge sooner
        self.assertLess(self.line_says(base=3.0), want)

if __name__ == "__main__":
    unittest.main()
