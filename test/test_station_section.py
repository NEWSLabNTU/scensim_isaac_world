"""The Isaac road layer's section per station (roadmap 086 C5a): road_detail.StationSection."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_layer as rl  # noqa: E402
from scensim_isaac_world import road_record as rr  # noqa: E402
from test_road_record import LINE, stations  # noqa: E402

V = np.linspace(-9.0, 9.0, 3601)


def level(**over):
    """A record with no crown, so each feature shows alone."""
    return stations(crown=np.zeros((2, 121)), **over)


class StationSection(unittest.TestCase):
    def test_a_constant_symmetric_level_record_is_sections_shape(self):
        for given in ({}, {"half_width": 3.0, "crown": 0.03, "shoulder": 0.8},
                      {"crown": 0.0, "ditch_depth": 0.5, "backslope": 0.3}):
            st, _ = rr.from_constants(given)
            a, b = rd.StationSection(st), rd.Section(**given)
            np.testing.assert_allclose(a.height_uv(np.full_like(V, 7.0), V), b.height(V),
                                       atol=1e-12)
            self.assertAlmostEqual(a.extent(), b.extent(), places=12)

    def test_the_bank_rolls_the_formation_about_the_centreline(self):
        s = rd.StationSection(level(bank=np.full(121, 0.05)))
        v = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(s.height_uv(np.full(5, 30.0), v), -math.tan(0.05) * v,
                                   atol=1e-12)

    def test_the_crown_line_sits_crown_offset_left_of_the_centreline(self):
        s = rd.StationSection(stations(crown_offset=np.full(121, 0.6)))
        h = s.height_uv(np.full_like(V, 10.0), V)
        on = np.abs(V) <= 2.5
        self.assertAlmostEqual(float(V[on][np.argmax(h[on])]), 0.6, places=2)
        self.assertAlmostEqual(float(h[on].max()), 0.0, places=12)

    def test_each_side_has_its_own_crown(self):
        s = rd.StationSection(stations(crown=np.array([[0.02] * 121, [0.06] * 121])))
        h = s.height_uv(np.full(2, 10.0), np.array([2.0, -2.0]))
        np.testing.assert_allclose(h, [-0.04, -0.12], atol=1e-12)

    def test_the_lip_drops_the_shoulder_over_its_face(self):
        u = np.full(3, 50.0)
        step = rd.StationSection(level(lip_h=np.full((2, 121), 0.10)))
        np.testing.assert_allclose(step.height_uv(u, np.array([2.5, 2.501, 2.9])),
                                   [0.0, -0.10, -0.10], atol=1e-12)
        ramp = rd.StationSection(level(lip_h=np.full((2, 121), 0.10),
                                       lip_angle=np.full((2, 121), 45.0)))
        np.testing.assert_allclose(ramp.height_uv(u, np.array([2.5, 2.55, 2.6])),
                                   [0.0, -0.05, -0.10], atol=1e-9)
        ramp30 = rd.StationSection(level(lip_h=np.full((2, 121), 0.10),
                                         lip_angle=np.full((2, 121), 30.0)))
        face = 0.10 / math.tan(math.radians(30.0))
        self.assertAlmostEqual(float(ramp30.height_uv(u[:1], np.array([2.5 + face / 2]))[0]),
                               -0.05, places=9)

    def test_fields_are_linear_between_stations(self):
        crown = np.zeros((2, 121))
        crown[:, 41:] = 0.04
        s = rd.StationSection(stations(crown=crown))
        h = s.height_uv(np.array([40.0, 40.5, 41.0]), np.full(3, 2.0))
        np.testing.assert_allclose(h, [0.0, -0.04, -0.08], atol=1e-12)
        np.testing.assert_allclose(s.carriageway(np.array([0.0, 200.0])), [2.5, 2.5])

    def test_the_banked_section_returns_to_the_base_without_a_step(self):
        s = rd.StationSection(stations(bank=np.full(121, 0.1),
                                       lip_h=np.full((2, 121), 0.05)))
        v = np.linspace(-12, 12, 240001)
        h = s.height_uv(np.full_like(v, 60.0), v)
        lip_face = (np.abs(np.abs(v) - 2.5) < 1e-3)
        jumps = np.abs(np.diff(h))[~(lip_face[1:] | lip_face[:-1])]
        self.assertLess(float(jumps.max()), 1e-3)
        far = np.abs(v) > s.extent()
        self.assertTrue(np.all(h[far] == 0.0))
        self.assertTrue(np.all(np.abs(h[np.abs(v) > s.extent() - 0.5]) < 0.5))

    def test_the_layer_samples_the_record_per_station(self):
        d = tempfile.mkdtemp()
        bank = np.linspace(0.0, 0.08, 121)
        rr.save_stations(os.path.join(d, "r.npz"),
                         stations(bank=bank, crown_offset=np.full(121, 0.3)))
        c, = rl.parse({"corridors": [{"centreline": LINE, "seed": 1, "stations": "r.npz",
                                      "roughness": {"gd0": 1e-12}}]}, base=d)
        det = rd.Detail([c])
        try:
            u = np.array([10.0, 37.0, 80.0, 80.0, 80.0])
            v = np.array([0.0, 1.2, -2.0, 0.3, 4.0])
            got = det.height(u, v)  # the line runs along x, so (x, y) is (u, v)
        finally:
            det.close()
        want = c.section.height_uv(u, v)
        np.testing.assert_allclose(got, want, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
