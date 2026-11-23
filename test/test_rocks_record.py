"""086 C6b: rocks' zones from the road record's strips (shoulder, run-off, ditch); no Isaac."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import rocks as rk  # noqa: E402


def stations(runoff_left=1.0, runoff_right=0.0):
    u = np.array([0.0, 100.0])
    side = {"crown": [[0.02] * 2] * 2, "shoulder": [[0.75] * 2] * 2,
            "foreslope": [[1 / 3] * 2] * 2, "ditch_depth": [[0.3] * 2] * 2,
            "ditch_bottom": [[0.6] * 2] * 2, "backslope": [[0.5] * 2] * 2,
            "lip_h": [[0.0] * 2] * 2, "lip_angle": [[90.0] * 2] * 2,
            "runoff_len": [[runoff_left] * 2, [runoff_right] * 2]}
    return SimpleNamespace(u=u, common={"half_width": np.full(2, 3.25), "bank": np.zeros(2),
                                        "crown_offset": np.zeros(2)},
                           side={k: np.array(v) for k, v in side.items()})


class RecordZones(unittest.TestCase):
    def test_the_records_strips(self):
        st = stations()
        sec = rd.StationSection(st)
        z = rk.Zones()
        want = {0.0: "carriageway", 3.2: "carriageway", 3.3: "shoulder", 3.95: "shoulder",
                4.1: "runoff", 4.95: "runoff", 5.1: "ditch", -3.3: "shoulder",
                -4.1: "ditch"}  # no run-off on the right
        for v, name in want.items():
            self.assertEqual(z.zone(sec, v, 50.0, st), name, v)
        self.assertIsNone(z.zone(sec, sec.extent() + 1.0, 50.0, st))
        self.assertEqual(z.k(sec, 4.5, 50.0, st), z.runoff)

    def test_without_a_record_as_before(self):
        sec = rd.Section()
        z = rk.Zones()
        self.assertEqual(z.zone(sec, 0.0), "carriageway")
        self.assertEqual(z.zone(sec, sec.half_width + 0.01), "shoulder")


class FixedRocks(unittest.TestCase):
    """086 C6b: a fixture's rocks at fixed places (`rocks: {at: ...}`)."""

    def layer(self, at):
        from scensim_isaac_world import road_layer

        c, = road_layer.parse({"corridors": [{
            "centreline": [[0, 0], [500, 0]], "roughness": {"gd0": 1e-12},
            "rocks": {"carriageway": 0, "shoulder": 0, "runoff": 0, "ditch": 0, "at": at}}]})
        return c

    def test_each_rock_where_and_as_tall_as_asked(self):
        c = self.layer([[100.0, 2.425, 0.15, 0.0, 0], [200.0, 2.425, 0.15, 0.0, 1],
                        [400.0, 1.625, 0.35, 0.2, 0]])
        d = rd.Detail([c], workers=0)
        got = []
        for i in range(int(500 // d.tile) + 1):
            for j in (-1, 0, 1):
                got += rk.place(d, i, j)
        self.assertEqual(len(got), 3)
        got.sort(key=lambda e: e[1][0])
        want = [(100, 2.425, 0.15, 0, 0), (200, 2.425, 0.15, 0, 1), (400, 1.625, 0.35, 0.2, 0)]
        for (rock, p, yaw, loose), (u, v, h, e, lo) in zip(got, want):
            self.assertAlmostEqual(p[0], u, places=6)
            self.assertAlmostEqual(p[1], v, places=6)
            surface = float(c.surface_base(np.array([u]), np.array([v]))[0]
                            + d.height(np.array([p[0]]), np.array([p[1]]))[0])
            top = p[2] + rock.verts[:, 2].max()
            self.assertAlmostEqual(top - surface, h, places=6)  # proud by `height`
            self.assertEqual(loose, bool(lo))
            self.assertAlmostEqual(yaw, 0.0)
        self.assertTrue(5.0 < got[1][0].mass < 30.0, got[1][0].mass)  # R4's ~11 kg

    def test_bad_rows_are_refused(self):
        from scensim_isaac_world import road_layer

        for bad in ([[100.0, 2.4, 0.15, 0.0]], [[100.0, 2.4, -0.1, 0.0, 0]],
                    [[100.0, 2.4, 0.15, 1.0, 0]], [[100.0, 2.4, 0.15, 0.0, 2]]):
            with self.assertRaises(road_layer.RoadDetailError, msg=str(bad)):
                self.layer(bad)


if __name__ == "__main__":
    unittest.main()
