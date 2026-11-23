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


if __name__ == "__main__":
    unittest.main()
