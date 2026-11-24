"""Issue 147: the look-ahead asks the tile worker for every tile within the frame end's reach of a
car that touches a corridor, as the frame's end waits for them (issue 144), so the frame's end
finds them made; a frame's end waiting over SLOW_WAIT_MS for them is logged. No Isaac."""
import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_layer, rock_layer, surface_layer  # noqa: E402


def winding(n=600, step=0.5):
    s = np.arange(n) * step
    h = 0.6 * np.sin(s / 90.0) + 0.25 * np.sin(s / 17.0)
    x = np.concatenate([[0.0], np.cumsum(np.cos(h[:-1]) * step)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(h[:-1]) * step)])
    return rd.Corridor(np.stack([x, y, np.zeros(n)], axis=1))


def car(x, y, vel=(10.0, 0.0)):
    return SimpleNamespace(last=((x, y, 0.0), (1.0, 0.0, 0.0, 0.0), (vel[0], vel[1], 0.0)))


class Ahead(unittest.TestCase):
    def test_the_reach_is_the_frame_ends(self):
        self.assertEqual(road_layer.AROUND,
                         max(rock_layer.ROCK_REACH, rock_layer.COLLISION_REACH))

    def test_every_tile_the_frame_end_waits_for_is_asked_for_ahead(self):
        c = winding()
        lay = road_layer.RoadLayer([c], workers=0)
        d = lay.detail
        try:
            for k in range(0, 600, 60):
                x, y = (float(v) for v in c.centreline[k][:2])
                want = {q for q in d.tiles_near([x], [y], road_layer.AROUND) if d.touches(*q)}
                self.assertLessEqual(want, lay._ahead(car(x, y)), k)
        finally:
            d.close()

    def test_tiles_off_every_corridor_are_not_asked_for(self):
        c = winding()
        lay = road_layer.RoadLayer([c], workers=0)
        d = lay.detail
        try:
            x, y = (float(v) for v in c.centreline[300][:2])
            got = lay._ahead(car(x, y))
            around = d.tiles_near([x], [y], road_layer.AROUND)
            self.assertTrue(all(d.touches(*q) for q in got & around))
        finally:
            d.close()


class SlowWait(unittest.TestCase):
    def surface(self, wait_s):
        s = surface_layer.SurfaceLayer.__new__(surface_layer.SurfaceLayer)
        s.wait_ms, s.wait_missed = [], 0
        s.road_colliders, s.road_rocks = object(), object()
        s.near_pose = [(1.0, 2.0, 0.0, 5.0)]
        clock = iter([0.0, wait_s])
        s.road = SimpleNamespace(ready_around=lambda *a, **k: [])
        return s, clock

    def test_a_wait_over_the_bound_is_logged_and_one_under_it_is_not(self):
        for wait_s, logged in ((0.08, True), (0.01, False)):
            s, clock = self.surface(wait_s)
            with mock.patch("time.perf_counter", side_effect=lambda: next(clock)):
                if logged:
                    with self.assertLogs("scensim_isaac_world", "WARNING") as cm:
                        s._wait_for_tiles()
                    self.assertIn("issue 147", cm.output[0])
                    self.assertIn("80 ms", cm.output[0])
                else:
                    with self.assertNoLogs("scensim_isaac_world", "WARNING"):
                        s._wait_for_tiles()
            self.assertEqual(len(s.wait_ms), 1)
            self.assertTrue(math.isclose(s.wait_ms[0], wait_s * 1e3))


if __name__ == "__main__":
    unittest.main()
