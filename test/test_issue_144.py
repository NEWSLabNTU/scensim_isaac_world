"""Issue 144: a frame's end waits, at most FRAME_WAIT_S, for the road's tiles within the rocks'
and the collision tiles' reach of every car before it updates them, so the tiles and rocks a frame
has depend on where the cars are and not on the tile worker's wall clock. No Isaac."""
import concurrent.futures
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_layer, rock_layer, surface_layer  # noqa: E402


def surface(road, colliders=True, rocks=True):
    s = surface_layer.SurfaceLayer.__new__(surface_layer.SurfaceLayer)
    s.road, s.road_meshes, s.tiles, s.grids, s.soil_xy = road, None, {}, {}, {}
    s.wait_ms, s.wait_missed = [], 0
    calls = []
    s.road_colliders = SimpleNamespace(update=lambda pts: calls.append("colliders")) \
        if colliders else None
    s.road_rocks = SimpleNamespace(update=lambda pts: calls.append("rocks")) if rocks else None
    s.near_pose = [(10.0, 20.0, 0.0, 5.0), (300.0, -40.0, 1.0, 2.0)]
    return s, calls


class Wait(unittest.TestCase):
    def test_the_wait_comes_before_the_tiles_and_rocks_for_every_car(self):
        order = []
        road = SimpleNamespace(ready_around=mock.Mock(side_effect=lambda *a, **k: order.append(
            "wait") or []))
        s, calls = surface(road)
        s.road_colliders.update = lambda pts: order.append("colliders")
        s.road_rocks.update = lambda pts: order.append("rocks")
        s.flush()
        self.assertEqual(order, ["wait", "colliders", "rocks"])
        (points, reach), kw = road.ready_around.call_args
        self.assertEqual(points, [(10.0, 20.0), (300.0, -40.0)])
        self.assertEqual(reach, max(rock_layer.ROCK_REACH, rock_layer.COLLISION_REACH))
        self.assertEqual(kw["timeout"], surface_layer.FRAME_WAIT_S)
        self.assertEqual(len(s.wait_ms), 1)

    def test_no_road_no_cars_or_nothing_to_update_waits_for_nothing(self):
        road = SimpleNamespace(ready_around=mock.Mock(return_value=[]))
        s, _ = surface(road)
        s.near_pose = []
        s.flush()
        s, _ = surface(road, colliders=False, rocks=False)
        s.flush()
        road.ready_around.assert_not_called()
        s, _ = surface(None)
        s.flush()

    def test_tiles_still_not_ready_are_counted_and_logged(self):
        road = SimpleNamespace(ready_around=mock.Mock(return_value=[(1, 2), (3, 4)]))
        s, calls = surface(road)
        with self.assertLogs("scensim_isaac_world", "WARNING") as cm:
            s.flush()
        self.assertEqual(s.wait_missed, 2)
        self.assertIn("issue 144", cm.output[0])
        self.assertEqual(calls, ["colliders", "rocks"])  # the frame goes on


def winding(n=600, step=0.5):
    s = np.arange(n) * step
    h = 0.6 * np.sin(s / 90.0) + 0.25 * np.sin(s / 17.0)
    x = np.concatenate([[0.0], np.cumsum(np.cos(h[:-1]) * step)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(h[:-1]) * step)])
    return rd.Corridor(np.stack([x, y, np.zeros(n)], axis=1))


class Paces(unittest.TestCase):
    """The proof's offline form: a slow worker, the same drive at two paces, and the tiles each
    frame's end finds ready in reach the same, frame by frame."""

    def drive(self, sleep_s, timeout=surface_layer.FRAME_WAIT_S):
        """Frames along a winding road; each frame's end waits as `flush` does, and records the
        tiles ready within reach. The worker makes a tile 20 ms after it is asked for, on the
        wall clock; `sleep_s` is the frame's own wall time (the pace)."""
        c = winding()
        lay = road_layer.RoadLayer([c], workers=0)
        pool = concurrent.futures.ThreadPoolExecutor(1)

        def submit(fn, i, j):
            return pool.submit(lambda: (time.sleep(0.02), fn(i, j))[1])
        lay.detail.pool = SimpleNamespace(submit=submit)
        reach = max(rock_layer.ROCK_REACH, rock_layer.COLLISION_REACH)
        seen = []
        try:
            for k in range(0, 560, 40):
                p = tuple(c.centreline[k][:2])
                left = lay.ready_around([p], reach, timeout=timeout)
                if timeout:
                    self.assertEqual(left, [])
                d = lay.detail
                seen.append(sorted(q for q in d.tiles_near([p[0]], [p[1]], reach)
                                   if d.touches(*q) and d.ready(q)))
                time.sleep(sleep_s)
        finally:
            pool.shutdown(wait=True)
        return seen

    def test_two_paces_give_the_same_tiles_frame_by_frame(self):
        waited = self.drive(0.0)
        self.assertEqual(waited, self.drive(0.05))
        # the control: without the wait the fast drive's frames find fewer tiles ready
        bare = self.drive(0.0, timeout=0.0)
        self.assertLess(sum(map(len, bare)), sum(map(len, waited)))


if __name__ == "__main__":
    unittest.main()
