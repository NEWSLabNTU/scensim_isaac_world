"""Issue 142: since issue 134 every tile was projected on every corridor, and a tile the road
does not reach took ten times as long in the worker (44 to 424 ms), so the rocks ahead loaded
about 30 m later. A corridor that cannot reach the tile is skipped again; heights, base road,
zones and the rocks placed are bit for bit those of the full projection; no Isaac."""
import math
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import rocks as rk  # noqa: E402


def winding(n=1500, step=0.5, **kw):
    s = np.arange(n) * step
    h = 0.6 * np.sin(s / 90.0) + 0.25 * np.sin(s / 17.0)
    x = np.concatenate([[0.0], np.cumsum(np.cos(h[:-1]) * step)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(h[:-1]) * step)])
    z = 0.02 * s
    return rd.Corridor(np.stack([x, y, z], axis=1), **kw)


def keys_around(d, c, every=150, ring=3):
    """Tiles along the road and up to `ring` tiles off it: on and off the road."""
    cl = c.centreline
    return sorted({(int(np.floor(cl[k, 0] / d.tile)) + di, int(np.floor(cl[k, 1] / d.tile)) + dj)
                   for k in range(0, len(cl), every)
                   for di in range(-ring, ring + 1) for dj in range(-ring, ring + 1)})


def corridors():
    return [winding(roughness=rd.Roughness(cls="C"), potholes=rd.Potholes(per_100m=3),
                    rocks=rk.Zones(shoulder=0.01, runoff=0.01, ditch=0.03)),
            winding(900, roughness=rd.Roughness(cls="B"))]


def made(d, key):
    t, _, extra, look = d._timed(*key)
    return t, extra, look


class Skip(unittest.TestCase):
    def test_tiles_on_and_off_the_road_are_as_the_full_projection(self):
        cs = corridors()
        # a 0.2 m grid: the skip does not depend on the spacing, and the test stays short
        new = rd.Detail(cs, spacing=0.2, workers=0, extras="scensim_isaac_world.rocks:place")
        full = rd.Detail(cs, spacing=0.2, workers=0, extras="scensim_isaac_world.rocks:place")
        full._reaches = lambda c, i, j: True  # project every corridor, as since issue 134
        keys = keys_around(new, cs[0], every=700, ring=2)
        off = 0
        for key in keys:
            t1, e1, (b1, z1) = made(new, key)
            t2, e2, (b2, z2) = made(full, key)
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(b1, b2)
            np.testing.assert_array_equal(z1, z2)
            self.assertEqual(repr(e1), repr(e2))
            off += not new.touches(*key)
        self.assertGreater(off, len(keys) // 3)  # the test covers tiles off the road

    def test_a_tile_off_the_road_costs_a_fraction_of_one_on_it(self):
        """CPU time, not wall clock (issue 132): a tile no corridor reaches is not projected."""
        c = winding(3000)
        d = rd.Detail([c], workers=0)
        on = (int(np.floor(c.centreline[1500, 0] / d.tile)),
              int(np.floor(c.centreline[1500, 1] / d.tile)))
        far = (on[0], on[1] + 6)
        self.assertTrue(d.touches(*on))
        self.assertFalse(d.touches(*far))
        t = time.process_time()
        for _ in range(3):
            d._timed(*on)
        t_on = time.process_time() - t
        t = time.process_time()
        for _ in range(3):
            d._timed(*far)
        t_far = time.process_time() - t
        self.assertLess(t_far, t_on / 4, (t_far, t_on))

    def test_the_skip_is_the_old_centre_test(self):
        c = winding(600)
        d = rd.Detail([c], spacing=0.2, workers=0)
        with mock.patch.object(c, "uv", wraps=c.uv) as uv:
            self.assertIsNone(d._uvs(0, 40)[0])  # 640 m off a road that never goes there
        self.assertEqual(uv.call_count, 1)  # the centre only
        for key in keys_around(d, c, ring=1):
            self.assertEqual(d._uvs(*key)[0] is not None, d.touches(*key))


def line(points, step=0.5):
    """A corridor through `points` ((x, y) corners), a station every `step` m."""
    pts = [np.asarray(points[0], float)]
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    p = np.array(pts)
    return rd.Corridor(np.concatenate([p, np.zeros((len(p), 1))], axis=1))


def hairpin():
    """+x from 0 to 100 m, a 10 m half circle, back -x at y 20."""
    arc = [(100 + 10 * math.sin(a), 10 - 10 * math.cos(a)) for a in np.linspace(0, math.pi, 32)]
    return line([(0.0, 0.0)] + arc + [(0.0, 20.0)])


class Blocks(unittest.TestCase):
    """Step (iii): a tile's projection culled per 2 m block, a block's segments at once."""

    def check(self, c, keys):
        d = rd.Detail([c], workers=0)
        for k in keys:
            X, Y = d._grid(*k)
            whole = c._uv(np.stack([X.ravel(), Y.ravel()], axis=1))
            for a, b in zip(whole, c.uv(X, Y)):
                np.testing.assert_array_equal(a, b)

    def test_tiles_are_the_per_tile_culls_bit_for_bit(self):
        c = winding(3000)
        d = rd.Detail([c], workers=0)
        self.check(c, keys_around(d, c, every=500, ring=1))

    def test_a_hairpin_and_its_ties(self):
        c = hairpin()
        self.check(c, [(i, j) for i in range(-1, 8) for j in range(-1, 3)])

    def test_a_tile_costs_a_fraction_of_the_per_tile_cull(self):
        c = winding(3000)
        d = rd.Detail([c], workers=0)
        k = (int(np.floor(c.centreline[1500, 0] / d.tile)),
             int(np.floor(c.centreline[1500, 1] / d.tile)))
        X, Y = d._grid(*k)
        p = np.stack([X.ravel(), Y.ravel()], axis=1)
        t = time.process_time()
        c._uv(p)
        whole = time.process_time() - t
        t = time.process_time()
        c.uv(X, Y)
        blocks = time.process_time() - t
        self.assertLess(blocks, whole / 2, (blocks, whole))


class Ahead(unittest.TestCase):
    """Step (ii): the prefetch follows the corridors."""

    def layer(self, cs):
        from scensim_isaac_world import road_layer
        return road_layer.RoadLayer(cs, workers=0)

    def car(self, x, y, yaw, vel=(0.0, 0.0)):
        q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        return SimpleNamespace(last=((x, y, 0.0), q, (vel[0], vel[1], 0.0)))

    def test_a_hairpin_is_asked_for_before_the_car_turns(self):
        lay = self.layer([hairpin()])
        keys = lay._ahead(self.car(80.0, 0.0, 0.0, (15.0, 0.0)))
        self.assertIn((5, 1), keys)  # the leg back, x 80 to 96 at y 20
        lay.detail.close()

    def test_two_corridors_both_followed(self):
        lay = self.layer([line([(0.0, 0.0), (400.0, 0.0)]),
                          line([(200.0, 0.0), (200.0, 400.0)])])
        keys = lay._ahead(self.car(196.0, 0.0, 0.0, (10.0, 0.0)))
        self.assertIn((15, 0), keys)  # along the first, x 240 to 256
        self.assertIn((12, 2), keys)  # along the second, y 32 to 48
        self.assertNotIn((15, 3), keys)  # between them
        lay.detail.close()

    def test_off_every_corridor_the_heading_line(self):
        lay = self.layer([line([(0.0, 0.0), (400.0, 0.0)])])
        keys = lay._ahead(self.car(100.0, 300.0, 0.0, (10.0, 0.0)))
        s = np.linspace(0.0, 1.0, int(48.0 / 8.0) + 1)
        want = lay.detail.tiles_near(np.r_[100.0, 100.0 + 48.0 * s], np.r_[300.0, 300.0 + 0 * s],
                                     6.0)
        self.assertEqual(keys, want)
        lay.detail.close()

    def test_a_standing_car_ignores_its_velocitys_noise(self):
        lay = self.layer([line([(0.0, 0.0), (400.0, 0.0)])])
        rng = np.random.default_rng(142)
        first = lay._ahead(self.car(100.0, 0.0, 0.0, (0.0, 0.0)))
        for _ in range(20):
            a = rng.uniform(0, 2 * math.pi)
            noise = (1e-3 * math.cos(a), 1e-3 * math.sin(a))
            self.assertEqual(lay._ahead(self.car(100.0, 0.0, 0.0, noise)), first)
        self.assertIn((8, 0), first)  # ahead along the road, x 128 to 144
        lay.detail.close()


class Grid(unittest.TestCase):
    """Step (iv): a call's segments from a grid, not every segment (the rocks' many small sets)."""

    def test_small_sets_near_and_far_are_the_plain_loops(self):
        c = winding(3000)
        rng = np.random.default_rng(142)
        lo, hi = c.centreline.min(axis=0) - 200, c.centreline.max(axis=0) + 200
        every = lambda self, p, among=None: range(len(self.seg_len))  # noqa: E731
        for _ in range(300):
            n = int(rng.integers(1, 60))
            centre = rng.uniform(lo, hi) if rng.random() < 0.3 else \
                c.centreline[int(rng.integers(len(c.centreline)))] + rng.normal(0, 5, 2)
            p = centre + rng.normal(0, rng.uniform(0.1, 6.0), (n, 2))
            got = c.uv(p[:, 0], p[:, 1])
            with mock.patch.object(rd.Corridor, "_candidates", every):
                want = c._uv(p)
            for a, b in zip(got, want):
                np.testing.assert_array_equal(a, b)

    def test_a_small_set_keeps_few_segments(self):
        c = winding(3000)
        p = c.centreline[1500] + np.array([[0.0, 0.0], [1.0, 1.0]])
        self.assertLess(len(c._near_segments(p)), 200)


class Placed(unittest.TestCase):
    """Step (v): a placed car's tiles and rocks before its first step."""

    def test_ready_around_inline_and_with_a_slow_worker(self):
        import concurrent.futures

        from scensim_isaac_world import road_layer
        c = winding(600)
        lay = road_layer.RoadLayer([c], workers=0)
        p = tuple(c.centreline[300])
        self.assertEqual(lay.ready_around([p], 32.0), [])
        keys = {k for k in lay.detail.tiles_near([p[0]], [p[1]], 32.0) if lay.detail.touches(*k)}
        self.assertTrue(keys and all(lay.detail.ready(k) for k in keys))
        stuck = road_layer.RoadLayer([c], workers=0)
        stuck.detail.pool = SimpleNamespace(submit=lambda fn, i, j: concurrent.futures.Future())
        t = time.monotonic()
        left = stuck.ready_around([p], 32.0, timeout=0.2)
        self.assertGreaterEqual(time.monotonic() - t, 0.2)
        self.assertEqual(set(left), keys)

    def surface(self, left=()):
        from scensim_isaac_world import surface_layer
        s = surface_layer.SurfaceLayer.__new__(surface_layer.SurfaceLayer)
        s.wheels, s.road_meshes = {}, None
        s.road = SimpleNamespace(ready_around=mock.Mock(return_value=list(left)),
                                 reset=mock.Mock())
        s.road_colliders = mock.Mock()
        rocks = SimpleNamespace(queue=list(range(10)), calls=0)

        def update(points):
            rocks.calls += 1
            del rocks.queue[:4]  # four prims a frame
        rocks.update = update
        s.road_rocks = rocks
        s.pose_of = {"/other": (5.0, 5.0, 0.0, 3.0, 0.8, 0.1, 0.14)}
        s.near_pose = list(s.pose_of.values())
        return s

    def test_placed_drains_the_rocks_and_keeps_the_others(self):
        s = self.surface()
        s.pose_of["/car"] = (0.0, 0.0, 0.0, 9.0, 0.8, 0.1, 0.14)
        yaw = 0.5
        left = s.placed("/car", ((100.0, 200.0, 0.0), (math.cos(yaw / 2), 0.0, 0.0,
                                                       math.sin(yaw / 2))))
        self.assertEqual(left, [])
        self.assertEqual(s.road_rocks.queue, [])
        self.assertEqual(s.road_rocks.calls, 3)
        self.assertEqual(s.pose_of["/other"], (5.0, 5.0, 0.0, 3.0, 0.8, 0.1, 0.14))
        x, y, a, v, *rest = s.pose_of["/car"]
        self.assertEqual((x, y, v, tuple(rest)), (100.0, 200.0, 0.0, (0.8, 0.1, 0.14)))
        self.assertAlmostEqual(a, yaw)
        s.road.reset.assert_called_once_with("/car")
        s.road_colliders.update.assert_called_once()

    def test_tiles_not_ready_are_logged(self):
        s = self.surface(left=[(1, 2)])
        with self.assertLogs("scensim_isaac_world", "WARNING") as cm:
            s.placed("/car", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
        self.assertIn("not ready", cm.output[0])


if __name__ == "__main__":
    unittest.main()
