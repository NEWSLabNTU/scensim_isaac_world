"""Issue 142: since issue 134 every tile was projected on every corridor, and a tile the road
does not reach took ten times as long in the worker (44 to 424 ms), so the rocks ahead loaded
about 30 m later. A corridor that cannot reach the tile is skipped again; heights, base road,
zones and the rocks placed are bit for bit those of the full projection; no Isaac."""
import os
import sys
import time
import unittest
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


if __name__ == "__main__":
    unittest.main()
