"""Issue 134: the road's render mesh made every frame anew the tiles in reach that hold none of
the road, two uv passes each, and a new tile's base road and zones were made on the frame's
thread. A tile's look (base road, zones) now comes with the tile from the same projection, and
a tile without the road is remembered; results as before, bit for bit; no Isaac."""
import os
import sys
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_layer  # noqa: E402


def winding(n=1500, step=0.5, **kw):
    s = np.arange(n) * step
    h = 0.6 * np.sin(s / 90.0) + 0.25 * np.sin(s / 17.0)
    x = np.concatenate([[0.0], np.cumsum(np.cos(h[:-1]) * step)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(h[:-1]) * step)])
    z = 0.02 * s
    return rd.Corridor(np.stack([x, y, z], axis=1), **kw)


def keys_along(d, c, every=200):
    cl = c.centreline
    return sorted({(int(np.floor(cl[k, 0] / d.tile)) + di, int(np.floor(cl[k, 1] / d.tile)) + dj)
                   for k in range(0, len(cl), every) for di in (-1, 0, 1) for dj in (-1, 0, 1)})


def old_surface(d, i, j):
    """surface_tile as before issue 134."""
    X, Y = d._grid(i, j)
    base = np.zeros(X.shape)
    for c in d.corridors:
        if not c.z.any() and not c.section_in_base:
            continue
        u, v, near = c.uv(X, Y)
        base = np.where(near.reshape(X.shape), c.surface_base(u, v).reshape(X.shape), base)
    return base + d.tile_of(i, j)


def old_zones(d, i, j):
    X, Y = d._grid(i, j)
    out = np.zeros(X.shape, dtype=np.int8)
    for c in d.corridors:
        u, v, near = c.uv(X, Y)
        u, v, near = u.reshape(X.shape), v.reshape(X.shape), near.reshape(X.shape)
        road = near & (np.abs(v) <= c.section.edge(u, v))
        out = np.maximum(out, np.where(road, d.ROAD, np.where(near, d.VERGE, d.OUTSIDE)))
    return out


class Look(unittest.TestCase):
    def test_the_look_is_as_before_and_made_with_the_tile(self):
        c = winding(roughness=rd.Roughness(cls="C"), potholes=rd.Potholes(per_100m=3))
        d = rd.Detail([c], workers=0)
        for key in keys_along(d, c):
            np.testing.assert_array_equal(d.surface_tile(*key), old_surface(d, *key))
            np.testing.assert_array_equal(d.zones_tile(*key), old_zones(d, *key))
        self.assertEqual(d.look_fallbacks, 0)  # each look came with its tile

    def test_a_worker_process_ships_the_look(self):
        c = winding(300)
        d = rd.Detail([c], workers=1, processes=True)
        try:
            keys = keys_along(d, c, every=100)[:6]
            d.prefetch(keys)
            for _ in range(600):
                if all(d.ready(k) for k in keys):
                    break
                time.sleep(0.05)
            for key in keys:
                np.testing.assert_array_equal(d.zones_tile(*key), old_zones(d, *key))
            self.assertEqual(d.look_fallbacks, 0)
        finally:
            d.close()

    def test_the_look_is_evicted_with_its_tile(self):
        c = winding(300)
        d = rd.Detail([c], workers=0, max_tiles=4)
        keys = keys_along(d, c, every=60)[:10]
        for key in keys:
            d.tile_of(*key)
        self.assertEqual(set(d.looks), set(d.tiles))
        self.assertLessEqual(len(d.looks), 4)

    def test_reading_a_made_tile_costs_a_fraction_of_making_its_look(self):
        c = winding(3000)
        d = rd.Detail([c], workers=0)
        key = keys_along(d, c, every=1500)[4]
        d.tile_of(*key)
        t = time.process_time()
        for _ in range(5):
            d.surface_tile(*key)
            d.zones_tile(*key)
        cached = time.process_time() - t
        t = time.process_time()
        old_surface(d, *key)
        old_zones(d, *key)
        made = time.process_time() - t
        self.assertLess(cached / 5, 0.25 * made)


class Meshes(unittest.TestCase):
    def test_a_tile_without_the_road_is_looked_at_once(self):
        pxr = mock.MagicMock()
        c = winding(300)
        layer = mock.MagicMock()
        layer.detail = rd.Detail([c], workers=0)
        layer.ox = layer.oy = 0.0
        d = layer.detail
        # a tile that touches the corridor's box but none of its reach: no quads
        empty = next(k for k in keys_along(d, c, every=20)
                     if d.touches(*k) and not (d.zones_tile(*k) > 0).any())
        with mock.patch.dict(sys.modules, {"pxr": pxr}):
            ms = road_layer.MeshStream(mock.MagicMock(), layer, reach=1.0)
            x, y = (np.array(empty) + 0.5) * d.tile
            calls = []
            real = d.zones_tile

            def counted(*k):
                calls.append(k)
                return real(*k)

            with mock.patch.object(d, "zones_tile", side_effect=counted):
                for _ in range(5):
                    ms.update([(x, y, 0.0)])
        self.assertEqual(calls.count(empty), 1)
        self.assertIsNone(ms.meshes[empty])
        with mock.patch.dict(sys.modules, {"pxr": pxr}):
            ms.update([])
            ms.remove()
        self.assertEqual(ms.meshes, {})


if __name__ == "__main__":
    unittest.main()
