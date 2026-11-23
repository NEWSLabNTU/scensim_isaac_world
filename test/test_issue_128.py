"""Issue 128: Corridor.uv projected every point onto every centreline segment, 7 s a tile on
Haruna's 3055-station record. It now keeps the segments that can be nearest (`_candidates`);
the result is the full loop's, bit for bit, ties included; no Isaac."""
import os
import sys
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402


def winding(n=3000, step=0.5):
    """A winding road of n stations at `step` m: a slow curve with a tighter wiggle on it."""
    s = np.arange(n) * step
    h = 0.6 * np.sin(s / 90.0) + 0.25 * np.sin(s / 17.0)
    x = np.concatenate([[0.0], np.cumsum(np.cos(h[:-1]) * step)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(h[:-1]) * step)])
    return rd.Corridor(np.stack([x, y, np.zeros(n)], axis=1))


def full(c, x, y):
    """uv over every segment, as before the fix."""
    with mock.patch.object(rd.Corridor, "_candidates", lambda self, p: range(len(self.seg_len))):
        return c.uv(x, y)


def grid(c, i, j, tile=16.0, n=81):
    g = i * tile + np.arange(n + 1) * tile / n, j * tile + np.arange(n + 1) * tile / n
    X, Y = np.meshgrid(*g)
    return X.ravel(), Y.ravel()


class Exact(unittest.TestCase):
    def test_tiles_along_and_off_the_road_are_unchanged(self):
        c = winding()
        cl, tile = c.centreline, 16.0
        keys = {(int(np.floor(cl[k, 0] / tile)) + di, int(np.floor(cl[k, 1] / tile)) + dj)
                for k in range(0, len(cl), 150) for di, dj in ((0, 0), (1, 0), (0, -1), (3, 3))}
        for i, j in sorted(keys):
            x, y = grid(c, i, j)
            got, want = c.uv(x, y), full(c, x, y)
            for g, w in zip(got, want):
                np.testing.assert_array_equal(g, w)

    def test_a_point_equidistant_from_two_segments_takes_the_first_as_before(self):
        # a hairpin: the two legs 4 m apart, the points on the line between them
        n = 200
        leg = np.arange(n) * 0.5
        line = np.concatenate([np.stack([leg, np.zeros(n)], 1),
                               np.stack([leg[::-1], np.full(n, 4.0)], 1)])
        c = rd.Corridor(np.concatenate([line, np.zeros((len(line), 1))], axis=1))
        x, y = np.linspace(10.0, 80.0, 50), np.full(50, 2.0)
        got, want = c.uv(x, y), full(c, x, y)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)
        self.assertTrue((got[0] < c.length / 2).all())  # the first leg's, as the loop took it

    def test_few_segments_are_kept_for_a_tile(self):
        c = winding()
        x, y = grid(c, int(c.centreline[1500, 0] // 16), int(c.centreline[1500, 1] // 16))
        kept = c._candidates(np.stack([x, y], axis=1))
        self.assertLess(len(kept), 200)
        self.assertGreater(len(kept), 0)


class Cost(unittest.TestCase):
    def test_a_tile_of_a_long_record_is_fast(self):
        """Haruna's tile (321 x 321 points, 3054 segments) took 7.0 s before the fix."""
        c = winding(3055)
        k = 1500
        i, j = int(c.centreline[k, 0] // 16), int(c.centreline[k, 1] // 16)
        x, y = grid(c, i, j, n=320)
        t = time.perf_counter()
        c.uv(x, y)
        self.assertLess(time.perf_counter() - t, 1.0)


if __name__ == "__main__":
    unittest.main()
