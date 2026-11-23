"""Issue 137: a wheel off the ground at a refit of the base plane keeps its last hit, not the
old plane's intercept at the world's origin; no Isaac."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_layer  # noqa: E402

RL = 2  # the rear left, in road_layer.ORDER


def hit(x, y, z):
    return SimpleNamespace(on_ground=True, position=(x, y, z))


def off():
    return SimpleNamespace(on_ground=False, position=(0.0, 0.0, 0.0))


def corners(cx, cy):
    """The four wheels' (x, y) of a car at (cx, cy) heading +y, in road_layer.ORDER: front
    left, front right, rear left, rear right (track 1.65 m, wheelbase 2.9 m)."""
    return [(cx - 0.83, cy + 2.9), (cx + 0.83, cy + 2.9), (cx - 0.83, cy), (cx + 0.83, cy)]


def car():
    c = road_layer._Car.__new__(road_layer._Car)
    c.v = SimpleNamespace(contacts=None)
    c.plane = None
    c.last_hit = [None] * road_layer.WHEELS
    return c


def base(plane, k, x, y):
    a, b, x0, y0, cs = plane
    return a * (x - x0) + b * (y - y0) + cs[k]


def today(hits, k, x, y):
    """Before issue 137, on the ground (every wheel): z = a x + b y + c_k at the origin."""
    xs, ys, zs = (np.array([h.position[i] for h in hits]) for i in range(3))
    a, b = road_layer._fit_slope(xs, ys, zs)
    h = hits[k].position
    return a * x + b * y + (h[2] - a * h[0] - b * h[1])


def refit(cx, cy, b1=-0.0785, b2=-0.0756, z0=-82.0):
    """Four wheels on a road of slope b1 along y, then the car 0.3 m on and three on a
    road of slope b2 (the rear left off the ground), as at issue 137's step 1039."""
    c = car()
    p1 = corners(cx, cy)
    first = [hit(x, y, z0 + b1 * y) for x, y in p1]
    c.v.contacts = first
    c._planes()
    p2 = corners(cx, cy + 0.3)
    zc = z0 + b1 * (cy + 0.3)  # the new road through the car's own height there
    second = [hit(x, y, zc + b2 * (y - cy - 0.3)) for x, y in p2]
    second[RL] = off()
    c.v.contacts = second
    c._planes()
    return c, first, p1


class OffGroundAtARefit(unittest.TestCase):
    def test_far_from_the_origin_the_wheel_keeps_its_hit(self):
        """349 m out (issue 137's place): the rear left's base at its last hit is that hit's
        height under the new slopes; the old intercept put it 1.01 m higher."""
        c, first, p1 = refit(80.8, 349.3)
        x, y, z = first[RL].position
        self.assertAlmostEqual(base(c.plane, RL, x, y), z, delta=1e-9)
        # what the old intercept under the new slopes gave there: the throw's 1 m
        a, b = c.plane[0], c.plane[1]
        stale = a * x + b * y + (z - (-0.0785) * y)  # the old c: z - b1 y at the origin
        self.assertGreater(abs(stale - z), 0.9)

    def test_at_the_origin_it_is_unchanged(self):
        """The same with the rear left's last hit at the origin: the old intercept was its
        hit, so both give the same base."""
        c, first, p1 = refit(0.83, 0.0)
        x, y, z = first[RL].position
        self.assertEqual((x, y), (0.0, 0.0))
        a, b = c.plane[0], c.plane[1]
        old = a * x + b * y + (z - (-0.0785) * y)
        for px, py in ((x, y), (x + 1.0, y - 2.0)):
            self.assertAlmostEqual(base(c.plane, RL, px, py), old + a * (px - x) + b * (py - y),
                                   delta=1e-12)

    def test_all_four_on_the_ground_is_todays_plane(self):
        """Four hits 350 m out on an uneven road: each wheel's base is today's (to float
        rounding: about the hits' centre, not the origin), and passes through its hit."""
        rng = np.random.default_rng(137)
        c = car()
        hits = [hit(x, y, -82.0 - 0.0785 * y + rng.normal(0, 0.02))
                for x, y in corners(80.8, 349.3)]
        c.v.contacts = hits
        c._planes()
        for k, h in enumerate(hits):
            x, y, z = h.position
            self.assertAlmostEqual(base(c.plane, k, x, y), z, delta=1e-9)
            for dx, dy in ((0.0, 0.0), (0.4, -0.7), (-1.2, 2.5)):
                self.assertAlmostEqual(base(c.plane, k, x + dx, y + dy),
                                       today(hits, k, x + dx, y + dy), delta=1e-9)

    def test_fewer_than_three_keeps_the_plane_and_records_the_hits(self):
        c, first, p1 = refit(80.8, 349.3)
        kept = c.plane
        c.v.contacts = [hit(1.0, 2.0, 3.0), off(), off(), off()]
        self.assertIs(c._planes(), kept)
        self.assertEqual(c.last_hit[0], (1.0, 2.0, 3.0))

    def test_a_wheel_never_on_the_ground_takes_the_hits_mean(self):
        c = car()
        c.v.contacts = [hit(x, y, 5.0) for x, y in corners(0.0, 0.0)]
        c.v.contacts[RL] = off()
        c._planes()
        self.assertAlmostEqual(base(c.plane, RL, -0.83, 0.0), 5.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
