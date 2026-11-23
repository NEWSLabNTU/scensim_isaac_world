"""086 C6b's budget: fixture_cost's cars follow a winding centreline (`Line`) or, as before,
the straight line from the course's first point to its last (`Straight`); no Isaac."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world.course.fixture_cost import Line, Straight  # noqa: E402

R = 60.0


def circle():
    """A left turn of radius R through 180 deg, at 0.5 m."""
    s = np.arange(0.0, math.pi * R, 0.5)
    t = s / R
    return Line(s, R * np.sin(t), R * (1 - np.cos(t)), 0.01 * s)


class Follow(unittest.TestCase):
    def test_the_lanes_are_offsets_across_the_heading(self):
        line = circle()
        for s in (10.0, 50.0, 120.0):
            centre = np.array([0.0, R])
            # left of a left turn is towards its centre
            self.assertAlmostEqual(np.linalg.norm(line.xy(s, 1.6) - centre), R - 1.6, delta=0.01)
            self.assertAlmostEqual(np.linalg.norm(line.xy(s, -1.6) - centre), R + 1.6, delta=0.01)
            self.assertAlmostEqual(line.heading(s), s / R, delta=0.01)
            self.assertAlmostEqual(line.z_at(s), 0.01 * s, delta=1e-9)

    def test_a_position_finds_its_s(self):
        line = circle()
        for s in (5.2, 77.7, 150.1):
            for v in (-1.6, 0.0, 1.6):
                self.assertAlmostEqual(line.s_of(line.xy(s, v)), s, delta=0.05)

    def test_the_straight_line_is_as_before(self):
        x, y = np.array([3.0, 7.0, 103.0]), np.array([4.0, 4.0, 4.0 + 100.0])
        st = Straight(x, y)
        h = math.atan2(100.0, 100.0)
        d, n = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
        np.testing.assert_allclose(st.xy(20.0, 1.6), np.array([3.0, 4.0]) + 20 * d + 1.6 * n)
        self.assertAlmostEqual(st.s_of(st.xy(20.0, 1.6)), 20.0)
        self.assertEqual(st.heading(5.0), h)


if __name__ == "__main__":
    unittest.main()
