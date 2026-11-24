"""086 C6b's budget: fixture_cost's cars follow a winding centreline (`Line`) or, as before,
the straight line from the course's first point to its last (`Straight`); no Isaac."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world.course import fixture_cost as fc  # noqa: E402
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


class GuardCourse(unittest.TestCase):
    """Issue 145: the guard's drive covers the whole course, not the first 400 frames (280 m of
    Haruna's 1527, before its first base contact at about 282 m)."""

    def test_a_course_knows_its_length(self):
        self.assertAlmostEqual(circle().length, math.pi * R, delta=0.5)
        self.assertAlmostEqual(Straight(np.array([0.0, 30.0]), np.array([0.0, 40.0])).length, 50.0)

    def test_the_guard_drives_to_the_end_of_harunas_course(self):
        for speed in (10.0, 13.9):
            n = fc.guard_frames(1527.0, fc.START, speed)
            way = fc.START + fc.warm_distance(speed) + n * fc.FRAME * speed
            self.assertGreaterEqual(way, 1527.0 - fc.END_MARGIN)
            # and not much beyond: the slack only
            self.assertLessEqual(way, 1527.0 - fc.END_MARGIN + fc.GUARD_SLACK_S * speed + speed)
        self.assertGreater(fc.guard_frames(1527.0, fc.START, fc.SPEED), 2 * fc.FRAMES)

    def test_a_short_course_keeps_the_timed_frames(self):
        self.assertEqual(fc.guard_frames(100.0, fc.START, fc.SPEED), fc.FRAMES)

    def test_the_warm_up_distance(self):
        self.assertAlmostEqual(fc.warm_distance(10.0), 20.0 + 10.0 * 2.0)  # 4 s ramp, 2 s held
        self.assertAlmostEqual(fc.warm_distance(20.0), 1.25 * 36.0)  # still ramping at 6 s


if __name__ == "__main__":
    unittest.main()
