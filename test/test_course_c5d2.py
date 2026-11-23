"""086 C5d2: the fixture with guardrails and trees (the record's barriers, a forest photo); no
Isaac."""
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world.course import fixture, props  # noqa: E402

COURSES = os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world", "courses")


def load(name):
    with open(os.path.join(COURSES, name)) as f:
        return yaml.safe_load(f)


class FixtureC5d2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = load("fixture_c5d2.yaml")
        cls.c = fixture.centreline(cls.course)
        cls.st = fixture.stations(cls.course, cls.c)

    def test_it_is_c5bs_road(self):
        base = load("fixture_c5.yaml")
        st0 = fixture.stations(base, fixture.centreline(base))
        for k, v in st0.common.items():
            np.testing.assert_array_equal(self.st.common[k], v, err_msg=k)
        for k, v in st0.side.items():
            if k != "barrier_offset":
                np.testing.assert_array_equal(self.st.side[k], v, err_msg=k)
        self.assertTrue(np.all(np.isnan(st0.side["barrier_offset"])))

    def test_a_guardrail_each_side_past_the_shoulder(self):
        b = self.st.side["barrier_offset"]
        np.testing.assert_array_equal(b, np.full(b.shape, 4.5))  # 3.25 + 0.75 + 0.5
        segs = props.guardrails(self.c, self.st, 3.0)
        # one run each side (left first), along the whole course
        self.assertEqual(len(segs) % 2, 0)
        n = len(segs) // 2
        # the left rail is on the inside of the left bend (2.5 rad): 4.5 m x 2.5 shorter,
        # the right that much longer; both together twice the course, less a segment each
        turn = float(self.c["heading"][-1] - self.c["heading"][0])
        left, right = sum(s[4] for s in segs[:n]), sum(s[4] for s in segs[n:])
        self.assertAlmostEqual(right - left, 2 * 4.5 * turn, delta=0.5)
        self.assertAlmostEqual(left + right, 2 * 510.0, delta=6.0)
        # the first segment of each side starts 4.5 m off the centreline's first point
        for k, side in ((0, 1.0), (n, -1.0)):
            x, y = segs[k][0] - self.c["x"][0], segs[k][1] - self.c["y"][0]
            self.assertAlmostEqual(y, side * 4.5, places=6)
            self.assertAlmostEqual(x, 0.0, places=6)

    def test_the_forest_photo_grows_trees(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            for colour, want in (("forest", True), ("grey", False)):
                out = os.path.join(d, f"{colour}.jpg")
                fixture.photo(out, colour=colour)
                r, g, b = (float(v) for v in np.asarray(Image.open(out).convert("RGB"))[0, 0])
                forest = g > 1.05 * r and g > b and (r + g + b) / 3 < 140  # props.trees' rule
                self.assertEqual(forest, want, colour)
            with self.assertRaises(ValueError):
                fixture.photo(os.path.join(d, "x.jpg"), colour="lava")


if __name__ == "__main__":
    unittest.main()
