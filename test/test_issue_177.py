"""scensim issue 177: a fixture's road detail reaches the world server. The course builder's
record step writes the map's `road_detail.yaml` for a fixture with features (the course's
record in course/ stays as it was), re-writes its own, refuses anyone else's; the world server
names a course record whose features no map file carries. No Isaac."""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_layer  # noqa: E402
from scensim_isaac_world.course import fixture, record  # noqa: E402

# The builder's steps project with pyproj (centreline.Frame): the course is the system Python's
# work, not the world server's in Isaac's venv, which lacks it (issues 090, 149). The world
# server's side below builds its record without it.
NO_PYPROJ = None if importlib.util.find_spec("pyproj") else (
    "no pyproj in this Python: the course builder runs under the system Python, not Isaac's")

COURSES = os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world", "courses")


def course_path(name):
    return os.path.join(COURSES, name + ".yaml")


def build(map_dir, name="fixture_c6a", steps=("centreline", "record")):
    from scensim_isaac_world.course import __main__ as course_main

    with contextlib.redirect_stdout(io.StringIO()):
        return course_main.main([course_path(name), map_dir, *steps])


def recorded(map_dir, name="fixture_c6a", map_file=True):
    """A fixture's course record (and, with `map_file`, its map file) without the builder's
    projection: what the world server finds on disk."""
    with open(course_path(name)) as f:
        course = yaml.safe_load(f)
    work = os.path.join(map_dir, "course")
    os.makedirs(work, exist_ok=True)
    feats = course["fixture"].get("features")
    record.write(work, fixture.stations(course, fixture.centreline(course)), in_mesh=True,
                 features=feats)
    if map_file:
        record.write_map(map_dir, course["name"], feats)


def features_of(name):
    with open(course_path(name)) as f:
        return yaml.safe_load(f)["fixture"]["features"]


@unittest.skipIf(NO_PYPROJ, NO_PYPROJ or "")
class TheMapsRoadDetail(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.map = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_fixture_with_features_gets_the_maps_road_detail(self):
        self.assertEqual(build(self.map), 0)
        path = road_layer.find(self.map)
        self.assertTrue(os.path.exists(path), path)
        c, = road_layer.load(path)
        rec, = road_layer.load(os.path.join(self.map, "course", "road_detail.yaml"))
        want = features_of("fixture_c6a")
        self.assertAlmostEqual(c.length, 640.0, places=0)
        # the map's corridor is the record's: the same features everywhere on it
        u = np.linspace(0.0, 640.0, 6401)
        for v in (-2.425, -0.8, 0.0, 0.8, 2.425):
            np.testing.assert_array_equal(c.features(u, np.full(len(u), v)),
                                          rec.features(u, np.full(len(u), v)))
        s, v, depth = want["potholes"]["at"][0][:3]
        self.assertAlmostEqual(float(c.features(np.array([s + 0.1]), np.array([v]))[0]), -depth)
        with open(path) as f:
            entry = yaml.safe_load(f)["corridors"][0]
        self.assertEqual(entry["stations"], "course/section.npz")
        self.assertEqual(entry["cleats"], want["cleats"])
        self.assertTrue(entry["section_in_base"])

    def test_the_course_record_is_as_before(self):
        build(self.map)
        with open(os.path.join(self.map, "course", "road_detail.yaml")) as f:
            entry = yaml.safe_load(f)["corridors"][0]
        self.assertEqual(entry["stations"], "section.npz")
        self.assertEqual(entry["washboard"], features_of("fixture_c6a")["washboard"])

    def test_the_record_step_alone_keeps_the_features(self):
        build(self.map, steps=("centreline",))
        self.assertEqual(build(self.map, steps=("record",)), 0)
        with open(os.path.join(self.map, "course", "road_detail.yaml")) as f:
            entry = yaml.safe_load(f)["corridors"][0]
        self.assertEqual(entry["cleats"], features_of("fixture_c6a")["cleats"])
        self.assertTrue(os.path.exists(road_layer.find(self.map)))

    def test_a_fixture_without_features_writes_no_map_file(self):
        self.assertEqual(build(self.map, name="fixture_c5"), 0)
        self.assertFalse(os.path.exists(os.path.join(self.map, "road_detail.yaml")))

    def test_its_own_file_is_written_again(self):
        build(self.map)
        path = road_layer.find(self.map)
        with open(path, "a") as f:
            f.write("# stale\n")
        self.assertEqual(build(self.map, steps=("record",)), 0)
        with open(path) as f:
            self.assertNotIn("# stale", f.read())

    def test_a_file_without_the_header_is_refused(self):
        build(self.map, steps=("centreline",))
        path = os.path.join(self.map, "road_detail.yaml")
        mine = "corridors:\n  - centreline: [[0, 0], [10, 0]]\n"
        with open(path, "w") as f:
            f.write(mine)
        with self.assertRaises(SystemExit) as e:
            build(self.map, steps=("record",))
        self.assertIn(path, str(e.exception))
        with open(path) as f:
            self.assertEqual(f.read(), mine)

    def test_another_courses_file_is_refused(self):
        build(self.map, name="fixture_c6b")
        path = os.path.join(self.map, "road_detail.yaml")
        with open(path) as f:
            theirs = f.read()
        with self.assertRaises(SystemExit) as e:
            build(self.map, name="fixture_c6a", steps=("record",))
        self.assertIn("fixture_c6b", str(e.exception))
        with open(path) as f:
            self.assertEqual(f.read(), theirs)


class TheWorldServerSaysSo(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.map = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_unread_features_are_named(self):
        recorded(self.map, map_file=False)
        msg = road_layer.course_record_unread(self.map)
        self.assertIsNotNone(msg)
        self.assertIn(os.path.join(self.map, "course", "road_detail.yaml"), msg)
        self.assertIn("record", msg)

    def test_nothing_to_say_with_the_maps_file_or_no_features(self):
        recorded(self.map)
        self.assertTrue(os.path.exists(road_layer.find(self.map)))
        self.assertIsNone(road_layer.course_record_unread(self.map))
        with tempfile.TemporaryDirectory() as plain:
            self.assertIsNone(road_layer.course_record_unread(plain))
            recorded(plain, name="fixture_c5")
            self.assertIsNone(road_layer.course_record_unread(plain))

if __name__ == "__main__":
    unittest.main()
