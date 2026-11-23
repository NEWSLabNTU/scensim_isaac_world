"""086 C5d: Autoware's maps from the record and the meshes (lanelet heights, the divided cloud);
no Isaac."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world.course import centreline as cl  # noqa: E402
from scensim_isaac_world.course import check, fixture, maps  # noqa: E402

COURSES = os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world", "courses")


def load(name):
    with open(os.path.join(COURSES, name)) as f:
        return yaml.safe_load(f)


class LaneletHeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = load("fixture_c5.yaml")
        cls.c = fixture.centreline(cls.course)
        cls.st = fixture.stations(cls.course, cls.c)
        cls.frame = cl.Frame(cls.course["origin"])

    def write(self, d, section):
        out = os.path.join(d, "lanelet2_map.osm")
        maps.lanelets(self.c, self.course, self.frame, out, 40.0, section)
        return check.osm_bounds(out)

    def test_a_bound_takes_the_records_height(self):
        sec = rd.StationSection(self.st)
        with tempfile.TemporaryDirectory() as d:
            ways = self.write(d, sec)
        cor = rd.Corridor(np.stack([self.c["x"], self.c["y"], self.c["z"]], 1))
        p = np.unique(np.concatenate(ways), axis=0)
        u, v, _ = cor.uv(p[:, 0], p[:, 1])
        # a bound is at +-half_width or 0; its 4-decimal plan position can put it a tenth of a
        # millimetre past the lip's step, so the record is read at the bound's own offset
        v = np.where(np.abs(v) > 3.0, np.sign(v) * 3.25, 0.0)
        want = np.interp(u, self.c["s"] - self.c["s"][0], self.c["z"]) + sec.height_uv(u, v)
        # ele has 4 decimals, and u read back from the plan position on the bend moves it
        self.assertLess(float(np.abs(p[:, 2] - want).max()), 1e-3)
        # on the 6 % bank (s 210-300, leaning left: the inside of the left bend low) the left
        # edge lies 0.195 m under the centreline, where the old lanelets put it level
        k = (u > 220) & (u < 290) & (v > 3.2)
        self.assertTrue(np.any(k))
        rise = p[k, 2] - np.interp(u[k], self.c["s"] - self.c["s"][0], self.c["z"])
        self.assertAlmostEqual(float(np.median(rise)), -3.25 * 0.06, delta=0.002)

    def test_a_level_record_gives_the_heights_of_before(self):
        flat = dict(self.course, fixture=dict(self.course["fixture"], section={
            "columns": ["s", "crown_left", "crown_right"], "rows": [[0.0, 0.0, 0.0],
                                                                    [510.0, 0.0, 0.0]]}))
        sec = rd.StationSection(fixture.stations(flat, self.c))
        with tempfile.TemporaryDirectory() as d:
            new = self.write(d, sec)
            old = self.write(d, None)
        for a, b in zip(new, old):
            np.testing.assert_array_equal(a, b)


class DividedCloud(unittest.TestCase):
    def cloud(self, n=5000, seed=3):
        rng = np.random.default_rng(seed)
        return (rng.random((n, 3)) * [350.0, 260.0, 5.0] - [120.0, 30.0, 0.0]).astype(np.float32)

    def test_over_the_limit_it_is_tiles_holding_every_point(self):
        pts = self.cloud()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "pointcloud_map.pcd")
            tiles = maps.write_cloud(pts, out, "c", divide=1000)
            self.assertTrue(os.path.isdir(out))
            meta = yaml.safe_load(open(os.path.join(d, maps.METADATA)))
            self.assertEqual(meta.pop("x_resolution"), maps.TILE)
            meta.pop("y_resolution")
            self.assertEqual(len(meta), tiles)
            got = np.concatenate([maps.read_pcd(os.path.join(out, f)) for f in sorted(meta)])
            key = lambda a: np.sort(a.view("V12").ravel())  # noqa: E731
            np.testing.assert_array_equal(key(got), key(pts))
            self.assertIn("c_-2_-1.pcd", meta)  # x from -120, y from -30: the cell's corner
            self.assertEqual(meta["c_-2_-1.pcd"], [-200, -100])
            os.makedirs(os.path.join(d, "course"))
            json.dump({"pcd_points": len(pts)}, open(os.path.join(d, "course", "maps.json"), "w"))
            res = check.check_cloud(d)
            self.assertTrue(res["pass"], res)
            # under the limit a rebuild writes one file, the tiles and metadata gone
            self.assertEqual(maps.write_cloud(pts, out, "c"), 0)
            self.assertTrue(os.path.isfile(out))
            self.assertFalse(os.path.exists(os.path.join(d, maps.METADATA)))
            np.testing.assert_array_equal(maps.read_pcd(out), pts)

    def test_a_tile_missing_from_the_metadata_fails(self):
        pts = self.cloud()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "pointcloud_map.pcd")
            maps.write_cloud(pts, out, "c", divide=1000)
            meta = os.path.join(d, maps.METADATA)
            lines = open(meta).read().splitlines()
            open(meta, "w").write("\n".join(lines[:-1]) + "\n")
            self.assertFalse(check.check_cloud(d)["pass"])


if __name__ == "__main__":
    unittest.main()
