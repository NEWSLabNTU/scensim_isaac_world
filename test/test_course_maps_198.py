"""scensim issue 198: the course map's cloud holds the road the lidar sees, not the terrain
under the road ribbon. `maps.point_cloud` used to sample the terrain grid under the ribbon
too and keep the first point per 0.2 m voxel, the terrain's, 0.13 m below the road. No Isaac."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world.course import maps  # noqa: E402

ROAD_Z, TERRAIN_Z, HALF = -0.03, -0.16, 2.0


def world(terrain_z=TERRAIN_Z):
    """A flat terrain grid (1 m, x 0..40, y -10..10) and a flat ribbon (x 0..40, |y| < HALF) at
    ROAD_Z, the centreline along x; the meshes' dict as `course.geometry` gives it."""
    xs, ys = np.arange(0.0, 41.0), np.arange(-10.0, 11.0)
    gx, gy = np.meshgrid(xs, ys)
    gz = np.full_like(gx, terrain_z) if np.isscalar(terrain_z) else terrain_z(gx, gy)
    terrain = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    rx = np.arange(0.0, 40.5, 0.5)
    left = np.stack([rx, np.full_like(rx, HALF), np.full_like(rx, ROAD_Z)], axis=1)
    right = np.stack([rx, np.full_like(rx, -HALF), np.full_like(rx, ROAD_Z)], axis=1)
    road = np.concatenate([left, right])
    n = len(rx)
    faces = np.array([[i, i + 1, n + i + 1, n + i] for i in range(n - 1)])
    m = {"terrain_shape": np.array([len(ys), len(xs)]), "terrain_points": terrain,
         "road_points": road, "road_faces": faces}
    c = {"x": np.arange(0.0, 40.1, 0.1), "y": np.zeros(401)}
    return m, c


def cloud(m, c):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "pointcloud_map.pcd")
        maps.point_cloud(m, c, out)
        return maps.read_pcd(out).astype(np.float64)


class RoadNotTerrain(unittest.TestCase):
    def test_no_point_lies_under_the_ribbon(self):
        p = cloud(*world())
        under = (np.abs(p[:, 1]) < HALF) & (p[:, 0] > 0.0) & (p[:, 0] < 40.0)
        self.assertGreater(under.sum(), 1000)
        self.assertEqual(int((p[under, 2] < ROAD_Z - 0.02).sum()), 0)

    def test_the_roads_cells_are_at_the_ribbon(self):
        p = cloud(*world())
        inner = (np.abs(p[:, 1]) < HALF - 0.2) & (p[:, 0] > 0.2) & (p[:, 0] < 39.8)
        q = p[inner]
        cell = np.floor(q[:, :2] / 0.2).astype(np.int64)
        keys, inv = np.unique(cell, axis=0, return_inverse=True)
        top = np.full(len(keys), -9.0)
        np.maximum.at(top, inv.ravel(), q[:, 2])
        self.assertLess(float(np.abs(top - ROAD_Z).max()), 0.01)

    def test_the_terrain_beside_the_ribbon_is_kept(self):
        p = cloud(*world())
        beside = np.abs(p[:, 1]) > HALF + 0.5
        self.assertGreater(beside.sum(), 1000)
        self.assertLess(float(np.abs(p[beside, 2] - TERRAIN_Z).max()), 1e-5)

    def test_terrain_above_the_ribbon_is_kept(self):
        # a bank over the ribbon's edge (terrain above the road there) is what the lidar sees
        bank = lambda gx, gy: np.where(gy >= 1.0, 0.5, TERRAIN_Z)  # noqa: E731
        p = cloud(*world(bank))
        over = (p[:, 1] > 1.5) & (p[:, 1] < HALF) & (p[:, 0] > 1.0) & (p[:, 0] < 39.0) & (p[:, 2] > 0.2)
        self.assertGreater(int(over.sum()), 0)


if __name__ == "__main__":
    unittest.main()
