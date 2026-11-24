"""Issue 140: each detail collision tile is made with an authored contact offset of
TILE_CONTACT_OFFSET (0.02 m) and a rest offset of 0 (with a stand-in for pxr); no Isaac. Left to
PhysX's default, a tile's reach to the skid plate was 0.33 m, and the solver took a 29 deg face
8.3 cm under the plate for a strike."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import rock_layer  # noqa: E402


class TileOffset(unittest.TestCase):
    def make(self, offset):
        n = 320
        d = SimpleNamespace(n=n, spacing=0.05, tile=16.0,
                            tiles_near=lambda xs, ys, reach: {(0, 0)},
                            touches=lambda i, j: True, ready=lambda k: True,
                            surface_tile=lambda i, j: np.zeros((n + 1, n + 1)))
        c = rock_layer.DetailColliders.__new__(rock_layer.DetailColliders)
        c.stage = mock.MagicMock()
        c.layer = SimpleNamespace(detail=d, ox=0.0, oy=0.0)
        c.group = "/World/vehicleCollisionGroups/roadDetail"
        c.n_c = n // rock_layer.COLLISION_STEP
        c.coll_idx = c.coll_cnt = None
        c.coll, c.coll_ms, c.serial, c.adds, c.drops = {}, [], 0, set(), set()
        pxr = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"pxr": pxr}), \
                mock.patch.object(rock_layer, "TILE_CONTACT_OFFSET", offset):
            c.update([(1.0, 1.0)])
        return c, pxr.PhysxSchema.PhysxCollisionAPI

    def test_a_tile_gets_its_offsets(self):
        c, api = self.make(rock_layer.TILE_CONTACT_OFFSET)
        self.assertEqual(rock_layer.TILE_CONTACT_OFFSET, 0.02)
        self.assertEqual(len(c.coll), 1)
        api.Apply.assert_called_once()
        col = api.Apply.return_value
        col.CreateContactOffsetAttr.assert_called_once_with(0.02)
        col.CreateRestOffsetAttr.assert_called_once_with(0.0)

    def test_none_leaves_physx_its_own(self):
        c, api = self.make(None)
        self.assertEqual(len(c.coll), 1)
        api.Apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
