"""Issue 034's rule across loads (issue 152's record): every collider path the road makes is new
in the session, not only in its stream. Each load makes a new stream; one that counted from 1
again remade the preload's 768-prim rock pool at the preload's paths, and omni.physx took each
for a collider in two collision groups. No Isaac."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import rock_layer  # noqa: E402

G = "/World/vehicleCollisionGroups"


def tiles(pxr):
    """A DetailColliders with only what update uses, one tile in reach, as a new load makes."""
    dc = rock_layer.DetailColliders.__new__(rock_layer.DetailColliders)
    dc.stage, dc.group, dc.coll, dc.coll_ms, dc.serial = object(), f"{G}/roadDetail", {}, [], 0
    dc.n_c = 2
    dc.coll_idx = dc.coll_cnt = None
    dc.layer = SimpleNamespace(ox=0.0, oy=0.0, detail=SimpleNamespace(
        tiles_near=lambda xs, ys, r: [(3, 4)], touches=lambda i, j: True,
        ready=lambda k: True, surface_tile=lambda i, j: np.zeros((5, 5)), spacing=1.0, tile=4.0))
    dc.adds, dc.drops = set(), set()
    return dc


class SessionPaths(unittest.TestCase):
    def test_the_serial_never_repeats(self):
        a = [rock_layer.next_serial() for _ in range(5)]
        self.assertEqual(a, sorted(set(a)))

    def test_two_loads_make_the_same_tile_at_two_paths(self):
        rel = SimpleNamespace(t=[])
        rel.GetTargets = lambda: list(rel.t)
        rel.SetTargets = lambda ts: setattr(rel, "t", list(ts))
        pxr = mock.MagicMock()
        pxr.Sdf.Path.side_effect = str
        pxr.UsdPhysics.CollisionGroup.Get.return_value.GetCollidersCollectionAPI.return_value \
            .GetIncludesRel.return_value = rel
        paths = []
        with mock.patch.dict(sys.modules, {"pxr": pxr}), \
                mock.patch.object(rock_layer, "_mesh", return_value=mock.MagicMock()):
            for _ in range(2):  # the preload's stream, then the scenario's
                dc = tiles(pxr)
                dc.update([(0.0, 0.0, 0.0)])
                paths.append(dc.coll[(3, 4)])
                rel.t = []
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(p.startswith(f"{rock_layer.COLL_ROOT}/t_3_4_") for p in paths))

    def test_every_collider_path_takes_the_session_serial(self):
        src = open(os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world",
                                "rock_layer.py")).read()
        self.assertNotIn("self.serial += 1", src)
        self.assertEqual(src.count("self.serial = next_serial()"), 3)


if __name__ == "__main__":
    unittest.main()
