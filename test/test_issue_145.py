"""Issue 145: PhysX fixes a collider's filtering when it takes the collider in, so every collider
the rock stream and the detail tiles make is in its collision group before control returns to
the caller, which steps next. The self-check (`intake_check`) names any that is not: an error in
tests (INTAKE_STRICT), a WARN live. With a stand-in for pxr; no Isaac."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import rock_layer  # noqa: E402

G = "/World/vehicleCollisionGroups"


class Check(unittest.TestCase):
    def test_nothing_left_out_is_nothing(self):
        self.assertEqual(rock_layer.intake_check({}, "here", strict=True), 0)

    def test_strict_raises_naming_the_prim_and_its_group(self):
        with self.assertRaises(rock_layer.IntakeError) as e:
            rock_layer.intake_check({"/World/RoadRocks/r7": f"{G}/rocks"}, "RockStream.reserve",
                                    strict=True)
        self.assertIn("/World/RoadRocks/r7 (not in /World/vehicleCollisionGroups/rocks)",
                      str(e.exception))
        self.assertIn("RockStream.reserve", str(e.exception))

    def test_live_warns_once_a_collider(self):
        with self.assertLogs("scensim_isaac_world", level="WARNING") as cm:
            n = rock_layer.intake_check({"/a": "g1", "/b": "g2"}, "here", strict=False)
        self.assertEqual(n, 2)
        self.assertEqual(len(cm.output), 2)
        self.assertIn("/a is not in g1", cm.output[0])

    def test_the_module_switch_decides_when_not_given(self):
        with mock.patch.object(rock_layer, "INTAKE_STRICT", True):
            with self.assertRaises(rock_layer.IntakeError):
                rock_layer.intake_check({"/a": "g"}, "here")


def stream():
    """A RockStream with only what update's group handling uses."""
    s = rock_layer.RockStream.__new__(rock_layer.RockStream)
    s.pending, s.rock_g, s.static_g = {}, f"{G}/rocks", f"{G}/rockTiles"
    s.own_colliders = False
    s.layer = SimpleNamespace(ox=0.0, oy=0.0)
    s._activate = lambda points: None
    return s


class Stream(unittest.TestCase):
    def test_update_writes_what_it_made_before_returning(self):
        s = stream()
        written = []

        def flush():
            for g, (adds, _) in s.pending.items():
                written.extend((g, p) for p in adds)
            s.pending.clear()

        s._stream_rocks = lambda xs, ys, pts: (s._group(s.rock_g, "/r1", True),
                                               s._group(s.static_g, "/s_1", True))
        s._flush_groups = flush
        with mock.patch.object(rock_layer, "INTAKE_STRICT", True):
            s.update([(0.0, 0.0, 0.0)])
        self.assertEqual(sorted(written), [(s.static_g, "/s_1"), (s.rock_g, "/r1")])

    def test_a_flush_that_leaves_one_out_fails_loudly(self):
        s = stream()
        s._stream_rocks = lambda xs, ys, pts: s._group(s.rock_g, "/World/RoadRocks/r9", True)
        s._flush_groups = lambda: None  # a flush that writes nothing
        with mock.patch.object(rock_layer, "INTAKE_STRICT", True):
            with self.assertRaises(rock_layer.IntakeError) as e:
                s.update([(0.0, 0.0, 0.0)])
        self.assertIn("/World/RoadRocks/r9", str(e.exception))

    def test_a_drop_is_not_a_collider_left_out(self):
        s = stream()
        s._group(s.rock_g, "/r1", True)
        s._group(s.rock_g, "/r1", False)  # made and released in one frame
        self.assertEqual(s._unjoined(), {})


class Tiles(unittest.TestCase):
    def test_a_tile_not_in_its_group_after_the_edit_fails_loudly(self):
        """DetailColliders writes its tiles into `roadDetail` in the same update; the check reads
        the group back after the edit."""
        class Rel:
            def __init__(self, keep):
                self.t, self.keep = [], keep

            def GetTargets(self):
                return list(self.t)

            def SetTargets(self, ts):
                self.t = [t for t in ts if self.keep(str(t))]

        for keep, fails in ((lambda p: True, False), (lambda p: False, True)):
            rel = Rel(keep)
            pxr = mock.MagicMock()
            pxr.Sdf.Path.side_effect = str
            pxr.UsdPhysics.CollisionGroup.Get.return_value.GetCollidersCollectionAPI.return_value \
                .GetIncludesRel.return_value = rel
            dc = rock_layer.DetailColliders.__new__(rock_layer.DetailColliders)
            dc.stage, dc.group, dc.coll, dc.coll_ms, dc.serial = object(), f"{G}/roadDetail", {}, [], 0
            dc.n_c = 2
            dc.coll_idx = dc.coll_cnt = None
            dc.layer = SimpleNamespace(ox=0.0, oy=0.0, detail=SimpleNamespace(
                tiles_near=lambda xs, ys, r: [(0, 0)], touches=lambda i, j: True,
                ready=lambda k: True, surface_tile=lambda i, j: np.zeros((5, 5)),
                spacing=1.0, tile=4.0))
            dc.adds, dc.drops = set(), set()
            with mock.patch.dict(sys.modules, {"pxr": pxr}), \
                    mock.patch.object(rock_layer, "_mesh", return_value=mock.MagicMock()), \
                    mock.patch.object(rock_layer, "INTAKE_STRICT", True):
                if fails:
                    with self.assertRaises(rock_layer.IntakeError):
                        dc.update([(0.0, 0.0, 0.0)])
                else:
                    dc.update([(0.0, 0.0, 0.0)])
                    self.assertEqual(len(rel.t), 1)


if __name__ == "__main__":
    unittest.main()
