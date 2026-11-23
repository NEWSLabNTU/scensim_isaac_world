"""Issue 123: the road's collision groups made before the simulation starts, by the world as
by the rock stream (with a stand-in for pxr), and the rock pool's prims joining their group at
a frame's end rather than as they are made; no Isaac."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from scensim_isaac_world import rock_layer  # noqa: E402
from scensim_isaac_world import rocks as rk  # noqa: E402

G = "/World/vehicleCollisionGroups"


class Rel:
    def __init__(self):
        self.targets = []

    def GetTargets(self):
        return list(self.targets)

    def AddTarget(self, t):
        self.targets.append(t)

    def SetTargets(self, ts):
        self.targets = list(ts)


class Group:
    def __init__(self, path):
        self.path, self.filtered, self.includes = path, Rel(), Rel()

    def GetPath(self):
        return self.path

    def CreateFilteredGroupsRel(self):
        return self.filtered

    def GetCollidersCollectionAPI(self):
        return SimpleNamespace(CreateIncludesRel=lambda: self.includes,
                               GetIncludesRel=lambda: self.includes)


def fake_pxr(groups, calls):
    def define(stage, path):
        calls.append(("define", path))
        return groups.setdefault(path, Group(path))

    pxr = mock.MagicMock()
    pxr.Sdf.Path.side_effect = str
    pxr.UsdPhysics.CollisionGroup.Define.side_effect = define
    pxr.UsdPhysics.CollisionGroup.Get.side_effect = lambda stage, path: groups[path]
    vehicle = mock.MagicMock()
    vehicle.ensure_vehicle_scene.side_effect = lambda stage: calls.append(("vehicle", None))
    return {"pxr": pxr, "aib": mock.MagicMock(), "aib.isaac": mock.MagicMock(),
            "aib.isaac.vehicle": vehicle}


class DefineGroups(unittest.TestCase):
    def test_the_groups_and_their_filters(self):
        groups, calls = {}, []
        with mock.patch.dict(sys.modules, fake_pxr(groups, calls)):
            rock_layer.define_groups(object())
            rock_layer.define_groups(object())  # again, as the rock stream does: no change
        self.assertEqual(calls[0], ("vehicle", None))  # the vehicles' groups first (issue 039)
        q, base = f"{G}/groundQuery", f"{G}/roadBase"
        self.assertEqual({p: g.filtered.targets for p, g in groups.items()},
                         {base: [], f"{G}/rocks": [q, base], f"{G}/rockTiles": [q, base],
                          f"{G}/roadDetail": [q]})
        self.assertTrue(all(g.includes.targets == [] for g in groups.values()))

    def test_the_stream_sets_the_base_members_on_the_groups_made_before(self):
        groups, calls = {}, []
        layer = SimpleNamespace(detail=SimpleNamespace(n=8))
        with mock.patch.dict(sys.modules, fake_pxr(groups, calls)):
            rock_layer.define_groups(object())
            made = len([c for c in calls if c[0] == "define"])
            rock_layer.RockStream(mock.MagicMock(), layer, None, ground=("/World/env/Road",),
                                  prewarm=False, colliders=SimpleNamespace(coll={}))
        self.assertEqual(groups[f"{G}/roadBase"].includes.targets, ["/World/env/Road"])
        self.assertEqual(groups[f"{G}/rocks"].filtered.targets,
                         [f"{G}/groundQuery", f"{G}/roadBase"])  # not filtered twice
        self.assertEqual(len(groups), 4)
        self.assertGreater(len(calls), made)  # defined again (idempotent), none new

    def test_the_world_makes_them_before_the_simulation_starts(self):
        with open(os.path.join(ROOT, "scensim_isaac_world", "world.py")) as f:
            src = f.read()
        self.assertLess(src.index("rock_layer.define_groups(self.stage)"),
                        src.index("self.world.reset()"))


class Pool(unittest.TestCase):
    def test_the_pool_joins_its_group_at_a_frames_end(self):
        """Made and grouped in one step, omni.physx warned for each pool prim (769 at a
        load); the pool's paths now wait in `pending` for the next frame's end."""
        from scensim_isaac_world import road_detail as rd

        c = rd.Corridor(np.array([[0.0, 0.0], [2000.0, 0.0]]), seed=5,
                        roughness=rd.Roughness(cls="C"), rocks=rk.Zones(carriageway=0.01))
        s = rock_layer.RockStream.__new__(rock_layer.RockStream)
        s.layer = SimpleNamespace(detail=rd.Detail([c]))
        s.pool, s.reserved, s.reserved_for, s.prewarm_ms = {}, {}, 0, []
        s.made, s.prewarmed, s.serial, s.pending = 0, 0, 0, {}
        s.rock_g = f"{G}/rocks"

        def new_prim(r, pos, quat):
            s.serial += 1
            s.made += 1
            s._group(s.rock_g, f"/p{s.serial}", True)
            return f"/p{s.serial}"

        s._new_prim = new_prim
        s._park = lambda *a: None
        s._flush_groups = mock.Mock()
        n = s.prewarm()
        self.assertGreater(n, 0)
        s._flush_groups.assert_not_called()
        self.assertEqual(len(s.pending[s.rock_g][0]), n)


if __name__ == "__main__":
    unittest.main()
