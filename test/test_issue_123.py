"""Issue 123: the road's collision groups made before the simulation starts, by the world as
by the rock stream (with a stand-in for pxr), and the rock pool's prims joining their group
before the next physics step (issue 145); no Isaac."""
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
    def test_the_pool_joins_its_group_before_reserve_returns(self):
        """Issue 145: the pool's prims are written into `rocks` before the next physics step;
        joining at the next frame's end (issue 123's way, against its 769 warnings) left them
        unfiltered, since PhysX fixes a collider's filtering when it takes it in."""
        from scensim_isaac_world import road_detail as rd

        c = rd.Corridor(np.array([[0.0, 0.0], [2000.0, 0.0]]), seed=5,
                        roughness=rd.Roughness(cls="C"), rocks=rk.Zones(carriageway=0.01))
        s = rock_layer.RockStream.__new__(rock_layer.RockStream)
        s.layer = SimpleNamespace(detail=rd.Detail([c]))
        s.pool, s.reserved, s.reserved_for, s.prewarm_ms = {}, {}, 0, []
        s.made, s.prewarmed, s.serial, s.pending = 0, 0, 0, {}
        s.rock_g = f"{G}/rocks"
        written = []

        def new_prim(r, pos, quat):
            s.serial += 1
            s.made += 1
            s._group(s.rock_g, f"/p{s.serial}", True)
            return f"/p{s.serial}"

        def flush():
            written.extend(s.pending.get(s.rock_g, (set(), set()))[0])
            s.pending.clear()

        s._new_prim = new_prim
        s._park = lambda *a: None
        s._flush_groups = flush
        with mock.patch.object(rock_layer, "INTAKE_STRICT", True):
            n = s.prewarm()
        self.assertGreater(n, 0)
        self.assertEqual(len(written), n)  # all of them, before prewarm returned
        self.assertEqual(s._unjoined(), {})


class BaseCheck(unittest.TestCase):
    """The load-time self-check: a rock on the base road within its first second awake."""

    def stream(self):
        s = rock_layer.RockStream.__new__(rock_layer.RockStream)
        s.base_paths = ("/World/env/Road",)
        s.base_check, s.base_checked, s.base_hits = {}, set(), []
        return s

    def header(self, a0, a1, c0, c1, lost=False):
        return SimpleNamespace(type="CONTACT_LOST" if lost else "CONTACT_FOUND", actor0=a0,
                               actor1=a1, collider0=c0, collider1=c1)

    def run_check(self, s, headers, dt=1 / 60):
        pxr = mock.MagicMock()
        pxr.PhysicsSchemaTools.intToSdfPath.side_effect = str  # the stand-in: paths as ints
        with mock.patch.dict(sys.modules, {"pxr": pxr}):
            return s.check_base(headers, dt)

    def test_a_rock_on_the_base_is_an_error_once(self):
        s = self.stream()
        s.base_check["/World/RoadRocks/r1"] = 1.0
        h = self.header("/World/RoadRocks/r1", "/World/env/Road", "/World/RoadRocks/r1",
                        "/World/env/Road/mesh_3")
        with self.assertLogs("scensim_isaac_world", "ERROR") as cm:
            hits = self.run_check(s, [h])
        self.assertEqual(hits, [("/World/RoadRocks/r1", "/World/env/Road/mesh_3")])
        self.assertIn("issue 123", cm.output[0])
        self.assertEqual(self.run_check(s, [h]), [])  # checked once, not every step
        self.assertEqual(len(s.base_hits), 1)

    def test_its_tile_and_a_lost_contact_are_not(self):
        s = self.stream()
        s.base_check["/World/RoadRocks/r1"] = 1.0
        tile = self.header("/World/RoadCollision/t_1_0_0", "/World/RoadRocks/r1",
                           "/World/RoadCollision/t_1_0_0", "/World/RoadRocks/r1")
        lost = self.header("/World/RoadRocks/r1", "/World/env/Road", "/World/RoadRocks/r1",
                           "/World/env/Road", lost=True)
        roadside = self.header("/World/RoadRocks/r1", "/World/env/RoadSigns",
                               "/World/RoadRocks/r1", "/World/env/RoadSigns/s")  # not a child
        self.assertEqual(self.run_check(s, [tile, lost, roadside]), [])

    def test_the_check_ends_after_a_second_awake(self):
        s = self.stream()
        s.base_check["/World/RoadRocks/r1"] = 1.0
        for _ in range(60):
            self.run_check(s, [])
        self.assertEqual(s.base_check, {})
        h = self.header("/World/RoadRocks/r1", "/World/env/Road", "/World/RoadRocks/r1",
                        "/World/env/Road")
        self.assertEqual(self.run_check(s, [h]), [])


if __name__ == "__main__":
    unittest.main()
