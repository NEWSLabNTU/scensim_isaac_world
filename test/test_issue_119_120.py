"""Issues 119 and 120: the detail's collision tiles for every road layer, in a group the
wheels' ground query filters out (with a stand-in for pxr), and on a section_in_base corridor
the tiles' surface and the rocks' rests with the section in it; no Isaac."""
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_layer  # noqa: E402
from scensim_isaac_world import rocks as rk  # noqa: E402


def stations(n=3, length=200.0):
    u = np.linspace(0.0, length, n)
    side = {"crown": 0.02, "shoulder": 0.75, "foreslope": 1 / 3, "ditch_depth": 0.3,
            "ditch_bottom": 0.6, "backslope": 0.5, "lip_h": 0.0, "lip_angle": 90.0,
            "runoff_len": 0.0}
    return SimpleNamespace(u=u, common={"half_width": np.full(n, 3.25), "bank": np.zeros(n),
                                        "crown_offset": np.zeros(n)},
                           side={k: np.full((2, n), v) for k, v in side.items()})


def corridor(in_base, rocks=None):
    st = stations()
    return rd.Corridor(np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]]),
                       roughness=rd.Roughness(gd0=1e-12), stations=st,
                       section=rd.StationSection(st) if not in_base else rd.Section(),
                       section_in_base=in_base, rocks=rocks)


class SectionInBase(unittest.TestCase):
    """Issue 120."""

    def test_the_tiles_stand_on_the_true_surface(self):
        c = corridor(True)
        d = rd.Detail([c], workers=0)
        ss = rd.StationSection(c.stations)
        i, j = int(100 // d.tile), 0
        h = d.surface_tile(i, j)
        X, Y = np.meshgrid(i * d.tile + np.arange(d.n + 1) * d.spacing,
                           j * d.tile + np.arange(d.n + 1) * d.spacing)
        u, v, near = c.uv(X.ravel(), Y.ravel())
        true = c.base_z(u) + ss.height_uv(u, v) + c.features(u, v)
        err = np.abs(h.ravel() - true)[near]
        self.assertLess(float(err.max()), 1e-3)
        self.assertGreater(float(np.abs(ss.height_uv(u, v))[near].max()), 0.05)  # a real crown

    def test_a_corridor_without_it_is_unchanged(self):
        c = corridor(False)
        d = rd.Detail([c], workers=0)
        i, j = int(100 // d.tile), 0
        X, Y = np.meshgrid(i * d.tile + np.arange(d.n + 1) * d.spacing,
                           j * d.tile + np.arange(d.n + 1) * d.spacing)
        # as before the fix: base_z where the corridor reaches (all 0 here), plus the tile
        np.testing.assert_array_equal(d.surface_tile(i, j), d.tile_of(i, j))
        u, v, _ = c.uv(X.ravel(), Y.ravel())
        np.testing.assert_array_equal(c.surface_base(u, v), c.base_z(u))

    def test_a_rock_rests_on_the_true_surface(self):
        c = corridor(True, rocks=rk.Zones(carriageway=0.0, shoulder=0.05, runoff=0.0, ditch=0.1,
                                          embed=(0.0, 0.0)))
        d = rd.Detail([c], workers=0)
        ss = rd.StationSection(c.stations)
        n = 0
        for i in range(int(200 // d.tile)):
            for j in (-1, 0):
                for rock, p, yaw, loose in rk.place(d, i, j):
                    u, v, _ = c.uv(np.array([p[0]]), np.array([p[1]]))
                    true = float(c.base_z(u)[0] + ss.height_uv(u, v)[0]
                                 + d.height(np.array([p[0]]), np.array([p[1]]))[0])
                    self.assertAlmostEqual(p[2] + rock.verts[:, 2].min(), true, places=6)
                    n += 1
        self.assertGreater(n, 0)


class Colliders(unittest.TestCase):
    """Issue 119, with a stand-in for pxr: tiles near the car, in the roadDetail group that
    filters out the wheels' ground query; gone when the car leaves; no rocks needed."""

    def test_tiles_without_rocks_in_a_group_the_wheels_do_not_query(self):
        pxr = mock.MagicMock()
        pxr.Vt.IntArray.FromNumpy.side_effect = lambda a: a
        pxr.Sdf.Path.side_effect = str
        targets = []
        rel = pxr.UsdPhysics.CollisionGroup.Get.return_value.GetCollidersCollectionAPI \
            .return_value.GetIncludesRel.return_value
        rel.GetTargets.side_effect = lambda: list(targets)
        rel.SetTargets.side_effect = lambda t: targets.__setitem__(slice(None), list(t))
        vehicle = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"pxr": pxr, "aib": mock.MagicMock(),
                                           "aib.isaac": mock.MagicMock(),
                                           "aib.isaac.vehicle": vehicle}):
            from scensim_isaac_world import rock_layer

            c = corridor(True)
            layer = road_layer.RoadLayer([c], (0.0, 0.0, 0.0), None)
            self.assertIsNone(layer.rocks)  # no rocks on this road
            stage = mock.MagicMock()
            dc = rock_layer.DetailColliders(stage, layer)
            filt = pxr.UsdPhysics.CollisionGroup.Define.return_value.CreateFilteredGroupsRel \
                .return_value.AddTarget
            filt.assert_called_with("/World/vehicleCollisionGroups/groundQuery")
            self.assertEqual(dc.group, "/World/vehicleCollisionGroups/roadDetail")
            # the layer's contact asks for the tiles ahead (`before_substep`), made in the tile
            # worker; a tile not made yet waits for a later frame
            d = layer.detail
            keys = {k for k in d.tiles_near([100.0], [1.6], rock_layer.COLLISION_REACH)
                    if d.touches(*k)}
            d.prefetch(keys)
            for _ in range(600):
                if all(d.ready(k) for k in keys):
                    break
                time.sleep(0.05)
            dc.update([(100.0, 1.6, 0.0)])
            self.assertEqual(set(dc.coll), keys)
            self.assertEqual(sorted(targets), sorted(dc.coll.values()))
            self.assertTrue(all(p.startswith(rock_layer.COLL_ROOT) for p in targets))
            dc.update([])  # the car gone: no tiles, none in the group
            self.assertEqual(dc.coll, {})
            self.assertEqual(targets, [])
            layer.close()


if __name__ == "__main__":
    unittest.main()
