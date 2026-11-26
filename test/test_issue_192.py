"""Issue 192: a soil world's cut road was reported as the static world. The contact filter
(`contacts.rows`, which `IsaacWorld._collect_contacts` calls) on stub PhysX headers: once a soil
zone is built, the soil zones' own ground (`/World/soil`) is ground, whatever the impulse, and
every other row is as before. No Isaac."""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import contacts as C  # noqa: E402

EGO, NPC = "/World/ego", "/World/npc"
CUT = "/World/soil/sand_road/cut/m0"
ROCK = "/World/RoadRocks/r1"
GROUND = ["/World/env/Road", "/World/RoadCollision"]   # a world's ground_paths and the detail's
WITH_SOIL = [*GROUND, C.SOIL_ROOT]                     # what _make_tiles adds


class Report:
    """A frame's contact report: one header per pair, its points' impulses (N*s vectors)."""

    def __init__(self):
        self.ids, self.headers, self.data = {}, [], []

    def id(self, path):
        return self.ids.setdefault(path, len(self.ids) + 1)

    def pair(self, a, b, impulses, kind="CONTACT_PERSIST", separation=0.01):
        self.headers.append(SimpleNamespace(
            type=f"ContactEventType.{kind}", actor0=self.id(a), actor1=self.id(b),
            collider0=self.id(a), collider1=self.id(b), contact_data_offset=len(self.data),
            num_contact_data=len(impulses)))
        self.data += [SimpleNamespace(impulse=v, separation=separation) for v in impulses]
        return self

    def rows(self, ground, reporters=(1,), on_ground=None):
        by_path = {EGO: 1, NPC: 2}
        path_of = {v: k for k, v in self.ids.items()}.__getitem__
        return C.rows(self.headers, self.data, path_of, by_path, set(reporters),
                      lambda p: C.is_under(p, ground), on_ground)


class SoilGround(unittest.TestCase):
    def test_the_ego_on_its_cut_road_is_no_contact_at_rest_or_bottoming_out(self):
        for impulse in ((0.0, 0.0, 0.0), (0.0, 0.0, 500.0)):
            r = Report().pair(EGO, CUT, [impulse])
            self.assertEqual(r.rows(WITH_SOIL), [], impulse)

    def test_before_a_soil_zone_the_cut_path_reports_as_it_did(self):
        r = Report().pair(EGO, CUT, [(0.0, 0.0, 0.0)])
        self.assertEqual(r.rows(GROUND), [(1, 0, CUT, 0.0)])

    def test_a_rock_is_still_a_contact_its_impulses_summed(self):
        r = Report().pair(EGO, ROCK, [(3.0, 4.0, 0.0), (0.0, 0.0, 2.0)])
        self.assertEqual(r.rows(WITH_SOIL), [(1, 0, ROCK, 7.0)])

    def test_the_other_rules_are_unchanged(self):
        r = (Report().pair(EGO, "/World/env/Road/kerb", [(0, 0, 9.0)])        # ground path
             .pair(EGO, "/World/RoadCollision/t3", [(0, 0, 9.0)])             # detail tiles
             .pair(EGO, NPC, [(0, 0, 4.0)])                                   # two reporters
             .pair(EGO, ROCK, [(0, 0, 9.0)], kind="CONTACT_LOST"))            # lost
        self.assertEqual(sorted(r.rows(WITH_SOIL, reporters=(1, 2))),
                         [(1, 2, NPC, 4.0), (2, 1, EGO, 4.0)])

    def test_a_dropped_soil_pair_reaches_the_log_hook_with_its_separation(self):
        seen = []
        r = Report().pair(EGO, CUT, [(0, 0, 0.0), (0, 0, 0.0)], separation=0.018)
        r.data[1].separation = 0.012
        self.assertEqual(r.rows(WITH_SOIL, on_ground=lambda h, m, t: seen.append(
            (m, t, C.separation(h, r.data)))), [])
        self.assertEqual(seen, [(EGO, CUT, 0.012)])
        self.assertTrue(C.is_under("/World/soil", [C.SOIL_ROOT]))
        self.assertFalse(C.is_under("/World/soilmore/x", [C.SOIL_ROOT]))


if __name__ == "__main__":
    unittest.main()
