"""The road record: a corridor's section as constants or per station (roadmap 086 C1)."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world import road_layer as rl  # noqa: E402
from scensim_isaac_world import road_record as rr  # noqa: E402

LINE = [[0.0, 0.0], [120.0, 0.0]]


def stations(n=121, **over):
    """A constant record over 0..n-1 m, with `over` (field: array) on top."""
    st, problems = rr.from_constants({})
    assert not problems
    u = np.arange(n, dtype=float)
    common = {k: np.repeat(v, n) for k, v in st.common.items()}
    side = {k: np.repeat(v, n, axis=1) for k, v in st.side.items()}
    text = {k: np.repeat(v, n, axis=1) for k, v in st.text.items()}
    for k, v in over.items():
        (common if k in common else side if k in side else text)[k] = np.asarray(v)
    return rr.Stations(u, common, side, text)


class Constants(unittest.TestCase):
    def test_no_section_is_todays_default_section(self):
        st, problems = rr.from_constants(None)
        self.assertEqual(problems, [])
        section, why = rr.detail_section(st, rd.Section)
        self.assertEqual((section, why), (rd.Section(), ""))

    def test_todays_keys_give_todays_section(self):
        given = {"half_width": 3.0, "crown": 0.03, "shoulder": 0.5}
        st, problems = rr.from_constants(given)
        self.assertEqual(problems, [])
        self.assertEqual(rr.detail_section(st, rd.Section)[0], rd.Section(**given))

    def test_a_side_takes_a_number_or_left_and_right(self):
        st, problems = rr.from_constants({"crown": {"left": 0.02, "right": 0.05},
                                          "hazard": {"left": "ditch", "right": "drop"},
                                          "barrier_offset": {"right": 4.2}})
        self.assertEqual(problems, [])
        np.testing.assert_array_equal(st.side["crown"][:, 0], [0.02, 0.05])
        self.assertEqual(list(st.text["hazard"][:, 0]), ["ditch", "drop"])
        self.assertTrue(math.isnan(st.side["barrier_offset"][0, 0]))
        self.assertEqual(st.side["barrier_offset"][1, 0], 4.2)
        section, why = rr.detail_section(st, rd.Section)
        self.assertIsNone(section)
        self.assertIn("crown changes along the road or between the sides", why)

    def test_every_problem_is_listed(self):
        st, problems = rr.from_constants({"half_width": -1, "bank": 0.5, "hazard": "cliff",
                                          "crown": {"middle": 0.1}, "tilt": 3,
                                          "lip_angle": "steep"}, "corridors[0].section")
        text = "\n".join(problems)
        for want in ("unknown ['tilt']", "half_width -1.0 out of range",
                     "bank 0.5 out of range", "hazard 'cliff' is not one of",
                     "crown: unknown ['middle'] (left, right)", "lip_angle: a number"):
            self.assertIn(want, text)
        self.assertTrue(all(p.startswith("corridors[0].section") for p in problems))

    def test_keys_name_every_field_with_its_default(self):
        k = rr.keys()
        self.assertEqual(k["half_width"][0], 2.5)
        self.assertIn("hazard.left|right", k)
        self.assertIn("corridors[].section.*", rl.keys())
        self.assertIn("lip_h.left|right", rl.keys()["corridors[].section.*"])


class Stations(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write(self, st, name="road.npz"):
        rr.save_stations(os.path.join(self.dir, name), st)
        return name

    def corridor(self, **c):
        return rl.parse({"corridors": [dict({"centreline": LINE, "seed": 4}, **c)]},
                        base=self.dir)

    def test_a_constant_record_is_the_same_road_as_its_constants(self):
        name = self.write(stations())
        a, = self.corridor(stations=name)
        b, = self.corridor()
        self.assertEqual(a.section, b.section)
        self.assertEqual(len(a.stations), 121)
        da, db = rd.Detail([a]), rd.Detail([b])
        for key in ((0, 0), (1, -1), (3, 0)):
            self.assertEqual(da.surface_tile(*key).tobytes(), db.surface_tile(*key).tobytes())
            self.assertEqual(da.zones_tile(*key).tobytes(), db.zones_tile(*key).tobytes())

    def test_a_record_round_trips(self):
        st = stations(kappa_v=np.linspace(0, 0.01, 121),
                      hazard=np.array([["ditch"] * 121, ["barrier"] * 121]))
        loaded, problems = rr.load_stations(os.path.join(self.dir, self.write(st)), 120.0)
        self.assertEqual(problems, [])
        np.testing.assert_array_equal(loaded.common["kappa_v"], st.common["kappa_v"])
        self.assertEqual(loaded.text["hazard"][1, 5], "barrier")

    def test_missing_arrays_take_their_defaults(self):
        np.savez(os.path.join(self.dir, "few.npz"), u=np.arange(5.0))
        st, problems = rr.load_stations(os.path.join(self.dir, "few.npz"))
        self.assertEqual(problems, [])
        np.testing.assert_array_equal(st.side["crown"], np.full((2, 5), 0.04))
        self.assertEqual(st.text["hazard"].shape, (2, 5))

    def test_a_bad_record_lists_every_problem(self):
        np.savez(os.path.join(self.dir, "bad.npz"), u=np.array([0.0, 2.0, 1.0, 300.0]),
                 half_width=np.array([2.0, 2.0, 0.0, 2.0]), crown_left=np.zeros(3),
                 hazard_right=np.array(["drop", "lava", "drop", "drop"]), tilt=np.zeros(4))
        st, problems = rr.load_stations(os.path.join(self.dir, "bad.npz"), 120.0, "bad")
        text = "\n".join(problems)
        for want in ("unknown array(s) ['tilt']", "u must start at 0 and increase",
                     "the last station, u 300, lies past the centreline's end (120 m)",
                     "half_width 0.0 out of range", "at station 2",
                     "crown_left has shape (3,), u has 4 stations", "hazard 'lava'"):
            self.assertIn(want, text)

    def test_a_record_the_road_layer_cannot_shape_yet_is_refused_by_the_world_server(self):
        bank = np.zeros(121)
        bank[60:] = 0.05
        name = self.write(stations(bank=bank, lip_h=np.full((2, 121), 0.08)))
        with self.assertRaises(rl.RoadDetailError) as e:
            self.corridor(stations=name)
        self.assertIn("roadmap 086 C5 (it is banked; it has a lip)", str(e.exception))
        notes = []
        c, = rl.parse({"corridors": [{"centreline": LINE, "stations": name}]},
                      base=self.dir, notes=notes)
        self.assertEqual(c.section, rd.Section())
        self.assertIn("first station's left side", notes[0])

    def test_section_or_stations_not_both_and_stations_is_an_npz(self):
        with self.assertRaises(rl.RoadDetailError) as e:
            rl.parse({"corridors": [{"centreline": LINE, "section": {}, "stations": "a.npz"},
                                    {"centreline": LINE, "stations": "a.csv"},
                                    {"centreline": LINE, "stations": "gone.npz"}]},
                     base=self.dir)
        text = str(e.exception)
        self.assertIn("corridors[0]: section or stations, not both", text)
        self.assertIn("corridors[1].stations: an .npz file's name", text)
        self.assertIn("corridors[2].stations (gone.npz): cannot read", text)

    def test_load_finds_the_stations_beside_the_yaml(self):
        name = self.write(stations())
        path = os.path.join(self.dir, rl.FILE)
        with open(path, "w") as f:
            f.write(f"schema: 1\ncorridors:\n  - centreline: {LINE}\n    stations: {name}\n")
        c, = rl.load(path)
        self.assertEqual(len(c.stations), 121)


class Schema(unittest.TestCase):
    def test_absent_is_one_and_newer_is_refused(self):
        self.assertTrue(rl.parse({"corridors": [{"centreline": LINE}]}))
        self.assertTrue(rl.parse({"schema": 1, "corridors": [{"centreline": LINE}]}))
        for bad, want in ((2, "schema 2 is newer than this reader's (1)"),
                          ("1", "schema must be a whole number"),
                          (True, "schema must be a whole number")):
            with self.assertRaises(rl.RoadDetailError) as e:
                rl.parse({"schema": bad, "corridors": []})
            self.assertIn(want, str(e.exception))


if __name__ == "__main__":
    unittest.main()
