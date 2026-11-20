"""A course's scenario names the stack's vehicle catalog and its map by `$(env SCENSIM_MAPS)`, and
the `scenario` step reruns alone from a published course's saved centreline (scensim issue 087);
synthetic, no network."""
import json
import math
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scensim_isaac_world.course import __main__ as course_main  # noqa: E402
from scensim_isaac_world.course import scenario  # noqa: E402

COURSE = {"name": "bend", "road": {"lane_width": 3.25}}


def centreline():
    """A 200 m straight, then a 90 deg left bend of 20 m radius, every 0.5 m."""
    s = np.arange(0.0, 200.0 + 20.0 * math.pi / 2, 0.5)
    x, y, h, k = [], [], [], []
    for d in s:
        if d <= 200.0:
            x.append(d), y.append(0.0), h.append(0.0), k.append(0.0)
        else:
            a = (d - 200.0) / 20.0
            x.append(200.0 + 20.0 * math.sin(a)), y.append(20.0 - 20.0 * math.cos(a))
            h.append(a), k.append(1.0 / 20.0)
    return {"s": s, "x": np.array(x), "y": np.array(y), "z": np.zeros(len(s)),
            "heading": np.array(h), "curvature": np.array(k)}


HAIRPINS = [{"s": 215.0, "radius": 20.0, "turn_deg": 90.0, "direction": "left"}]


class CourseScenario(unittest.TestCase):
    def test_the_scenario_names_the_stacks_catalog_and_its_map_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_dir = os.path.join(tmp, "bend")
            os.makedirs(os.path.join(map_dir, "course"))
            out = scenario.build(centreline(), HAIRPINS, COURSE, map_dir)
            root = ET.parse(out["scenario"]).getroot()
            self.assertEqual(root.find("CatalogLocations/VehicleCatalog/Directory").get("path"),
                             "$(env SCENSIM_VEHICLES)")
            self.assertEqual(root.find("RoadNetwork/LogicFile").get("filepath"),
                             "$(env SCENSIM_MAPS)/bend")
            self.assertNotIn(tmp, Path(out["scenario"]).read_text())  # no absolute path
            self.assertEqual(out["drift_points"], 1)

    def test_the_scenario_step_reruns_from_the_saved_centreline(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_dir = os.path.join(tmp, "bend")
            work = os.path.join(map_dir, "course")
            os.makedirs(work)
            c = centreline()
            np.savez(os.path.join(work, "centreline.npz"), **c)
            with open(os.path.join(work, "centreline.json"), "w") as f:
                json.dump({"hairpins": HAIRPINS}, f)
            direct = scenario.build(c, HAIRPINS, COURSE, map_dir)
            first = Path(direct["scenario"]).read_text()
            saved_c, saved_h = course_main.saved_centreline(work)
            again = scenario.build(saved_c, saved_h, COURSE, map_dir)
            self.assertEqual(again, direct)
            self.assertEqual(Path(again["scenario"]).read_text(), first)


if __name__ == "__main__":
    unittest.main()
