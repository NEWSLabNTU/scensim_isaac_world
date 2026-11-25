"""Issue 168: base_drop, the depth the world's road render goes under the detail."""
import logging
import os
import tempfile
import unittest

from scensim_isaac_world import base_drop, road_layer

YAML = """# a test road (issue 168)
schema: 1
corridors:
  - centreline: [[0.0, 0.0], [40.0, 0.0]]
    section_in_base: true
    # nothing rough
    roughness: {gd0: 1.0e-12}
    potholes: {at: [[20.0, 0.0, 0.05, 0.5, 0.8]]}
  - centreline: [[0.0, 50.0], [40.0, 50.0]]
    section_in_base: true
    roughness: {gd0: 1.0e-12}
    ruts: {depth: 0.03}
ground: [/World/env/Terrain, /World/env/Road, /World/env/Lines]
"""

class BaseDrop(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "road_detail.yaml")
        with open(self.path, "w") as f:
            f.write(YAML)

    def corridors(self):
        return road_layer.load(self.path)

    def test_hash_ignores_base_drop_and_sees_changes(self):
        e = base_drop.entries(self.path)[0]
        h = base_drop.entry_hash(e, self.dir)
        self.assertEqual(h, base_drop.entry_hash({**e, "base_drop": {"depth": 1}}, self.dir))
        self.assertNotEqual(h, base_drop.entry_hash({**e, "seed": 3}, self.dir))

    def test_bound(self):
        c = self.corridors()
        self.assertAlmostEqual(base_drop.bound(c[0]), 0.05, places=4)
        self.assertAlmostEqual(base_drop.bound(c[1]), 0.03, places=4)
        # class C roughness: 3 sd of its PSD over the band (about 9 mm sd)
        sd = base_drop.roughness_sd(road_layer.rd.Roughness(cls="C"))
        self.assertTrue(0.008 < sd < 0.011, sd)

    def test_sampled_finds_the_pothole(self):
        self.assertAlmostEqual(base_drop.sampled(self.corridors()[0]), 0.05, places=3)

    def test_a_level_world_road_counts_the_section(self):
        """section_in_base false: the detail mesh draws the crown and the shoulders, the
        world's road is level, so the fall is below the base too; the bound covers it."""
        with open(self.path, "w") as f:
            f.write(YAML.replace("    section_in_base: true\n", ""))
        c = self.corridors()[0]
        s = base_drop.sampled(c)
        self.assertGreater(s, 0.05)
        self.assertGreaterEqual(base_drop.bound(c) + 1e-6, s)

    def test_write_keeps_the_file_and_round_trips(self):
        base_drop.write(self.path, [(0.0512, "a" * 64), (0.0301, "b" * 64)])
        text = open(self.path).read()
        self.assertIn("# a test road (issue 168)", text)
        self.assertIn("# nothing rough", text)
        es = base_drop.entries(self.path)
        self.assertEqual(es[0]["base_drop"], {"depth": 0.0512, "sha256": "a" * 64})
        self.assertEqual(es[1]["base_drop"]["sha256"], "b" * 64)
        self.assertEqual(es[0]["potholes"], {"at": [[20.0, 0.0, 0.05, 0.5, 0.8]]})
        self.assertEqual(len(self.corridors()), 2)  # road_layer accepts the key
        base_drop.write(self.path, [(0.06, "c" * 64), (0.04, "d" * 64)])  # replaced, not added
        es = base_drop.entries(self.path)
        self.assertEqual(es[0]["base_drop"]["depth"], 0.06)
        self.assertEqual(sum(ln.lstrip("- ").startswith("base_drop:")
                             for ln in open(self.path).read().split("\n")), 2)

    def test_stored_used_when_current_bound_when_stale(self):
        cs = self.corridors()
        es = base_drop.entries(self.path)
        base_drop.write(self.path, [(0.0499, base_drop.entry_hash(es[0], self.dir)),
                                    (0.0299, "0" * 64)])
        with self.assertLogs("scensim_isaac_world", logging.WARNING) as logs:
            ds = base_drop.depths(self.path, cs)
        self.assertEqual(ds[0], (0.0499, "stored"))
        self.assertEqual(ds[1][1], "bound")
        self.assertIn("base_drop", logs.output[0])
        self.assertIn("--write", logs.output[0])
        self.assertAlmostEqual(base_drop.drop(self.path, cs), 0.0499 + base_drop.MARGIN, places=6)

    def test_render_paths(self):
        g = ["/World/env/Terrain", "/World/env/Road", "/World/env/Lines"]
        self.assertEqual(base_drop.render_paths(self.path, g), ["/World/env/Road"])
        with open(self.path, "a") as f:
            f.write("base_render: [/World/env/Road, /World/env/Lines]\n")
        self.assertEqual(base_drop.render_paths(self.path, g),
                         ["/World/env/Road", "/World/env/Lines"])
        self.assertEqual(len(self.corridors()), 2)  # road_layer accepts base_render
        with open(self.path, "w") as f:
            f.write(YAML + "base_render: []\n")
        self.assertEqual(base_drop.render_paths(self.path, g), [])

if __name__ == "__main__":
    unittest.main()
