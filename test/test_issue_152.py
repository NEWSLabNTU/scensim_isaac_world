"""Issue 152: the rocks' base ground (`roadBase`) never holds the detail's collision tiles. The
live world counted `/World/RoadCollision` as ground and handed that list to the rock stream, so
each tile was in `roadBase` and `roadDetail` at once and issue 123's self-check counted a rock
resting on a tile as on the base. No Isaac."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import names, rock_layer, world_server  # noqa: E402

C = rock_layer.COLL_ROOT


class BaseGround(unittest.TestCase):
    def test_the_world_ground_without_the_tiles(self):
        world = ["/World/env/Terrain", "/World/env/Road", "/World/env/Lines", C]
        self.assertEqual(rock_layer.base_ground(world),
                         ["/World/env/Terrain", "/World/env/Road", "/World/env/Lines"])

    def test_the_detail_files_ground_first_and_never_the_tiles(self):
        self.assertEqual(rock_layer.base_ground(["/World/env/Road", C], ["/World/base", C]),
                         ["/World/base"])

    def test_a_root_that_holds_the_tiles_covers_them(self):
        for p in (C, C + "/", "/World", "/World/", "/"):
            self.assertTrue(rock_layer.covers_tiles(p), p)
        for p in ("/World/env/Road", C + "Other", "/World/RoadRocks", C + "/t_1_2_3"):
            self.assertFalse(rock_layer.covers_tiles(p), p)

    def test_the_stream_refuses_a_ground_that_holds_the_tiles(self):
        with self.assertRaises(ValueError) as e:
            rock_layer.RockStream(object(), object(), object(), ground=["/World/env/Road", C])
        self.assertIn("issue 152", str(e.exception))


class Preload(unittest.TestCase):
    """Part 2: the world server preloaded its world with no ground paths, so the world's ground
    entered PhysX outside `roadBase` and the scenario's load, keeping that world, wrote
    `roadBase` after the ground's intake. The preload now names the world file's ground paths,
    and a load keeps the world only with the ground it was loaded with."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.map = os.path.join(self.tmp.name, "haruna")
        os.makedirs(os.path.join(self.map, "isaac"))
        open(os.path.join(self.map, "isaac", "world.usdc"), "w").close()

    def tearDown(self):
        self.tmp.cleanup()

    def world(self, text):
        with open(os.path.join(self.map, "isaac", "world.yaml"), "w") as f:
            f.write(text)

    def test_the_preload_names_the_world_files_ground_paths(self):
        self.world('usd: isaac/world.usdc\nground_paths: ["/World/env/Terrain", '
                   '"/World/env/Road"]\n')
        usd = os.path.join(self.map, "isaac", "world.usdc")
        self.assertEqual(world_server.preload_list(self.map),
                         [(usd, ["/World/env/Terrain", "/World/env/Road"])])

    def test_a_world_file_without_ground_paths_and_a_usd_file_preload_with_none(self):
        self.world("usd: isaac/world.usdc\n")
        usd = os.path.join(self.map, "isaac", "world.usdc")
        self.assertEqual(world_server.preload_list(f"{self.map},{usd}"), [(usd, [])])

    def test_a_load_with_the_preloads_ground_keeps_the_world(self):
        g = ["/World/env/Terrain", "/World/env/Road"]
        self.assertTrue(names.keep_loaded_world("/w.usdc", list(g), "/w.usdc", g))

    def test_a_load_with_other_ground_loads_the_world_again(self):
        g = ["/World/env/Terrain", "/World/env/Road"]
        self.assertFalse(names.keep_loaded_world("/w.usdc", g, "/w.usdc", []))
        self.assertFalse(names.keep_loaded_world("/w.usdc", g, "/w.usdc", g[:1]))
        self.assertFalse(names.keep_loaded_world("/v.usdc", g, "/w.usdc", g))
        self.assertFalse(names.keep_loaded_world("", [], "", []))

    def test_the_world_keeps_only_through_that_rule_and_warns_naming_both_lists(self):
        src = open(os.path.join(os.path.dirname(__file__), "..", "scensim_isaac_world",
                                "world.py")).read()
        body = src[src.index("    def _load_env("):src.index("    def _load_surfaces(")]
        keep = body.index("names.keep_loaded_world(")
        self.assertLess(keep, body.index("# The same world (preloaded"))
        warn = body[body.index('log.warning("issue 152'):]
        self.assertIn("list(self.loaded_ground or [])", warn[:600])
        self.assertIn("list(ground_paths)", warn[:600])
        self.assertIn("self.load(usd, list(ground_paths))", src)


if __name__ == "__main__":
    unittest.main()
