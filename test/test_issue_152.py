"""Issue 152: the rocks' base ground (`roadBase`) never holds the detail's collision tiles. The
live world counted `/World/RoadCollision` as ground and handed that list to the rock stream, so
each tile was in `roadBase` and `roadDetail` at once and issue 123's self-check counted a rock
resting on a tile as on the base. No Isaac."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import rock_layer  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
