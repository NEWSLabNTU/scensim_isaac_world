"""scensim issue 161: the Isaac world server declares READBACK_AFTER_STEP, with what it needs,
and every capability it declares is one its protocol pin knows (the shell looks each one up at
start, so an unknown name would stop the server)."""
import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "deps" / "scensim_world_protocol" / "scensim_world_server"))
from scensim_world_server import protocol as pb  # noqa: E402


def declared():
    tree = ast.parse((HERE.parents[1] / "scensim_isaac_world" / "world.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "IsaacWorld":
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and stmt.targets[0].id == "capabilities"):
                    return ast.literal_eval(stmt.value)
    raise AssertionError("IsaacWorld.capabilities not found")


class Issue161(unittest.TestCase):
    def test_the_read_back_after_the_step_is_declared_with_what_it_needs(self):
        caps = declared()
        self.assertIn("READBACK_AFTER_STEP", caps)
        self.assertIn("ENVIRONMENT_APPLIED", caps)

    def test_k9f_a_the_road_read_back_and_zones_are_declared(self):
        caps = declared()
        self.assertIn("ROAD_CONDITION_APPLIED", caps)   # needs ROAD_CONDITION
        self.assertIn("ROAD_CONDITION", caps)
        self.assertIn("ROAD_ZONES", caps)               # needs ROAD_CONDITION_APPLIED

    def test_every_declared_capability_is_known_to_the_protocol_pin(self):
        unknown = [c for c in declared() if c not in pb.CAPABILITY]
        self.assertEqual(unknown, [])


if __name__ == "__main__":
    unittest.main()
