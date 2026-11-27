"""scensim issue 195 L1b: the world server refuses to start, before Isaac does, when Isaac's
message bindings lack a package aib needs, naming the command that builds it. Without the
check it died minutes later, importing it."""
import re
import unittest
from pathlib import Path

from scensim_isaac_world import bindings

BUILD_MSGS = (Path(__file__).resolve().parents[1] / "deps" / "autoware_isaac_bridge" / "src"
              / "aib" / "scripts" / "build_msgs.sh")


def build_msgs_defaults():
    m = re.search(r"\|\| PKGS=\(([^)]*)\)", BUILD_MSGS.read_text())
    return tuple(m.group(1).split())


class BindingsTest(unittest.TestCase):
    def test_the_required_packages_are_what_build_msgs_builds_by_default(self):
        self.assertEqual(sorted(bindings.REQUIRED), sorted(build_msgs_defaults()))

    def test_the_heading_needs_autoware_sensing_msgs(self):
        self.assertIn("autoware_sensing_msgs", bindings.REQUIRED)

    def test_a_missing_package_is_named_with_its_fix(self):
        have = set(bindings.REQUIRED) - {"autoware_sensing_msgs"}
        missing = bindings.missing(lambda name: object() if name in have else None)
        self.assertEqual(missing, ["autoware_sensing_msgs"])
        why = bindings.refusal(missing)
        self.assertIn("autoware_sensing_msgs", why)
        self.assertIn("scensim isaac msgs", why)
        self.assertIn("build_msgs.sh autoware_sensing_msgs", why)

    def test_nothing_missing_is_no_refusal(self):
        self.assertEqual(bindings.missing(lambda name: object()), [])


if __name__ == "__main__":
    unittest.main()
