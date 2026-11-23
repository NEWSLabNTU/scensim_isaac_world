"""Nothing in the world server or its spikes asks Isaac for its asset root, NVIDIA's S3 bucket
(scensim issue 117): the flat ground is aib.isaac.ground's, built locally. Pure: reads the
sources."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT_CALLS = re.compile(r"\b(add_default_ground_plane|get_assets_root_path)\s*\(")


class NoAssetRootTest(unittest.TestCase):
    def test_no_module_or_spike_fetches_isaacs_asset_root(self):
        found = [f"{p.relative_to(ROOT)}:{i}"
                 for d in ("scensim_isaac_world", "spikes", "scripts")
                 for p in sorted((ROOT / d).rglob("*.py"))
                 for i, line in enumerate(p.read_text().splitlines(), 1)
                 if ASSET_ROOT_CALLS.search(line.split("#")[0])]
        self.assertEqual(found, [], "these reach NVIDIA's S3 at start; use aib.isaac.ground")


if __name__ == "__main__":
    unittest.main()
