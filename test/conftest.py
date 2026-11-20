"""The tests run from the source tree: the world server shell (scensim_world_server, which
holds the shared surface class table) from the deps/ checkout, as scripts/isaac_env.sh puts it
on PYTHONPATH for a stand-alone run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps" / "scensim_world_protocol"
                       / "scensim_world_server"))
