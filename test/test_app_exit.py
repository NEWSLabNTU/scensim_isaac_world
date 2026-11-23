"""A run that raises exits nonzero (scensim issue 118): the world server and the two course
drives under aib.fake_kit, which stands in for Isaac (its World raises), so no Isaac is needed.
Before the fix each closed the app in a `finally`, and Kit's close ends the process with 0."""
import os
import socket
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIB = os.path.join(ROOT, "deps", "autoware_isaac_bridge", "src", "aib")
SERVER = os.path.join(ROOT, "deps", "scensim_world_protocol", "scensim_world_server")
FORCED = "fake_kit: forced failure building the World"  # aib.fake_kit.FORCED
COURSE = os.path.join(ROOT, "scensim_isaac_world", "courses", "fixture_c5.yaml")


def run_fake(module, *args, extra=""):
    """The entry point under aib.fake_kit: (exit status, stdout + stderr)."""
    env = dict(os.environ, AIB_SRC=AIB, FAKE_KIT_EXTRA=extra,
               PYTHONPATH=os.pathsep.join([AIB, ROOT, SERVER]))
    for k in ("ISAAC_KIT_ARGS", "DISPLAY"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, "-m", "aib.fake_kit", module, *args], env=env,
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EntryPointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, ROOT)
        from scensim_isaac_world.course import fixture, record

        cls.tmp = tempfile.TemporaryDirectory()
        cls.map = os.path.join(cls.tmp.name, "map")
        work = os.path.join(cls.map, "course")
        os.makedirs(work)
        with open(COURSE) as f:
            co = yaml.safe_load(f)
        c = fixture.centreline(co)
        np.savez(os.path.join(work, "centreline.npz"),
                 **{k: v for k, v in c.items() if k != "hairpins"})
        record.write(work, fixture.stations(co, c), in_mesh=True)
        cls.cases = os.path.join(cls.tmp.name, "cases.yaml")
        with open(cls.cases, "w") as f:
            yaml.safe_dump({"cases": [{"segment": 0, "from": 50.0}]}, f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_failed(self, code, output):
        self.assertNotEqual(code, 0, output)
        self.assertIn(FORCED, output)  # failed where it builds the World, with its traceback

    def test_world_server(self):
        code, output = run_fake("scensim_isaac_world.world_server", "--port", str(free_port()),
                                extra="zmq")
        self.assert_failed(code, output)

    def test_arc_drive(self):
        out = os.path.join(self.tmp.name, "arc")
        self.assert_failed(*run_fake("scensim_isaac_world.course.arc_drive", "--map", self.map,
                                     "--out", out))

    def test_lip_drive(self):
        out = os.path.join(self.tmp.name, "lip")
        self.assert_failed(*run_fake("scensim_isaac_world.course.lip_drive", "--map", self.map,
                                     "--course", self.cases, "--out", out))


if __name__ == "__main__":
    unittest.main()
