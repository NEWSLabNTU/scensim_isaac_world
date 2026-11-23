"""A run that raises exits nonzero (scensim issue 118): the world server, the course tools and
the step-response script under aib.fake_kit, which stands in for Isaac (its World raises), so no
Isaac is needed. Before the fix each closed the app in a `finally` or left it to the app's
atexit close, and Kit's close ends the process with 0."""
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


# the ROS modules an ego's node imports (aib.ros), stood in for too
ROS = ("rclpy,rosgraph_msgs,sensor_msgs,std_msgs,geometry_msgs,nav_msgs,builtin_interfaces,"
       "diagnostic_msgs,tf2_msgs,autoware_vehicle_msgs,autoware_control_msgs,tier4_vehicle_msgs,"
       "autoware_perception_msgs,autoware_adapi_v1_msgs")
COURSE_C6A = os.path.join(ROOT, "scensim_isaac_world", "courses", "fixture_c6a.yaml")
# a script, not a module: run by path under aib.fake_kit
RUN_PATH = """
import os, runpy, sys
from aib import fake_kit
fake_kit.install([m for m in os.environ["FAKE_KIT_EXTRA"].split(",") if m],
                 os.environ.get("FAKE_KIT_FAIL", ""))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_fake(module, *args, extra="", fail="", path=False):
    """The entry point under aib.fake_kit: (exit status, stdout + stderr)."""
    env = dict(os.environ, AIB_SRC=AIB, FAKE_KIT_EXTRA=extra, FAKE_KIT_FAIL=fail,
               PYTHONPATH=os.pathsep.join([AIB, ROOT, SERVER]))
    for k in ("ISAAC_KIT_ARGS", "DISPLAY"):
        env.pop(k, None)
    how = ["-c", RUN_PATH] if path else ["-m", "aib.fake_kit"]
    r = subprocess.run([sys.executable, *how, module, *args], env=env,
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


def course_map(root, course):
    """A map directory holding `course`'s record (course/), as the course tools read it."""
    from scensim_isaac_world.course import fixture, record

    work = os.path.join(root, "course")
    os.makedirs(work)
    with open(course) as f:
        co = yaml.safe_load(f)
    c = fixture.centreline(co)
    np.savez(os.path.join(work, "centreline.npz"),
             **{k: v for k, v in c.items() if k != "hairpins"})
    record.write(work, fixture.stations(co, c), in_mesh=True)
    return root


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EntryPointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, ROOT)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.map = course_map(os.path.join(cls.tmp.name, "map"), COURSE)
        cls.straight = course_map(os.path.join(cls.tmp.name, "c6a"), COURSE_C6A)
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

    def test_fixture_drive(self):
        out = os.path.join(self.tmp.name, "fixture_drive")
        self.assert_failed(*run_fake("scensim_isaac_world.course.fixture_drive", "--map",
                                     self.straight, "--out", out))

    def test_fixture_cost(self):
        out = os.path.join(self.tmp.name, "fixture_cost")
        self.assert_failed(*run_fake("scensim_isaac_world.course.fixture_cost", "--map",
                                     self.straight, "--out", out))

    def test_isaac_check(self):
        # it builds no World: its first Isaac import, inside the run, fails instead
        code, output = run_fake("scensim_isaac_world.course.isaac_check", self.straight,
                                fail="import:omni.physx")
        self.assertNotEqual(code, 0, output)
        self.assertIn("fake_kit: forced failure at import:omni.physx", output)

    def test_isaac_step(self):
        self.assert_failed(*run_fake(os.path.join(ROOT, "scripts", "step_response",
                                                  "isaac_step.py"),
                                     os.path.join(AIB, "config", "ego.yaml"),
                                     os.path.join(self.tmp.name, "step"), "fake", path=True,
                                     extra=ROS))


if __name__ == "__main__":
    unittest.main()
