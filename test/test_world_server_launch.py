"""world_server.launch.py's refusals (csb issue 044), without Isaac: a configuration the node
would refuse is refused before it starts, and the node's usage error ends the launch."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import launch  # noqa: F401
    HAVE_LAUNCH = True
except ImportError:
    HAVE_LAUNCH = False

PATH = Path(__file__).resolve().parents[1] / "launch" / "world_server.launch.py"


def load():
    spec = importlib.util.spec_from_file_location("world_server_launch", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_LAUNCH, "no ROS 2 launch (source /opt/ros/humble/setup.bash)")
class LaunchRefusals(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.isaac = Path(self._tmp.name) / "isaac-sim"  # what install_isaac.sh makes
        (self.isaac / "venv" / "bin").mkdir(parents=True)
        (self.isaac / "venv" / "bin" / "activate").write_text("")
        (self.isaac / "venv/lib/python3.12/site-packages/isaacsim").mkdir(parents=True)

    def ok(self, m, **kw):
        var = dict(m.ARGS)
        var.update({k: v for k, v in kw.items() if k in m.ARGS})
        env = {"ISAAC_ROOT": str(self.isaac)}
        env.update({k: v for k, v in kw.items() if k not in m.ARGS})
        return m.problems(var, env)

    def test_isaac_root_has_no_default_and_must_be_an_install(self):
        """077 G75: unset ISAAC_ROOT is refused (no lab default), and so is a root without
        install_isaac.sh's venv and Isaac Sim in it."""
        m = load()
        got = self.ok(m, ISAAC_ROOT="")
        self.assertEqual(len(got), 1, got)
        self.assertIn("ISAAC_ROOT is not set", got[0])
        self.assertIn("isaac.root in ~/.config/swb/host.yaml", got[0])
        empty = Path(self._tmp.name) / "empty"
        empty.mkdir()
        got = self.ok(m, ISAAC_ROOT=str(empty))
        self.assertIn(f"ISAAC_ROOT={empty} is not an Isaac Sim install: no "
                      f"{empty}/venv/bin/activate", got[0])
        (self.isaac / "venv/lib/python3.12/site-packages/isaacsim").rmdir()
        self.assertIn("site-packages/isaacsim", self.ok(m)[0])

    def test_a_window_needs_a_display(self):
        m = load()
        self.assertIn("DISPLAY is unset", self.ok(m, render="window")[0])
        self.assertIn("DISPLAY is empty", self.ok(m, render="window", DISPLAY="  ")[0])
        self.assertEqual(self.ok(m, render="window", DISPLAY=":8"), [])
        self.assertEqual(self.ok(m, render="headless"), [])
        self.assertEqual(self.ok(m), [])  # the defaults: render auto

    def test_what_the_world_server_would_refuse_is_refused_here(self):
        m = load()
        for kw, word in [({"render": "bogus"}, "render:=bogus is not one of auto, window"),
                         ({"follow": "yes"}, "follow:=yes"),
                         ({"record_view": "top"}, "record_view:=top"),
                         ({"wheels_topic": "maybe"}, "wheels_topic:=maybe"),
                         ({"physics_hz": "0"}, "physics_hz:=0 is not a rate > 0"),
                         ({"physics_hz": "fast"}, "physics_hz:=fast"),
                         ({"ISAAC_PORT": "x"}, "ISAAC_PORT=x is not a whole number"),
                         ({"ego": "/nonexistent/ego.yaml"}, "ego:=/nonexistent/ego.yaml: no such"),
                         ({"world": "/nonexistent/a.usda"}, "world:= /nonexistent/a.usda: no such")]:
            with self.subTest(**kw):
                got = self.ok(m, **kw)
                self.assertEqual(len(got), 1, got)
                self.assertIn(word, got[0])
        # what the server takes, and a name only the server's environment resolves
        self.assertEqual(self.ok(m, follow="editor", record_view="road", wheels_topic="0",
                                 physics_hz="30", ego="none", world=str(PATH.parent),
                                 EGO_ROS_DOMAIN_ID="7"), [])
        self.assertEqual(self.ok(m, world="${SWB_NOT_SET_HERE}/Town01/Town01.usda"), [])

    def test_a_world_under_home_is_taken_as_the_server_takes_it(self):
        """names.expand expands ~ as well as variables: a ~ entry that exists is not refused
        (HOME pointed at a directory of the test's own)."""
        import os
        import tempfile
        from unittest import mock
        m = load()
        with tempfile.TemporaryDirectory() as home:
            (Path(home) / "Town01.usda").write_text("#usda 1.0\n")
            with mock.patch.dict(os.environ, {"HOME": home}):
                self.assertEqual(self.ok(m, world="~/Town01.usda"), [])
                self.assertEqual(self.ok(m, world="$HOME/Town01.usda"), [])
                self.assertIn("no such world file", self.ok(m, world="~/Town02.usda")[0])

    def test_only_a_usage_error_shuts_the_launch_down(self):
        from launch.actions import EmitEvent
        from launch.events import Shutdown
        m = load()
        self.assertEqual(m.on_exit(SimpleNamespace(returncode=1), None), [])
        self.assertEqual(m.on_exit(SimpleNamespace(returncode=-9), None), [])  # a deadline
        acts = m.on_exit(SimpleNamespace(returncode=m.EXIT_USAGE), None)
        emits = [a for a in acts if isinstance(a, EmitEvent)]
        self.assertEqual(len(emits), 1)
        self.assertIsInstance(emits[0].event, Shutdown)
        self.assertIn("not respawning", emits[0].event.reason)

    def test_the_xml_passes_every_argument_and_the_node_respawns(self):
        m = load()
        xml = (PATH.parent / "swb_isaac_world.launch.xml").read_text()
        for name in m.ARGS:
            self.assertIn(f'<arg name="{name}" value="$(var {name})"/>', xml)
            self.assertIn(f'<arg name="{name}" default=', xml)
        src = PATH.read_text()
        self.assertIn("respawn=True", src)
        # the port and the ego's domain as substitutions: a SetEnvironmentVariable before the
        # include is not in os.environ under play_launch's parser
        self.assertIn('EnvironmentVariable("ISAAC_PORT", default_value="5590")', src)
        self.assertIn('EnvironmentVariable("EGO_ROS_DOMAIN_ID", default_value="1")', src)


if __name__ == "__main__":
    unittest.main()


class IsaacEnvRefusals(unittest.TestCase):
    """scripts/isaac_env.sh, which the world server's and the entry points' wrappers source:
    no ISAAC_ROOT default (077 G75), exit 2 on an unset root or one that is not an install."""

    def source(self, root):
        import subprocess
        script = PATH.parents[1] / "scripts" / "isaac_env.sh"
        env = {"PATH": "/usr/bin:/bin", "ISAAC_ROOT": root}
        return subprocess.run(["bash", "-c", f'source "{script}" || exit $?'], env=env,
                              capture_output=True, text=True)

    def test_unset_and_not_an_install_are_refused(self):
        import tempfile
        r = self.source("")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("ISAAC_ROOT is not set", r.stderr)
        with tempfile.TemporaryDirectory() as d:
            r = self.source(d)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn(f"ISAAC_ROOT={d} is not an Isaac Sim install: no {d}/venv/bin/activate",
                          r.stderr)
