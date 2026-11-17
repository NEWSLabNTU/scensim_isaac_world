"""The host settings roadmap 080 renamed: one list in three places (the package, the shell
scripts and the launch file), and every old name set is refused naming the new."""
import re
import subprocess
import unittest
from pathlib import Path

from scensim_isaac_world import renamed

ROOT = Path(__file__).resolve().parents[1]


def launch_list():
    src = (ROOT / "launch" / "world_server.launch.py").read_text()
    block = src[src.index("RENAMED_ENV = ("):src.index("def renamed_refused")]
    return tuple(re.findall(r'\("(\w+)", "(\w+)"\)', block))


def shell_list():
    text = (ROOT / "scripts" / "renamed_env.sh").read_text()
    return tuple(tuple(p.split(":")) for p in re.findall(r"^\s+(SWB_\w+:SCENSIM_\w+)$", text, re.M))


class Renamed(unittest.TestCase):
    def test_the_three_lists_are_one(self):
        self.assertEqual(launch_list(), renamed.RENAMED_ENV)
        self.assertEqual(shell_list(), renamed.RENAMED_ENV)
        for old, new in renamed.RENAMED_ENV:
            self.assertEqual(old[len("SWB_"):], new[len("SCENSIM_"):])

    def test_every_old_name_set_is_refused_naming_the_new(self):
        self.assertEqual(renamed.refused({"SCENSIM_CARLA_USD": "/a"}), [])
        self.assertEqual(renamed.refused({"SWB_CARLA_USD": ""}),
                         ["SWB_CARLA_USD is renamed SCENSIM_CARLA_USD (roadmap 080, scensim)"])

    def test_the_shell_scripts_refuse_too(self):
        env = {"PATH": "/usr/bin:/bin", "SWB_VEHICLE_USD": "/v"}
        r = subprocess.run(["bash", "-c", f"source {ROOT / 'scripts' / 'renamed_env.sh'}"],
                           env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SWB_VEHICLE_USD is renamed SCENSIM_VEHICLE_USD", r.stderr)
        r = subprocess.run(["bash", "-c", f"source {ROOT / 'scripts' / 'renamed_env.sh'}"],
                           env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
