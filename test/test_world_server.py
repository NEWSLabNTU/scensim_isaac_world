"""The world server's guards against a wedged Kit (docs/issues/011), without Isaac.

Both guards are the request-deadline child process's (`scensim_world_server.watchdog`), started
before Kit; nothing of them runs in the world server's process (csb issue 042).
"""
import ast
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "deps" / "scensim_world_protocol" / "scensim_world_server"))
from scensim_isaac_world import world_server  # noqa: E402
from scensim_world_server import watchdog  # noqa: E402

SOURCE = (HERE.parents[1] / "scensim_isaac_world" / "world_server.py").read_text()


class GuardTest(unittest.TestCase):
    def test_no_python_log_consumer_and_no_timer_in_the_process(self):
        """042: a Python omni.log consumer deadlocked a PhysX cooking task on the GIL, and a
        timer in the process cannot watch a thread holding the GIL."""
        names = {n.attr if isinstance(n, ast.Attribute) else n.id
                 for n in ast.walk(ast.parse(SOURCE)) if isinstance(n, (ast.Attribute, ast.Name))}
        for name in ("add_message_consumer", "set_message_consumer", "Timer", "omni"):
            self.assertNotIn(name, names)
        self.assertFalse(hasattr(world_server, "fatal_consumer"))
        self.assertFalse(hasattr(world_server, "startup_watchdog"))

    def test_the_start_is_watched_from_before_kit_and_the_child_serves_the_requests(self):
        main = SOURCE[SOURCE.index("def main("):]
        armed = main.index('watchdog.step("Kit\'s start and the preload", '
                           'deadline=STARTUP_LIMIT)')
        self.assertLess(main.index("watchdog = StepWatchdog()"), armed)
        self.assertLess(armed, main.index("from isaacsim import SimulationApp"))
        self.assertLess(main.index("world.preload(usd, ground)"), main.index("startup.__exit__("))
        self.assertIn("watchdog.watch_log(kit_log, KIT_FATAL)", main)
        self.assertIn("watchdog=watchdog)", main[main.index("run(world, socket"):])
        self.assertEqual(world_server.STARTUP_LIMIT, 1200.0)
        # a Kit log still missing after the preload (and KIT_LOG_GRACE) is an ERROR
        after = main[main.index("startup.__exit__("):main.index("run(world, socket")]
        self.assertIn("log.error(", after)
        self.assertIn("KIT_LOG_GRACE", after)
        self.assertEqual(world_server.KIT_LOG_GRACE, 60.0)

    def test_a_kit_fatal_line_ends_the_world_server_and_others_do_not(self):
        """011: a FATAL message in Kit's log (as Kit writes its file) ends the world server."""
        d = tempfile.mkdtemp()
        path = os.path.join(d, "kit_20261117_034501.log")
        with open(path, "w") as f:
            f.write("2026-11-16T19:45:44Z [43,659ms] [Warning] [omni.physx.plugin] Collisions "
                    "are supported currently only in one collision group.\n"
                    "2026-11-16T19:45:45Z [43,865ms] [Info] [omni.physx.cooking.plugin] "
                    "ConvexMeshCookingTask: polygon limit reached\n"
                    "2026-11-16T19:45:46Z [44,000ms] [Error] [omni.rtx] an error\n")
        r, w = os.pipe()
        os.write(w, f"log\t{world_server.KIT_FATAL}\t{path}\n".encode())
        fatal = []

        def later():
            time.sleep(0.8)
            with open(path, "a") as f:
                f.write("2026-10-26T12:28:36Z [150,000ms] [Fatal] [omni.rtx] Out of resource "
                        "descriptors!\n")
            time.sleep(3.0)
            os.close(w)
        t = threading.Thread(target=later)
        t.start()
        with os.fdopen(r, "rb", buffering=0) as stream:
            watchdog.watch(1234, stream, expire=lambda *a: self.fail("expired"),
                           fatal=lambda *a: fatal.append(a[2]))
        t.join()
        shutil.rmtree(d)
        self.assertEqual(len(fatal), 1)
        self.assertIn("[Fatal] [omni.rtx] Out of resource descriptors!", fatal[0])

    def test_a_start_past_its_limit_ends_the_world_server(self):
        """011: not up in STARTUP_LIMIT ends the world server (here at 0.3 s)."""
        r, w = os.pipe()
        os.write(w, b"begin 0.3 Kit's start and the preload\n")
        expired = []
        with os.fdopen(r, "rb", buffering=0) as stream:
            watchdog.watch(1234, stream, expire=lambda *a: expired.append(a[1]),
                           fatal=lambda *a: self.fail("fatal"))
        os.close(w)
        self.assertEqual(expired, ["Kit's start and the preload"])


class RefusalTest(unittest.TestCase):
    """044: the world server's refusals before Kit exit 2, on which its launch ends."""

    def test_a_missing_world_is_refused_with_exit_2(self):
        with self.assertRaises(SystemExit) as cm:
            world_server.preload_list("/nonexistent/world.usda")
        self.assertEqual(cm.exception.code, world_server.EXIT_USAGE)

    def test_every_refusal_before_kit_is_a_usage_exit(self):
        main = SOURCE[SOURCE.index("def world_usd("):SOURCE.index("from isaacsim import")]
        self.assertNotIn("raise SystemExit(", main)


if __name__ == "__main__":
    unittest.main()
