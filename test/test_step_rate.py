"""SSv2's frame against the world's physics step (scensim issue 051, roadmap 077 G84); no Isaac."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scensim_isaac_world import step_rate  # noqa: E402


def bridge_substeps(frame_dt, max_dt):
    """scenario_world_bridge remote/mod.rs physics_steps."""
    import math
    n = max(1, math.ceil(frame_dt / max_dt - 1e-9))
    return n, frame_dt / n


class StepRate(unittest.TestCase):
    def test_the_fitting_rates_are_the_whole_divisors(self):
        self.assertEqual(step_rate.fitting_rates(60), [60, 30, 20, 15, 12, 10, 6, 5, 4, 3, 2, 1])
        self.assertIn(120, step_rate.fitting_rates(240))
        self.assertNotIn(100, step_rate.fitting_rates(240))
        self.assertEqual(step_rate.fitting_rates(100)[:3], [100, 50, 25])

    def test_a_frame_of_whole_steps_runs_at_the_worlds_own_dt(self):
        for frame_hz, physics_hz in ((20, 60), (60, 60), (30, 240), (100, 100), (100, 300),
                                     (10, 60)):
            frame, physics = 1 / frame_hz, 1 / physics_hz
            self.assertIsNone(step_rate.frame_problem(frame, physics), (frame_hz, physics_hz))
            # and the bridge then steps at exactly that dt, which step() takes
            n, dt = bridge_substeps(frame, physics)
            self.assertIsNone(step_rate.step_problem(dt, physics), (frame_hz, physics_hz))

    def test_g84s_frame_is_refused_naming_what_fits(self):
        why = step_rate.frame_problem(0.01, 1 / 60)   # frame_rate 100 on physics_hz 60
        self.assertIn("SSv2's step 10 ms (100 Hz) is not a whole number of this world's 60 Hz",
                      why)
        self.assertIn("60, 30, 20, 15, 12, 10, 6, 5, 4, 3, 2, 1 Hz", why)
        self.assertIn("a multiple of 100 Hz", why)
        # 120 Hz frames on 60 Hz physics: shorter than one step
        self.assertIsNotNone(step_rate.frame_problem(1 / 120, 1 / 60))
        self.assertIsNotNone(step_rate.frame_problem(1 / 25, 1 / 60))

    def test_a_step_off_the_worlds_dt_is_refused(self):
        n, dt = bridge_substeps(0.01, 1 / 60)
        self.assertEqual(n, 1)
        why = step_rate.step_problem(dt, 1 / 60)
        self.assertIn("a physics step of 10 ms is not this world's 16.6667 ms (60 Hz)", why)

    def test_no_step_time_is_not_judged(self):
        self.assertIsNone(step_rate.frame_problem(0.0, 1 / 60))


if __name__ == "__main__":
    unittest.main()
