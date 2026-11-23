"""Phase 036 smoke test: start Isaac Sim headless, build a world, step it, report timings."""
import os
import time

t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

# windowed on a display (VNC), else headless; Kit settings of the host (ISAAC_KIT_ARGS), as the
# world server takes them: e.g. --portable-root for an install's own caches (scensim roadmap 095)
app = SimulationApp({"headless": not os.environ.get("DISPLAY"),
                     "extra_args": os.environ.get("ISAAC_KIT_ARGS", "").split()})
t_app = time.time() - t0

from isaacsim.core.api import World  # noqa: E402
import omni.kit.app  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
from aib.isaac import ground  # noqa: E402
ground.add_ground_plane(world.stage)  # local: no asset root (scensim issue 117)
world.reset()
t1 = time.time()
for _ in range(600):
    world.step(render=True)
t_step = (time.time() - t1) / 600
ver = omni.kit.app.get_app().get_build_version()
print(f"SMOKE OK kit {ver}; app start {t_app:.1f} s; step+render {1e3 * t_step:.2f} ms; sim time {world.current_time:.2f} s")
app.close()
