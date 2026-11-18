"""The world's physics step against SSv2's frame (scensim issue 051, roadmap 077 G84).

The world builds PhysX at ``1 / physics_hz`` and the ego is tuned and warmed up on that step.
The bridge splits each SSv2 frame into substeps no longer than it (``max_physics_dt``), so a
frame that is a whole number of physics steps is stepped at the world's own dt. Any other frame
would need ``set_simulation_dt`` at run time, after which the PhysX vehicle stopped moving
(077 G84: frame_rate 100 on physics_hz 60, 0.0 m driven). So the world takes no other step:
it refuses the frame at load and the dt at each step, naming the frame rates that fit.
"""

TOLERANCE = 1e-6   # relative: 0.05 / (1 / 60) is 3.0000000000000004


def fitting_rates(physics_hz: float) -> list:
    """The whole-number frame rates (Hz) that are a whole number of physics steps, highest
    first: physics_hz / n for n = 1, 2, ... while a whole number of Hz."""
    out = []
    n = 1
    while physics_hz / n >= 1 - TOLERANCE:
        f = physics_hz / n
        if abs(f - round(f)) <= TOLERANCE * f:
            out.append(int(round(f)))
        n += 1
    return out


def _rates(physics_hz: float) -> str:
    return ", ".join(str(f) for f in fitting_rates(physics_hz)) or "none"


def frame_problem(frame_dt: float, physics_dt: float):
    """Why a frame of ``frame_dt`` s cannot run on a world stepping at ``physics_dt`` s, or
    None when it is a whole number of steps."""
    if not (frame_dt > 0 and physics_dt > 0):
        return None
    n = round(frame_dt / physics_dt)
    if n >= 1 and abs(n * physics_dt - frame_dt) <= TOLERANCE * frame_dt:
        return None
    hz = 1 / physics_dt
    return (f"SSv2's step {frame_dt * 1000:g} ms ({1 / frame_dt:g} Hz) is not a whole number "
            f"of this world's {hz:g} Hz physics steps, and the ego is not run on another step "
            f"(scensim issue 051): use an SSv2 frame rate of {_rates(hz)} Hz, or a physics "
            f"rate that is a multiple of {1 / frame_dt:g} Hz")


def step_problem(dt: float, physics_dt: float):
    """Why a substep of ``dt`` s is not this world's, or None."""
    if abs(dt - physics_dt) <= TOLERANCE * physics_dt:
        return None
    hz = 1 / physics_dt
    return (f"a physics step of {dt * 1000:g} ms is not this world's {physics_dt * 1000:g} ms "
            f"({hz:g} Hz), and the ego is not run on another step (scensim issue 051): SSv2's "
            f"frame must be a whole number of {hz:g} Hz steps ({_rates(hz)} Hz)")
