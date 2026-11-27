"""Isaac's message bindings the world server needs (scensim issue 195 L1b): Autoware's message
packages built for Isaac's Python into ``$ISAAC_ROOT/msgs`` (aib's ``build_msgs.sh``, which
builds these by default; a test keeps the two lists equal). Checked before Isaac starts, so a
host that lacks one is told how to build it instead of the server dying minutes later at an
import."""
import importlib.util
import sys

REQUIRED = ("autoware_vehicle_msgs", "autoware_control_msgs", "autoware_perception_msgs",
            "autoware_sensing_msgs")


def missing(find=importlib.util.find_spec):
    """The REQUIRED packages `find` (a module name to a spec, or None) does not locate."""
    out = []
    for name in REQUIRED:
        if name in sys.modules:  # imported already, or stood in for (aib's fake kit)
            continue
        try:
            spec = find(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            out.append(name)
    return out


def refusal(names):
    return (f"Isaac's message bindings lack {', '.join(names)}, which the world server needs "
            f"(aib publishes and reads them): run `scensim isaac msgs`, or with ISAAC_ROOT set "
            f"aib's src/aib/scripts/build_msgs.sh {' '.join(names)}")
