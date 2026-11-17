"""The Isaac world server node of scensim_isaac_world.launch.xml, with its refusals (csb issue 044).

scensim_isaac_world.launch.xml declares the arguments and documents them; this file holds the
node. The world server is respawned when it dies (issue 011, roadmap 077 G70), so a
configuration it refuses would be refused again every 10 s, without end. Two guards here:

- What can be checked before the node exists is checked here, and the launch fails once with
  a message and starts nothing (`ros2 launch` and play_launch then exit 1): every choice
  argument against its values, physics_hz a rate > 0, the port and the ego's domain whole
  numbers, an ego config and the worlds existing, and `render:=window` with DISPLAY set and
  not empty. This works under both launchers.
- The world server's usage errors exit 2 (`EXIT_USAGE`: its entry script's refusals, and
  argparse's). An exit with that code shuts the launch down, naming the reason, instead of
  respawning: the same arguments would be refused again. Any other exit respawns as before.
  This backstop works under `ros2 launch` only: play_launch 0.35.0 and 0.36.0 ignore this
  handler and respawn (issue 044), so what can be caught above is caught above.
"""
import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, LogInfo, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

EXIT_USAGE = 2
ARGS = {  # name: default, as scensim_isaac_world.launch.xml declares them
    "ego": "default", "world": "none", "follow": "false", "record": "",
    "record_view": "offroad", "render": "auto", "physics_hz": "60", "state_dir": "",
    "wheels_topic": "true",
}


CHOICES = {  # what the entry script (scripts/scensim_isaac_world_server.in) and argparse take
    "render": ("auto", "window", "headless"),
    "follow": ("true", "1", "false", "0", "editor"),
    "record_view": ("offroad", "road"),
    "wheels_topic": ("true", "1", "false", "0"),
}


def _number(text, kind):
    try:
        return kind(text)
    except ValueError:
        return None


# What scripts/install_isaac.sh makes under ISAAC_ROOT, and isaac_env.sh checks (077 G75)
ISAAC_INSTALL = ("venv/bin/activate", "venv/lib/python3.12/site-packages/isaacsim")


def isaac_root_problems(environ):
    """ISAAC_ROOT unset, or not an Isaac Sim install: isaac_env.sh's refusals, as messages."""
    root = environ.get("ISAAC_ROOT", "")
    if not root:
        return ["ISAAC_ROOT is not set: it names Isaac Sim 6.1's pip install "
                "(scripts/install_isaac.sh); set isaac.root in ~/.config/scensim/host.yaml, or "
                "export ISAAC_ROOT"]
    for need in ISAAC_INSTALL:
        if not os.path.exists(os.path.join(root, need)):
            return [f"ISAAC_ROOT={root} is not an Isaac Sim install: no {root}/{need} "
                    "(scripts/install_isaac.sh makes it)"]
    return []


# The host settings roadmap 080 renamed (`(old, new)`): an old one set is refused, naming the
# new. The same list as scensim_isaac_world/renamed.py and scripts/renamed_env.sh (a test
# keeps the three equal); inline, as this file runs under the system's Python.
RENAMED_ENV = (
    ("SWB_CARLA_USD", "SCENSIM_CARLA_USD"),
    ("SWB_VEHICLE_USD", "SCENSIM_VEHICLE_USD"),
    ("SWB_ISAAC_DIR", "SCENSIM_ISAAC_DIR"),
    ("SWB_TRACKED_FRICTION", "SCENSIM_TRACKED_FRICTION"),
    ("SWB_VEH_KIND", "SCENSIM_VEH_KIND"),
    ("SWB_VEH_ONLY", "SCENSIM_VEH_ONLY"),
    ("SWB_VEH_META", "SCENSIM_VEH_META"),
    ("SWB_PHYS_OUT", "SCENSIM_PHYS_OUT"),
    ("SWB_REPO", "SCENSIM_REPO"),
)


def renamed_refused(environ):
    return [f"{old} is renamed {new} (roadmap 080, scensim)"
            for old, new in RENAMED_ENV if old in environ]


def problems(var, environ):
    """What the world server would refuse in these arguments (`var`, name -> value) and this
    environment, as messages; empty when nothing is. Only what is cheap and sure to check:
    the world server's own checks stay behind these."""
    out = []
    for name, ok in CHOICES.items():
        if var[name] not in ok:
            out.append(f"{name}:={var[name]} is not one of {', '.join(ok)}")
    hz = _number(var["physics_hz"], float)
    if hz is None or not hz > 0:
        out.append(f"physics_hz:={var['physics_hz']} is not a rate > 0 (Hz)")
    out += renamed_refused(environ)   # the host settings roadmap 080 renamed
    for env in ("ISAAC_PORT", "EGO_ROS_DOMAIN_ID"):
        v = environ.get(env, "")
        if v and _number(v, int) is None:
            out.append(f"{env}={v} is not a whole number")
    out += isaac_root_problems(environ)
    display = environ.get("DISPLAY", "")
    if var["render"] == "window" and not display.strip():
        out.append("render:=window needs DISPLAY (Isaac's window), and DISPLAY is "
                   f"{'empty' if display else 'unset'}: set DISPLAY (e.g. DISPLAY=:8, a VNC "
                   "display), or use render:=headless or render:=auto")
    if var["ego"] not in ("default", "none", "") and not os.path.isfile(var["ego"]):
        out.append(f"ego:={var['ego']}: no such file (default, none or an aib ego config)")
    if var["world"] not in ("none", ""):
        for entry in (e.strip() for e in var["world"].split(",")):
            # as the world server's names.expand: variables, then ~
            path = os.path.expanduser(os.path.expandvars(entry))
            # a name the environment does not resolve here is the world server's to judge
            if entry and "$" not in path and not os.path.exists(path):
                out.append(f"world:= {entry}: no such world file, map directory or USD file")
    return out


def on_exit(event, _context):
    """Shut the launch down on a usage error; anything else is left to the respawn."""
    if event.returncode != EXIT_USAGE:
        return []
    reason = (f"scensim_isaac_world: the world server refused its arguments (exit {EXIT_USAGE}); "
              "not respawning it: the same arguments would be refused again (its log says why)")
    return [LogInfo(msg=reason), EmitEvent(event=Shutdown(reason=reason))]


def _setup(context):
    var = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    # The environment from os.environ: play_launch's parser has no perform on
    # EnvironmentVariable (stack.launch.py reads the environment the same way). DISPLAY is
    # inherited by every launcher here, never set with SetEnvironmentVariable; the port and
    # the domain may be (below), so a value missed here is checked by the world server.
    bad = problems(var, os.environ)
    if bad:
        raise RuntimeError("scensim_isaac_world: refused before the world server starts: " +
                           "; ".join(bad))
    # respawn_delay: 10 s lets the dead server's GPU memory and port go before the next one
    # claims them.
    node = Node(package="scensim_isaac_world", executable="scensim_isaac_world_server",
                name="scensim_isaac_world", output="screen", respawn=True, respawn_delay=10.0,
                # the port and the ego's domain as substitutions, as the XML's $(env ...): a
                # SetEnvironmentVariable before this include (stack.launch.py, not nested) is
                # in them under both launchers, and not in os.environ under play_launch's
                arguments=["--port", EnvironmentVariable("ISAAC_PORT", default_value="5590"),
                           "--ego", var["ego"], "--ego-domain",
                           EnvironmentVariable("EGO_ROS_DOMAIN_ID", default_value="1"),
                           "--world", var["world"],
                           "--follow", var["follow"], "--render", var["render"],
                           f"--record={var['record']}", "--record-view", var["record_view"],
                           "--physics-hz", var["physics_hz"], f"--state-dir={var['state_dir']}",
                           "--wheels-topic", var["wheels_topic"]])
    return [node, RegisterEventHandler(OnProcessExit(target_action=node, on_exit=on_exit))]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(k, default_value=d) for k, d in ARGS.items()]
                             + [OpaqueFunction(function=_setup)])
