"""The Isaac Sim world server (roadmaps 037, 046): `python -m scensim_isaac_world.world_server`.

Starts Isaac Sim (headless unless `--display`), builds the world, and serves the world
protocol on a ZMQ REP socket bound to localhost (the shell `scensim_world_server`, which holds the
owner lease) until a client sends `Shutdown` or the process gets SIGINT/SIGTERM; it also
ends when its parent does (a supervisor killed hard). Run it through `just isaac-world`, which
sets Isaac's environment.

    python -m scensim_isaac_world.world_server --port 5590 [--display :8] [--ego <aib ego config>]
        [--world <world>[,<world>...]] [--follow]

A world is a world file (`<map>/isaac/world.yaml`, as scensim's `LoadWorld` reads it), a map
directory (its `isaac/world.yaml`) or a USD file. Several, comma separated, are all loaded and
rendered once before the server answers (roadmap 043: a town's first render compiles its
materials, minutes, which a scenario's `LoadWorld` should not have to); the first one listed
stays loaded. A world file's map directory with a `surfaces.yaml` (roadmap 047) gives that
world its surface map, checked here before Isaac starts.

A world file's `usd:` may name its stage through the environment (`${SCENSIM_CARLA_USD}/...`), and
a course world finds its props through `SCENSIM_CARLA_USD` when its `carla-usd` link is absent
(`names`). With `--state-dir` what the server writes for a map goes to `<state dir>/<map>/`,
never into the map directory.
"""
import argparse
import logging
import os
import sys
import time

from . import names

# A world server that cannot go on ends, for its launch to respawn it fresh (docs/issues/011:
# one stopped answering after RTX's "Out of resource descriptors!", another sat in Kit's start
# for 16 minutes; neither exited). Both guards are the request-deadline child process's
# (`scensim_world_server.watchdog`, roadmap 077 G70), started before Kit, which ends this process
# from outside: Kit's start and the preload are a block with STARTUP_LIMIT as its deadline,
# and the child watches Kit's log file for a FATAL line (KIT_FATAL). Nothing runs in this
# process for either: a Python omni.log consumer took the GIL for every Kit message, and a
# PhysX cooking task logging from a worker thread waited for it while the main thread, which
# held it, waited for the task (csb issue 042); a timer here needs the GIL too, so it cannot
# watch a thread that holds the GIL.
#
# Kit's start, the worlds' preload and their first renders, s: a cold Town01 takes about 7
# minutes; isaac_up.sh waits 30 (ISAAC_READY_TIMEOUT) for an answer.
STARTUP_LIMIT = 1200.0
# What marks a FATAL message in Kit's log file: `<time> [<ms>] [Fatal] [<channel>] <message>`.
KIT_FATAL = "] [Fatal] ["
# s after Kit's start by which Kit's log must exist, or the world server logs an ERROR.
KIT_LOG_GRACE = 60.0
# A refused configuration (argparse's own exit too): its launch shuts down on it instead of
# respawning the world server to be refused again (csb issue 044).
EXIT_USAGE = 2


class Refused(SystemExit):
    """`raise Refused(message)`: the message on stderr, exit EXIT_USAGE."""

    def __init__(self, message):
        print(f"scensim_isaac_world: {message}", file=sys.stderr, flush=True)
        super().__init__(EXIT_USAGE)
        self.message = message

    def __str__(self):
        return self.message


def world_file(entry):
    """The world file a `--world` entry names (a map directory's `isaac/world.yaml`), or
    None for a USD file (`names.world_file`)."""
    return names.world_file(entry)


def world_usd(entry):
    """The USD stage a `--world` entry names: a world file's `usd` (names expanded, relative
    to its map directory, as scensim resolves it), a map directory's `isaac/world.yaml`, or a USD
    file."""
    try:
        path = world_file(entry)
        if path is None:
            return os.path.abspath(names.expand(entry))
        usd = names.world_usd_of(path)
    except names.UnresolvedName as e:
        raise Refused(str(e)) from e
    if not usd:
        raise Refused(f"{entry}: the world file names no usd (a flat ground plane needs no "
                         "preload)")
    return usd


def world_ground(entry):
    """The ground paths a `--world` entry's world file names (`ground_paths`), [] for a USD
    file: the preload loads the world with them, as scensim's `LoadWorld` will (issue 152)."""
    path = world_file(entry)
    return names.world_ground_of(path) if path is not None else []


def preload_list(worlds):
    """`--world`'s comma-separated entries as (USD path, ground paths), checked before Isaac
    starts; one a USD, the first entry's."""
    out = {}
    for e in (e.strip() for e in worlds.split(",") if e.strip()):
        out.setdefault(world_usd(e), world_ground(e))
    missing = [u for u in out if not os.path.exists(u)]
    if missing:
        raise Refused(f"world(s) not found: {', '.join(missing)}")
    return list(out.items())


def kit_log_file():
    """The file Kit's own logger writes (its `/log/file` setting, tokens resolved), or None."""
    import carb.settings
    import carb.tokens

    path = carb.settings.get_settings().get("/log/file")
    if not path:
        return None
    return carb.tokens.get_tokens_interface().resolve(str(path)) or str(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=5590, help="REP port on 127.0.0.1")
    ap.add_argument("--display", default="",
                    help="X display for Isaac's window (e.g. :8, the VNC one); empty: headless")
    ap.add_argument("--follow", action="store_true",
                    help="with --display: the viewport follows the ego from behind, no UI "
                         "(for watching or recording a run)")
    ap.add_argument("--ui", action="store_true",
                    help="with --follow: keep Isaac's editor UI and full-size render (a demo "
                         "that shows the editor)")
    ap.add_argument("--world", default="",
                    help="worlds to load and render before serving, comma separated: world "
                         "files, map directories or USD files (a first render compiles the "
                         "world's materials); the first stays loaded, and scensim's LoadWorld of "
                         "it keeps it")
    ap.add_argument("--ego", default="",
                    help="aib ego config (autoware_isaac_bridge's config/ego.yaml): the ego "
                         "is aib's PhysX vehicle driven by Autoware (038, 045); empty: a "
                         "kinematic box (037). aib must be on PYTHONPATH")
    ap.add_argument("--ego-domain", type=int, default=0,
                    help="the ego's ROS domain, over the ego config's (the ego stack's "
                         "ROS_DOMAIN_ID); 0: the config's")
    ap.add_argument("--record", default="",
                    help="a directory: each scenario's first ego's dash and wheel pictures "
                         "and soil state, every rendered frame (`recorder`), for a video; "
                         "empty: none")
    ap.add_argument("--record-view", default="offroad", choices=("offroad", "road"),
                    help="with --record: the pictures' size and chase camera (`recorder.VIEWS`): "
                         "offroad (960 x 540, a high chase from behind and to the left) or road "
                         "(1280 x 720, 7 m behind, 3.4 m up, 15 deg down, 80 deg: the NTU twin's, "
                         "the same as CARLA's)")
    ap.add_argument("--physics-hz", type=float, default=60.0,
                    help="the physics rate, Hz: PhysX, the egos' onboard computers and the soil "
                         "step at it, and the bridge splits SSv2's step into steps no longer "
                         "than 1 / this (Hello's max_physics_dt). The ego's commands are taken "
                         "once per SSv2 step: a faster controller needs a shorter SSv2 step")
    ap.add_argument("--state-dir", default="",
                    help="where the server writes for a map, and a project's own map files: "
                         "<dir>/<map>/ (ruts, surfaces.yaml, road_detail.yaml; `names`); "
                         "empty: into the map directory")
    ap.add_argument("--wheels-topic", default="true", choices=("true", "false"),
                    help="publish /scensim/wheels (`wheels`: each ego wheel's ground and tire "
                         "state, DiagnosticArray, 20 Hz) on the ego's domain")
    args = ap.parse_args(argv)
    from . import renamed
    gone = renamed.refused(os.environ)
    if gone:
        raise Refused("; ".join(gone))
    if not args.physics_hz > 0:
        raise Refused(f"--physics-hz {args.physics_hz}: a rate > 0")
    from scensim_world_server import lifetime
    from scensim_world_server.server import bind, run

    lifetime.die_with_parent()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("scensim_isaac_world")

    preload = preload_list(args.world)
    socket = bind(args.port)
    from .surface_layer import register
    from .surfaces import SurfaceError

    entries = [e.strip() for e in args.world.split(",") if e.strip()]
    try:
        # A world's surface map (<map>/surfaces.yaml, roadmap 047), checked before Isaac starts.
        surface_maps = register([w for w in map(world_file, entries) if w],
                                state_dir=args.state_dir or None)
    except SurfaceError as e:
        raise Refused(f"surface map refused:\n{e}") from e
    for usd, files in surface_maps.items():
        log.info("surface map %s for %s%s%s", files.surfaces, usd,
                 "" if os.path.exists(files.surfaces) else
                 " (none yet: looked for at each LoadWorld)",
                 f"; its state in {files.state}" if files.state else "")
    if args.display:
        os.environ["DISPLAY"] = args.display
    else:
        os.environ.pop("DISPLAY", None)
    started = time.monotonic()
    from scensim_world_server.watchdog import StepWatchdog

    # Before Kit: Kit's start and the preload are watched from outside this process (issue
    # 042), and the same child then holds the requests' deadlines (`run`).
    watchdog = StepWatchdog()
    startup = watchdog.step("Kit's start and the preload", deadline=STARTUP_LIMIT)
    startup.__enter__()
    from aib import app_exit  # as the world's ground (aib.isaac.ground), aib is on the path
    from isaacsim import SimulationApp

    # Kit uploads a crash's minidump to NVIDIA by default: kept on disk only (its dumpDir).
    config = {"headless": not args.display, "extra_args": ["--/crashreporter/url="]}
    # Host-specific Kit settings (a host setting, ISAAC_KIT_ARGS): e.g. one GPU on a host
    # whose other GPU is busy, "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=1".
    config["extra_args"] += os.environ.get("ISAAC_KIT_ARGS", "").split()
    if args.display and args.follow and not args.ui:
        # A small render keeps the frame inside its 50 ms beside the lidar (1080p: 49.5 ms
        # p50 on a loaded host, 720p: 52); the window scales it up.
        config.update(hide_ui=True, width=854, height=480, window_width=1920,
                      window_height=1080)
    app = SimulationApp(config)
    code = 0
    try:
        _serve(args, app, socket, run, watchdog, startup, surface_maps, preload, started, log)
    except BaseException as e:  # app.close() would end the process with 0 (scensim issue 118)
        code = app_exit.code_of(e)
    finally:
        app_exit.close(app, code)


def _serve(args, app, socket, run, watchdog, startup, surface_maps, preload, started, log):
    """Kit is up: the world, its preload and the requests, until the app stops."""
    kit_log = kit_log_file()
    if kit_log:
        if not os.path.exists(kit_log):
            log.warning("Kit's log %s does not exist (yet): watched all the same", kit_log)
        watchdog.watch_log(kit_log, KIT_FATAL)
        log.info("Kit's log %s: a FATAL line in it ends this world server", kit_log)
    else:
        log.warning("Kit names no log file (/log/file): a Kit FATAL does not end this world "
                    "server")

    from .world import IsaacWorld

    world = IsaacWorld(physics_dt=1 / args.physics_hz,
                       ego_config=args.ego or None, ego_domain=args.ego_domain or None,
                       follow=bool(args.display and args.follow), surface_maps=surface_maps,
                       record=args.record or None, record_view=args.record_view,
                       state_dir=args.state_dir or None,
                       wheels_topic=args.wheels_topic == "true")
    # The first listed is preloaded last, so it is the one loaded when the bridge first asks.
    for usd, ground in reversed(preload):
        world.preload(usd, ground)
    startup.__exit__(None, None, None)
    # Kit opens its log at its start: one still missing now, at least 60 s on (KIT_LOG_GRACE),
    # means the child watches a file that never appears and a Kit FATAL ends nothing (042's
    # review: a token in /log/file could resolve to another name than Kit's file). Checked
    # here, not by a timer, which the GIL could hold up.
    if kit_log:
        wait = KIT_LOG_GRACE - (time.monotonic() - started)
        if wait > 0 and not os.path.exists(kit_log):
            time.sleep(wait)
        if not os.path.exists(kit_log):
            log.error("Kit's log %s does not exist %.0f s after Kit's start: a Kit FATAL does "
                      "not end this world server", kit_log, time.monotonic() - started)
    log.info("Isaac Sim %s up in %.1f s, physics at %g Hz", world.isaac_version,
             time.monotonic() - started, 1 / world.max_physics_dt)
    run(world, socket, should_stop=lambda: not app.is_running(), watchdog=watchdog)


if __name__ == "__main__":
    main()
