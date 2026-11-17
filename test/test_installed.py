"""The Isaac world server from an installed, relocated tree (roadmap 082, the tarball): a plain
(non-symlink) merged colcon install of this package and what it imports -- the protocol's
Python shell, aib, sim_vehicle_spec, sim_vehicle_ecu -- moved to another directory, runs its
imports from there, nothing from a source tree.

Needs ROS 2 Humble and colcon; the Isaac half also ISAAC_ROOT (Isaac Sim's pip install).
Skipped without them. Slow (a colcon build): ~1-2 min.
"""
import os
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
AIB = REPO / "deps" / "autoware_isaac_bridge"
SOURCES = {
    "scensim_isaac_world": REPO,
    "scensim_world_server": REPO / "deps" / "scensim_world_protocol" / "scensim_world_server",
    "aib": AIB / "src" / "aib",
    "sim_vehicle_spec": AIB / "deps" / "sim_vehicle_spec",
    "sim_vehicle_ecu": AIB / "deps" / "sim_vehicle_ecu",
}
ROS = Path("/opt/ros/humble/setup.bash")
ISAAC_ROOT = os.environ.get("ISAAC_ROOT", "")
# never copied into the workspace: build products, the source tree's own deps, spikes
SKIP = shutil.ignore_patterns("build", "install", "log", "deps", "spikes", "__pycache__",
                              ".git", "play_log")
# the same, keeping deps/ (COLCON_IGNOREd in it, as in a checkout)
SKIP_KEEP_DEPS = shutil.ignore_patterns("build", "install", "log", "spikes", "__pycache__",
                                        ".git", "play_log")

pytestmark = pytest.mark.skipif(not ROS.is_file() or shutil.which("colcon") is None,
                                reason="needs ROS 2 Humble and colcon")

ECU_STAND_IN = {
    # sim_vehicle_ecu has no package.xml yet (118 adds it, roadmap 082 item 2): a stand-in
    # ament_python wrapper around its own source, as it will install
    "package.xml": textwrap.dedent("""\
        <?xml version="1.0"?>
        <package format="3"><name>sim_vehicle_ecu</name><version>0.0.0</version>
        <description>test stand-in</description><maintainer email="t@e.com">t</maintainer>
        <license>Apache-2.0</license><export><build_type>ament_python</build_type></export>
        </package>
        """),
    "setup.py": textwrap.dedent("""\
        from setuptools import find_packages, setup
        setup(name="sim_vehicle_ecu", version="0.0.0",
              packages=find_packages(include=["sim_vehicle_ecu", "sim_vehicle_ecu.*"]),
              data_files=[("share/ament_index/resource_index/packages",
                           ["resource/sim_vehicle_ecu"]),
                          ("share/sim_vehicle_ecu", ["package.xml"])])
        """),
    "resource/sim_vehicle_ecu": "",
}


def build(ws):
    """A plain merged build from a clean environment: a build run with other workspaces
    sourced records them as underlays in its setup.bash, and their packages (an old aib) would
    then be found from it."""
    env = {"HOME": os.environ["HOME"], "USER": os.environ.get("USER", ""),
           "PATH": f"{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin"}
    return subprocess.run(["bash", "-c", f"source {ROS} && colcon build --merge-install"],
                          cwd=ws, env=env, capture_output=True, text=True)


@pytest.fixture(scope="module")
def prefix(tmp_path_factory):
    """The merged install, built plainly, then moved: ``<tmp>/moved``."""
    root = tmp_path_factory.mktemp("installed")
    src = root / "ws" / "src"
    for name, path in SOURCES.items():
        shutil.copytree(path, src / name, ignore=SKIP)
    ecu = src / "sim_vehicle_ecu"
    if not (ecu / "package.xml").is_file():
        for rel, text in ECU_STAND_IN.items():
            (ecu / rel).parent.mkdir(parents=True, exist_ok=True)
            (ecu / rel).write_text(text)
    r = build(root / "ws")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    moved = root / "moved"
    (root / "ws" / "install").rename(moved)
    shutil.rmtree(root / "ws")            # no build or source tree left to fall back on
    return moved


@pytest.fixture(scope="module")
def sim_only(tmp_path_factory):
    """A source build on a simulation-only host: scensim_isaac_world (its checkout with deps/,
    which colcon skips), the shell and sim_vehicle_spec; no aib, no sim_vehicle_ecu in any
    prefix (autoware_isaac_bridge is the vehicle side's, not checked out). ``(ws, prefix)``,
    not moved."""
    ws = tmp_path_factory.mktemp("sim_only") / "ws"
    shutil.copytree(REPO, ws / "src" / "scensim_isaac_world", ignore=SKIP_KEEP_DEPS)
    for name in ("scensim_world_server", "sim_vehicle_spec"):
        shutil.copytree(SOURCES[name], ws / "src" / name, ignore=SKIP)
    r = build(ws)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert not list((ws / "install").glob("*/python3*/*-packages/aib"))
    return ws, ws / "install"


def isaac_probe(prefix, mods):
    """As scensim_isaac_world_server does it, up to its `python -m`: AMENT_PREFIX_PATH kept,
    the system ROS environment dropped, the installed isaac_env.sh sourced; then each of
    ``mods`` imported, its file printed, and where aib's config was found."""
    probe = ("import importlib, os, sys\n"
             f"for m in {mods!r}:\n"
             "    print(m, importlib.import_module(m).__file__)\n"
             "print('python', sys.version.split()[0])\n"
             "print('aib_config', os.environ.get('SCENSIM_AIB_CONFIG', ''))\n")
    return clean(f"source {ROS} && source {prefix}/setup.bash && "
                 "export SCENSIM_PREFIX_PATH=\"$AMENT_PREFIX_PATH\" && "
                 "unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH "
                 "COLCON_PREFIX_PATH ROS_VERSION ROS_PYTHON_VERSION VIRTUAL_ENV && "
                 f"source {prefix}/share/scensim_isaac_world/scripts/isaac_env.sh && "
                 f"python -c {shlex.quote(probe)}")


def clean(script, cwd="/"):
    env = {"HOME": os.environ["HOME"], "USER": os.environ.get("USER", ""),
           "PATH": "/usr/bin:/bin", "ISAAC_ROOT": ISAAC_ROOT}
    return subprocess.run(["bash", "-c", script], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=300)


def test_the_package_installs_with_its_data(prefix):
    pkg = next(prefix.glob("local/lib/python3*/dist-packages/scensim_isaac_world"))
    for f in ("world_server.py", "walker_blueprints.json", "courses/tsukuba.yaml"):
        assert (pkg / f).is_file(), f
    assert (prefix / "share/scensim_isaac_world/scripts/isaac_env.sh").is_file()
    assert (prefix / "share/scensim_isaac_world/scripts/renamed_env.sh").is_file()
    assert (prefix / "share/aib/config/ego.yaml").is_file()


def test_a_system_python_entry_runs_from_the_moved_prefix(prefix):
    r = clean(f"source {ROS} && source {prefix}/setup.bash && "
              "ros2 run scensim_isaac_world road_detail --help")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "usage" in r.stdout.lower()


@pytest.mark.skipif(not (Path(ISAAC_ROOT) / "venv" / "bin" / "activate").is_file(),
                    reason="needs ISAAC_ROOT: Isaac Sim's pip install")
def test_the_world_servers_imports_come_from_the_moved_prefix_in_isaacs_python(prefix):
    """As scensim_isaac_world_server does it, up to its `python -m`: AMENT_PREFIX_PATH kept,
    the system ROS environment dropped, the installed isaac_env.sh sourced; then what the
    world server imports outside Kit, each from the moved prefix."""
    mods = ["scensim_isaac_world.world_server", "scensim_isaac_world.catalogue",
            "scensim_world_server.server", "scensim_world_server.protocol", "aib.rig",
            "sim_vehicle_spec", "sim_vehicle_ecu"]
    probe = ("import importlib, sys\n"
             f"for m in {mods!r}:\n"
             "    print(m, importlib.import_module(m).__file__)\n"
             "print('python', sys.version.split()[0])\n")
    r = clean(f"source {ROS} && source {prefix}/setup.bash && "
              "export SCENSIM_PREFIX_PATH=\"$AMENT_PREFIX_PATH\" && "
              "unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH "
              "COLCON_PREFIX_PATH ROS_VERSION ROS_PYTHON_VERSION VIRTUAL_ENV && "
              f"source {prefix}/share/scensim_isaac_world/scripts/isaac_env.sh && "
              f"python -c {shlex.quote(probe)}")
    assert r.returncode == 0, r.stdout + r.stderr
    files = dict(line.split(" ", 1) for line in r.stdout.splitlines())
    assert files["python"].startswith("3.12"), files["python"]
    for m in mods:
        assert files[m].startswith(str(prefix)), (m, files[m])
        assert str(REPO) not in files[m], (m, files[m])


NEEDS_ISAAC = pytest.mark.skipif(not (Path(ISAAC_ROOT) / "venv" / "bin" / "activate").is_file(),
                                 reason="needs ISAAC_ROOT: Isaac Sim's pip install")


@NEEDS_ISAAC
def test_a_simulation_only_source_build_takes_aib_from_its_checkouts_deps(sim_only):
    """133: no prefix has aib or sim_vehicle_ecu; the prefix the package was installed to falls
    back to the source tree it was built from (deps/), the rest still from the prefix."""
    ws, prefix = sim_only
    tree = ws / "src" / "scensim_isaac_world"
    mods = ["scensim_isaac_world.world_server", "scensim_world_server.server", "aib.rig",
            "sim_vehicle_spec", "sim_vehicle_ecu"]
    r = isaac_probe(prefix, mods)
    assert r.returncode == 0, r.stdout + r.stderr
    files = dict(line.split(" ", 1) for line in r.stdout.splitlines())
    for m in ("scensim_isaac_world.world_server", "scensim_world_server.server"):
        assert files[m].startswith(str(prefix)), (m, files[m])
    # aib from a checkout brings its own pinned sim_vehicle_spec and sim_vehicle_ecu
    # (aib.deps.use puts its deps/ first), as it does in every source-tree run
    for m in ("aib.rig", "sim_vehicle_spec", "sim_vehicle_ecu"):
        assert files[m].startswith(str(tree / "deps" / "autoware_isaac_bridge")), (m, files[m])
    assert files["aib_config"] == str(tree / "deps/autoware_isaac_bridge/src/aib/config")


def test_a_relocated_install_never_reads_the_tree_it_was_built_from(sim_only, tmp_path):
    """The same install copied elsewhere (a tarball), its source tree still on disk: no aib,
    refused by name rather than taken from that tree."""
    ws, prefix = sim_only
    moved = tmp_path / "moved"
    shutil.copytree(prefix, moved, symlinks=True)
    r = clean(f"source {ROS} && source {moved}/setup.bash && "
              "export SCENSIM_PREFIX_PATH=\"$AMENT_PREFIX_PATH\" && unset AMENT_PREFIX_PATH && "
              f"source {moved}/share/scensim_isaac_world/scripts/isaac_env.sh")
    assert r.returncode != 0
    assert "isaac_env.sh: no aib (the Isaac ego" in r.stderr, r.stderr
