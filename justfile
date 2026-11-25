# scensim_isaac_world: Isaac Sim behind the world protocol. Its dependencies, scensim_world_protocol
# (the Python shell) and autoware_isaac_bridge (the ego), are submodules under deps/
# (git submodule update --init --recursive). ISAAC_ROOT: the Isaac Sim 6.2 pip install.

isaac_root := env_var_or_default('ISAAC_ROOT', '')  # no default (077 G75): set it in the environment

default:
    @just --list

# Build the ROS package (the launch node wrapper, launch file, config)
build:
    #!/usr/bin/env bash
    set -e
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install --base-paths .

# Tests that need no Isaac (the pure modules); pxr and warp ones skip outside Isaac's Python.
# Needs the deps/ checked out recursively (aib finds sim_vehicle_ecu in its own deps/). Shielded
# from a user-site numpy 2 (pip --user) that breaks the system packages built for numpy 1
# (PYTHONNOUSERSITE, as scensim's lane), with the world protocol's shell from deps/ (scensim
# issue 184). The ROS-only tests skip where ROS 2's launch is not importable.
test:
    PYTHONNOUSERSITE=1 PYTHONPATH=deps/scensim_world_protocol/scensim_world_server${PYTHONPATH:+:$PYTHONPATH} python3 -m unittest discover -s test

# The tests, then pyflakes
check: test
    python3 -m pyflakes scensim_isaac_world test scripts

# Autoware's message bindings for Isaac's Python (once per Isaac install)
msgs:
    ISAAC_ROOT={{isaac_root}} scripts/build_msgs.sh
