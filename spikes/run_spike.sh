#!/bin/bash
# Phase 036: run a spike under Isaac's Python with the bundled ROS 2 libraries.
# Usage: run_spike.sh <rmw: fastrtps|cyclonedds> <log> <spike.py> [args...]
set -u
rmw=$1; log=$2; shift 2
# the log and the spike as given, resolved before the cd into ISAAC_ROOT (a relative path from
# where it was run would otherwise name a file under the install; 241, roadmap 095)
log=$(realpath -m "$log"); spike=$(realpath -m "$1"); shift
mkdir -p "$(dirname "$log")"   # a fresh ISAAC_ROOT has no logs/ yet (243)
export RMW_IMPLEMENTATION=rmw_${rmw}_cpp
source "$(dirname "$0")/../scripts/isaac_env.sh"
cd "${ISAAC_ROOT:?ISAAC_ROOT is not set: the Isaac Sim 6.2 pip install}"
exec python "$spike" "$@" > "$log" 2>&1
