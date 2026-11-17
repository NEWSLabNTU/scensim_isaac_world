# Isaac Sim 6.1 environment (roadmap 036), for scensim's Isaac side: aib's (autoware_isaac_bridge,
# the submodule at deps/autoware_isaac_bridge; roadmap 045) -- the pip install's venv, Isaac's
# bundled ROS 2 Humble libraries, Autoware's messages rebuilt for it ($ISAAC_ROOT/msgs, shared
# by both repositories) -- plus aib's Python package on PYTHONPATH, which the world server
# imports for its ego, and the world server shell (`scensim_world_server`): from the
# installed prefixes (SCENSIM_PREFIX_PATH), or, sourced from a source tree, from its deps/.
# Do not source the system Humble into it. RMW_IMPLEMENTATION and CYCLONEDDS_URI pass through, so the stack's DDS settings apply.
# ISAAC_ROOT has no default (roadmap 077 G75): a fresh host's install is wherever it was put.
# What counts as an install is what scripts/install_isaac.sh makes: the venv and Isaac Sim in it.
# the host settings renamed by roadmap 080 are refused, naming the new
source "$(dirname "${BASH_SOURCE[0]}")/renamed_env.sh" || return 2 2>/dev/null || exit 2
if [ -z "${ISAAC_ROOT:-}" ]; then
    echo "isaac_env.sh: ISAAC_ROOT is not set: it names Isaac Sim 6.1's pip install" \
         "(scripts/install_isaac.sh); set isaac.root in ~/.config/scensim/host.yaml, or export ISAAC_ROOT" >&2
    return 2 2>/dev/null || exit 2
fi
for _scensim_need in venv/bin/activate venv/lib/python3.12/site-packages/isaacsim; do
    if [ ! -e "$ISAAC_ROOT/$_scensim_need" ]; then
        echo "isaac_env.sh: ISAAC_ROOT=$ISAAC_ROOT is not an Isaac Sim install: no" \
             "$ISAAC_ROOT/$_scensim_need (scripts/install_isaac.sh makes it)" >&2
        unset _scensim_need
        return 2 2>/dev/null || exit 2
    fi
done
unset _scensim_need
_scensim_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_scensim_here/../scensim_isaac_world" ]; then
    # A source tree (SCENSIM_ISAAC_DIR, a developer's): aib and the shell from its deps/.
    _scensim_aib="$(cd "$_scensim_here/../deps/autoware_isaac_bridge/src/aib" && pwd)" || {
        echo "isaac_env.sh: no aib at deps/autoware_isaac_bridge (git submodule update --init --recursive)" >&2
        return 1 2>/dev/null || exit 1
    }
    # shellcheck source=/dev/null
    source "$_scensim_aib/scripts/isaac_env.sh"
    # The world protocol's server shell (roadmap 046), beside this package.
    _scensim_shell="$(cd "$_scensim_here/../deps/scensim_world_protocol/scensim_world_server" && pwd)"
    export PYTHONPATH="$_scensim_shell:$_scensim_aib:$PYTHONPATH"
    unset _scensim_aib _scensim_shell
else
    # Installed (share/scensim_isaac_world/scripts; roadmap 082): every package from the
    # prefixes the launch ran under, SCENSIM_PREFIX_PATH (the scripts keep AMENT_PREFIX_PATH
    # there before dropping the system ROS environment). Their site-packages go after Isaac's
    # paths, so Isaac's own ROS 2 Python (its messages) is found first.
    if [ -z "${SCENSIM_PREFIX_PATH:-}" ]; then
        echo "isaac_env.sh: SCENSIM_PREFIX_PATH is not set: source scensim's setup.bash and run" \
             "this through ros2 run scensim_isaac_world (or set SCENSIM_ISAAC_DIR to a source tree)" >&2
        return 2 2>/dev/null || exit 2
    fi
    _scensim_aib_env=""
    _scensim_py=""
    IFS=: read -r -a _scensim_prefixes <<< "$SCENSIM_PREFIX_PATH"
    for _scensim_p in "${_scensim_prefixes[@]}"; do
        if [ -z "$_scensim_aib_env" ] && [ -f "$_scensim_p/share/aib/scripts/isaac_env.sh" ]; then
            _scensim_aib_env="$_scensim_p/share/aib/scripts/isaac_env.sh"
        fi
    done
    if [ -z "$_scensim_aib_env" ]; then
        echo "isaac_env.sh: no aib in SCENSIM_PREFIX_PATH (share/aib/scripts/isaac_env.sh): build or" \
             "install aib (autoware_isaac_bridge) and source its setup.bash" >&2
        return 1 2>/dev/null || exit 1
    fi
    # shellcheck source=/dev/null
    source "$_scensim_aib_env"
    for _scensim_m in scensim_isaac_world scensim_world_server aib sim_vehicle_spec sim_vehicle_ecu; do
        _scensim_found=""
        for _scensim_p in "${_scensim_prefixes[@]}"; do
            # ament_python's site-packages; ament_cmake_python's dist-packages (Debian's layout)
            for _scensim_d in "$_scensim_p"/lib/python3*/site-packages \
                              "$_scensim_p"/local/lib/python3*/dist-packages \
                              "$_scensim_p"/lib/python3*/dist-packages; do
                if [ -d "$_scensim_d/$_scensim_m" ]; then _scensim_found="$_scensim_d"; break 2; fi
            done
        done
        if [ -z "$_scensim_found" ]; then
            echo "isaac_env.sh: no Python package $_scensim_m in SCENSIM_PREFIX_PATH's" \
                 "site-packages or dist-packages: build or install it and source its setup.bash" >&2
            return 1 2>/dev/null || exit 1
        fi
        case ":$_scensim_py:" in *":$_scensim_found:"*) ;; *) _scensim_py="$_scensim_py:$_scensim_found" ;; esac
    done
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH}$_scensim_py"
    PYTHONPATH="${PYTHONPATH#:}"
    unset _scensim_aib_env _scensim_py _scensim_prefixes _scensim_p _scensim_m _scensim_found _scensim_d
fi
unset _scensim_here
