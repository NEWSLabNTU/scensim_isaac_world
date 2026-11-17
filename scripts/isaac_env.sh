# Isaac Sim 6.1 environment (roadmap 036), for scensim's Isaac side: aib's (autoware_isaac_bridge,
# the submodule at deps/autoware_isaac_bridge; roadmap 045) -- the pip install's venv, Isaac's
# bundled ROS 2 Humble libraries, Autoware's messages rebuilt for it ($ISAAC_ROOT/msgs, shared
# by both repositories) -- plus aib's Python package on PYTHONPATH, which the world server
# imports for its ego, and the world server shell (`scensim_world_server`). Do not source the system Humble into it. RMW_IMPLEMENTATION and
# CYCLONEDDS_URI pass through, so the stack's DDS settings apply.
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
_scensim_aib="$(cd "$(dirname "${BASH_SOURCE[0]}")/../deps/autoware_isaac_bridge/src/aib" && pwd)" || {
    echo "isaac_env.sh: no aib at deps/autoware_isaac_bridge (git submodule update --init --recursive)" >&2
    return 1 2>/dev/null || exit 1
}
# shellcheck source=/dev/null
source "$_scensim_aib/scripts/isaac_env.sh"
# The world protocol's server shell (roadmap 046), beside this package.
_scensim_shell="$(cd "$(dirname "${BASH_SOURCE[0]}")/../deps/swb_world_protocol/scensim_world_server" && pwd)"
export PYTHONPATH="$_scensim_shell:$_scensim_aib:$PYTHONPATH"
unset _scensim_aib _scensim_shell
