# Host settings renamed by roadmap 080 (scensim), for the shell scripts: an old name set in the
# environment is refused, naming the new one (scensim_isaac_world/renamed.py holds the same
# list; a test keeps them equal). Sourced: `source .../renamed_env.sh || exit 2`.
_scensim_renamed=(
    SWB_CARLA_USD:SCENSIM_CARLA_USD
    SWB_VEHICLE_USD:SCENSIM_VEHICLE_USD
    SWB_ISAAC_DIR:SCENSIM_ISAAC_DIR
    SWB_TRACKED_FRICTION:SCENSIM_TRACKED_FRICTION
    SWB_VEH_KIND:SCENSIM_VEH_KIND
    SWB_VEH_ONLY:SCENSIM_VEH_ONLY
    SWB_VEH_META:SCENSIM_VEH_META
    SWB_PHYS_OUT:SCENSIM_PHYS_OUT
    SWB_REPO:SCENSIM_REPO
)
_scensim_refused=0
for _pair in "${_scensim_renamed[@]}"; do
    if printenv "${_pair%%:*}" >/dev/null; then   # set in the environment, empty or not
        echo "${_pair%%:*} is renamed ${_pair##*:} (roadmap 080, scensim)" >&2
        _scensim_refused=1
    fi
done
unset _pair _scensim_renamed
[ "$_scensim_refused" = 0 ]
