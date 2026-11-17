"""Host settings renamed by roadmap 080 (scensim): `(old, new)`. An old name set in the
environment is refused, naming the new one, never read as an alias. Pure Python, no Isaac.
`scripts/renamed_env.sh` holds the same list for the shell scripts (a test keeps them equal).

A world file's `${SWB_CARLA_USD}` is not a setting but a data-format name in published map
datasets: `names.expand` resolves it to SCENSIM_CARLA_USD's value until the datasets carry the
new token (080's data-format exceptions)."""

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


def refused(environ):
    """One message per old name set in `environ`: `SWB_X is renamed SCENSIM_X`."""
    return [f"{old} is renamed {new} (roadmap 080, scensim)"
            for old, new in RENAMED_ENV if old in environ]
