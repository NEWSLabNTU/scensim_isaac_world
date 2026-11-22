"""A course from open data (roadmap 058): a real road's section, with its terrain and aerial
photo, as an Isaac world and Autoware's maps.

    python3 -m scensim_isaac_world.course <course.yaml> <map dir> [steps...]

`courses/<name>.yaml` names the road (an Overpass filter), the section's two ends, the lanes and
the map frame's origin. The data is fetched once into `<map dir>/raw/` (OpenStreetMap through
Overpass; GSI's 5 m DEM and aerial photo tiles) and reused offline after. The geometry and the
Autoware maps run under the system Python (pyproj, lanelet2); the USD world is written by
Isaac's Python from the geometry those steps leave in `<map dir>/course/`.

The `record` step (scensim roadmap 086 C3, after `geometry`) writes the course's road record,
`course/road_detail.yaml` and `course/section.npz` (`record`): the section per station and
what lies past each edge, which the guardrails are placed from. On its own
(`... <map dir> record`) it reads the earlier steps' files and fetches nothing.

A fixture course (086 C5b, `courses/fixture_c5.yaml`) has a `fixture:` block in place of the
road's data: its plan, grade and section table (`fixture`). Nothing is fetched; its record is
the table, and its road mesh carries the record's crown, bank and lip. `check` compares a
course's road mesh (and with `--layer` the road layer's tiles) with its record, and
`isaac_check`, under Isaac's Python, Isaac's road by PhysX's scene queries:

    python3 -m scensim_isaac_world.course.check <map dir> [--layer]
    python -m scensim_isaac_world.course.isaac_check <map dir>    # Isaac's Python
"""
