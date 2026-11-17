cmake_minimum_required(VERSION 3.14)
project(scensim_isaac_world)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# The Python package under the prefix (roadmap 082): <prefix>/lib/python3.X/site-packages/
# scensim_isaac_world, data files with it (walker_blueprints.json, courses/, examples/). The
# scripts run it from there in Isaac's Python; SCENSIM_ISAAC_DIR (a source tree) overrides it
# for a developer.
ament_python_install_package(${PROJECT_NAME})
# Where this was built from and installed to: a source build on a simulation-only host has no
# aib (or sim_vehicle_ecu) in any prefix (autoware_isaac_bridge is the vehicle side's, and this
# tree's deps/ are COLCON_IGNOREd), so isaac_env.sh falls back to this tree's deps/ -- only
# while it runs from this prefix: a relocated install (a tarball) never reads the source tree.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/built_from
     "SCENSIM_BUILT_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}\nSCENSIM_BUILT_PREFIX=${CMAKE_INSTALL_PREFIX}\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/built_from DESTINATION share/${PROJECT_NAME})
# isaac_env.sh (and renamed_env.sh, which it sources), for the scripts below
install(PROGRAMS scripts/isaac_env.sh scripts/renamed_env.sh
        DESTINATION share/${PROJECT_NAME}/scripts)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
# The examples a user copies into a map or project (the rough road's road_detail.yaml files;
# roadmap 076 A9): share/scensim_isaac_world/examples/.
install(DIRECTORY scensim_isaac_world/examples/ DESTINATION share/${PROJECT_NAME}/examples)

# `scensim_isaac_world_server`: the Isaac world server as a launch node (scensim_isaac_world.launch.xml,
# roadmap 043). It runs the installed Python package in Isaac's own Python
# (scripts/isaac_env.sh); SCENSIM_ISAAC_DIR, a source tree, overrides it at run time.
configure_file(scripts/scensim_isaac_world_server.in ${CMAKE_CURRENT_BINARY_DIR}/scensim_isaac_world_server @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/scensim_isaac_world_server DESTINATION lib/${PROJECT_NAME})

# `ros2 run scensim_isaac_world <entry>`: a module of the installed package (SCENSIM_ISAAC_DIR:
# a source tree's; scripts/scensim_isaac_world_entry.in), in Isaac's Python (isaac) or the system's (system).
#   make_world   a map's Isaac world: export, physics layer, world file (`make_world`)
#   road_detail  check or picture a road_detail.yaml (`road_layer`)
#   surfaces     check or picture a surfaces.yaml (`surfaces`)
foreach(entry "make_world:scensim_isaac_world.make_world:isaac"
              "road_detail:scensim_isaac_world.road_layer:system"
              "surfaces:scensim_isaac_world.surfaces:system")
  string(REPLACE ":" ";" parts "${entry}")
  list(GET parts 0 ENTRY_NAME)
  list(GET parts 1 ENTRY_MODULE)
  list(GET parts 2 ENTRY_PYTHON)
  configure_file(scripts/scensim_isaac_world_entry.in ${CMAKE_CURRENT_BINARY_DIR}/${ENTRY_NAME} @ONLY)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${ENTRY_NAME} DESTINATION lib/${PROJECT_NAME})
endforeach()

ament_package()
