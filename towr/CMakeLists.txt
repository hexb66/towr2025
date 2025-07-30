cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(towr VERSION 1.3.2)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(ifopt 2.0.1 REQUIRED)


###########
## Build ##
###########
IF(NOT CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "CMAKE_BUILD_TYPE not set to Release -> impacts performance")
endif()

# so dynamic library libtowr.so retains link to ifopt_core.so
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
)

# The motion-planning problem formulated through variables and constraints
add_library(${PROJECT_NAME} SHARED
  # sample formulation usage
  src/nlp_formulation.cc
  src/parameters.cc
  # variables
  src/variables/nodes_variables.cc
  src/variables/nodes_variables_all.cc
  src/variables/nodes_variables_phase_based.cc
  src/variables/phase_durations.cc
  # models
  src/models/robot_model.cc
  src/models/dynamic_model.cc
  src/models/single_rigid_body_dynamics.cc
  # constraints
  src/constraints/time_discretization_constraint.cc
  src/constraints/base_motion_constraint.cc
  src/constraints/terrain_constraint.cc
  src/constraints/terrain_constraint_hard.cc
  src/constraints/swing_constraint.cc
  src/constraints/force_constraint.cc
  src/constraints/torque_constraint.cc
  src/constraints/total_duration_constraint.cc
  src/constraints/dynamic_constraint.cc
  src/constraints/range_of_motion_constraint.cc
  src/constraints/spline_acc_constraint.cc
  src/constraints/linear_constraint.cc
  src/constraints/base_height_constraint.cc
  # costs
  src/costs/node_cost.cc
  src/costs/soft_constraint.cc
  src/costs/base_height_cost.cc
  # initialization
  src/initialization/gait_generator.cc
  src/initialization/monoped_gait_generator.cc
  src/initialization/biped_gait_generator.cc
  src/initialization/quadruped_gait_generator.cc
  # terrain
  src/terrain/height_map_examples.cc
  src/terrain/height_map.cc
  # helpers
  src/helpers/state.cc
  src/helpers/polynomial.cc
  src/helpers/spline.cc
  src/helpers/phase_spline.cc
  src/helpers/node_spline.cc
  src/helpers/nodes_observer.cc
  src/helpers/spline_holder.cc
  src/helpers/euler_converter.cc
  src/helpers/phase_durations_observer.cc
)
target_link_libraries(${PROJECT_NAME} 
  PUBLIC 
    ifopt::ifopt_core
)
target_include_directories(${PROJECT_NAME} 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


#############
## Testing ##
#############
enable_testing()
# example problem solved with IPOPT
add_executable(${PROJECT_NAME}-example 
  test/hopper_example.cc
)
target_link_libraries(${PROJECT_NAME}-example  
  PRIVATE
    ${PROJECT_NAME} 
    ifopt::ifopt_ipopt
)
add_test(${PROJECT_NAME}-example ${PROJECT_NAME}-example)

add_executable(${PROJECT_NAME}-example-2 
  test/biped_example.cc
)
target_link_libraries(${PROJECT_NAME}-example-2  
  PRIVATE
    ${PROJECT_NAME} 
    ifopt::ifopt_ipopt
)

# unit tests of costs/constraints
find_package(GTest QUIET)
if (TARGET GTest::GTest) # only build when modern targets exists

  add_executable(${PROJECT_NAME}-test
    test/dynamic_constraint_test.cc
    test/dynamic_model_test.cc
  )
  target_link_libraries(${PROJECT_NAME}-test
    PRIVATE
      ${PROJECT_NAME}
      GTest::GTest GTest::Main
  )
  add_test(${PROJECT_NAME}-test ${PROJECT_NAME}-test)
endif()


#############
## Install ##
#############
include(GNUInstallDirs) # for correct libraries locations across platforms
set(config_package_location "share/${PROJECT_NAME}/cmake") # for .cmake find-scripts installs
install(
  TARGETS ${PROJECT_NAME} ${PROJECT_NAME}-example 
  EXPORT ${PROJECT_NAME}-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/towr
)
# Copy header files
install(
  DIRECTORY include/towr/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/towr
)
# Auto generate and install the find scripts helpers
install(
  EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE towr::
  DESTINATION ${config_package_location}
)
# generate the file TowrConfigVersion.cmake
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/towr-config-version.cmake")
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${version_file}
  VERSION ${towr_VERSION}
  COMPATIBILITY SameMajorVersion
)
# install the two files in a place where cmake looks for them
install(
  FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/towr-config.cmake"   # self-written
    ${version_file}                                         # generated
  DESTINATION ${config_package_location}
)
# Install a Catkin 'package.xml' file. This is required by REP-136.
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
