cmake_minimum_required(VERSION 3.8)
project(depth_estimation_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

if(NOT mono_depth_estimation_FOUND)
  message(WARNING "skipping depth_estimation_ros, not found mono_depth_estimation")
  return()
endif()

ament_auto_add_library(depth_estimation_ros SHARED
  src/depth_estimation_ros.cpp
)
rclcpp_components_register_node(
  depth_estimation_ros
  PLUGIN "depth_estimation_ros::DepthEstimationNode"
  EXECUTABLE depth_estimation_ros_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  # INSTALL_TO_SHARE
  # launch
)
