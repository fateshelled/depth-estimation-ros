cmake_minimum_required(VERSION 3.8)
project(lingbot_depth_estimation)

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

option(DEPTH_ESTIMATION_ROS_USE_TENSORRT "Use TensorRT" ON)
option(JETSON "Use Jetson TensorRT/CUDA paths" OFF)
if(JETSON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include/aarch64-linux-gnu/ -L/usr/lib/aarch64-linux-gnu/")
  set(DEPTH_ESTIMATION_ROS_USE_TENSORRT ON)
endif()

if(NOT DEPTH_ESTIMATION_ROS_USE_TENSORRT)
  message(FATAL_ERROR "DEPTH_ESTIMATION_ROS_USE_TENSORRT must be ON")
endif()

set(ENABLE_TENSORRT ON)

find_package(CUDA REQUIRED)
if(NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "NvInfer.h was not found; set TENSORRT_ROOT")
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/include/lingbot_depth_estimation/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/lingbot_depth_estimation/config.h"
)

ament_auto_add_library(lingbot_depth_estimation SHARED
  src/lingbot_depth_estimation_tensorrt.cpp
)
target_include_directories(lingbot_depth_estimation PUBLIC ${TENSORRT_INCLUDE_DIR})
ament_target_dependencies(lingbot_depth_estimation CUDA)
ament_export_dependencies(CUDA)
target_link_libraries(lingbot_depth_estimation nvinfer)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
