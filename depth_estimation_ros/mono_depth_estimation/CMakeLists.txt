cmake_minimum_required(VERSION 3.8)
project(mono_depth_estimation)

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
option(JETSON                            "Use Jetson"   OFF)

if(JETSON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include/aarch64-linux-gnu/ -L/usr/lib/aarch64-linux-gnu/")
  set(DEPTH_ESTIMATION_ROS_USE_TENSORRT ON)
endif()

if(NOT DEPTH_ESTIMATION_ROS_USE_TENSORRT)
  message(FATAL_ERROR "DEPTH_ESTIMATION_ROS_USE_TENSORRT must be ON")
  return()
endif()

set(ENABLE_TENSORRT OFF)

if(DEPTH_ESTIMATION_ROS_USE_TENSORRT)
  find_package(CUDA REQUIRED)
  if (NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
  endif()
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

  set(ENABLE_TENSORRT ON)
  set(TARGET_SRC ${TARGET_SRC} src/mono_depth_estimation_tensorrt.cpp)
  set(TARGET_LIBS ${TARGET_LIBS} nvinfer nvinfer_plugin)
  set(TARGET_DPENDENCIES ${TARGET_DPENDENCIES} CUDA)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/include/mono_depth_estimation/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/mono_depth_estimation/config.h"
)

ament_auto_add_library(mono_depth_estimation SHARED ${TARGET_SRC})
ament_target_dependencies(mono_depth_estimation ${TARGET_DPENDENCIES})
ament_export_dependencies(${TARGET_DPENDENCIES})
target_link_libraries(mono_depth_estimation ${TARGET_LIBS})

if (DEPTH_ESTIMATION_ROS_USE_TFLITE)
  target_include_directories(mono_depth_estimation PUBLIC ${TFLITE_INCLUDES})
  ament_export_include_directories(${TFLITE_INCLUDES})
  install(DIRECTORY ${TFLITE_LIB_PATH}/ DESTINATION lib)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
