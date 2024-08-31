#pragma once

#include "config.h"

#ifdef ENABLE_OPENVINO
    #include "mono_depth_estimation_openvino.hpp"
#endif

#ifdef ENABLE_TENSORRT
    #include "mono_depth_estimation_tensorrt.hpp"
#endif

#ifdef ENABLE_ONNXRUNTIME
    #include "mono_depth_estimation_onnxruntime.hpp"
#endif

#ifdef ENABLE_TFLITE
    #include "mono_depth_estimation_tflite.hpp"
#endif

