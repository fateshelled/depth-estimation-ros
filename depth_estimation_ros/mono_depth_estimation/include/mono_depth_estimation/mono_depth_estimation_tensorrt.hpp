#pragma once

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "core.hpp"
#include "tensorrt_logging.h"

namespace depth_estimation_ros{
    using namespace nvinfer1;

    #define CHECK(status) \
        do\
        {\
            auto ret = (status);\
            if (ret != 0)\
            {\
                std::cerr << "CUDA Failure: " << ret << std::endl;\
                abort();\
            }\
        } while (0)


    class MonoDepthEstimationTensorRT: public AbcMonoDepthEstimation{
        public:
            MonoDepthEstimationTensorRT(
                const std::string &path_to_engine,
                const std::vector<double>& mean, const std::vector<double>& std,
                int device=0);
            ~MonoDepthEstimationTensorRT();
            cv::Mat inference(const cv::Mat& frame) override;

        private:
            void doInference(const float* input, float* output);

            int DEVICE_ = 0;
            Logger gLogger_;
            std::unique_ptr<IRuntime> runtime_;
            std::unique_ptr<ICudaEngine> engine_;
            std::unique_ptr<IExecutionContext> context_;
            int input_size_;
            int output_size_;
            const int inputIndex_ = 0;
            const int outputIndex_ = 1;
            void *inference_buffers_[2];
            std::vector<float> input_blob_;
            std::vector<float> output_blob_;

    };
} // namespace depth_estimation_ros
