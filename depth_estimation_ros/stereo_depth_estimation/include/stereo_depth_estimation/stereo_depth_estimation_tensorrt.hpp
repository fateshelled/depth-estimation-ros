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


    class StereoDepthEstimationTensorRT: public AbcStereoDepthEstimation{
        public:
            StereoDepthEstimationTensorRT(
                const std::string &path_to_engine,
                bool input_normalize,
                const std::vector<double>& mean, const std::vector<double>& std,
                bool swap_r_b,
                int device=0);
            ~StereoDepthEstimationTensorRT();
            cv::Mat inference(const cv::Mat& left, const cv::Mat& right) override;

        private:
            void doInference(const float* left, const float* right, float* output);

            int DEVICE_ = 0;
            Logger gLogger_;
            std::unique_ptr<IRuntime> runtime_;
            std::unique_ptr<ICudaEngine> engine_;
            std::unique_ptr<IExecutionContext> context_;
            int input_size_;
            int output_size_;
            const int input_left_Index_ = 0;
            const int input_right_Index_ = 1;
            const int outputIndex_ = 2;
            void *inference_buffers_[3];
            std::vector<float> input_left_blob_;
            std::vector<float> input_right_blob_;
            std::vector<float> output_blob_;

    };
} // namespace depth_estimation_ros
