// Copyright 2026 fateshelled
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "lingbot_depth_estimation/core.hpp"

namespace depth_estimation_ros
{
class LingbotDepthEstimationTensorRT : public AbcLingbotDepthEstimation
{
public:
  explicit LingbotDepthEstimationTensorRT(const std::string & engine_path, int device = 0);
  ~LingbotDepthEstimationTensorRT();

  LingbotDepthEstimationTensorRT(const LingbotDepthEstimationTensorRT &) = delete;
  LingbotDepthEstimationTensorRT & operator=(const LingbotDepthEstimationTensorRT &) = delete;

  // color_bgr must be CV_8UC3. depth_m must be CV_32FC1 in meters.
  cv::Mat inference(const cv::Mat & color_bgr, const cv::Mat & depth_m) override;
  int input_width() const noexcept override {return input_w_;}
  int input_height() const noexcept override {return input_h_;}

private:
  struct Tensor
  {
    std::string name;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    std::size_t elements{0};
    std::size_t bytes{0};
    void * device{nullptr};
  };

  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * message) noexcept override;
  };

  static std::size_t element_size(nvinfer1::DataType dtype);
  static std::size_t volume(const nvinfer1::Dims & dims);
  static void check_cuda(cudaError_t status, const char * operation);
  Tensor make_tensor(
    const std::string & preferred_name, nvinfer1::TensorIOMode mode,
    int fallback_ordinal);
  void encode_image(const cv::Mat & color_bgr);
  void encode_depth(const cv::Mat & depth_m);
  cv::Mat decode_output();

  int device_id_{0};
  int input_w_{0};
  int input_h_{0};
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{nullptr};
  Tensor image_;
  Tensor depth_;
  Tensor output_;
  std::vector<float> image_float_;
  std::vector<float> depth_float_;
  std::vector<float> output_float_;
  std::vector<std::uint16_t> image_half_;
  std::vector<std::uint16_t> depth_half_;
  std::vector<std::uint16_t> output_half_;
};
}  // namespace depth_estimation_ros
