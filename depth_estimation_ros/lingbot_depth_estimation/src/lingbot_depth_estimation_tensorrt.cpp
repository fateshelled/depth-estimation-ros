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

#include "lingbot_depth_estimation/lingbot_depth_estimation_tensorrt.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace depth_estimation_ros
{
namespace
{
template<typename T>
std::unique_ptr<T> make_trt_unique(T * pointer, const char * what)
{
  if (pointer == nullptr) {
    throw std::runtime_error(std::string("TensorRT failed to create ") + what);
  }
  return std::unique_ptr<T>(pointer);
}

void float_to_half(const std::vector<float> & source, std::vector<std::uint16_t> & destination)
{
  static_assert(sizeof(__half) == sizeof(std::uint16_t));
  if (destination.size() != source.size()) {
    destination.resize(source.size());
  }
  for (std::size_t i = 0; i < source.size(); ++i) {
    const __half value = __float2half(source[i]);
    const __half_raw raw = value;
    destination[i] = raw.x;
  }
}
}  // namespace

void LingbotDepthEstimationTensorRT::Logger::log(
  Severity severity, const char * message) noexcept
{
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TensorRT] " << message << '\n';
  }
}

void LingbotDepthEstimationTensorRT::check_cuda(
  cudaError_t status, const char * operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::size_t LingbotDepthEstimationTensorRT::element_size(nvinfer1::DataType dtype)
{
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(std::uint16_t);
    default: throw std::runtime_error("LingBot-Depth supports only FP32 and FP16 tensors");
  }
}

std::size_t LingbotDepthEstimationTensorRT::volume(const nvinfer1::Dims & dims)
{
  std::size_t result = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      throw std::runtime_error("LingBot-Depth requires a fixed-shape TensorRT engine");
    }
    result *= static_cast<std::size_t>(dims.d[i]);
  }
  return result;
}

LingbotDepthEstimationTensorRT::Tensor LingbotDepthEstimationTensorRT::make_tensor(
  const std::string & preferred_name, nvinfer1::TensorIOMode mode, int fallback_ordinal)
{
  const char * selected = nullptr;
  int ordinal = 0;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) != mode) {
      continue;
    }
    if (preferred_name == name) {
      selected = name;
      break;
    }
    if (ordinal++ == fallback_ordinal) {
      selected = name;
    }
  }
  if (selected == nullptr) {
    throw std::runtime_error("TensorRT engine is missing tensor: " + preferred_name);
  }

  Tensor tensor;
  tensor.name = selected;
  tensor.dims = engine_->getTensorShape(selected);
  tensor.dtype = engine_->getTensorDataType(selected);
  tensor.elements = volume(tensor.dims);
  tensor.bytes = tensor.elements * element_size(tensor.dtype);
  check_cuda(cudaMalloc(&tensor.device, tensor.bytes), "cudaMalloc");
  return tensor;
}

LingbotDepthEstimationTensorRT::LingbotDepthEstimationTensorRT(
  const std::string & engine_path, int device)
: device_id_(device)
{
  check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");

  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Could not open TensorRT engine: " + engine_path);
  }
  const auto end = file.tellg();
  if (end <= 0) {
    throw std::runtime_error("TensorRT engine is empty: " + engine_path);
  }
  std::vector<char> serialized(static_cast<std::size_t>(end));
  file.seekg(0, std::ios::beg);
  if (!file.read(serialized.data(), static_cast<std::streamsize>(serialized.size()))) {
    throw std::runtime_error("Could not read TensorRT engine: " + engine_path);
  }

  runtime_ = make_trt_unique(nvinfer1::createInferRuntime(logger_), "runtime");
  engine_ = make_trt_unique(
    runtime_->deserializeCudaEngine(serialized.data(), serialized.size()), "engine");
  context_ = make_trt_unique(engine_->createExecutionContext(), "execution context");

  image_ = make_tensor("image", nvinfer1::TensorIOMode::kINPUT, 0);
  depth_ = make_tensor("depth", nvinfer1::TensorIOMode::kINPUT, 1);
  output_ = make_tensor("depth_refined", nvinfer1::TensorIOMode::kOUTPUT, 0);

  if (image_.dims.nbDims != 4 || image_.dims.d[0] != 1 || image_.dims.d[1] != 3) {
    throw std::runtime_error("image tensor must have shape [1,3,H,W]");
  }
  input_h_ = image_.dims.d[2];
  input_w_ = image_.dims.d[3];
  const bool valid_depth_shape =
    (depth_.dims.nbDims == 3 && depth_.dims.d[0] == 1 &&
    depth_.dims.d[1] == input_h_ && depth_.dims.d[2] == input_w_) ||
    (depth_.dims.nbDims == 4 && depth_.dims.d[0] == 1 && depth_.dims.d[1] == 1 &&
    depth_.dims.d[2] == input_h_ && depth_.dims.d[3] == input_w_);
  if (!valid_depth_shape || output_.elements != static_cast<std::size_t>(input_h_ * input_w_)) {
    throw std::runtime_error("depth input/output dimensions do not match image dimensions");
  }

  if (!context_->setTensorAddress(image_.name.c_str(), image_.device) ||
    !context_->setTensorAddress(depth_.name.c_str(), depth_.device) ||
    !context_->setTensorAddress(output_.name.c_str(), output_.device))
  {
    throw std::runtime_error("Failed to bind TensorRT tensor addresses");
  }
  check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  image_float_.resize(image_.elements);
  depth_float_.resize(depth_.elements);
  output_float_.resize(output_.elements);
  if (image_.dtype == nvinfer1::DataType::kHALF) {
    image_half_.resize(image_.elements);
  }
  if (depth_.dtype == nvinfer1::DataType::kHALF) {
    depth_half_.resize(depth_.elements);
  }
  if (output_.dtype == nvinfer1::DataType::kHALF) {
    output_half_.resize(output_.elements);
  }
}

LingbotDepthEstimationTensorRT::~LingbotDepthEstimationTensorRT()
{
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
  for (Tensor * tensor : std::array<Tensor *, 3>{&image_, &depth_, &output_}) {
    if (tensor->device != nullptr) {
      cudaFree(tensor->device);
    }
  }
}

void LingbotDepthEstimationTensorRT::encode_image(const cv::Mat & color_bgr)
{
  const std::size_t plane = static_cast<std::size_t>(input_h_ * input_w_);
  for (int y = 0; y < input_h_; ++y) {
    const auto * row = color_bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input_w_; ++x) {
      const std::size_t i = static_cast<std::size_t>(y * input_w_ + x);
      image_float_[i] = static_cast<float>(row[x][2]) / 255.0F;
      image_float_[plane + i] = static_cast<float>(row[x][1]) / 255.0F;
      image_float_[2 * plane + i] = static_cast<float>(row[x][0]) / 255.0F;
    }
  }
}

void LingbotDepthEstimationTensorRT::encode_depth(const cv::Mat & depth_m)
{
  for (int y = 0; y < input_h_; ++y) {
    const auto * row = depth_m.ptr<float>(y);
    std::copy(row, row + input_w_, depth_float_.begin() + y * input_w_);
  }
}

cv::Mat LingbotDepthEstimationTensorRT::decode_output()
{
  if (output_.dtype == nvinfer1::DataType::kHALF) {
    for (std::size_t i = 0; i < output_half_.size(); ++i) {
      __half_raw raw;
      raw.x = output_half_[i];
      const __half value(raw);
      output_float_[i] = __half2float(value);
    }
  }
  cv::Mat result(input_h_, input_w_, CV_32FC1);
  std::copy(output_float_.begin(), output_float_.end(), result.ptr<float>());
  return result;
}

cv::Mat LingbotDepthEstimationTensorRT::inference(
  const cv::Mat & color_bgr, const cv::Mat & depth_m)
{
  if (color_bgr.type() != CV_8UC3 || depth_m.type() != CV_32FC1) {
    throw std::invalid_argument("LingBot-Depth expects CV_8UC3 color and CV_32FC1 depth");
  }
  if (color_bgr.cols != input_w_ || color_bgr.rows != input_h_ ||
    depth_m.cols != input_w_ || depth_m.rows != input_h_)
  {
    throw std::invalid_argument(
            "Input images do not match fixed engine size " + std::to_string(input_w_) + "x" +
            std::to_string(input_h_));
  }

  encode_image(color_bgr);
  const void * image_host = image_float_.data();
  const void * depth_host = nullptr;
  if (image_.dtype == nvinfer1::DataType::kHALF) {
    float_to_half(image_float_, image_half_);
    image_host = image_half_.data();
  }
  if (depth_.dtype == nvinfer1::DataType::kHALF) {
    encode_depth(depth_m);
    float_to_half(depth_float_, depth_half_);
    depth_host = depth_half_.data();
  } else if (depth_m.isContinuous()) {
    depth_host = depth_m.ptr<float>();
  } else {
    encode_depth(depth_m);
    depth_host = depth_float_.data();
  }

  check_cuda(cudaMemcpyAsync(
      image_.device, image_host, image_.bytes, cudaMemcpyHostToDevice, stream_),
    "copy image to CUDA");
  check_cuda(cudaMemcpyAsync(
      depth_.device, depth_host, depth_.bytes, cudaMemcpyHostToDevice, stream_),
    "copy depth to CUDA");
  if (!context_->enqueueV3(stream_)) {
    throw std::runtime_error("TensorRT inference failed");
  }
  void * output_host = output_float_.data();
  if (output_.dtype == nvinfer1::DataType::kHALF) {
    output_host = output_half_.data();
  }
  check_cuda(cudaMemcpyAsync(
      output_host, output_.device, output_.bytes, cudaMemcpyDeviceToHost, stream_),
    "copy output from CUDA");
  check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return decode_output();
}
}  // namespace depth_estimation_ros
