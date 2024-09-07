#include "stereo_depth_estimation/stereo_depth_estimation_tensorrt.hpp"

namespace depth_estimation_ros
{

    StereoDepthEstimationTensorRT::StereoDepthEstimationTensorRT(
        const std::string &path_to_engine,
        const std::vector<double>& mean, const std::vector<double>& std,
        int device)
        : AbcStereoDepthEstimation(mean, std),
          DEVICE_(device)
    {
        cudaSetDevice(this->DEVICE_);
        // create a model using the API directly and serialize it to a stream
        std::vector<char> trtModelStream;
        size_t size{0};

        std::ifstream file(path_to_engine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }
        else
        {
            std::string msg = "invalid arguments path_to_engine: ";
            msg += path_to_engine;
            throw std::runtime_error(msg.c_str());
        }

        this->runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(this->gLogger_));
        assert(this->runtime_ != nullptr);
        this->engine_ = std::unique_ptr<ICudaEngine>(this->runtime_->deserializeCudaEngine(trtModelStream.data(), size));
        assert(this->engine_ != nullptr);
        this->context_ = std::unique_ptr<IExecutionContext>(this->engine_->createExecutionContext());
        assert(this->context_ != nullptr);

        const auto input_left_name = this->engine_->getIOTensorName(this->input_left_Index_);
        const auto input_left_dims = this->engine_->getTensorShape(input_left_name);
        assert(input_left_dims.nbDims == 4);
        this->input_h_ = input_left_dims.d[2];
        this->input_w_ = input_left_dims.d[3];
        this->input_size_ = 1;
        for (int j = 0; j < input_left_dims.nbDims; ++j)
        {
            this->input_size_ *= input_left_dims.d[j];
        }

        const auto input_right_name = this->engine_->getIOTensorName(this->input_right_Index_);
        const auto input_right_dims = this->engine_->getTensorShape(input_right_name);
        assert(input_right_dims.nbDims == 4);
        for (int j = 0; j < input_right_dims.nbDims; ++j)
        {
            assert(input_left_dims.d[j] == input_right_dims.d[j]);
        }

        std::cout << "INPUT_HEIGHT: " << this->input_h_ << std::endl;
        std::cout << "INPUT_WIDTH:  " << this->input_w_ << std::endl;

        const auto output_name = this->engine_->getIOTensorName(this->outputIndex_);
        auto output_dims = this->engine_->getTensorShape(output_name);
        assert(output_dims.nbDims == 4);
        this->output_c_ = output_dims.d[1];
        this->output_h_ = output_dims.d[2];
        this->output_w_ = output_dims.d[3];
        this->output_size_ = 1;
        for (int j = 0; j < output_dims.nbDims; ++j)
        {
            this->output_size_ *= output_dims.d[j];
        }
        std::cout << "OUTPUT_HEIGHT: " << this->output_h_ << std::endl;
        std::cout << "OUTPUT_WIDTH:  " << this->output_w_ << std::endl;

        // allocate buffer
        this->input_left_blob_.resize(this->input_size_);
        this->input_right_blob_.resize(this->input_size_);
        this->output_blob_.resize(this->output_size_);

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(this->engine_->getNbIOTensors() == 3);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        assert(this->engine_->getTensorDataType(input_left_name) == nvinfer1::DataType::kFLOAT);
        assert(this->engine_->getTensorDataType(input_right_name) == nvinfer1::DataType::kFLOAT);
        assert(this->engine_->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);

        // Create GPU buffers on device
        CHECK(cudaMalloc(&this->inference_buffers_[this->input_left_Index_], this->input_size_ * sizeof(float)));
        CHECK(cudaMalloc(&this->inference_buffers_[this->input_right_Index_], this->input_size_ * sizeof(float)));
        CHECK(cudaMalloc(&this->inference_buffers_[this->outputIndex_], this->output_size_ * sizeof(float)));

        assert(this->context_->setInputShape(input_left_name, input_left_dims));
        assert(this->context_->setInputShape(input_right_name, input_right_dims));
        assert(this->context_->allInputDimensionsSpecified());

        assert(this->context_->setInputTensorAddress(input_left_name, this->inference_buffers_[this->input_left_Index_]));
        assert(this->context_->setInputTensorAddress(input_right_name, this->inference_buffers_[this->input_right_Index_]));
        assert(this->context_->setOutputTensorAddress(output_name, this->inference_buffers_[this->outputIndex_]));
    }

    StereoDepthEstimationTensorRT::~StereoDepthEstimationTensorRT()
    {
        CHECK(cudaFree(inference_buffers_[this->input_left_Index_]));
        CHECK(cudaFree(inference_buffers_[this->input_right_Index_]));
        CHECK(cudaFree(inference_buffers_[this->outputIndex_]));
    }

    cv::Mat StereoDepthEstimationTensorRT::inference(const cv::Mat &left, const cv::Mat &right)
    {
        // preprocess
        auto pre_left = preprocess(left);
        blobFromImage(pre_left, input_left_blob_.data());
        auto pre_right = preprocess(right);
        blobFromImage(pre_right, input_right_blob_.data());

        // inference
        this->doInference(input_left_blob_.data(), input_right_blob_.data(), output_blob_.data());

        // postprocess
        auto output = postprocess(output_blob_.data(), left.size());

        return output;
    }

    void StereoDepthEstimationTensorRT::doInference(
        const float *left, const float *right, float *output)
    {
        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(
            cudaMemcpyAsync(
                this->inference_buffers_[this->input_left_Index_],
                left,
                3 * this->input_h_ * this->input_w_ * sizeof(float),
                cudaMemcpyHostToDevice, stream));
        CHECK(
            cudaMemcpyAsync(
                this->inference_buffers_[this->input_right_Index_],
                right,
                3 * this->input_h_ * this->input_w_ * sizeof(float),
                cudaMemcpyHostToDevice, stream));

        bool success = context_->executeV2(this->inference_buffers_);
        if (!success)
            throw std::runtime_error("failed inference");

        CHECK(
            cudaMemcpyAsync(
                output,
                this->inference_buffers_[this->outputIndex_],
                this->output_size_ * sizeof(float),
                cudaMemcpyDeviceToHost, stream));

        CHECK(cudaStreamSynchronize(stream));

        // Release stream
        CHECK(cudaStreamDestroy(stream));
    }

} // namespace depth_estimation_ros
