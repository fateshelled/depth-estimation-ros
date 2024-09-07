#pragma once

#include <opencv2/opencv.hpp>

namespace depth_estimation_ros
{
    class AbcStereoDepthEstimation
    {
    public:
        AbcStereoDepthEstimation(
            bool input_normalize, const std::vector<double>& mean, const std::vector<double>& std,
            bool swap_r_b
        )
        : input_normalize_(input_normalize), mean_(mean), std_(std), swap_r_b_(swap_r_b)
        {
            if (mean.size() != 3 || std.size() != 3)
            {
                std::string msg = "invalid mean or std size. must be 3.";
                throw std::runtime_error(msg.c_str());
            }
            mean_std_.resize(3);
            std255_inv_.resize(3);
            for (size_t i = 0; i < 3; ++i)
            {
                mean_std_[i] = mean_[i] / std_[i];
                std255_inv_[i] = 1.0 / (std_[i] * 255.0);
            }
        }
        virtual cv::Mat inference(const cv::Mat &left, const cv::Mat &right) = 0;

    protected:
        int input_h_;
        int input_w_;
        int output_c_;
        int output_h_;
        int output_w_;
        bool input_normalize_;
        std::vector<double> mean_ = {0.485, 0.456, 0.406};
        std::vector<double> std_ = {0.229, 0.224, 0.225};
        std::vector<float> std255_inv_ = {
            1.0 / (255.0 * 0.229), 1.0 / (255.0 * 0.224), 1.0 / (255.0 * 0.225)};
        std::vector<float> mean_std_ = {
            -0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225};
        bool swap_r_b_;

        cv::Mat preprocess(const cv::Mat &img)
        {
            cv::Mat output;
            cv::resize(img, output, cv::Size(this->input_w_, this->input_h_), 0.0, 0.0, cv::INTER_LINEAR);
            if (this->swap_r_b_)
            {
                cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
            }
            return output;
        }

        cv::Mat postprocess(float* out_blob, const cv::Size output_size, int disparity_channel=0)
        {
            cv::Mat output(
                this->input_h_, this->input_w_, CV_32FC1,
                out_blob + this->output_h_ * this->output_w_ * disparity_channel);
            cv::resize(output, output, output_size, 0.0, 0.0, cv::INTER_LINEAR);
            return output;
        }

        // for NCHW
        void blobFromImage(const cv::Mat &img, float *blob_data)
        {
            const size_t channels = 3;
            const size_t img_h = img.rows;
            const size_t img_w = img.cols;
            const size_t img_hw = img_h * img_w;
            float *blob_data_ch0 = blob_data;
            float *blob_data_ch1 = blob_data + img_hw;
            float *blob_data_ch2 = blob_data + img_hw * 2;
            // HWC -> CHW
            if (this->input_normalize_)
            {
                for (size_t i = 0; i < img_hw; ++i)
                {
                    // blob = (img / 255.0 - mean) / std
                    const size_t src_idx = i * channels;
                    blob_data_ch0[i] = static_cast<float>(img.data[src_idx + 0]) * this->std255_inv_[0] + this->mean_std_[0];
                    blob_data_ch1[i] = static_cast<float>(img.data[src_idx + 1]) * this->std255_inv_[1] + this->mean_std_[1];
                    blob_data_ch2[i] = static_cast<float>(img.data[src_idx + 2]) * this->std255_inv_[2] + this->mean_std_[2];
                }
            }
            else
            {
                for (size_t i = 0; i < img_hw; ++i)
                {
                    const size_t src_idx = i * channels;
                    blob_data_ch0[i] = static_cast<float>(img.data[src_idx + 0]);
                    blob_data_ch1[i] = static_cast<float>(img.data[src_idx + 1]);
                    blob_data_ch2[i] = static_cast<float>(img.data[src_idx + 2]);
                }
            }
        }
    };
}
