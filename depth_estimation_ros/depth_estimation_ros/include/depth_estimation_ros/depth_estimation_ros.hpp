#pragma once

#include <cmath>
#include <chrono>
#include <vector>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
// #include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <std_msgs/msg/header.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/approximate_epsilon_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include "mono_depth_estimation/mono_depth_estimation.hpp"
#include "stereo_depth_estimation/stereo_depth_estimation.hpp"
#include "lingbot_depth_estimation/lingbot_depth_estimation.hpp"

namespace depth_estimation_ros{
    class DepthEstimationNode : public rclcpp::Node
    {
    public:
        DepthEstimationNode(const rclcpp::NodeOptions &);
    private:
        enum class ModelType
        {
            MODEL_TYPE_MONO,
            MODEL_TYPE_STEREO_DISPARITY,
            MODEL_TYPE_LINGBOT,
        };

        void initialize();
        void mono_image_callback(const sensor_msgs::msg::Image::UniquePtr);
        void stereo_image_callback(
            const sensor_msgs::msg::Image::SharedPtr, const sensor_msgs::msg::CameraInfo::SharedPtr,
            const sensor_msgs::msg::Image::SharedPtr, const sensor_msgs::msg::CameraInfo::SharedPtr);
        void lingbot_image_callback(
            const sensor_msgs::msg::Image::ConstSharedPtr,
            const sensor_msgs::msg::Image::ConstSharedPtr);
        void lingbot_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr);
        void update_point_cloud_lut(int, int, float, float, float, float);

        std::unique_ptr<depth_estimation_ros::AbcMonoDepthEstimation> mono_depth_;
        std::unique_ptr<depth_estimation_ros::AbcStereoDepthEstimation> stereo_depth_;
        std::unique_ptr<depth_estimation_ros::AbcLingbotDepthEstimation> lingbot_depth_;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_mono_image_;

        // Subscribe stereo images
        message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_image_, sub_right_image_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_left_info_, sub_right_info_;
        using ExactPolicy = message_filters::sync_policies::ExactTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo,
            sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>;
        using ApproximatePolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo,
            sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>;
        using ApproximateEpsilonPolicy = message_filters::sync_policies::ApproximateEpsilonTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo,
            sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>;
        using ExactSync = message_filters::Synchronizer<ExactPolicy>;
        using ApproximateSync = message_filters::Synchronizer<ApproximatePolicy>;
        using ApproximateEpsilonSync = message_filters::Synchronizer<ApproximateEpsilonPolicy>;
        std::shared_ptr<ExactSync> exact_sync_;
        std::shared_ptr<ApproximateSync> approximate_sync_;
        std::shared_ptr<ApproximateEpsilonSync> approximate_epsilon_sync_;

        message_filters::Subscriber<sensor_msgs::msg::Image> sub_lingbot_rgb_, sub_lingbot_depth_;
        using RgbdExactPolicy = message_filters::sync_policies::ExactTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        using RgbdApproximatePolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        using RgbdExactSync = message_filters::Synchronizer<RgbdExactPolicy>;
        using RgbdApproximateSync = message_filters::Synchronizer<RgbdApproximatePolicy>;
        std::shared_ptr<RgbdExactSync> rgbd_exact_sync_;
        std::shared_ptr<RgbdApproximateSync> rgbd_approximate_sync_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_lingbot_camera_info_;
        sensor_msgs::msg::CameraInfo::SharedPtr lingbot_camera_info_;

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_colored_depth_image_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pcl2_;

        const std::string window_name_ = "depth_estimation_ros";

        bool imshow_ = true;
        bool publish_depth_image_ = true;
        bool publish_disparity_image_ = true;
        bool publish_colored_depth_image_ = true;
        bool publish_point_cloud2_ = true;

        double baseline_meter_ = 0.050; // D435: 50mm
        double depth_scale_ = 0.001;
        double depth_offset_ = 0.0;
        double max_depth_meter_ = 20.0;
        double min_depth_meter_ = 0.0;
        double lingbot_input_depth_scale_ = 0.001;

        std::vector<float> point_cloud_x_lut_;
        std::vector<float> point_cloud_y_lut_;
        int point_cloud_lut_width_ = 0;
        int point_cloud_lut_height_ = 0;
        float point_cloud_lut_fx_ = 0.0f;
        float point_cloud_lut_fy_ = 0.0f;
        float point_cloud_lut_cx_ = 0.0f;
        float point_cloud_lut_cy_ = 0.0f;

    };
}
