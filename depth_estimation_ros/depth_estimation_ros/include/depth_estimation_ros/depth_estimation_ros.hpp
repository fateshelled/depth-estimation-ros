#pragma once

#include <cmath>
#include <chrono>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
// #include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include "mono_depth_estimation/mono_depth_estimation.hpp"

namespace depth_estimation_ros{
    class DepthEstimationNode : public rclcpp::Node
    {
    public:
        DepthEstimationNode(const rclcpp::NodeOptions &);
    private:
        void initialize();
        void mono_image_callback(const sensor_msgs::msg::Image::UniquePtr);
        // void stereo_image_callback(
        //     const sensor_msgs::msg::Image::SharedPtr, const sensor_msgs::msg::Image::SharedPtr);

        std::unique_ptr<depth_estimation_ros::AbcMonoDepthEstimation> mono_depth_;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_mono_image_;

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_colored_depth_image_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pcl2_;

        const std::string window_name_ = "depth_estimation_ros";
        bool imshow_ = true;
        bool publish_point_cloud2_ = true;
        bool publish_depth_image_ = true;
        bool publish_colored_depth_image_ = true;
    };
}
