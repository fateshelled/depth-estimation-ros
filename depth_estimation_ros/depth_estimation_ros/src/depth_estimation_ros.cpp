#include "depth_estimation_ros/depth_estimation_ros.hpp"

namespace depth_estimation_ros
{
    DepthEstimationNode::DepthEstimationNode(const rclcpp::NodeOptions &options)
        : Node("depth_estimation_ros", options)
    {
        this->initialize();
    }

    void DepthEstimationNode::initialize()
    {

        const auto model_path = this->declare_parameter(
            "model_path",
            "src/depth-estimation-ros/depth_anything_v2_vits.sim.fp16.engine");
        const auto model_type = this->declare_parameter("model_type", "mono");
        const auto backend = this->declare_parameter("backend", "tensorrt");
        const auto tensorrt_device = this->declare_parameter("tensorrt_device", 0);

        const auto input_mean = this->declare_parameter("model_input_mean",
            std::vector<double>{0.485, 0.456, 0.406});
        const auto input_std = this->declare_parameter("model_input_std",
            std::vector<double>{0.229, 0.224, 0.225});

        // this->publish_point_cloud2_ = this->declare_parameter("publish_point_cloud2", false);
        this->publish_depth_image_ = this->declare_parameter("publish_depth_image", true);
        this->publish_colored_depth_image_ = this->declare_parameter("publish_colored_depth_image", true);
        this->imshow_ = this->declare_parameter("imshow", true);

        if (model_type == "mono")
        {

            if (backend == "tensorrt")
            {
#ifdef ENABLE_TENSORRT
                RCLCPP_INFO(this->get_logger(), "Model Type is TensorRT");
                this->mono_depth_ = std::make_unique<MonoDepthEstimationTensorRT>(
                    model_path,
                    input_mean, input_std,
                    tensorrt_device);
#else
                RCLCPP_ERROR(this->get_logger(), "depth_estimation is not built with TensorRT");
                rclcpp::shutdown();
#endif
            }
            else if (backend == "openvino")
            {
                RCLCPP_ERROR(this->get_logger(), "not support openvino yet.");
                rclcpp::shutdown();
            }
            else if (backend == "onnxruntime")
            {
                RCLCPP_ERROR(this->get_logger(), "not support openvino yet.");
                rclcpp::shutdown();
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "invalid backend %s.", backend.c_str());
                rclcpp::shutdown();
            }
            RCLCPP_INFO(this->get_logger(), "model loaded");
        }


        if (model_type == "mono")
        {
            this->sub_mono_image_ = this->create_subscription<sensor_msgs::msg::Image>(
                "image_raw",
                // rclcpp::SensorDataQoS(),
                10,
                std::bind(&DepthEstimationNode::mono_image_callback, this, std::placeholders::_1));
        }
        else if (model_type == "stereo")
        {
            RCLCPP_ERROR(this->get_logger(), "not support stereo model yet.");
            rclcpp::shutdown();
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "invalid model type %s.", model_type.c_str());
            rclcpp::shutdown();
        }

        if (this->publish_depth_image_)
        {
            this->pub_depth_image_ = this->create_publisher<sensor_msgs::msg::Image>(
                "depth_estimation_ros/depth/image_raw", 10);
        }
        if (this->publish_colored_depth_image_)
        {
            this->pub_colored_depth_image_ = this->create_publisher<sensor_msgs::msg::Image>(
                "depth_estimation_ros/color/image_raw", 10);
        }
        // if (this->publish_point_cloud2_)
        // {
        //     RCLCPP_WARN(this->get_logger(), "not support publish point_cloud2 yet.");
        //     // this->pub_pcl2_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        //     //     "depth_estimation_ros/point_cloud2", rclcpp::SensorDataQoS());
        // }

        if (this->imshow_)
        {
            cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
        }
    }

    void DepthEstimationNode::mono_image_callback(const sensor_msgs::msg::Image::UniquePtr ptr)
    {
        auto img = cv_bridge::toCvCopy(*ptr, "bgr8");
        cv::Mat frame = img->image;

        auto now = std::chrono::system_clock::now();
        auto depth = this->mono_depth_->inference(frame);
        auto end = std::chrono::system_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - now);
        RCLCPP_INFO(this->get_logger(), "Inference time: %5ld us", elapsed.count());

        if (this->publish_depth_image_)
        {
            cv::Mat depth_u16;
            cv::normalize(
                depth, depth_u16,
                0, std::numeric_limits<uint16_t>::max(),
                cv::NORM_MINMAX, CV_16U);

            sensor_msgs::msg::Image::SharedPtr pub_img =
                cv_bridge::CvImage(img->header, "mono16", depth_u16).toImageMsg();
            this->pub_depth_image_->publish(*pub_img);
        }

        if (this->publish_colored_depth_image_ || this->imshow_)
        {
            cv::Mat colored;
            cv::normalize(
                depth, colored,
                0, std::numeric_limits<uint8_t>::max(), cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(colored, colored, cv::COLORMAP_JET);

            if (this->publish_colored_depth_image_)
            {
                sensor_msgs::msg::Image::SharedPtr pub_img =
                    cv_bridge::CvImage(img->header, "bgr8", colored).toImageMsg();
                this->pub_colored_depth_image_->publish(*pub_img);
            }
            if (this->imshow_)
            {
                cv::imshow(window_name_, colored);
                auto key = cv::waitKey(1);
                if (key == 27)
                {
                    rclcpp::shutdown();
                }
            }
        }
    }

}

RCLCPP_COMPONENTS_REGISTER_NODE(depth_estimation_ros::DepthEstimationNode)
