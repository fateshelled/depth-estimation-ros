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
            // "src/depth-estimation-ros/depth_anything_v2_vits.sim.fp16.engine"
            "src/depth-estimation-ros/crestereo_init_iter2_360x640.fp16.engine"
        );
        const auto model_type = this->declare_parameter("model_type", "stereo");
        const auto backend = this->declare_parameter("backend", "tensorrt");
        const auto tensorrt_device = this->declare_parameter("tensorrt_device", 0);

        const auto input_normalize = this->declare_parameter("model_input_normalize", true);
        const auto input_mean = this->declare_parameter("model_input_mean",
            std::vector<double>{0.485, 0.456, 0.406});
        const auto input_std = this->declare_parameter("model_input_std",
            std::vector<double>{0.229, 0.224, 0.225});
        const auto swap_r_b = this->declare_parameter("model_swap_r_b", false);

        this->baseline_meter_ = this->declare_parameter("stereo_baseline_meter", 0.05);
        this->depth_scale_ = this->declare_parameter("depth_scale", 0.001);
        this->depth_offset_ = this->declare_parameter("depth_offset", 0.0);
        this->max_depth_meter_ = this->declare_parameter("max_depth_meter", 20.0);
        this->min_depth_meter_ = this->declare_parameter("min_depth_meter", 0.0);

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
                    input_normalize,
                    input_mean, input_std,
                    swap_r_b,
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
        else
        {
            if (backend == "tensorrt")
            {
#ifdef ENABLE_TENSORRT
                RCLCPP_INFO(this->get_logger(), "Model Type is TensorRT");
                this->stereo_depth_ = std::make_unique<StereoDepthEstimationTensorRT>(
                    model_path,
                    input_normalize,
                    input_mean, input_std,
                    swap_r_b,
                    tensorrt_device);
#else
                RCLCPP_ERROR(this->get_logger(), "depth_estimation is not built with TensorRT");
                rclcpp::shutdown();
#endif
            }

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
            auto queue_size = this->declare_parameter("message_filter.queue_size", 5);
            auto approximate_sync = this->declare_parameter("message_filter.approximate_sync", false);
            auto approximate_sync_tol = this->declare_parameter("message_filter.approximate_sync_tolerance_seconds", 0.0);

            sub_left_image_.subscribe(this, "/camera/camera/infra1/image_rect_raw");
            sub_right_image_.subscribe(this, "/camera/camera/infra2/image_rect_raw");
            sub_left_info_.subscribe(this, "/camera/camera/infra2/camera_info");
            sub_right_info_.subscribe(this, "/camera/camera/infra2/camera_info");

            // Synchronize callbacks
            if (approximate_sync) {
                if (approximate_sync_tol == 0.0) {
                    approximate_sync_.reset(
                        new ApproximateSync(
                            ApproximatePolicy(queue_size),
                            sub_left_image_, sub_left_info_, sub_right_image_, sub_right_info_));
                    approximate_sync_->registerCallback(&DepthEstimationNode::stereo_image_callback, this);
                } else {
                    approximate_epsilon_sync_.reset(
                        new ApproximateEpsilonSync(
                            ApproximateEpsilonPolicy(queue_size, rclcpp::Duration::from_seconds(approximate_sync_tol)),
                            sub_left_image_, sub_left_info_, sub_right_image_, sub_right_info_));
                    approximate_epsilon_sync_->registerCallback(&DepthEstimationNode::stereo_image_callback, this);
                }
            } else {
                exact_sync_.reset(
                    new ExactSync(ExactPolicy(queue_size), sub_left_image_, sub_left_info_, sub_right_image_, sub_right_info_));
                exact_sync_->registerCallback(&DepthEstimationNode::stereo_image_callback, this);
            }
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

    void DepthEstimationNode::stereo_image_callback(
            const sensor_msgs::msg::Image::SharedPtr left_ptr,
            const sensor_msgs::msg::CameraInfo::SharedPtr left_info_ptr,
            const sensor_msgs::msg::Image::SharedPtr right_ptr,
            const sensor_msgs::msg::CameraInfo::SharedPtr)
    {
        cv::Mat left = cv_bridge::toCvCopy(*left_ptr, "bgr8")->image;
        cv::Mat right = cv_bridge::toCvCopy(*right_ptr, "bgr8")->image;

        auto now = std::chrono::system_clock::now();
        auto disparity_map = this->stereo_depth_->inference(left, right);
        auto end = std::chrono::system_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - now);
        RCLCPP_INFO(this->get_logger(), "Inference time: %5ld us", elapsed.count());

        auto fx = left_info_ptr->p[0];
        cv::Mat depth_f32 = (fx * this->baseline_meter_) / disparity_map + this->depth_offset_;

        // remove out of range
        depth_f32.setTo(0.0, depth_f32 < this->min_depth_meter_);
        depth_f32.setTo(0.0, depth_f32 > this->max_depth_meter_);

        if (this->publish_depth_image_)
        {
            cv::Mat depth_u16;
            cv::normalize(
                depth_f32 / this->depth_scale_, depth_u16,
                0, std::numeric_limits<uint16_t>::max(), cv::NORM_MINMAX, CV_16U);
            sensor_msgs::msg::Image::SharedPtr pub_img =
                cv_bridge::CvImage(left_ptr->header, "mono16", depth_u16).toImageMsg();
            this->pub_depth_image_->publish(*pub_img);
        }

        if (this->publish_colored_depth_image_ || this->imshow_)
        {
            cv::Mat colored;
            cv::normalize(
                depth_f32, colored,
                0, std::numeric_limits<uint8_t>::max(), cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(colored, colored, cv::COLORMAP_JET);

            if (this->publish_colored_depth_image_)
            {
                sensor_msgs::msg::Image::SharedPtr pub_img =
                    cv_bridge::CvImage(left_ptr->header, "bgr8", colored).toImageMsg();
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
