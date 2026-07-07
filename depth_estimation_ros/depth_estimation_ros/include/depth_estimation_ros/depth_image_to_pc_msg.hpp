#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <opencv2/core.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace depth_estimation_ros
{
    namespace
    {
        struct PointXYZ
        {
            float x;
            float y;
            float z;
        };
        static_assert(sizeof(PointXYZ) == sizeof(float) * 3);
    }

    template <class img_type=float>
    inline sensor_msgs::msg::PointCloud2::SharedPtr depth_image_to_pc_msg(
        const cv::Mat &depth_image, const std_msgs::msg::Header& header,
        float fx, float fy, float cx, float cy,
        const float * x_lut = nullptr, const float * y_lut = nullptr
    )
    {
        sensor_msgs::msg::PointCloud2::SharedPtr msg(new sensor_msgs::msg::PointCloud2);
        msg->header = header;

        sensor_msgs::msg::PointField field;
        field.count = 1;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;

        field.offset = 0;
        field.name = "x";
        msg->fields.push_back(field);

        field.offset += sizeof(float);
        field.name = "y";
        msg->fields.push_back(field);

        field.offset += sizeof(float);
        field.name = "z";
        msg->fields.push_back(field);

        msg->height = 1;
        msg->width = 0;
        msg->point_step = sizeof(float) * 3;
        msg->row_step = 0;
        msg->is_bigendian = false;
        msg->is_dense = true;

        const auto max_points = static_cast<size_t>(depth_image.cols) *
            static_cast<size_t>(depth_image.rows);
        msg->data.resize(max_points * sizeof(PointXYZ));

        const float fx_inv = 1.0f / fx;
        const float fy_inv = 1.0f / fy;
        auto * dst = msg->data.data();
        size_t valid_points = 0;

        if (x_lut != nullptr && y_lut != nullptr)
        {
            for (int y = 0; y < depth_image.rows; ++y)
            {
                const auto * row = depth_image.ptr<img_type>(y);
                const float y_factor = y_lut[y];
                for (int x = 0; x < depth_image.cols; ++x)
                {
                    const float d = static_cast<float>(row[x]);
                    if (d <= 0.0f || !std::isfinite(d))
                    {
                        continue;
                    }

                    const PointXYZ point{d * x_lut[x], d * y_factor, d};
                    std::memcpy(dst, &point, sizeof(PointXYZ));
                    dst += sizeof(PointXYZ);
                    ++valid_points;
                }
            }
        }
        else
        {
            for (int y = 0; y < depth_image.rows; ++y)
            {
                const auto * row = depth_image.ptr<img_type>(y);
                const float y_factor = (static_cast<float>(y) - cy) * fy_inv;
                for (int x = 0; x < depth_image.cols; ++x)
                {
                    const float d = static_cast<float>(row[x]);
                    if (d <= 0.0f || !std::isfinite(d))
                    {
                        continue;
                    }

                    const PointXYZ point{
                        d * (static_cast<float>(x) - cx) * fx_inv,
                        d * y_factor,
                        d};
                    std::memcpy(dst, &point, sizeof(PointXYZ));
                    dst += sizeof(PointXYZ);
                    ++valid_points;
                }
            }
        }

        msg->width = static_cast<uint32_t>(valid_points);
        msg->data.resize(valid_points * sizeof(PointXYZ));
        msg->row_step = msg->width * msg->point_step;

        return msg;
    }
}
