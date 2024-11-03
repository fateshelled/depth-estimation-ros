#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace depth_estimation_ros
{
    namespace
    {
        #pragma pack(1)
        struct PointXYZ
        {
            float x;
            float y;
            float z;
        };
    }
    template <class img_type=uint16_t>
    inline sensor_msgs::msg::PointCloud2::SharedPtr depth_image_to_pc_msg(
        const cv::Mat &depth_image, const std_msgs::msg::Header& header,
        float fx, float fy, float cx, float cy, float scale
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

        float fx_inv = 1.0 / fx;
        float fy_inv = 1.0 / fy;
        std::vector<PointXYZ> tmp_cloud(depth_image.cols * depth_image.rows);
        depth_image.forEach<img_type>(
            [&depth_image, &tmp_cloud, &cx, &cy, &fx_inv, &fy_inv, &scale]
            (const img_type &depth, const int * position) -> void {
                const auto d = static_cast<float>(depth)* scale;
                const int h = position[0];
                const int w = position[1];
                const size_t pos = h * depth_image.cols + w;
                tmp_cloud[pos].x = d * ((float)w - cx) * fx_inv;
                tmp_cloud[pos].y = d * ((float)h - cy) * fy_inv;
                tmp_cloud[pos].z = d;
            }
        );

        msg->data.reserve((sizeof(float) / sizeof(uint8_t)) * tmp_cloud.size());
        auto ptr = reinterpret_cast<uint8_t *>(tmp_cloud.data());
        for (size_t i = 0; i < tmp_cloud.size(); ++i)
        {
            if (tmp_cloud[i].z <= 0.0f) continue;
            auto data = ptr + (i * sizeof(PointXYZ));
            msg->data.insert(msg->data.end(), data, data + sizeof(PointXYZ));
            msg->width += 1;
        }
        msg->row_step = msg->width * msg->point_step;

        return msg;
    }
}
