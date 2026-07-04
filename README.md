# depth-estimation-ros

## Support Backend

- TensorRT

## Support Model

### Mono Depth

- Depth-Anything-V2
  - Reference of model converting to TensorRT Engine
    - <https://github.com/spacewalk01/depth-anything-tensorrt>

### Stereo

- CREStereo
  - ONNX model download
    - <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo>
- Fast-ACVNet
  - ONNX model download
    - <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/338_Fast-ACVNet>

### RGB-D Refinement

- LingBot-Depth (fixed-shape TensorRT engine)
  - Reference of model converting to TensorRT Engine
    - <https://github.com/Ar-Ray-code/lingbot-depth-trt>

## Topic

### Subscribe

#### Mono Depth

- `image_raw` (sensor_msgs::msg::Image)

#### Stereo Depth

- `/camera/camera/infra1/image_rect_raw` (sensor_msgs::msg::Image)
- `/camera/camera/infra2/image_rect_raw` (sensor_msgs::msg::Image)
- `/camera/camera/infra1/camera_info` (sensor_msgs::msg::CameraInfo)
- `/camera/camera/infra2/camera_info` (sensor_msgs::msg::CameraInfo)

#### LingBot-Depth

- `/camera/camera/color/image_raw` (sensor_msgs::msg::Image, `bgr8`/RGB-convertible)
- `/camera/camera/aligned_depth_to_color/image_raw` (sensor_msgs::msg::Image, `16UC1` or `32FC1`)
- `/camera/camera/color/camera_info` (sensor_msgs::msg::CameraInfo, used only for point clouds)

The RGB and depth images must already be spatially aligned and must match the fixed engine size.

### Publish

- `depth_estimation_ros/depth/image_raw` (sensor_msgs::msg::Image)
  - Depth image
  - format: mono16

- `depth_estimation_ros/color/image_raw` (sensor_msgs::msg::Image)
  - Colorized depth image.
  - format: bgr8
  - colormap: jet

## Parameters

- `model_path` (str)
  - model absolute or relative path.
- `backend` (str)
  - default: "tensorrt"
  - inference backend.
- `model_type` (str)
  - default: "stereo"
  - specify `mono`, `stereo`, or `lingbot`.
- `model_input_normalize` (bool)
  - default: true
  - Whether to normalize model input value with mean and std.
- `model_input_mean` (vector of double)
  - default: {0.485, 0.456, 0.406}
- `model_input_std` (vector of double)
  - default: {0.229, 0.224, 0.225}
- `model_swap_r_b` (bool)
  - default: false
  - Whether to swap model input Red channel and Blue channel.
- `stereo_baseline_meter` (double)
  - default: 0.05
  - Specify distance between left and right camera.
- `depth_scale` (double)
  - default: 0.001
  - Scale value when converted to uint16_t image.
- `depth_offset` (double)
  - default: 0.0
- `max_depth_meter` (double)
  - default: 20.0
  - Replace distance values greater than this threshold with zeros.
- `min_depth_meter` (double)
  - default: 0.0
  - Replace distance values smaller than this threshold with zeros
- `publish_colored_depth_image` (bool)
  - default: true
  - Whether to publish colored_depth_image.
- `publish_depth_image` (bool)
  - default: true
  - Whether to publish depth_image.
- `imshow` (bool)
  - default: true
  - Whether to show colored depth image.
- `tensorrt_device` (int)
  - `default`: 0
  - CUDA device ID used by TensorRT.
- `lingbot.rgb_topic` (str)
  - default: `/camera/camera/color/image_raw`
- `lingbot.depth_topic` (str)
  - default: `/camera/camera/aligned_depth_to_color/image_raw`
- `lingbot.camera_info_topic` (str)
  - default: `/camera/camera/color/camera_info`
- `lingbot.input_depth_scale` (double)
  - default: `0.001`
  - meters per unit for a `16UC1` input. A `32FC1` input is always interpreted as meters.
- `message_filter.approximate_sync` (bool)
  - default: `false`; set to `true` if the RGB and depth timestamps are not identical.
- `message_filter.queue_size` (int)
  - default: `5`.

## LingBot-Depth example

Build the workspace after generating the engine described in [lingbot-depth-trt](https://github.com/Ar-Ray-code/lingbot-depth-trt):

```bash
cd depth-estimation-ros
colcon build --symlink-install
source install/setup.bash

# launch RealSense
ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:=640,480,30 \
  depth_module.depth_profile:=640,480,30 \
  align_depth.enable:=true \
  pointcloud.enable:=True

# launch lingbot-depth
ros2 run depth_estimation_ros depth_estimation_ros_node --ros-args \
  -p model_type:=lingbot \
  -p model_path:=/absolute/path/to/lingbot_depth_nt1200.engine \
  -p publish_depth_image:=true \
  -p depth_scale:=0.001
```

The refined output is published as metric `16UC1` depth on
`depth_estimation_ros/depth/image_raw`; each unit is `depth_scale` meters.
