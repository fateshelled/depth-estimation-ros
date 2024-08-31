# depth-estimation-ros

## Support Backend
- TensorRT

## Support Model
### Mono Depth
- Depth-Anything-V2
    - Reference of model converting to TensorRT Engine
        - https://github.com/spacewalk01/depth-anything-tensorrt

### Stereo
- not support yet.

## Topic

### Subscribe
- `image_raw` (sensor_msgs::msg::Image)

### Publish
- `depth_estimation_ros/depth/image_raw` (sensor_msgs::msg::Image)
    - format: mono16

- `depth_estimation_ros/color/image_raw` (sensor_msgs::msg::Image)
    - format: bgr8
    - colormat: jet

## Parameters
- `model_path` (str)
    - model absolute or relative path.
- `backend` (str)
    - default: "tensorrt"
    - inference backend.
- `model_type` (str)
    - default: "mono"
    - specify "mono" or "stereo" model.
- `model_input_mean` (vector of double)
    - default: {0.485, 0.456, 0.406}
- `model_input_std` (vector of double)
    - default: {0.229, 0.224, 0.225}
- `publish_colored_depth_image` (bool)
    - default: true
    - Whether to publish colored_depth_image.
- `publish_depth_image` (bool)
    - default: true
    - Whether to publish depth_image.
<!-- - `publish_point_cloud2` (bool)
    - default: false
    - Whether to publish point_cloud2.
    - not support yet. -->
- `imshow` (bool)
    - default: true
    - Whether to show colored depth image.
- `tensorrt_device` (int)
    - `default`: 0
    - CUDA device ID used by TensorRT.
