# depth-estimation-ros

## Support Backend
- TensorRT

## Support Model
### Mono Depth
- Depth-Anything-V2
    - Reference of model converting to TensorRT Engine
        - https://github.com/spacewalk01/depth-anything-tensorrt

### Stereo
- CREStereo
    - ONNX model download
        - https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo
- Fast-ACVNet
    - ONNX model download
        - https://github.com/PINTO0309/PINTO_model_zoo/tree/main/338_Fast-ACVNet

## Topic

### Subscribe
#### Mono Depth
- `image_raw` (sensor_msgs::msg::Image)

#### Stereo Depth
- `/camera/camera/infra1/image_rect_raw` (sensor_msgs::msg::Image)
- `/camera/camera/infra2/image_rect_raw` (sensor_msgs::msg::Image)
- `/camera/camera/infra1/camera_info` (sensor_msgs::msg::CameraInfo)
- `/camera/camera/infra2/camera_info` (sensor_msgs::msg::CameraInfo)

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
    - default: "mono"
    - specify "mono" or "stereo" model.
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
