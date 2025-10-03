Multi-Camera Object Detection with Hailo AI Kit
================================================
This example performs real-time object detection using a Hailo AI Kit on Raspberry Pi 5.
It supports multiple USB cameras simultaneously, processing frames with latest-frame capture techniques to minimize latency.
The application can handle single inputs (images/video/camera) or automatically detect and process multiple USB cameras in parallel, displaying each camera feed in separate OpenCV windows with object detection annotations.

![output example](./obj_det.gif)

Requirements
------------

### Hardware
- Raspberry Pi 5
- Hailo AI Kit for Raspberry Pi 5
- One or more USB cameras (UVC compatible)

### Software
- Raspberry Pi OS (64-bit recommended)
- Hailo AI Kit installation (follow the official guide):
  https://www.raspberrypi.com/documentation/accessories/ai-kit.html#install
- HailoRT==4.22.0
- OpenCV >= 4.5.4
    ```shell script
    sudo apt-get install -y libopencv-dev python3-opencv
    ```
- Boost
    ```shell script
    sudo apt-get install libboost-all-dev
    ```
- CMake >= 3.16
- Gtk


Supported Models
----------------
This example expects the HEF to contain HailoRT-Postprocess. 

Because of that, this example only supports detections models that allow HailoRT-Postprocess:
- YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLOv11
- YOLOX
- SSD
- CenterNet

### Pre-trained Models
The repository includes several YOLOv11 models optimized for Hailo:
- `yolov11n.hef` - Nano (fastest, lowest accuracy)
- `yolov11l.hef` - Large (slower, higher accuracy)
- `yolov11m.hef` - Medium
- `yolov11x.hef` - Extra Large

For additional models, you can download YOLOv11 Large from:
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov11l/2024-10-02/yolo11l.zip


Usage
-----
0. Make sure you have installed all of the requirements, including the Hailo AI Kit.

1. Prepare your hardware:
   - Connect multiple USB cameras to your Raspberry Pi 5
   - Ensure cameras have proper permissions:
     ```shell script
     sudo chmod 777 /dev/video*
     ```
   - Verify camera detection:
     ```shell script
     ls /dev/video*
     ```

2. Clone this repository:
    ```shell script
    git clone <repository-url>
    cd rpi-5-halio-pwm
    ``` 

2. Download sample resources:
    ```shell script
    ./download_resources.sh
    ```
    The following files will be downloaded:
    ```
    full_mov_slow.mp4
    bus.jpg
    yolov8n.hef
    ```

3. Compile the project on the development machine  
    ```shell script
    ./build.sh
    ```
    This creates the directory hierarchy build/x86_64 and compile an executable file called obj_det

5. Run the example:

    ```shell script
    ./build/x86_64/obj_det --net <hef_path> --input <image_or_video_or_camera_path>
    ```

Arguments
---------

- ``-n, --net``: Path to the pre-trained model file (HEF).
- ``-i, --input``: Path to the input (image, folder, video file, camera, or leave empty for multi-camera mode).
- ``-b, --batch_size (optional)``: Number of images in one batch. Defaults to 1.
- ``-s (optional)``: A flag for saving the output video of a camera input.

Running the Example
-------------------

### Multi-Camera Mode (New Feature!)
- **Automatic multi-camera detection** (leave input empty or use "multicamera"):
    ```shell script
    ./build/x86_64/obj_det --net yolov11l.hef --batch_size 4
    ```
    or
    ```shell script
    ./build/x86_64/obj_det --net yolov11l.hef --input multicamera --batch_size 2
    ```
    - Automatically detects all connected USB cameras
    - Opens separate OpenCV windows for each camera
    - Uses latest-frame capture to minimize latency
    - Press 'q' in any window to exit
    - Real-time FPS display in terminal

### Single Input Modes
- For a video:
    ```shell script
    ./build/x86_64/obj_det --net yolov11l.hef --input full_mov_slow.mp4 --batch_size 16
    ```
    Output video is saved as processed_video.mp4

- For a single image:
    ```shell script
    ./build/x86_64/obj_det -n yolov11l.hef -i bus.jpg
    ```
    Output image is saved as processed_image_0.jpg

- For a directory of images:
    ```shell script
    ./build/x86_64/obj_det -n yolov11l.hef -i images -b 4
    ````
    Each image is saved as processed_image_i.jpg
    
- For single camera, enabling saving the output:
    ```shell script
    ./build/x86_64/obj_det --net yolov11l.hef --input /dev/video0 --batch_size 2 -s
    ```
    Output video is saved as processed_video.mp4

Multi-Camera Features
--------------------
- **Automatic USB Camera Detection**: Scans `/dev/video*` devices and tests connectivity
- **Latest Frame Capture**: Uses `CAP_PROP_BUFFERSIZE=1` and advanced grab/retrieve techniques to minimize latency
- **Real-time Processing**: Processes frames from multiple cameras simultaneously using Hailo's batch processing
- **Individual Windows**: Each camera gets its own OpenCV display window with camera identification overlay
- **Adaptive Frame Rate**: Automatically adjusts processing speed based on number of connected cameras
- **Graceful Error Handling**: Continues processing if individual cameras disconnect
- **Performance Monitoring**: Real-time FPS display and frame processing statistics

Performance Results
-------------------
Real-world testing on Raspberry Pi 5 with Hailo AI Kit shows excellent performance:

### YOLOv11x Performance (4 USB Cameras)
```
Network: yolov11x.hef
Input Shape: (640, 640, 3)
Output Shape: (80, 100, 0) - NMS Postprocessed

Multi-camera Results:
- 4 USB cameras simultaneously
- Sustained ~10 FPS processing rate
- Automatic camera failure recovery
- Real-time object detection on all feeds
```

**Sample Output:**
```
Camera 0 initialized successfully
Camera 4 initialized successfully  
Camera 8 initialized successfully
Camera 12 initialized successfully
Multi-camera post-processing started. Press 'q' in any window to exit.

Multi-camera FPS: 10 | Total frames processed: 31
Multi-camera FPS: 10 | Total frames processed: 41
Multi-camera FPS: 10 | Total frames processed: 51
```

### Performance by Model Size
| Model      | Cameras | Avg FPS | Latency | Notes |
|------------|---------|---------|---------|-------|
| YOLOv11n   | 4       | ~15-20  | Low     | Fastest, good for real-time |
| YOLOv11l   | 4       | ~12-15  | Medium  | Balanced performance/accuracy |
| YOLOv11x   | 4       | ~10     | Higher  | Best accuracy, still real-time |
| YOLOv11x   | 2       | ~18-20  | Medium  | Higher FPS with fewer cameras |

*Results may vary based on camera resolution, USB bandwidth, and system load.*

### Test System Specifications
- **Hardware**: Raspberry Pi 5 (8GB RAM recommended)
- **AI Accelerator**: Hailo-8 AI Kit
- **Cameras**: 4x USB UVC compatible cameras
- **OS**: Raspberry Pi OS 64-bit
- **Camera Resolution**: Default (varies by camera, typically 640x480 or 1280x720)

### Optimization Tips
- **For Higher FPS**: Use YOLOv11n or reduce number of cameras
- **For Better Accuracy**: Use YOLOv11l or YOLOv11x with fewer cameras  
- **USB Bandwidth**: Distribute cameras across different USB ports/hubs
- **Memory Usage**: Monitor with `htop`, consider increasing swap if needed
- **Camera Issues**: Some cameras may disconnect/reconnect during operation - the system automatically handles this

Notes
----------------
- Last HailoRT version checked - ``HailoRT v4.22.0``
- Optimized for Raspberry Pi 5 with Hailo AI Kit
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp 
- There should be no spaces between "=" given in the command line arguments and the file name itself
- The example only works for detection models that have the NMS on-Hailo (either on the NN-core or on the CPU)
- **Multi-Camera Specific**:
    - Requires UVC-compatible USB cameras
    - Tested with up to 4 simultaneous cameras
    - Performance depends on camera resolution and batch size
    - Recommended batch size: 2-4 for multi-camera mode
- When using camera as input:
    - To exit gracefully from openCV window, press 'q', 'Q', or ESC.
    - Camera paths are usually found under `/dev/video0`, `/dev/video1`, etc.
    - Ensure you have the permissions for all cameras:
        ```shell script
        sudo chmod 777 /dev/video*
        ```
    - Force OpenCV to use V4L2 instead of GStreamer for better performance:
      ```shell script
        export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0
        export OPENCV_VIDEOIO_PRIORITY_V4L2=100
      ```
- Using multiple models on same device:
    - If you need to run multiple models on the same virtual device (vdevice), use the AsyncModelInfer constructor that accepts two arguments. Initialize each model using the same group_id. 
    - Example:
      ```
         std::string group_id = "<group_id>";
         AsyncModelInfer model1("<hef1_path>", group_id);
         AsyncModelInfer model2("<hef2_path>", group_id);
      ```
    - By assigning the same group_id to models from different HEF files, you enable the runtime to treat them as part of the same group, allowing them to share resources and run more efficiently on the same hardware.

Troubleshooting Multi-Camera Issues
----------------------------------
- **No cameras detected**: 
  - Check `ls /dev/video*` to verify camera devices exist
  - Ensure proper permissions: `sudo chmod 777 /dev/video*`
  - Try connecting cameras one at a time to identify problematic devices
  
- **Camera fails to open**:
  - Some cameras may be busy or used by other processes
  - Check with `lsof /dev/video*` to see which processes are using cameras
  - Restart the application or reboot if cameras remain locked
  
- **Low FPS or high latency**:
  - Reduce batch size (try `--batch_size 1` or `2`)
  - Use lower resolution cameras if possible
  - Close unnecessary applications to free up system resources
  
- **OpenCV/Display issues**:
  - Make sure you're running in a graphical environment (not headless SSH)
  - For remote access, use X11 forwarding: `ssh -X user@raspberrypi`
  - Install X11 dependencies: `sudo apt-get install libgtk-3-dev`

- **Memory issues with multiple cameras**:
  - Monitor memory usage with `htop` or `free -h`
  - Reduce number of simultaneous cameras
  - Consider using swap if running low on RAM

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
