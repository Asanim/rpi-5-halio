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

### Pre-trained Models
Compatable with YOLOv11 models optimized for Hailo:
- `yolov11n.hef` - Nano (fastest, lowest accuracy)
- `yolov11l.hef` - Large (slower, higher accuracy)
- `yolov11m.hef` - Medium
- `yolov11x.hef` - Extra Large

Download from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)


Usage
-----
0. Make sure you have installed all of the requirements, including the Hailo AI Kit.

1. Compile the project on the development machine  
    ```shell script
    ./build.sh
    ```
    This creates the directory hierarchy build/x86_64 and compile an executable file called obj_det

2. Run the example:

    ```shell script
    ./build/x86_64/obj_det multicamera --net yolov11x.hef --batch_size 2
    ```

Arguments
---------

- ``-n, --net``: Path to the pre-trained model file (HEF).
- ``-i, --input``: Path to the input (image, folder, video file, camera, or leave empty for multi-camera mode).
- ``-b, --batch_size (optional)``: Number of images in one batch. Defaults to 1.

Running the Example
-------------------

### Multi-Camera Mode
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