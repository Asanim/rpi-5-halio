#!/usr/bin/env python3
"""
Multi-Instance Hailo Detection Pipeline
Runs 4 parallel inference instances using YOLOv11l model with individual FPS tracking.
Based on the hailo-apps-infra detection_simple example.
"""

import os
import time
import threading
import multiprocessing
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo

from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class, GStreamerApp
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.defines import (
    RESOURCES_VIDEOS_DIR_NAME, 
    SIMPLE_DETECTION_VIDEO_NAME, 
    SIMPLE_DETECTION_APP_TITLE, 
    SIMPLE_DETECTION_PIPELINE, 
    RESOURCES_MODELS_DIR_NAME, 
    RESOURCES_SO_DIR_NAME, 
    SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME, 
    SIMPLE_DETECTION_POSTPROCESS_FUNCTION
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, 
    INFERENCE_PIPELINE, 
    USER_CALLBACK_PIPELINE, 
    DISPLAY_PIPELINE,
    QUEUE
)

class MultiInstanceCallback(app_callback_class):
    """Enhanced callback class with FPS tracking per instance"""
    
    def __init__(self, instance_id):
        super().__init__()
        self.instance_id = instance_id
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.detection_count = 0
        self.total_detections = 0
        
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.fps_frame_count += 1
        
        # Calculate FPS every 30 frames
        if self.fps_frame_count >= 30:
            time_diff = current_time - self.last_fps_time
            if time_diff > 0:
                self.current_fps = self.fps_frame_count / time_diff
            
            self.fps_frame_count = 0
            self.last_fps_time = current_time
            
        return self.current_fps

def create_app_callback(instance_id):
    """Create callback function for specific instance"""
    
    def app_callback(pad, info, user_data):
        user_data.increment()
        fps = user_data.calculate_fps()
        
        # Parse detections from buffer
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
            
        detection_info = []
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            for detection in detections:
                label = detection.get_label()
                confidence = detection.get_confidence()
                detection_info.append(f"{label}: {confidence:.2f}")
                user_data.total_detections += 1
        except Exception as e:
            print(f"Instance {instance_id}: Detection parsing error: {e}")
            
        user_data.detection_count = len(detection_info)
        
        # Print status every 60 frames to avoid spam
        if user_data.get_count() % 60 == 0:
            runtime = time.time() - user_data.start_time
            avg_fps = user_data.get_count() / runtime if runtime > 0 else 0
            print(f"Instance {instance_id}: Frame {user_data.get_count()}, "
                  f"Current FPS: {fps:.1f}, Avg FPS: {avg_fps:.1f}, "
                  f"Detections: {user_data.detection_count}, "
                  f"Total: {user_data.total_detections}")
                  
        return Gst.PadProbeReturn.OK
        
    return app_callback

class MultiInstanceDetectionApp(GStreamerApp):
    """Multi-instance detection application"""
    
    def __init__(self, instance_id=0, total_instances=4):
        # Setup parser
        parser = get_default_parser()
        parser.add_argument("--labels-json", default=None, help="Path to labels JSON file")
        parser.add_argument("--instances", type=int, default=4, help="Number of inference instances")
        parser.add_argument("--model-path", default=None, help="Path to HEF/ONNX model file")
        
        # Create callback for this instance
        self.instance_id = instance_id
        self.total_instances = total_instances
        user_data = MultiInstanceCallback(instance_id)
        app_callback = create_app_callback(instance_id)
        
        # Call parent constructor
        super().__init__(parser, user_data)
        
        # Model configuration
        self.video_width = 640
        self.video_height = 640
        self.batch_size = 1
        
        # Detection thresholds for YOLOv11l
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45
        
        # Check for YOLOv11l model
        yolo11_path = "/home/sam/rpi-5-halio-pwm/models/yolo11l.onnx"
        if self.options_menu.model_path:
            self.hef_path = self.options_menu.model_path
        elif os.path.exists(yolo11_path):
            self.hef_path = yolo11_path
            print(f"Using YOLOv11l model: {self.hef_path}")
        else:
            # Fallback to default model
            try:
                self.hef_path = get_resource_path(
                    pipeline_name=SIMPLE_DETECTION_PIPELINE,
                    resource_type=RESOURCES_MODELS_DIR_NAME,
                )
                print(f"Using default model: {self.hef_path}")
            except:
                raise ValueError("Could not find YOLOv11l model or default model")
        
        # Auto-detect architecture
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch
            
        # Set default video source if none specified
        if self.options_menu.input is None:
            # For multi-instance, prefer test video to avoid webcam conflicts
            # Only use webcam for instance 0
            if instance_id == 0 and os.path.exists("/dev/video0"):
                self.video_source = "/dev/video0"
                print(f"Instance {instance_id}: Using webcam: /dev/video0")
            else:
                self.video_source = get_resource_path(
                    pipeline_name=SIMPLE_DETECTION_PIPELINE,
                    resource_type=RESOURCES_VIDEOS_DIR_NAME,
                    model=SIMPLE_DETECTION_VIDEO_NAME
                )
                print(f"Instance {instance_id}: Using test video: {self.video_source}")
        else:
            self.video_source = self.options_menu.input
            print(f"Instance {instance_id}: Using input: {self.video_source}")
        
        # Post-processing configuration
        try:
            self.post_process_so = get_resource_path(
                pipeline_name=SIMPLE_DETECTION_PIPELINE,
                resource_type=RESOURCES_SO_DIR_NAME,
                model=SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME
            )
        except:
            self.post_process_so = None
            print("Warning: Could not find post-processing shared object")
            
        self.post_function_name = SIMPLE_DETECTION_POSTPROCESS_FUNCTION
        self.labels_json = self.options_menu.labels_json
        self.app_callback = app_callback
        
        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )
        
        # Create pipeline
        self.create_pipeline()
        
    def get_pipeline_string(self):
        """Generate GStreamer pipeline string for this instance"""
        
        # Source pipeline - each instance can share the same source or have individual sources
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width, 
            video_height=self.video_height,
            frame_rate=self.frame_rate, 
            sync=self.sync,
            no_webcam_compression=True,
            name=f'source_{self.instance_id}'
        )
        
        # Inference pipeline with unique vdevice group ID for each instance
        inference_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str,
            name=f'inference_{self.instance_id}',
            vdevice_group_id=self.instance_id + 1  # Unique vdevice group for parallel processing
        )
        
        # User callback pipeline
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        # Display pipeline - only show display for instance 0, others use fakesink
        if self.instance_id == 0:
            display_pipeline = DISPLAY_PIPELINE(
                video_sink=self.video_sink, 
                sync=self.sync, 
                show_fps=self.show_fps
            )
        else:
            # For other instances, just consume the data without display
            display_pipeline = f"{QUEUE(name=f'sink_queue_{self.instance_id}')} ! fakesink sync=false"
        
        # Complete pipeline string
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{inference_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
        
        print(f"Instance {self.instance_id} pipeline: {pipeline_string}")
        return pipeline_string

def monitor_fps():
    """Monitor and print basic status - detailed stats are printed by each instance"""
    def monitor_thread():
        while True:
            time.sleep(30)  # Print status every 30 seconds
            print("\\n" + "="*60)
            print("MULTI-INSTANCE DETECTION STATUS")
            print("Individual instance statistics are shown above.")
            print("Each instance reports its own FPS and detection counts.")
            print("="*60 + "\\n")
    
    monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True)
    monitor_thread_obj.start()

def run_single_instance(instance_id, total_instances):
    """Run a single instance of the detection app"""
    try:
        print(f"Starting instance {instance_id}...")
        # Initialize GStreamer in each process
        Gst.init(None)
        app = MultiInstanceDetectionApp(instance_id, total_instances)
        app.run()
    except Exception as e:
        print(f"Error in instance {instance_id}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run multiple instances"""
    try:
        print("Multi-Instance Hailo YOLOv11l Detection")
        print("========================================")
        
        # Configuration
        num_instances = 4
        
        # Check for YOLOv11l model
        yolo11_path = "/home/sam/rpi-5-halio-pwm/models/yolo11l.onnx"
        if os.path.exists(yolo11_path):
            print(f"Found YOLOv11l model: {yolo11_path}")
        else:
            print(f"Warning: YOLOv11l model not found at {yolo11_path}")
            print("Will use default model instead.")
        
        print(f"Starting {num_instances} parallel inference instances...")
        print("Each instance will run on a separate vdevice group for maximum parallelism.")
        print("Press Ctrl+C to stop all instances.\\n")
        
        # Start FPS monitoring thread
        monitor_fps()
        
        # Use multiprocessing instead of threading to avoid signal handler conflicts
        processes = []
        
        for i in range(num_instances):
            process = multiprocessing.Process(
                target=run_single_instance, 
                args=(i, num_instances)
            )
            processes.append(process)
            process.start()
            time.sleep(1)  # Small delay between instance starts
        
        # Wait for all processes
        try:
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            print("\\nShutting down all instances...")
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
            
    except KeyboardInterrupt:
        print("\\nApplication stopped by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set environment for hailo-apps-infra
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        os.environ["HAILO_ENV_FILE"] = str(env_file)
    
    # Set multiprocessing start method for compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    main()
