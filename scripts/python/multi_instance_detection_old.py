#!/usr/bin/env python3
"""
Multi-Instance Hailo YOLOv11l Detection Pipeline
Runs 4 parallel inference instances on the Hailo processor with individual FPS tracking.
"""

import os
import time
import threading
from pathlib import Path
from collections import defaultdict
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo

from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.defines import (
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
    """Callback class for tracking multiple inference instances with FPS"""
    
    def __init__(self, instance_id):
        super().__init__()
        self.instance_id = instance_id
        self.last_time = time.time()
        self.frame_times = []
        self.fps = 0.0
        self.detection_count = 0
        
    def calculate_fps(self):
        """Calculate FPS based on recent frame times"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only the last 30 frames for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff
        
        return self.fps

class MultiInstanceHailoApp:
    """Main application class for running multiple Hailo inference instances"""
    
    def __init__(self):
        self.num_instances = 4
        self.pipelines = []
        self.callbacks = []
        self.main_loop = None
        self.setup_gstreamer()
        self.setup_parameters()
        
    def setup_gstreamer(self):
        """Initialize GStreamer"""
        Gst.init(None)
        
    def setup_parameters(self):
        """Setup model and pipeline parameters"""
        # Auto-detect Hailo architecture
        self.arch = detect_hailo_arch()
        if self.arch is None:
            raise ValueError("Could not auto-detect Hailo architecture. Please specify manually.")
        print(f"Auto-detected Hailo architecture: {self.arch}")
        
        # Use YOLOv11l model
        self.hef_path = "/home/sam/rpi-5-halio-pwm/models/yolo11l.onnx"
        if not os.path.exists(self.hef_path):
            # Fallback to default model location
            try:
                self.hef_path = get_resource_path(
                    pipeline_name="simple_detection",
                    resource_type=RESOURCES_MODELS_DIR_NAME,
                )
            except:
                self.hef_path = "/home/sam/rpi-5-halio-pwm/models/yolo11l.onnx"
        
        print(f"Using model: {self.hef_path}")
        
        # Post-processing configuration
        try:
            self.post_process_so = get_resource_path(
                pipeline_name="simple_detection",
                resource_type=RESOURCES_SO_DIR_NAME,
                model=SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME
            )
        except:
            # Fallback if resource path fails
            self.post_process_so = None
            
        self.post_function_name = SIMPLE_DETECTION_POSTPROCESS_FUNCTION
        
        # Model parameters for YOLOv11l
        self.batch_size = 1
        self.video_width = 640
        self.video_height = 640
        self.frame_rate = 30
        
        # Detection thresholds
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45
        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )
        
        # Video source - use webcam or test video
        self.video_source = "/dev/video0"  # Change this to your preferred source
        
    def create_inference_callback(self, instance_id):
        """Create callback function for a specific instance"""
        def app_callback(pad, info, user_data):
            user_data.increment()
            fps = user_data.calculate_fps()
            
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
                
            detection_info = []
            try:
                for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
                    detection_info.append(f"{detection.get_label()}: {detection.get_confidence():.2f}")
                    user_data.detection_count += 1
            except Exception as e:
                print(f"Instance {instance_id}: Error processing detections: {e}")
                
            # Print status every 30 frames
            if user_data.get_count() % 30 == 0:
                print(f"Instance {instance_id}: Frame {user_data.get_count()}, "
                      f"FPS: {fps:.1f}, Detections: {len(detection_info)}")
                      
            return Gst.PadProbeReturn.OK
            
        return app_callback
        
    def create_pipeline_string(self, instance_id):
        """Create GStreamer pipeline string for one instance"""
        # Source pipeline (shared input, but each instance gets its own tee branch)
        if instance_id == 0:
            # Main source pipeline with tee for distribution
            source_pipeline = SOURCE_PIPELINE(
                video_source=self.video_source,
                video_width=self.video_width, 
                video_height=self.video_height,
                frame_rate=self.frame_rate, 
                sync=True,
                no_webcam_compression=True,
                name=f'source_{instance_id}'
            )
            source_pipeline += f" ! tee name=input_tee "
        else:
            # Subsequent instances get their feed from the tee
            source_pipeline = f"input_tee. "
            
        # Inference pipeline for this instance
        inference_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            additional_params=self.thresholds_str,
            name=f'inference_{instance_id}',
            vdevice_group_id=instance_id + 1  # Different vdevice group for each instance
        )
        
        # User callback pipeline
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        # Display pipeline (only for instance 0, others are fakesink)
        if instance_id == 0:
            display_pipeline = DISPLAY_PIPELINE(
                video_sink='autovideosink', 
                sync=True, 
                show_fps=True
            )
        else:
            display_pipeline = f"{QUEUE(name=f'sink_queue_{instance_id}')} ! fakesink sync=false"
            
        # Complete pipeline
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{QUEUE(name=f"queue_to_inference_{instance_id}")} ! '
            f'{inference_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
        
        return pipeline_string
        
    def create_single_source_multi_inference_pipeline(self):
        """Create a single pipeline with multiple inference branches"""
        # Base source
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width, 
            video_height=self.video_height,
            frame_rate=self.frame_rate, 
            sync=True,
            no_webcam_compression=True,
            name='main_source'
        )
        
        # Create tee and multiple inference branches
        pipeline_parts = [source_pipeline + " ! tee name=input_tee"]
        
        for i in range(self.num_instances):
            # Each branch gets its own inference pipeline
            inference_pipeline = INFERENCE_PIPELINE(
                hef_path=self.hef_path,
                post_process_so=self.post_process_so,
                post_function_name=self.post_function_name,
                batch_size=self.batch_size,
                additional_params=self.thresholds_str,
                name=f'inference_{i}',
                vdevice_group_id=i + 1  # Different vdevice group for each instance
            )
            
            # User callback pipeline
            user_callback_pipeline = USER_CALLBACK_PIPELINE()
            
            # Only first instance shows display, others use fakesink
            if i == 0:
                display_pipeline = DISPLAY_PIPELINE(
                    video_sink='autovideosink', 
                    sync=True, 
                    show_fps=True
                )
            else:
                display_pipeline = f"{QUEUE(name=f'sink_queue_{i}')} ! fakesink sync=false"
            
            # Add this branch to the pipeline
            branch = (
                f"input_tee. ! "
                f"{QUEUE(name=f'queue_to_inference_{i}')} ! "
                f"{inference_pipeline} ! "
                f"{user_callback_pipeline} ! "
                f"{display_pipeline}"
            )
            pipeline_parts.append(branch)
        
        # Join all parts
        complete_pipeline = " ".join(pipeline_parts)
        return complete_pipeline
        
    def setup_callbacks(self, pipeline):
        """Setup probe callbacks for each inference instance"""
        for i in range(self.num_instances):
            callback_data = MultiInstanceCallback(i)
            callback_func = self.create_inference_callback(i)
            self.callbacks.append((callback_data, callback_func))
            
            # Find the callback pipeline element for this instance
            callback_element = pipeline.get_by_name('identity')  # USER_CALLBACK_PIPELINE uses identity
            if callback_element:
                # Get the src pad and add probe
                src_pad = callback_element.get_static_pad('src')
                if src_pad:
                    src_pad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        callback_func,
                        callback_data
                    )
                    
    def print_fps_stats(self):
        """Print FPS statistics for all instances"""
        def stats_thread():
            while True:
                time.sleep(5)  # Print stats every 5 seconds
                print("\\n=== FPS Statistics ===")
                total_fps = 0
                for i, (callback_data, _) in enumerate(self.callbacks):
                    fps = callback_data.fps
                    total_fps += fps
                    print(f"Instance {i}: {fps:.1f} FPS, "
                          f"Frames: {callback_data.get_count()}, "
                          f"Detections: {callback_data.detection_count}")
                print(f"Total FPS: {total_fps:.1f}")
                print("========================\\n")
                
        stats_thread_obj = threading.Thread(target=stats_thread, daemon=True)
        stats_thread_obj.start()
        
    def run(self):
        """Run the multi-instance pipeline"""
        try:
            print(f"Starting multi-instance Hailo detection with {self.num_instances} instances...")
            print(f"Using model: {self.hef_path}")
            print(f"Video source: {self.video_source}")
            
            # Create pipeline string
            pipeline_string = self.create_single_source_multi_inference_pipeline()
            print(f"\\nPipeline: {pipeline_string}")
            
            # Create pipeline
            pipeline = Gst.parse_launch(pipeline_string)
            
            # Setup callbacks
            self.setup_callbacks(pipeline)
            
            # Start FPS monitoring thread
            self.print_fps_stats()
            
            # Create main loop
            self.main_loop = GLib.MainLoop()
            
            # Set up bus message handling
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_bus_message)
            
            # Start pipeline
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start pipeline")
                
            print("Pipeline started successfully!")
            print("Press Ctrl+C to stop...")
            
            # Run main loop
            try:
                self.main_loop.run()
            except KeyboardInterrupt:
                print("\\nStopping pipeline...")
                
            # Cleanup
            pipeline.set_state(Gst.State.NULL)
            
        except Exception as e:
            print(f"Error running pipeline: {e}")
            raise
            
    def on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, Debug: {debug}")
            self.main_loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("End of stream")
            self.main_loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Warning: {warn}, Debug: {debug}")

def main():
    """Main function"""
    try:
        # Check for YOLOv11l model
        yolo11_path = "/home/sam/rpi-5-halio-pwm/models/yolo11l.onnx"
        if not os.path.exists(yolo11_path):
            print(f"Warning: YOLOv11l model not found at {yolo11_path}")
            print("The script will attempt to use the default model instead.")
            
        # Create and run the multi-instance app
        app = MultiInstanceHailoApp()
        app.run()
        
    except KeyboardInterrupt:
        print("\\nApplication stopped by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Multi-Instance Hailo YOLOv11l Detection")
    print("========================================")
    main()
