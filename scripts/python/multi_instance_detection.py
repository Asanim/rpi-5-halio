#!/usr/bin/env python3
"""
Enhanced Multi-Instance Hailo Detection with YOLOv11l
Runs 4 parallel inference instances with individual FPS tracking.
Updated to use YOLOv11l model as outlined in Hailo Model Zoo.
"""

import os
import time
import threading
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp

# Enhanced callback class with better FPS tracking per instance
class EnhancedInstanceCallback(app_callback_class):
    """Enhanced callback class with accurate FPS tracking for each instance"""
    
    def __init__(self, instance_id):
        super().__init__()
        self.instance_id = instance_id
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.detection_count = 0
        self.total_detections = 0
        self.frame_times = []
        
    def calculate_fps(self):
        """Calculate accurate FPS"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 60 frames for calculation
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
            
        # Calculate FPS from frame times
        if len(self.frame_times) >= 2:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.current_fps = (len(self.frame_times) - 1) / time_span
                
        return self.current_fps

# Global storage for callback instances
instance_callbacks = {}

# Enhanced callback function factory
def create_enhanced_callback(instance_id):
    """Create an enhanced callback function for specific instance"""
    def enhanced_callback(pad, info, user_data):
        user_data.increment()
        fps = user_data.calculate_fps()
        
        # Process detections
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
            
        detections_this_frame = []
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
                label = detection.get_label()
                confidence = detection.get_confidence()
                detections_this_frame.append(f"{label}: {confidence:.2f}")
                user_data.total_detections += 1
        except Exception as e:
            if user_data.get_count() % 100 == 0:  # Only print error occasionally
                print(f"Instance {instance_id}: Detection error: {e}")
                
        user_data.detection_count = len(detections_this_frame)
        
        # Print comprehensive status every 90 frames
        if user_data.get_count() % 90 == 0:
            runtime = time.time() - user_data.start_time
            avg_fps = user_data.get_count() / runtime if runtime > 0 else 0
            
            print(f"🔍 Instance {instance_id}: Frame {user_data.get_count():6d} | "
                  f"FPS: {fps:5.1f} | Avg: {avg_fps:5.1f} | "
                  f"Detections: {user_data.detection_count:2d} | "
                  f"Total: {user_data.total_detections:6d}")
                  
        return Gst.PadProbeReturn.OK
        
    return enhanced_callback

class YOLOv11lDetectionApp(GStreamerDetectionApp):
    """Enhanced detection app specifically configured for YOLOv11l"""
    
    def __init__(self, instance_id=0):
        self.instance_id = instance_id
        
        # Create enhanced callback for this instance
        user_data = EnhancedInstanceCallback(instance_id)
        app_callback = create_enhanced_callback(instance_id)
        
        # Store globally for monitoring
        instance_callbacks[instance_id] = user_data
        
        # Initialize parent with our callback
        super().__init__(app_callback, user_data)
        
        # Override model path for YOLOv11l
        yolo11_path = "/home/sam/rpi-5-halio-pwm/models/yolo11l.onnx"
        if os.path.exists(yolo11_path):
            self.hef_path = yolo11_path
            print(f"✅ Instance {instance_id}: Using YOLOv11l model: {yolo11_path}")
        else:
            print(f"⚠️  Instance {instance_id}: YOLOv11l not found, using default model: {self.hef_path}")
        
        # Configure for YOLOv11l (larger model, different thresholds)
        self.batch_size = 1  # YOLOv11l works best with batch size 1
        
        # Updated thresholds for YOLOv11l
        nms_score_threshold = 0.25  # Lower threshold for YOLOv11l
        nms_iou_threshold = 0.45
        
        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )
        
        print(f"🎯 Instance {instance_id}: Configured with thresholds - Score: {nms_score_threshold}, IoU: {nms_iou_threshold}")
        
    def get_pipeline_string(self):
        """Override to add instance-specific configuration"""
        # Get the base pipeline string
        base_pipeline = super().get_pipeline_string()
        
        # Modify hailonet element to use different vdevice-group-id for each instance
        # This enables true parallel processing on different Hailo cores
        modified_pipeline = base_pipeline.replace(
            'hailonet ',
            f'hailonet vdevice-group-id={self.instance_id + 1} '
        )
        
        # For instances other than 0, replace display with fakesink to save resources
        if self.instance_id > 0:
            modified_pipeline = modified_pipeline.replace(
                'fpsdisplaysink',
                'fakesink sync=false'
            )
            
        return modified_pipeline

def run_instance(instance_id):
    """Run a single detection instance"""
    try:
        print(f"🚀 Starting instance {instance_id}...")
        
        # Set environment for this instance
        project_root = Path(__file__).resolve().parent.parent
        env_file = project_root / ".env"
        env_path_str = str(env_file)
        os.environ["HAILO_ENV_FILE"] = env_path_str
        
        # Create and run the app
        app = YOLOv11lDetectionApp(instance_id)
        app.run()
        
    except Exception as e:
        print(f"❌ Error in instance {instance_id}: {e}")
        import traceback
        traceback.print_exc()

def monitor_performance():
    """Monitor and display performance statistics for all instances"""
    def monitoring_thread():
        print("📊 Performance monitoring started...")
        time.sleep(5)  # Give instances time to start
        
        while True:
            time.sleep(15)  # Update every 15 seconds
            
            print("\\n" + "="*80)
            print("🎯 MULTI-INSTANCE YOLOv11l PERFORMANCE DASHBOARD")
            print("="*80)
            
            total_fps = 0
            total_frames = 0
            total_detections = 0
            
            for instance_id in sorted(instance_callbacks.keys()):
                callback = instance_callbacks[instance_id]
                runtime = time.time() - callback.start_time
                avg_fps = callback.get_count() / runtime if runtime > 0 else 0
                
                total_fps += avg_fps
                total_frames += callback.get_count()
                total_detections += callback.total_detections
                
                # Performance indicators
                perf_indicator = "🟢" if avg_fps > 20 else "🟡" if avg_fps > 10 else "🔴"
                
                print(f"{perf_indicator} Instance {instance_id}: "
                      f"Frames: {callback.get_count():7,d} | "
                      f"Current FPS: {callback.current_fps:6.1f} | "
                      f"Average FPS: {avg_fps:6.1f} | "
                      f"Detections: {callback.total_detections:6,d}")
            
            print("-" * 80)
            efficiency = (total_fps / (len(instance_callbacks) * 30)) * 100 if len(instance_callbacks) > 0 else 0
            
            print(f"📈 SUMMARY: {len(instance_callbacks)} instances | "
                  f"Combined FPS: {total_fps:.1f} | "
                  f"Total Frames: {total_frames:,d} | "
                  f"Total Detections: {total_detections:,d}")
            print(f"⚡ Efficiency: {efficiency:.1f}% of theoretical max (30 FPS per instance)")
            print("="*80 + "\\n")
    
    monitor_thread = threading.Thread(target=monitoring_thread, daemon=True)
    monitor_thread.start()

def main():
    """Main function to coordinate multiple YOLOv11l detection instances"""
    print("🎯 Multi-Instance Hailo YOLOv11l Detection System")
    print("="*60)
    
    # Configuration
    num_instances = 4
    
    # Pre-flight checks
    yolo11_path = "/home/sam/rpi-5-halio-pwm/models/yolo11l.onnx"
    if os.path.exists(yolo11_path):
        print(f"✅ Found YOLOv11l model: {yolo11_path}")
    else:
        print(f"⚠️  YOLOv11l model not found at {yolo11_path}")
        print("Will attempt to use default model instead.")
    
    print(f"🚀 Launching {num_instances} parallel inference instances...")
    print("📊 Each instance uses a separate vdevice-group-id for maximum parallelism")
    print("🎥 Instance 0 shows video output, others run headless for performance")
    print("📈 Performance monitoring will start in 5 seconds...")
    print("⏹️  Press Ctrl+C to stop all instances\\n")
    
    # Start performance monitoring
    monitor_performance()
    
    # Launch instances in separate threads
    threads = []
    
    for i in range(num_instances):
        thread = threading.Thread(target=run_instance, args=(i,), daemon=True)
        threads.append(thread)
        thread.start()
        time.sleep(0.5)  # Small delay between starts
        
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
            # Check if any thread died
            alive_threads = [t for t in threads if t.is_alive()]
            if len(alive_threads) < num_instances:
                print(f"⚠️  Warning: Only {len(alive_threads)}/{num_instances} instances running")
                
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down all instances...")
        print("Waiting for graceful cleanup...")
        time.sleep(2)

if __name__ == "__main__":
    main()
