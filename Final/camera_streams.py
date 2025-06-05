import cv2
import numpy as np
import time
import threading
import queue
import traceback
import os
from utils import resize_with_aspect_ratio, update_display


# Try importing RealSense library
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("RealSense library found")
except ImportError:
    REALSENSE_AVAILABLE = False
    print("RealSense library not found - depth camera functionality will be limited")

# Global variables for stream control

pipeline = None
frame_queue = queue.Queue(maxsize=10)
current_depth_image = None
depth_scale = 0.001  # Default scale, will be updated when starting RealSense

class CameraStreamer:
    """Base class for camera streaming"""
    def __init__(self):
        self.stop_flag = False
        self.is_streaming = False
        self.current_frame = None
        self.thread = None
    
    def start(self):
        """Start the camera streaming thread"""
        if self.is_streaming:
            print("Stream already running")
            return
        
        self.stop_flag = False
        self.thread = threading.Thread(target=self._stream_thread, daemon=True)
        self.thread.start()
        self.is_streaming = True
    
    def stop(self):
        """Stop the camera streaming thread"""
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1.0)
        self.is_streaming = False
    
    def get_frame(self):
        """Get the current frame"""
        return self.current_frame
    
    def _stream_thread(self):
        """Streaming thread to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _stream_thread method")

class RealSenseCamera(CameraStreamer):
    """RealSense depth camera streamer"""
    def __init__(self, width=1280, height=720, fps=30):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.depth_scale = 0.001
        self.depth_frame = None
        self.color_frame = None
        self.aligned_frames = None
        self.align = None
        self.calibration = {}
        self.is_calibrated = False
        
        # Create pipeline object once during initialization
        # But don't try to start it yet
        if REALSENSE_AVAILABLE:
            self.pipeline = rs.pipeline()
            
        # Load calibration from file if available
        self.load_calibration_from_file()
            
    def load_calibration_from_file(self):
        """Load calibration data from file if available"""
        try:
            import json
            if os.path.exists("realsense_calibration.json"):
                with open("realsense_calibration.json", "r") as f:
                    calibration_data = json.load(f)
                    intrinsics = calibration_data.get("intrinsics", {})
                    
                    self.calibration = {
                        "fx": intrinsics.get("fx", 0),
                        "fy": intrinsics.get("fy", 0),
                        "cx": intrinsics.get("ppx", 0),
                        "cy": intrinsics.get("ppy", 0)
                    }
                    self.is_calibrated = True
                    print("Loaded calibration from file")
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            
    def calibrate_camera(self):
        """Calibrate the camera if connected - only called when actually streaming"""
        if not REALSENSE_AVAILABLE or not self.pipeline or self.is_calibrated:
            return
            
        try:
            # We'll only do this if we're already streaming, using the active profile
            if hasattr(self, 'profile') and self.profile:
                color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
                intrinsics = color_profile.get_intrinsics()
                
                # Get depth scale
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                print(f"Depth Scale: {self.depth_scale}")
                
                # Save to calibration file
                calibration_data = {
                    "intrinsics": {
                        "fx": intrinsics.fx,
                        "fy": intrinsics.fy,
                        "ppx": intrinsics.ppx,
                        "ppy": intrinsics.ppy,
                        "width": intrinsics.width,
                        "height": intrinsics.height
                    }
                }
                    
                # Save to file
                import json
                with open("realsense_calibration.json", "w") as f:
                    json.dump(calibration_data, f, indent=4)
                
                # Update calibration dict
                self.calibration = {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.ppx,
                    "cy": intrinsics.ppy
                }
                
                self.is_calibrated = True
                print("Camera calibrated successfully")
                    
        except Exception as e:
            print(f"Error during camera calibration: {e}")
            
    def start(self):
        """Override start method to initialize RealSense camera first"""
        if self.start_realsense():
            super().start()
            return True
        return False
            
    def start_realsense(self):
        """Initialize and start RealSense camera with timeout protection"""
        global depth_scale
        
        if not REALSENSE_AVAILABLE:
            print("RealSense library not available")
            return False
            
        if not self.pipeline:
            print("RealSense pipeline not initialized")
            return False
        
        # Create a timeout to prevent hanging when no camera is connected
        import time
        
        try:
            # Create a config object
            config = rs.config() 
             
            
            # Try to check if devices are connected first to fail faster
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if devices.size() == 0:
                print("No RealSense devices detected")
                return False
                     
            # Enable streams
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start streaming with a timeout protection
            try:
                # Create a separate thread for the pipeline start to implement timeout
                pipeline_started = False
                start_error = None
                
                def start_pipeline_thread():
                    nonlocal pipeline_started, start_error
                    try:
                        profile = self.pipeline.start(config)
                        self.profile = profile
                        pipeline_started = True
                    except Exception as e:
                        start_error = e
                
                # Start pipeline in separate thread
                start_thread = threading.Thread(target=start_pipeline_thread)
                start_thread.daemon = True
                start_thread.start()
                
                # Wait for thread with timeout (2 seconds is usually enough)
                start_time = time.time()
                timeout = 2.0  # 2 seconds timeout
                
                while not pipeline_started and time.time() - start_time < timeout:
                    time.sleep(0.1)
                    if start_error:
                        raise start_error
                
                if not pipeline_started:
                    print("Timeout waiting for RealSense camera")
                    return False
                
                # Create alignment object
                self.align = rs.align(rs.stream.color)
                
                # Get depth scale
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                depth_scale = self.depth_scale  # Update global
                
                # Once we have a successful connection, attempt to calibrate
                self.calibrate_camera()
                
                print(f"RealSense camera started: {self.width}x{self.height} @ {self.fps}fps")
                return True
                
            except RuntimeError as e:
                print(f"Error starting RealSense camera: {e}")
                self.pipeline = None  # Reset pipeline to avoid further errors
                return False
                
        except Exception as e:
            print(f"General error initializing RealSense: {e}")
            return False
    
    def stop_realsense(self):
        """Stop RealSense camera"""
        try:
            if self.pipeline:
                self.pipeline.stop()
            return True
        except Exception as e:
            print(f"Error stopping RealSense camera: {e}")
            return False
    
    def stop(self):
        """Stop streaming"""
        super().stop()
        self.stop_realsense()
        print("RealSense camera stopped")
    
    def _stream_thread(self):
        """Stream from RealSense camera"""
        global current_depth_image, frame_queue
        
        # We no longer need to call start_realsense() here as it's called from start() method
        # which already invokes super().start() that calls this method
        if not self.pipeline or not self.align:
            print("RealSense pipeline or align object not initialized")
            return
        
        try:
            while not self.stop_flag:
                try:
                    # Wait for frames
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    
                    # Align depth frame to color frame
                    aligned_frames = self.align.process(frames)
                    
                    # Get aligned frames
                    aligned_depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if not aligned_depth_frame or not color_frame:
                        continue
                    
                    # Store frames
                    self.aligned_frames = aligned_frames
                    self.depth_frame = aligned_depth_frame
                    self.color_frame = color_frame
                    
                    # Convert frames to numpy arrays
                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Store current depth image for other processing
                    current_depth_image = depth_image
                    
                    # Store current color frame
                    self.current_frame = color_image
                    
                    # Add to queue for processing
                    try:
                        if not frame_queue.full():
                            frame_queue.put((color_image, depth_image), block=False)
                    except queue.Full:
                        # Queue is full, skip this frame
                        pass
                except Exception as inner_e:
                    print(f"Error processing frame: {inner_e}")
                    time.sleep(0.1)  # Wait a bit before trying again
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in RealSense stream: {e}")
            print(traceback.format_exc())
        finally:
            pass
            # Don't call stop_realsense here as it will be called by the stop() method
    
    def get_depth_frame(self):
        """Get the current depth frame"""
        return self.depth_frame
    
    def get_color_frame(self):
        """Get the current color frame"""
        return self.color_frame
    
    def get_intrinsics(self):
        """Get camera intrinsics"""
        return self.calibration

class IPCamera(CameraStreamer):
    """IP camera streamer"""
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.cap = None
    
    def _stream_thread(self):
        """Stream from IP camera"""
        try:
            # Open capture
            self.cap = cv2.VideoCapture(self.url)
            
            if not self.cap.isOpened():
                print(f"Failed to open IP camera at {self.url}")
                return
            
            while not self.stop_flag:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame from IP camera")
                    time.sleep(1.0)  # Wait before retry
                    continue
                
                # Store current frame
                self.current_frame = frame
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.03)  # ~30 FPS
        
        except Exception as e:
            print(f"Error in IP camera stream: {e}")
            print(traceback.format_exc())
        finally:
            if self.cap:
                self.cap.release()

class USBCamera(CameraStreamer):
    """USB camera streamer"""
    def __init__(self, device_id=0, width=1280, height=720):
        super().__init__()
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = None
    
    def _stream_thread(self):
        """Stream from USB camera"""
        try:
            # Open capture
            self.cap = cv2.VideoCapture(self.device_id)
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if not self.cap.isOpened():
                print(f"Failed to open USB camera {self.device_id}")
                return
            
            while not self.stop_flag:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame from USB camera")
                    time.sleep(1.0)  # Wait before retry
                    continue
                
                # Store current frame
                self.current_frame = frame
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.03)  # ~30 FPS
        
        except Exception as e:
            print(f"Error in USB camera stream: {e}")
            print(traceback.format_exc())
        finally:
            if self.cap:
                self.cap.release()

def update_camera_display(panel, camera, width, height):
    """
    Update display panel with camera frame
    
    Args:
        panel: tkinter panel to update
        camera: Camera object
        width: Display width
        height: Display height
    """
    if not camera.is_streaming:
        return
    
    frame = camera.get_frame()
    if frame is not None:
        update_display(panel, frame, width, height)
    
    # Schedule next update
    panel.after(30, lambda: update_camera_display(panel, camera, width, height))
