import cv2
import numpy as np
import time
import threading
import queue
import traceback
import os
from utils import resize_with_aspect_ratio, update_display

# Import centralized debug system
from debug_utils import debug_print

# Try importing RealSense library
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    debug_print("RealSense library found", "startup")
except ImportError:
    REALSENSE_AVAILABLE = False
    debug_print("RealSense library not found - depth camera functionality will be limited", "errors")

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
            debug_print("Stream already running", "cameras")
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
    def __init__(self, width=1280, height=720, fps=30, intrinsics=None, skip_cache_loading=False):
        super().__init__()
        # Force 1280x720 resolution for consistency
        self.width = 1280
        self.height = 720
        self.fps = 30
        self.pipeline = None
        self.depth_scale = 0.001
        self.depth_frame = None
        self.color_frame = None
        self.aligned_frames = None
        self.align = None
        self.calibration = intrinsics if intrinsics else {}
        self.is_calibrated = bool(intrinsics)
        self.depth_intrin = None
        
        # Create pipeline object once during initialization
        # But don't try to start it yet
        if REALSENSE_AVAILABLE:
            self.pipeline = rs.pipeline()
            
        # Load calibration from file if available (only if not provided in constructor AND not skipping cache)
        if not self.is_calibrated and not skip_cache_loading:
            self.load_calibration_from_file()
       

    def load_calibration_from_file(self):
        """Load calibration data from unified intrinsics file if available"""
        try:
            from camera_utils import load_camera_intrinsics
            
            # Try to load from unified intrinsics file
            realsense_data = load_camera_intrinsics("realsense")
            if realsense_data:
                self.calibration = {
                    "fx": realsense_data.get("fx", 0),
                    "fy": realsense_data.get("fy", 0),
                    "cx": realsense_data.get("cx", 0),
                    "cy": realsense_data.get("cy", 0),
                    "width": realsense_data.get("width", 1280),
                    "height": realsense_data.get("height", 720)
                }
                self.is_calibrated = True
                debug_print("Loaded RealSense calibration from unified intrinsics file", "startup")
                
        except Exception as e:
            debug_print(f"Error loading calibration file: {e}", "errors")
            
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
                debug_print(f"Depth Scale: {self.depth_scale}", "startup")
                
                # Get device info
                device = self.profile.get_device()
                device_serial = device.get_info(rs.camera_info.serial_number)
                device_product_line = device.get_info(rs.camera_info.product_line)
                
                # Save to unified intrinsics file
                unified_intrinsics_data = {
                    "camera_matrix": [
                        [intrinsics.fx, 0.0, intrinsics.ppx],
                        [0.0, intrinsics.fy, intrinsics.ppy],
                        [0.0, 0.0, 1.0]
                    ],
                    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],  # Default distortion
                    "width": intrinsics.width,
                    "height": intrinsics.height,
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.ppx,
                    "cy": intrinsics.ppy,
                    "device_serial": device_serial,
                    "device_product_line": device_product_line
                }
                
                # Save to unified intrinsics file
                from camera_utils import save_camera_intrinsics
                save_camera_intrinsics("realsense", unified_intrinsics_data)
                
                # Update calibration dict for internal use
                self.calibration = {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.ppx,
                    "cy": intrinsics.ppy,
                    "width": intrinsics.width,
                    "height": intrinsics.height
                }
                
                self.is_calibrated = True
                debug_print(f"Camera calibrated successfully and saved to unified intrinsics file", "startup")
                debug_print(f"Resolution: {intrinsics.width}x{intrinsics.height}", "startup")
                debug_print(f"Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}", "startup")
                    
        except Exception as e:
            debug_print(f"Error during camera calibration: {e}", "errors")
    def get_depth_intrinsics(self):
        """Return the cached depth intrinsics"""
        return self.depth_intrin if hasattr(self, 'depth_intrin') else None        
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
            debug_print("RealSense library not available", "errors")
            return False
            
        if not self.pipeline:
            debug_print("RealSense pipeline not initialized", "errors")
            return False
        
        try:
            # Create config object
            config = rs.config() 
            
            # Check for connected devices first
            ctx = rs.context()
            if ctx.query_devices().size() == 0:
                debug_print("No RealSense devices detected", "errors")
                return False
                    
            # Enable streams
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start streaming with timeout protection
            pipeline_started = False
            start_error = None
            
            def start_pipeline_thread():
                nonlocal pipeline_started, start_error
                try:
                    # Start pipeline and get profile
                    profile = self.pipeline.start(config)
                    self.profile = profile
                    
                    # Create alignment object
                    self.align = rs.align(rs.stream.color)

                    # Configure depth sensor
                    depth_sensor = profile.get_device().first_depth_sensor()
                    self.depth_scale = depth_sensor.get_depth_scale()
                    global depth_scale
                    depth_scale = self.depth_scale

                    # Set high accuracy preset if available
                    if depth_sensor.supports(rs.option.visual_preset):
                        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
                    
                    # Set maximum laser power if available
                    if depth_sensor.supports(rs.option.laser_power):
                        max_power = depth_sensor.get_option_range(rs.option.laser_power).max
                        depth_sensor.set_option(rs.option.laser_power, max_power)
                        debug_print(f"Laser power set to: {max_power}", "cameras")

                    # Cache depth intrinsics
                    depth_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                    self.depth_intrin = depth_profile.get_intrinsics()
                    debug_print(f"Cached depth intrinsics - fx: {self.depth_intrin.fx}, fy: {self.depth_intrin.fy}", "cameras")

                    # Calibrate camera
                    self.calibrate_camera()
                    
                    pipeline_started = True
                    
                except Exception as e:
                    start_error = e
            
            # Start pipeline in separate thread with timeout
            start_thread = threading.Thread(target=start_pipeline_thread)
            start_thread.daemon = True
            start_thread.start()
            
            # Wait for thread completion with timeout
            start_time = time.time()
            timeout = 5.0
            
            while not pipeline_started and time.time() - start_time < timeout:
                time.sleep(0.1)
                if start_error:
                    debug_print(f"Error starting pipeline: {start_error}", "errors")
                    self.pipeline = None
                    return False
            
            if not pipeline_started:
                debug_print("Timeout waiting for RealSense camera", "errors")
                return False

            debug_print(f"RealSense camera started: {self.width}x{self.fps}fps", "cameras")
            return True
            
        except Exception as e:
            debug_print(f"Error initializing RealSense: {e}", "errors")
            self.pipeline = None
            return False
    
    def stop_realsense(self):
        """Stop RealSense camera"""
        try:
            if self.pipeline:
                self.pipeline.stop()
            return True
        except Exception as e:
            debug_print(f"Error stopping RealSense camera: {e}", "errors")
            return False
    
    def stop(self):
        """Stop streaming"""
        super().stop()
        self.stop_realsense()
        debug_print("RealSense camera stopped", "cameras")
    
    def _stream_thread(self):
        """Stream from RealSense camera"""
        global current_depth_image, frame_queue
        
        # We no longer need to call start_realsense() here as it's called from start() method
        # which already invokes super().start() that calls this method
        if not self.pipeline or not self.align:
            debug_print("RealSense pipeline or align object not initialized", "errors")
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
                    debug_print(f"Error processing frame: {inner_e}", "errors")
                    time.sleep(0.1)  # Wait a bit before trying again
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.01)
                
        except Exception as e:
            debug_print(f"Error in RealSense stream: {e}", "errors")
            debug_print(traceback.format_exc(), "errors")
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
    def __init__(self, url, intrinsics=None):
        super().__init__()
        self.url = url
        self.cap = None
        self.intrinsics = intrinsics if intrinsics else {}
    
    def _stream_thread(self):
        """Stream from IP camera"""
        try:
            # Open capture
            self.cap = cv2.VideoCapture(self.url)
            
            if not self.cap.isOpened():
                debug_print(f"Failed to open IP camera at {self.url}", "errors")
                return
            
            while not self.stop_flag:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    debug_print("Failed to read frame from IP camera", "errors")
                    time.sleep(1.0)  # Wait before retry
                    continue
                
                # Store current frame
                self.current_frame = frame
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.03)  # ~30 FPS
        
        except Exception as e:
            debug_print(f"Error in IP camera stream: {e}", "errors")
            debug_print(traceback.format_exc(), "errors")
        finally:
            if self.cap:
                self.cap.release()
