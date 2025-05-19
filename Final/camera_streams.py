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
        self.calibration = {}
    
    def start_realsense(self):
        """Initialize and start RealSense camera"""
        global pipeline, depth_scale
        
        if not REALSENSE_AVAILABLE:
            print("RealSense library not available")
            return False
        
        try:
            # Create pipeline
            pipeline = rs.pipeline()
            
            # Create a config object
            config = rs.config()
            
            # Enable streams
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start streaming
            profile = pipeline.start(config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            self.depth_scale = depth_scale
            print(f"Depth Scale: {depth_scale}")
            
            # Create align object
            self.align = rs.align(rs.stream.color)
            
            # Try to load calibration
            try:
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intrinsics = color_profile.get_intrinsics()
                
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
                
            except Exception as e:
                print(f"Error saving calibration: {e}")
            
            print("RealSense camera started successfully")
            return True
            
        except Exception as e:
            print(f"Error starting RealSense camera: {e}")
            print(traceback.format_exc())
            return False
    
    def stop_realsense(self):
        """Stop RealSense camera"""
        global pipeline
        
        if pipeline:
            try:
                pipeline.stop()
                print("RealSense camera stopped")
            except Exception as e:
                print(f"Error stopping RealSense: {e}")
    
    def _stream_thread(self):
        """Stream from RealSense camera"""
        global current_depth_image, frame_queue
        
        if not self.start_realsense():
            print("Failed to start RealSense camera")
            return
        
        try:
            while not self.stop_flag:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                
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
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in RealSense stream: {e}")
            print(traceback.format_exc())
        finally:
            self.stop_realsense()
    
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
