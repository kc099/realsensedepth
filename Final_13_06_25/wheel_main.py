from turtle import done
import cv2
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import numpy as np
import math
import traceback
from datetime import datetime
import sqlite3
import queue
from PIL import Image, ImageTk  # Still needed for UI image handling

# Import from modular files
from utils import update_display, update_panel_image, update_clock, resize_with_aspect_ratio
from camera_streams import RealSenseCamera, IPCamera, USBCamera, current_depth_image, frame_queue
from image_processing import process_frame, load_model_in_background
from wheel_measurements import classify_wheel_model
from settings_manager import load_settings, save_settings, load_realsense_calibration
from signal_handler import SignalHandler
from database import init_db, add_inspection, get_daily_report, get_monthly_report, get_date_range_report
from reports_window import show_report_window
from settings_window import show_settings_window
from app_icon import set_app_icon

# Set the environment variable to avoid OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
SAVE_DIR = "captured_frames1"

# UI Color Scheme
BG_COLOR = "#AFE1AF"
TEXT_COLOR = "#333333"
BUTTON_COLOR = "#023020"
HIGHLIGHT_COLOR = "#5c8cd5"
PASS_COLOR = "#4caf50"
FAIL_COLOR = "#f44336"
REALSENSE_URL = "realsense://"
TOP_CAMERA_URL = "http://192.168.100.50:8080/stream-hd"  # Adjust based on your camera

# Global variables
current_settings = None
WHEEL_MODELS = None
stop_streaming = False
streaming_active = False
auto_capture_active = False
photo_count = 0
settings_win = None
reports_win = None
wheel_model_counts = {}
model_count_vars = {}  # Will hold StringVar for each model count
current_model_display = None  # Will hold the StringVar for current model in result frame
count_frame = None  # Add count_frame to global variables

# Panel size constants - optimized for 16:9 camera aspect ratio (1280x720)
SIDE_PANEL_WIDTH = 640
SIDE_PANEL_HEIGHT = 360  # 640/360 = 1.78 ≈ 16:9 ratio
TOP_PANEL_WIDTH = 640
TOP_PANEL_HEIGHT = 360   # 640/360 = 1.78 ≈ 16:9 ratio

# Global variables for real-world measurements
real_measurements = {}

# Global variables for camera objects
realsense_camera = None  # RealSense D435 camera for side view (depth)
top_camera = None        # IP camera for top view
side_camera = None       # This refers to the realsense_camera (for clarity)
signal_handler = None  # Will be initialized at startup

# Frame storage
frame_top = None
frame_side = None
# Initialize global variables before GUI creation
def init_globals():
    """Initialize global variables and configurations"""
    global current_settings, WHEEL_MODELS, realsense_camera, top_camera, side_camera    
    
    # Ensure captured frames directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Load settings
    current_settings, WHEEL_MODELS = load_settings()
    
    # Ensure required camera URL settings exist (fixes settings window display)
    if "top_camera_url" not in current_settings:
        current_settings["top_camera_url"] = TOP_CAMERA_URL
    
    if "side_camera_url" not in current_settings:
        current_settings["side_camera_url"] = REALSENSE_URL
    
    # Add capture_interval if missing (needed for settings_window.py)
    if "capture_interval" not in current_settings:
        # Check if auto_capture_interval exists and use that value if available
        if "auto_capture_interval" in current_settings:
            current_settings["capture_interval"] = current_settings["auto_capture_interval"]
        else:
            # Default to 5 seconds
            current_settings["capture_interval"] = 5.0
        
    # Initialize camera objects - only create objects, don't start streaming yet
    try:
        # Initialize RealSense camera
        realsense_camera = RealSenseCamera()
        side_camera = realsense_camera  # For clarity, side_camera is the same as realsense_camera
        
        # Initialize top camera
        top_camera = IPCamera(TOP_CAMERA_URL)
        
        print("Camera objects initialized successfully")
    except Exception as e:
        print(f"Error initializing camera objects: {e}")
        
    # Save settings to ensure they persist
    save_settings(current_settings, WHEEL_MODELS)
     # Load the AI model immediately (no background loading)
    print("Loading AI model...")
    from image_processing import load_model
    model = load_model()
    if model is not None:
        print("AI model loaded successfully")
    else:
        print("Warning: AI model failed to load")

def take_photo():
    """Trigger background measurement thread without blocking UI and without saving images"""
    if not streaming_active:
        status_label_main.config(text="Error: Streams not active")
        return
    
    # Start the measurement process in a background thread
    threading.Thread(target=run_measurement_thread, daemon=True).start()

def run_measurement_thread():
    """Complete measurement processing in background thread without saving images"""
    global photo_count, frame_top, frame_side, signal_handler
    
    try:
        # Initialize status
        root.after(0, lambda: status_label_main.config(text="Starting measurement..."))
        
        # Create local copies of frames to avoid threading issues
        local_top = frame_top.copy() if frame_top is not None else None
        local_side = frame_side.copy() if frame_side is not None else None
        
        # Check camera availability - FIXED: Properly handle numpy arrays
        frames_available = True
        if local_top is None and local_side is None:
            frames_available = False
        elif local_top is not None and isinstance(local_top, np.ndarray) and local_top.size == 0:
            frames_available = False
        elif local_side is not None and isinstance(local_side, np.ndarray) and local_side.size == 0:
            frames_available = False
            
        if not frames_available:
            root.after(0, lambda: status_label_main.config(text="Error: No frames available"))
            if signal_handler:
                signal_handler.send_measurement_data(None, 0.0, 0.0, WHEEL_MODELS)
            return
        
        # Determine camera scenario
        camera_scenario = "both"
        if local_top is not None and local_side is None:
            local_side = local_top.copy()
            camera_scenario = "top_only"
            print("[THREAD] Using top frame for both views")
        elif local_side is not None and local_top is None:
            local_top = local_side.copy()
            camera_scenario = "side_only"
            print("[THREAD] Using side frame for both views")
        
        # Get aligned depth frame if available
        depth_frame = None
        aligned_color_frame = None
        if realsense_camera and realsense_camera.is_streaming:
            try:
                if hasattr(realsense_camera, 'aligned_frames') and realsense_camera.aligned_frames:
                    aligned_depth_frame = realsense_camera.aligned_frames.get_depth_frame()
                    aligned_color_frame_rs = realsense_camera.aligned_frames.get_color_frame()
                    if aligned_depth_frame:
                        depth_frame = aligned_depth_frame
                        aligned_color_frame = np.asanyarray(aligned_color_frame_rs.get_data())  
                        local_side = aligned_color_frame.copy()  
                        print("[THREAD] Using aligned depth frame")
                elif hasattr(realsense_camera, 'get_depth_frame'):
                    depth_frame = realsense_camera.get_depth_frame()
                    print("[THREAD] Using unaligned depth frame")
            except Exception as e:
                print(f"[THREAD] Depth frame error: {e}")
        
        # Display captured frames in main panels
        root.after(0, lambda: update_display(side_panel, local_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit"))
        root.after(0, lambda: update_display(top_panel, local_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit"))
        
        # Get current model
        model_name = current_model_display.get().strip() if current_model_display and current_model_display.get() else "Unknown"
        
        # Process side view
        root.after(0, lambda: status_label_main.config(text="Processing side view..."))
        side_result = process_frame(
            local_side,
            is_top_view=False,
            camera_settings=current_settings,
            wheel_models=WHEEL_MODELS,
            selected_model=model_name,
            depth_frame=depth_frame
        )
        
        if not side_result:
            root.after(0, lambda: status_label_main.config(text="Error: Side processing failed"))
            if signal_handler:
                signal_handler.send_measurement_data(None, 0.0, 0.0, WHEEL_MODELS)
            return
        
        # Update side processed panel with visualization
        root.after(0, lambda: update_display(
            side_processed_panel, 
            side_result['visualization'], 
            SIDE_PANEL_WIDTH, 
            SIDE_PANEL_HEIGHT, 
            "fit"
        ))
        
        # Process top view
        root.after(0, lambda: status_label_main.config(text="Processing top view..."))
        top_result = process_frame(
            local_top,
            is_top_view=True,
            camera_settings=current_settings,
            wheel_models=WHEEL_MODELS,
            selected_model=model_name,
            wheel_height_mm=side_result['height_mm']
        )
        
        if not top_result:
            root.after(0, lambda: status_label_main.config(text="Error: Top processing failed"))
            if signal_handler:
                signal_handler.send_measurement_data(None, 0.0, 0.0, WHEEL_MODELS)
            return
        
        # Update top processed panel with visualization
        root.after(0, lambda: update_display(
            top_processed_panel, 
            top_result['visualization'], 
            TOP_PANEL_WIDTH, 
            TOP_PANEL_HEIGHT, 
            "fit"
        ))
        
        # Update photo count
        photo_count += 1
        
        # Generate timestamp for part number
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        part_no = f"INDIP {timestamp.split('_')[0]} {photo_count}"
        
        # Update all measurement displays in the UI
        root.after(0, lambda: [
            # Update measurement values
            measured_height_var.set(f"{side_result['height_mm']:.1f} mm"),
            measured_dia_var.set(f"{top_result['diameter_mm']:.1f} mm"),
            
            # Update side camera distance if available
            side_cam_distance_result.set(
                f"{side_result['side_camera_distance_m'] * 1000:.1f} mm" 
                if side_result.get('side_camera_distance_m') is not None 
                else "N/A"
            ),
            
            # Update model code
            result_model_code_var.set(
                str(signal_handler._get_model_code(model_name)) 
                if signal_handler 
                else "Err"
            ),
            
            # Update pass/fail status
            result_status_var.set(
                "Within Tolerance" if signal_handler and 
                signal_handler._determine_pass_fail(
                    model_name, 
                    top_result['diameter_mm'], 
                    side_result['height_mm'], 
                    WHEEL_MODELS
                ) == 1 
                else "Out of Tolerance" if signal_handler 
                else "Error: Check config"
            ),
            
            # Update wheel counts
            update_wheel_counts(),
            
            # Final status update
            status_label_main.config(text="Processing complete")
        ])
        
        # Save to database (without image paths)
        try:
            add_inspection(
                part_no,
                model_name,
                top_result['diameter_mm'],
                side_result['height_mm'],
                None,  # No image path for top
                None   # No image path for side
            )
            print("[THREAD] Saved to database")
        except Exception as e:
            print(f"[THREAD] Database error: {e}")
        
        # Send to Modbus
        if signal_handler:
            try:
                signal_handler.send_measurement_data(
                    model_name,
                    top_result['diameter_mm'],
                    side_result['height_mm'],
                    WHEEL_MODELS
                )
                print("[THREAD] Sent to Modbus")
            except Exception as e:
                print(f"[THREAD] Modbus error: {e}")
        
    except Exception as e:
        print(f"[THREAD ERROR] {str(e)}")
        traceback.print_exc()
        root.after(0, lambda: status_label_main.config(text="Error: Processing failed"))
    finally:
        print("[THREAD] Measurement processing complete")
        
def take_photo1():
    """
    Capture and process frames from cameras
    
    This function ensures the correct processing order:
    1. Side view first - to measure wheel height using depth data
    2. Top view second - to calculate diameter using the measured height
    """
    global photo_count, frame_top, frame_side, signal_handler, measured_height_var, measured_dia_var
    
    if not streaming_active:
        # messagebox.showerror("Error", "No streams available to process. Start the streams first!")
        return
    
    # Initialize frame variables
    has_top_camera = frame_top is not None
    has_side_camera = frame_side is not None
    
    if not has_top_camera and not has_side_camera:
        # messagebox.showerror("Error", "No frames available to process. Start the streams first!")
        return
    
    # Track which camera scenario we're in
    camera_scenario = "both"  # Default: both cameras available
    
    # Use the available frame for both if only one camera is working
    if has_top_camera and not has_side_camera:
        frame_side = frame_top.copy()  # Use top camera for side view if only side is unavailable
        print("Using top camera frame for side view processing")
        camera_scenario = "top_only"
    elif has_side_camera and not has_top_camera:
        frame_top = frame_side.copy()  # Use side camera for top view if only top is unavailable
        print("Using side camera frame for top view processing")
        camera_scenario = "side_only"
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    photo_count += 1
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    # Initialize filename variables to None
    filename_side = None
    filename_top = None
    
    # Save frames based on camera scenario
    if camera_scenario == "both":
        # Save both frames when both cameras are available
        if frame_side is not None:
            filename_side = os.path.join(SAVE_DIR, f"side_view_{timestamp}.jpg")
            cv2.imwrite(filename_side, frame_side)
            print(f"Saved side view image to {filename_side}")
        
        if frame_top is not None:
            filename_top = os.path.join(SAVE_DIR, f"top_view_{timestamp}.jpg")
            cv2.imwrite(filename_top, frame_top)
            print(f"Saved top view image to {filename_top}")
    elif camera_scenario == "top_only":
        # Save top camera frame with both names
        filename_top = os.path.join(SAVE_DIR, f"top_view_{timestamp}.jpg")
        filename_side = os.path.join(SAVE_DIR, f"side_view_from_top_{timestamp}.jpg")
        cv2.imwrite(filename_top, frame_top)
        cv2.imwrite(filename_side, frame_top)
        print(f"Saved top camera frame as both views")
    elif camera_scenario == "side_only":
        # Save side camera frame with both names
        filename_side = os.path.join(SAVE_DIR, f"side_view_{timestamp}.jpg")
        filename_top = os.path.join(SAVE_DIR, f"top_view_from_side_{timestamp}.jpg")
        cv2.imwrite(filename_side, frame_side)
        cv2.imwrite(filename_top, frame_side)
        print(f"Saved side camera frame as both views")
    
    # Process side view first to get wheel height
    side_measurements = None
    top_measurements = None
    
    # Get aligned depth and color frames from RealSense camera if available
    depth_frame = None
    aligned_color_frame = None
    
    if realsense_camera is not None and realsense_camera.is_streaming:
        try:
            # Get the current aligned frames from the camera
            # This ensures depth and color frames are properly synchronized
            if hasattr(realsense_camera, 'aligned_frames') and realsense_camera.aligned_frames:
                aligned_depth_frame = realsense_camera.aligned_frames.get_depth_frame()
                aligned_color_frame_rs = realsense_camera.aligned_frames.get_color_frame()
                
                if aligned_depth_frame and aligned_color_frame_rs:
                    # Convert RealSense frames to numpy arrays
                    depth_frame = aligned_depth_frame
                    aligned_color_frame = np.asanyarray(aligned_color_frame_rs.get_data())
                    
                    # Update frame_side with the properly aligned color frame
                    frame_side = aligned_color_frame.copy()
                    print("Using aligned depth and color frames from RealSense")
                else:
                    print("Warning: Could not get aligned frames, using existing frame_side")
            else:
                # Fallback: try to get depth frame directly (less ideal)
                if hasattr(realsense_camera, 'get_depth_frame'):
                    depth_frame = realsense_camera.get_depth_frame()
                    print("Using separate depth frame (not guaranteed to be aligned)")
        except Exception as e:
            print(f"Error getting aligned frames: {e}")
            # Fallback to existing method
            if hasattr(realsense_camera, 'get_depth_frame'):
                depth_frame = realsense_camera.get_depth_frame()
    
    # Process side view first
    status_label_main.config(text="Processing: Measuring wheel height from side view...")
    if frame_side is not None:
        side_measurements = process_frame(
            frame_side,
            is_top_view=False,
            camera_settings=current_settings,
            wheel_models=WHEEL_MODELS,
            selected_model=current_model_display.get() if current_model_display else None,
            depth_frame=depth_frame
        )
        
        if side_measurements:
            # Update side view display with visualization
            update_display(side_panel, side_measurements['visualization'], SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            update_display(side_processed_panel, side_measurements['visualization'], SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")  # ✅ added
            print(f"Side view height: {side_measurements['height_mm']:.1f}mm")

            # Update side camera distance result
            side_dist_m = side_measurements.get('side_camera_distance_m')
            if side_dist_m is not None:
                side_cam_distance_result.set(f"{side_dist_m * 1000:.1f} mm")
            else:
                side_cam_distance_result.set("N/A")
        else:
            print("Failed to process side view")
            status_label_main.config(text="Error: Failed to process side view")
            return
    
    # Process top view using height from side view
    status_label_main.config(text="Processing: Measuring wheel diameter from top view...")
    if frame_top is not None and side_measurements:
        top_measurements = process_frame(
            frame_top,
            is_top_view=True,
            camera_settings=current_settings,
            wheel_models=WHEEL_MODELS,
            selected_model=current_model_display.get() if current_model_display else None,
            wheel_height_mm=side_measurements['height_mm']
        )
        
        if top_measurements:
            # Update top view display with visualization
            update_display(top_panel, top_measurements['visualization'], TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
            update_display(top_processed_panel, top_measurements['visualization'], TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")  # ✅ added
            print(f"Top view diameter: {top_measurements['diameter_mm']:.1f}mm")
        else:
            print("Failed to process top view")
            status_label_main.config(text="Error: Failed to process top view")
            return
    
    # If we have both measurements, update the UI and send data
    if side_measurements and top_measurements:
        # Update status
        status_label_main.config(text="Measurements complete")
        
        # Get current model name string
        model_name_str = current_model_display.get() if current_model_display else "Unknown"
        
        # Get numeric model code for results display
        if signal_handler: # Ensure signal_handler is initialized
            numeric_code = signal_handler._get_model_code(model_name_str)
            result_model_code_var.set(str(numeric_code))
        else:
            result_model_code_var.set("Err")
        
        # Update measurements display
        height_mm = side_measurements['height_mm']
        diameter_mm = top_measurements['diameter_mm']
        
        # Update measurement variables instead of non-existent labels
        measured_height_var.set(f"{height_mm:.1f} mm")
        measured_dia_var.set(f"{diameter_mm:.1f} mm")
        
        # Check against model specifications if available
        if WHEEL_MODELS and model_name_str in WHEEL_MODELS:
            model_spec = WHEEL_MODELS[model_name_str]
            model_height = model_spec.get('height_mm')
            model_diameter = model_spec.get('diameter_mm')
            
            if model_height and model_diameter:
                height_error = height_mm - model_height
                diameter_error = diameter_mm - model_diameter
                
                # Update status based on tolerances (since error labels don't exist)
                height_tolerance = model_spec.get('height_tolerance_mm', 5.0)
                diameter_tolerance = model_spec.get('diameter_tolerance_mm', 5.0)
                
                height_ok = abs(height_error) <= height_tolerance
                diameter_ok = abs(diameter_error) <= diameter_tolerance
                
                # Update status based on tolerances
                if height_ok and diameter_ok:
                    status_label_main.config(text=f"Measurements OK - Height: {height_error:+.1f}mm, Diameter: {diameter_error:+.1f}mm")
                else:
                    status_label_main.config(text=f"Measurements NOK - Height: {height_error:+.1f}mm, Diameter: {diameter_error:+.1f}mm")
        part_no = f"INDIP {timestamp.split('_')[0]} {photo_count}"
        # Determine pass/fail status
        if signal_handler and WHEEL_MODELS: # Ensure signal_handler is initialized
            pass_fail_status = signal_handler._determine_pass_fail(model_name_str, diameter_mm, height_mm, WHEEL_MODELS)
            tolerance_text = "Within Tolerance" if pass_fail_status == 1 else "Out of Tolerance"
            result_status_var.set(tolerance_text)
            # Optionally, set label color based on status
            # result_status_label.config(foreground=(PASS_COLOR if pass_fail_status == 1 else FAIL_COLOR))
        else:
            result_status_var.set("Error: Check config")

        # Save measurements to database
        try:
            add_inspection(
                part_no,         # Assumes 'part_no' is available in this scope
                model_name_str,  # Using 'model_name_str' for 'model_type'
                diameter_mm,
                height_mm,
                filename_top,    # Corresponds to image_path_top
                filename_side    # Corresponds to image_path_side
            )
            print("Measurements saved to database")
        except Exception as e:
            print(f"Error saving to database: {e}")
        
        # Send measurements via Modbus if signal handler is available
        if signal_handler is not None:
            try:
                signal_handler.send_measurement_data(model_name_str, diameter_mm, height_mm, WHEEL_MODELS)
                print("Measurements sent via Modbus")
            except Exception as e:
                print(f"Error sending measurements via Modbus: {e}")
        
        # Update wheel model counts
        update_wheel_counts()
    else:
        status_label_main.config(text="Error: Incomplete measurements")
        print("Failed to get complete measurements")

def start_streaming():
    """Start streaming from cameras"""
    global streaming_active, realsense_camera, top_camera, stop_streaming, signal_handler
    
    if streaming_active:
        status_label_main.config(text="Streaming already active")
        return
    
    stop_streaming = False
    streaming_active = True
    status_label_main.config(text="Initializing cameras...")
    
    # Update UI
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    # photo_button.config(state=tk.NORMAL)
    # auto_photo_button.config(state=tk.NORMAL)
    
    # Try to start RealSense camera first (for side view)
    if realsense_camera is None:
        try:
            realsense_camera = RealSenseCamera()
            side_camera = realsense_camera
            print("RealSense camera object created")
        except Exception as e:
            print(f"Error creating RealSense camera: {e}")
    
    if realsense_camera:
        try:
            realsense_camera.start()
            status_label_side.config(text="RealSense camera connected")
        except Exception as e:
            print(f"Error starting RealSense camera: {e}")
            status_label_side.config(text="RealSense camera failed to connect")
    
    # Start top camera (event camera or fallback)
    if top_camera is None:
        try:
            top_camera = IPCamera(TOP_CAMERA_URL)
            print("Top camera object created")
        except Exception as e:
            print(f"Error creating top camera: {e}")
    
    if top_camera:
        try:
            top_camera.start()
            status_label_top.config(text="Top camera connected")
        except Exception as e:
            print(f"Error starting top camera: {e}")
            status_label_top.config(text="Top camera failed to connect")
    
    # Start frame update threads
    threading.Thread(target=update_frames, daemon=True).start()
    
    # Note: Signal handler is now started at application launch, not here
    # This ensures modbus frames are detected even when streaming is not active
    
    status_label_main.config(text="Streaming started")
def update_frames():
    """Update frames from cameras"""
    global frame_top, frame_side, stop_streaming
    
    while not stop_streaming:
        # Track which cameras are available
        has_realsense = realsense_camera and realsense_camera.is_streaming
        has_top = top_camera and top_camera.is_streaming
        
        # Get frames from cameras
        if has_realsense:
            frame_side = realsense_camera.get_frame()
        
        if has_top:
            frame_top = top_camera.get_frame()
        
        # Update displays based on available cameras
        if has_realsense and has_top:
            # Both cameras available - normal display
            update_display(side_panel, frame_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            update_display(top_panel, frame_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
        elif has_realsense and not has_top:
            # Only side camera - display in both panels
            update_display(side_panel, frame_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            update_display(top_panel, frame_side, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
        elif has_top and not has_realsense:
            # Only top camera - display in both panels
            update_display(side_panel, frame_top, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            update_display(top_panel, frame_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
        
        time.sleep(0.03)  # ~30 FPS

def stop_streaming_func():
    """Stop streaming from cameras"""
    global streaming_active, stop_streaming, realsense_camera, top_camera, signal_handler
    
    stop_streaming = True
    streaming_active = False
    
    # Stop all cameras without destroying objects, but only if they're streaming
    if realsense_camera and hasattr(realsense_camera, 'is_streaming') and realsense_camera.is_streaming:
        try:
            realsense_camera.stop()
            print("RealSense camera streaming stopped")
        except Exception as e:
            print(f"Error stopping RealSense camera: {e}")
    
    if top_camera and hasattr(top_camera, 'is_streaming') and top_camera.is_streaming:
        try:
            top_camera.stop()
            print("Top camera streaming stopped")
        except Exception as e:
            print(f"Error stopping top camera: {e}")
    
    # Update UI
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    # photo_button.config(state=tk.DISABLED)
    # auto_photo_button.config(state=tk.DISABLED)
    
    status_label_main.config(text="Streaming stopped")
    
    # Note: We don't stop the signal_handler as it needs to run continuously
    # to detect 24V signals even when streaming is stopped

# def auto_capture():
#     """Toggle automatic photo capture"""
#     global auto_capture_active
    
#     if not streaming_active:
#         messagebox.showerror("Error", "Start streaming first!")
#         return
    
#     auto_capture_active = not auto_capture_active
    
#     if auto_capture_active:
#         # Start auto capture
#         auto_photo_button.config(text="Stop Auto Capture")
#         interval = current_settings.get("auto_capture_interval", 5)  # Default 5 seconds
#         threading.Thread(target=auto_capture_thread, args=(interval,), daemon=True).start()
#     else:
#         # Stop auto capture
#         auto_photo_button.config(text="Start Auto Capture")

def auto_capture_thread(interval):
    """Thread for automatic photo capture"""
    global auto_capture_active
    
    while auto_capture_active and streaming_active and not stop_streaming:
        # Call take_photo on main thread
        root.after(0, take_photo)
        
        # Wait specified interval
        for _ in range(interval * 10):  # Check every 100ms if we should stop
            if not auto_capture_active or not streaming_active or stop_streaming:
                break
            time.sleep(0.1)


def update_wheel_counts():
    """Update the wheel count display from database with counts by model"""
    global wheel_model_counts
    
    try:
        conn = sqlite3.connect('wheel_inspection.db')
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM inspections")
        total_count = cursor.fetchone()[0] or 0
        total_count_var.set(str(total_count))
        
        # Get counts by model
        cursor.execute("SELECT model_type, COUNT(*) FROM inspections GROUP BY model_type")
        model_counts = cursor.fetchall()
        
        # Update wheel model count variables
        wheel_model_counts = {}
        for model, count in model_counts:
            wheel_model_counts[model] = count
            
        # Update UI elements if they exist
        for model in WHEEL_MODELS.keys():
            if model in wheel_model_counts and model in model_count_vars:
                model_count_vars[model].set(str(wheel_model_counts.get(model, 0)))
            elif model in model_count_vars:
                model_count_vars[model].set('0')
    except Exception as e:
        print(f"Error updating wheel counts: {e}")
    finally:
        if conn:
            conn.close()

def update_model_parameters():
    """Update UI with current model parameters"""
    # Reload settings to ensure we have the latest saved values
    global current_settings, WHEEL_MODELS, current_model_display
    current_settings, WHEEL_MODELS = load_settings()
    
    model_name = current_settings.get("selected_model", "10-13")
    model_data = WHEEL_MODELS.get(model_name, {})
    
    # Update model info display
    model_value.set(model_name)
    
    # Update current model display in results frame
    if current_model_display is not None:
        current_model_display.set(model_name)
        print(f"Updated model display to: {model_name}")
    
    # Show diameter range
    min_dia = model_data.get("min_dia", 0)
    max_dia = model_data.get("max_dia", 0)
    diameter_value.set(f"{min_dia}-{max_dia} mm")
    
    # Show expected height
    height = model_data.get("height", 0)
    height_value.set(f"{height} mm")
    
    # # Show tolerance
    # tolerance = model_data.get("tolerance", 0)
    # tolerance_value.set(f"{tolerance} mm")
    
    # Update camera heights from settings, not measured values
    top_cam_height = current_settings["calibration"]["base_height"]
    side_cam_height = current_settings["calibration"]["side_camera_height"]
    top_cam_height_var.set(f"{top_cam_height:.1f} mm")
    side_cam_height_var.set(f"{side_cam_height:.1f} mm")

# Close app function removed - app should only close when manually closed
def detect_object():
    """Detect object in the image using image_processing.py"""
    global realsense_camera, side_panel, side_processed_panel

    if not realsense_camera:
        messagebox.showinfo("Camera Required", "RealSense camera not connected")
        return
    
    if not streaming_active:
        # Start streaming if not already active
        start_streaming()
        
        # Wait a moment for camera to initialize
        root.update()
        time.sleep(1)
        
    try:
        # Import necessary modules
        import utils
        
        # Get depth and color frames directly from the camera object
        depth_frame = realsense_camera.get_depth_frame()
        color_frame = realsense_camera.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("No frames available - camera may not be streaming")
            return
            
        # The frames are already aligned in the camera_streams.py implementation
        # so we don't need to create an additional alignment object here
        
        # Use the detect_object function from image_processing module
        from image_processing import detect_object as process_object
        processed_image, depth_image, measurements = process_object(color_frame, depth_frame)
        
        if processed_image is not None and depth_image is not None:
            # Update UI with the images
            # utils.update_display(side_panel, depth_image, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            utils.update_display(side_processed_panel, processed_image, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            
            # Update status label with measurements
            distance = measurements.get("distance", 0)
            depth = measurements.get("depth", 0)
            height = measurements.get("height", 0)
            obj_class = measurements.get("class", "unknown")
            
            status_text = f"Detected object: {obj_class} - Distance: {distance:.3f}m, Depth: {depth:.3f}m"
            if height > 0:
                status_text += f", Height: {height:.1f}cm"
            
            status_label_main.config(text=status_text)
            
            # Print measurements to console
            print(f"Object Detection Results:")
            print(f"  Class: {obj_class}")
            print(f"  Distance: {distance:.3f} m (3D)")
            print(f"  Depth: {depth:.3f} m (Z)")
            print(f"  Height: {height:.1f} cm")
        else:
            print("Failed to process object detection")
            

    except Exception as e:
        print(f"Error detecting object: {e}")
        messagebox.showerror("Error", f"Failed to detect object: {str(e)}")
        traceback.print_exc()


def on_closing():
    """Handle window close event"""
    global stop_streaming, signal_handler, realsense_camera, top_camera, streaming_active
    
    # Set flag to stop all threads
    stop_streaming = True
    
    # Only attempt to stop cameras if they're still streaming
    if streaming_active:
        # This will call stop_streaming_func which properly stops the cameras
        stop_streaming_func()
    else:
        # If we're not streaming, we might still need to clean up camera objects
        # But only try to stop them if they are initialized AND streaming
        if realsense_camera and hasattr(realsense_camera, 'is_streaming') and realsense_camera.is_streaming:
            try:
                realsense_camera.stop()
                print("RealSense camera stopped properly")
            except Exception as e:
                print(f"Error stopping RealSense camera: {e}")
        
        if top_camera and hasattr(top_camera, 'is_streaming') and top_camera.is_streaming:
            try:
                top_camera.stop()
                print("Top camera stopped properly")
            except Exception as e:
                print(f"Error stopping top camera: {e}")
    
    # Always ensure streaming_active is set to False
    streaming_active = False
    
    # Make sure signal handler is stopped
    if signal_handler:
        try:
            signal_handler.stop_detection()
            print("Signal handler stopped")
        except Exception as e:
            print(f"Error stopping signal handler: {e}")
    
    print("Application closed properly")
    root.destroy()

def open_settings_window():
    """Open the settings window"""
    global settings_win
    
    if settings_win is not None and settings_win.winfo_exists():
        settings_win.lift()  # Bring to front if exists
        settings_win.focus_force()
        return
        
    settings_win = show_settings_window(root, current_settings, WHEEL_MODELS, update_model_parameters)
    
    # Handle window close
    settings_win.protocol("WM_DELETE_WINDOW", lambda: on_settings_close(settings_win))

def on_settings_close(window):
    """Handle settings window close"""
    global settings_win
    window.destroy()
    settings_win = None

def open_report_window():
    """Open the reports window"""
    global reports_win
    
    if reports_win is not None and reports_win.winfo_exists():
        reports_win.lift()  # Bring to front if exists
        reports_win.focus_force()
        return
        
    reports_win = show_report_window(root)
    
    # Handle window close - only if the function returns a window object
    if reports_win is not None:
        reports_win.protocol("WM_DELETE_WINDOW", lambda: on_reports_close(reports_win))

def on_reports_close(window):
    """Handle reports window close"""
    global reports_win
    window.destroy()
    reports_win = None
   
def upload_image(is_top_view=True):
    """Upload and process an image file"""
    file_path = filedialog.askopenfilename(
        title="Select Image", 
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    
    if not file_path:
        return
    
    try:
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Error", "Failed to read image file.")
            return
        
        # Check for associated metadata file
        base_name = os.path.splitext(file_path)[0]
        meta_file = f"{base_name}_metadata.csv"
        global metadata_file_path
        if os.path.exists(meta_file):
            metadata_file_path = meta_file
            print(f"Using metadata from: {meta_file}")
        
        if is_top_view:
            global frame_top
            frame_top = frame.copy()
            display_frame = resize_with_aspect_ratio(frame, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)
            update_display(top_panel, frame, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
            
            result = process_frame(
                frame,
                is_top_view=True,
                camera_settings=current_settings,
                wheel_models=WHEEL_MODELS,
                selected_model=current_model_display.get() if current_model_display else None
            )
            
            if result:
                # Update top view display with visualization
                update_display(top_processed_panel, result['visualization'], TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
                
                # Update result display
                if 'diameter_mm' in result:
                    measured_dia_var.set(f"{result['diameter_mm']:.2f} mm")
                    result_status_var.set("OK" if result.get("is_ok", False) else "NOT OK")
                    update_result_frame()
        else:
            global frame_side
            frame_side = frame.copy()
            display_frame = resize_with_aspect_ratio(frame, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)
            update_display(side_panel, frame, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            
            # For uploaded side view images, we don't have depth frame, so pass None
            result = process_frame(
                frame,
                is_top_view=False,
                camera_settings=current_settings,
                wheel_models=WHEEL_MODELS,
                selected_model=current_model_display.get() if current_model_display else None,
                depth_frame=None
            )
            
            if result:
                # Update side view display with visualization
                update_display(side_processed_panel, result['visualization'], SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
                
                # Update result display
                if 'height_mm' in result:
                    measured_height_var.set(f"{result['height_mm']:.2f} mm")
                    
                    # Update side camera distance result if available
                    if 'depth_mm' in result:
                        side_cam_distance_result.set(f"{result['depth_mm']:.1f} mm")
        
        status_label_main.config(text=f"Processed uploaded {'top' if is_top_view else 'side'} view image: {os.path.basename(file_path)}")
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error processing image: {e}")
        print(traceback.format_exc())

def signal_handler_callback(signal_type=""):
    """Handle modbus frame with complete measurement cycle"""
    global streaming_active
    
    if signal_type == "MODBUS_FRAME":
        print("Processing modbus frame: Starting measurement cycle")
        
        # If already streaming, just take a photo
        if streaming_active:
            root.after(0, take_photo)
            return
        
        # Start the streaming to initialize cameras
        root.after(0, start_streaming)
        
        # Wait for cameras to initialize (2 second delay)
        def delayed_photo_and_stop():
            # Take photo to process
            take_photo()
            
            # Wait for processing to complete before stopping
            def delayed_stop():
                # Only stop streaming when done, don't close the app
                stop_streaming_func()
                print("Measurement cycle completed - waiting for next modbus frame")
            
            # Give 1 second for processing to complete
            root.after(1000, delayed_stop)
        
        # Wait 2 seconds for cameras to initialize
        root.after(2000, delayed_photo_and_stop)

def update_count_frame():
    """Update the count frame with latest daily and monthly counts"""
    global count_frame
    
    if count_frame is None:
        print("Count frame not initialized")
        return
        
    # Get today's date and current month
    today = datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.now().strftime("%B")  # Full month name (e.g., "May")
    
    # Clear existing widgets in count frame
    for widget in count_frame.winfo_children():
        widget.destroy()
    
    # Configure grid for two columns
    count_frame.columnconfigure(0, weight=1)  # Daily report column
    count_frame.columnconfigure(1, weight=0)  # Separator column
    count_frame.columnconfigure(2, weight=1)  # Monthly report column
    
    # Create frames for daily and monthly reports
    daily_frame = ttk.Frame(count_frame)
    daily_frame.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=5)
    
    monthly_frame = ttk.Frame(count_frame)
    monthly_frame.grid(row=0, column=2, sticky="nsew", padx=(0, 5), pady=5)
    
    # Add vertical separator
    separator = ttk.Separator(count_frame, orient='vertical')
    separator.grid(row=0, column=1, sticky='ns', padx=5, pady=5)
    
    # Get today's counts
    today_total, today_model_counts, _ = get_daily_report(today)
    ttk.Label(daily_frame, text="Today's Count:", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, sticky="w", pady=5)
    ttk.Label(daily_frame, text=str(today_total), font=('Helvetica', 12, 'bold')).grid(row=0, column=1, sticky="w", pady=5)
    
    # Display today's model counts
    row = 1
    for model, count in today_model_counts:
        ttk.Label(daily_frame, text=f"{model}:", font=('Helvetica', 11)).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(daily_frame, text=str(count), font=('Helvetica', 11)).grid(row=row, column=1, sticky="w", pady=2)
        row += 1
    
    # Get current month's counts
    current_month_date = datetime.now().strftime("%Y-%m")
    month_total, month_model_counts = get_monthly_report(current_month_date[:4], current_month_date[5:7])
    
    # Display current month's total
    ttk.Label(monthly_frame, text=f"{current_month} Month's Count:", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, sticky="w", pady=5)
    ttk.Label(monthly_frame, text=str(month_total), font=('Helvetica', 12, 'bold')).grid(row=0, column=1, sticky="w", pady=5)
    
    # Display current month's model counts
    row = 1
    for model, count in month_model_counts:
        ttk.Label(monthly_frame, text=f"{model}:", font=('Helvetica', 11)).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(monthly_frame, text=str(count), font=('Helvetica', 11)).grid(row=row, column=1, sticky="w", pady=2)
        row += 1



# Main function to create UI and start application
def main():
    global root, side_panel, top_panel, side_processed_panel, top_processed_panel
    global start_button, stop_button, photo_button, auto_photo_button
    global status_label_main, status_label_side, status_label_top
    global model_value, diameter_value, height_value, tolerance_value
    global top_cam_height_var, side_cam_height_var, side_cam_distance_result
    global measured_dia_var, measured_height_var, result_status_var, result_model_code_var
    global top_result_text, side_result_text
    global total_count_var, passed_count_var, faulty_count_var
    global result_status_label, count_frame  # Add count_frame to global declaration
    
    # Initialize global variables
    init_globals()
    
    # Initialize database
    init_db()
    
    # Create main window
    root = tk.Tk()
    root.title("Wheel Inspection System")
    root.geometry("1400x900")
    root.configure(background=BG_COLOR)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Set custom application icon
    set_app_icon(root)
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TFrame', background=BG_COLOR)
    style.configure('TLabelframe', background=BG_COLOR, foreground=TEXT_COLOR)
    style.configure('TLabelframe.Label', background=BG_COLOR, foreground=TEXT_COLOR)
    style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR)
    style.configure('TButton', background=BUTTON_COLOR, foreground="white", font=('Helvetica', 12, 'bold'))
    style.map('TButton', background=[('active', HIGHLIGHT_COLOR)])
    
    # Initialize Tkinter variables
    height_adjustment_var = tk.StringVar(value=str(current_settings["calibration"]["base_height"]))
    model_value = tk.StringVar(value=current_settings["selected_model"])
    diameter_value = tk.StringVar(value="0.0 inch")
    height_value = tk.StringVar(value="0.0 mm")
    measured_dia_var = tk.StringVar(value="0.0 mm")
    measured_height_var = tk.StringVar(value="0.0 mm")
    
    # Initialize global current_model_display for result frame
    global current_model_display
    current_model_display = tk.StringVar(value=current_settings["selected_model"])
    result_status_var = tk.StringVar(value="N/A") # For Within Tolerance / Out of Tolerance
    result_model_code_var = tk.StringVar(value="N/A") # For numeric model code in results
    
    # Camera height variables
    top_cam_height_var = tk.StringVar(value=f"{current_settings['calibration']['base_height']:.1f} mm")
    side_cam_height_var = tk.StringVar(value=f"{current_settings['calibration']['side_camera_height']:.1f} mm")
    side_cam_distance_result = tk.StringVar(value="0.0 mm")  # For measured values
    
    # Initialize database counts
    conn = sqlite3.connect('wheel_inspection.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM inspections")
    total_count = cursor.fetchone()[0] or 0
    
    # Get counts by model
    cursor.execute("SELECT model_type, COUNT(*) FROM inspections GROUP BY model_type")
    model_counts = cursor.fetchall() or []
    conn.close()
    
    # Initialize global count variables
    total_count_var = tk.StringVar(value=str(total_count))
    
    # Create model count variables
    global model_count_vars, wheel_model_counts
    wheel_model_counts = {model: count for model, count in model_counts}
    
    # Initialize a StringVar for each model
    for model in WHEEL_MODELS.keys():
        model_count_vars[model] = tk.StringVar(value=str(wheel_model_counts.get(model, 0)))
    
    # Main layout
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.columnconfigure(2, weight=1)
    main_frame.rowconfigure(0, weight=0)  # Header row
    main_frame.rowconfigure(1, weight=0)  # Button row
    main_frame.rowconfigure(2, weight=0)  # Status row
    main_frame.rowconfigure(3, weight=0)  # Info panels row
    main_frame.rowconfigure(4, weight=1)  # Camera frames row - expands
    main_frame.rowconfigure(5, weight=0)
    # Header with logo and app title
    header_frame = ttk.Frame(main_frame)
    header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    header_frame.columnconfigure(0, weight=0)  # Logo
    header_frame.columnconfigure(1, weight=2)  # App title
    header_frame.columnconfigure(2, weight=0)  # Clock
    
    # Try to load logo
    try:
        logo_path = os.path.join(os.path.dirname(__file__), "Taurus_logo1.png")
        logo_img = Image.open(logo_path)
        logo_img = logo_img.resize((100, 100), Image.Resampling.LANCZOS)
        logo_imgtk = ImageTk.PhotoImage(image=logo_img)
        
        logo_label = ttk.Label(header_frame, image=logo_imgtk, background=BG_COLOR)
        logo_label.image = logo_imgtk
        logo_label.grid(row=0, column=0, sticky="w", padx=10)
    except Exception as e:
        print(f"Error loading logo: {e}")
        ttk.Label(header_frame, text="LOGO", font=('Helvetica', 14, 'bold')).grid(row=0, column=0, sticky="w", padx=10)
    
    # App title
    app_title_label = ttk.Label(
        header_frame,
        text="WHEEL DIMENSION ANALYZER",
        font=("Arial", 26, "bold"),
        anchor="center",
        foreground="blue"
    )
    app_title_label.grid(row=0, column=1, sticky="ew")
    
    # Clock
    clock_label = ttk.Label(
        header_frame,
        font=('Helvetica', 22, 'bold'),
        foreground='black'
    )
    clock_label.grid(row=0, column=2, sticky="e", padx=10)
    update_clock(clock_label)  # Start clock update
    
    # Buttons row
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    for i in range(12):  # Increased to accommodate more buttons
        button_frame.columnconfigure(i, weight=1)
    
    start_button = ttk.Button(button_frame, text="Start Streaming", command=start_streaming)
    start_button.grid(row=0, column=0, padx=3, pady=5)
    
    stop_button = ttk.Button(button_frame, text="Stop Streaming", command=stop_streaming_func, state=tk.DISABLED)
    stop_button.grid(row=0, column=2, padx=3, pady=5)
    
    # photo_button = ttk.Button(button_frame, text="Take Photo", command=take_photo)
    # photo_button.grid(row=0, column=2, padx=3, pady=5)
    
    # auto_photo_button = ttk.Button(button_frame, text="Start Auto Capture", command=auto_capture, state=tk.DISABLED)
    # auto_photo_button.grid(row=0, column=3, padx=3, pady=5)

    # detectObject_button = ttk.Button(button_frame, text="Detect Object", command=detect_object)
    # detectObject_button.grid(row=0, column=4, padx=3, pady=5)

    # upload_top_button = ttk.Button(button_frame, text="Upload Top View", command=lambda: upload_image(is_top_view=True))
    # upload_top_button.grid(row=0, column=4, padx=3, pady=5)
    
    # upload_side_button = ttk.Button(button_frame, text="Upload Side View", command=lambda: upload_image(is_top_view=False))
    # upload_side_button.grid(row=0, column=5, padx=3, pady=5)
    
    # Test signal handler functions (for debugging)
    def test_signal_handler():
        """Test function to simulate modbus signal"""
        if signal_handler:
            print("Testing signal handler...")
            signal_handler_callback("MODBUS_FRAME")
        else:
            print("Signal handler not initialized")

    def test_modbus_frame():
        """Test the enhanced Modbus frame with pass/fail results"""
        if signal_handler:
            print("Testing enhanced Modbus frame transmission...")
            # Test with sample measurements
            test_model = "13-16"
            test_diameter = 380.5  # mm (about 15 inches - should PASS for 13-16)
            test_height = 76.2     # mm (should PASS with 3mm tolerance)
            
            print(f"Test measurements: Model={test_model}, Diameter={test_diameter}mm, Height={test_height}mm")
            
            success = signal_handler.send_measurement_data(
                test_model, test_diameter, test_height, WHEEL_MODELS
            )
            
            if success:
                print("Test Modbus frame sent successfully!")
                print("Frame includes: diameter*100, height*100, model info, and PASS/FAIL result")
            else:
                print("Failed to send test Modbus frame")
        else:
            print("Signal handler not initialized")

    # Test buttons
    # test_signal_button = ttk.Button(button_frame, text="Test Signal", command=test_signal_handler)
    # test_signal_button.grid(row=0, column=6, padx=3, pady=5)
    
    # test_modbus_button = ttk.Button(button_frame, text="Test Modbus", command=test_modbus_frame)
    # test_modbus_button.grid(row=0, column=7, padx=3, pady=5)
    
    report_button = ttk.Button(button_frame, text="Generate Report", command=open_report_window)
    report_button.grid(row=0, column=4, padx=3, pady=5)
    
    settings_button = ttk.Button(button_frame, text="Settings", command=open_settings_window)
    settings_button.grid(row=0, column=6, padx=3, pady=5)
    
    # Status row
    status_frame = ttk.Frame(main_frame)
    status_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    status_frame.columnconfigure(0, weight=1)
    status_frame.columnconfigure(1, weight=1)
    
    status_label = ttk.Label(status_frame, text="Ready. Click 'Start Streaming' or 'Upload Image' to begin.", 
                            font=("Arial", 14, "italic"))
    status_label.grid(row=0, column=0, sticky="w", padx=5)
    
    status_label_main = ttk.Label(status_frame, text="Status: Ready", 
                                 font=("Arial", 14, "italic"))
    status_label_main.grid(row=0, column=1, sticky="w", padx=5)
    
    # Information panels (Model Data, Result, Wheel Count)
    info_frame = ttk.Frame(main_frame)
    info_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    info_frame.columnconfigure(0, weight=1)
    info_frame.columnconfigure(1, weight=1)
    info_frame.columnconfigure(2, weight=1)
    
    # Configure panel styles
    style.configure("Info.TLabelframe", background=BG_COLOR)
    style.configure("Info.TLabelframe.Label", font=('Helvetica', 13, 'bold'), background=BG_COLOR)
    
    # Model Data panel
    model_frame = ttk.LabelFrame(info_frame, text="Model Data", style="Info.TLabelframe")
    model_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=5)
    
    ttk.Label(model_frame, text="Model:", font=('Helvetica', 12)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(model_frame, textvariable=model_value, font=('Helvetica', 12)).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(model_frame, text="Diameter:", font=('Helvetica', 12)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(model_frame, textvariable=diameter_value, font=('Helvetica', 12)).grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(model_frame, text="Height:", font=('Helvetica', 12)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(model_frame, textvariable=height_value, font=('Helvetica', 12)).grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    # Tolerance references removed as requested
    
    ttk.Label(model_frame, text="Top Camera Height:", font=('Helvetica', 12)).grid(row=4, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(model_frame, textvariable=top_cam_height_var, font=('Helvetica', 12)).grid(row=4, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(model_frame, text="Side Camera Distance:", font=('Helvetica', 12)).grid(row=5, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(model_frame, textvariable=side_cam_height_var, font=('Helvetica', 12)).grid(row=5, column=1, sticky="w", padx=5, pady=5)
    
    # Result panel
    result_frame = ttk.LabelFrame(info_frame, text="Measurement Results", style="Info.TLabelframe")
    result_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    
    # Add model name at the top of results
    ttk.Label(result_frame, text="Model:", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, sticky="w", padx=5, pady=5)
    # Use the new result_model_code_var for numeric model code
    ttk.Label(result_frame, textvariable=result_model_code_var, font=('Helvetica', 12)).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
    # Display diameter measurement
    ttk.Label(result_frame, text="Measured Diameter:", font=('Helvetica', 12)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=measured_dia_var, font=('Helvetica', 12)).grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    # Display height measurement
    ttk.Label(result_frame, text="Measured Height:", font=('Helvetica', 12)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=measured_height_var, font=('Helvetica', 12)).grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(result_frame, text="Side Camera Distance:", font=('Helvetica', 12)).grid(row=4, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=side_cam_distance_result, font=('Helvetica', 12)).grid(row=4, column=1, sticky="w", padx=5, pady=5)
    
    # Result: Within Tolerance / Out of Tolerance
    ttk.Label(result_frame, text="Result:", font=('Helvetica', 12, 'bold')).grid(row=5, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=result_status_var, font=('Helvetica', 12, 'bold')).grid(row=5, column=1, sticky="w", padx=5, pady=5)
    
    # Result status references removed
    
    # Wheel count panel - updated to show model counts
    count_frame = ttk.LabelFrame(info_frame, text="Wheel Count by Model", style="Info.TLabelframe")
    count_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=5)
    
    # Initial update of count frame
    update_count_frame()
    
    # Camera views
    camera_frame = ttk.Frame(main_frame)
    camera_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
    camera_frame.columnconfigure(0, weight=1)
    camera_frame.columnconfigure(1, weight=1)
    camera_frame.rowconfigure(0, weight=1)

    # Side view
    side_frame = ttk.LabelFrame(camera_frame, text="Side View", style="Info.TLabelframe")
    side_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    side_frame.columnconfigure(0, weight=1)
    side_frame.rowconfigure(0, weight=1)
    side_frame.rowconfigure(1, weight=0)

    side_container = ttk.Frame(side_frame)
    side_container.grid(row=0, column=0, sticky="nsew")
    side_container.columnconfigure(0, weight=1)
    side_container.columnconfigure(1, weight=1)
    side_container.rowconfigure(0, weight=1)

    side_original = ttk.LabelFrame(side_container, text="Original")
    side_original.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
    
    # Create a fixed size frame for side panel
    side_panel_frame = ttk.Frame(side_original, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)
    side_panel_frame.pack(padx=5, pady=5)
    side_panel_frame.pack_propagate(False)  # Prevent resizing
    
    # Create label inside the fixed frame
    side_panel = ttk.Label(side_panel_frame)
    side_panel.place(x=-80, y=0, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)

    side_processed_frame = ttk.LabelFrame(side_container, text="Processed")
    side_processed_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    
    # Create a fixed size frame for side processed panel
    side_processed_panel_frame = ttk.Frame(side_processed_frame, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)
    side_processed_panel_frame.pack(padx=5, pady=5)
    side_processed_panel_frame.pack_propagate(False)  # Prevent resizing
    
    # Create label inside the fixed frame
    side_processed_panel = ttk.Label(side_processed_panel_frame)
    side_processed_panel.place(x=-80, y=0, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)

    status_label_side = ttk.Label(side_frame, text="Side camera not connected", font=("Arial", 11, "bold"))
    status_label_side.grid(row=1, column=0, sticky="w", pady=(0, 5))

    # Top view
    top_frame = ttk.LabelFrame(camera_frame, text="Top View", style="Info.TLabelframe")
    top_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    top_frame.columnconfigure(0, weight=1)
    top_frame.rowconfigure(0, weight=1)
    top_frame.rowconfigure(1, weight=0)

    top_container = ttk.Frame(top_frame)
    top_container.grid(row=0, column=0, sticky="nsew")
    top_container.columnconfigure(0, weight=1)
    top_container.columnconfigure(1, weight=1)
    top_container.rowconfigure(0, weight=1)

    top_original = ttk.LabelFrame(top_container, text="Original")
    top_original.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
    
    # Create a fixed size frame for top panel
    top_panel_frame = ttk.Frame(top_original, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)
    top_panel_frame.pack(padx=5, pady=5)
    top_panel_frame.pack_propagate(False)  # Prevent resizing
    
    # Create label inside the fixed frame
    top_panel = ttk.Label(top_panel_frame)
    top_panel.place(x=-100, y=0, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)

    top_processed_frame = ttk.LabelFrame(top_container, text="Processed")
    top_processed_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    
    # Create a fixed size frame for top processed panel
    top_processed_panel_frame = ttk.Frame(top_processed_frame, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)
    top_processed_panel_frame.pack(padx=5, pady=5)
    top_processed_panel_frame.pack_propagate(False)  # Prevent resizing
    
    # Create label inside the fixed frame
    top_processed_panel = ttk.Label(top_processed_panel_frame)
    top_processed_panel.place(x=-100, y=0, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)

    status_label_top = ttk.Label(top_frame, text="Top camera not connected", font=("Arial", 11, "bold"))
    status_label_top.grid(row=1, column=0, sticky="w", pady=(0, 5))
    
    company_frame = ttk.Frame(main_frame)
    company_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

    # Configure columns for alignment
    company_frame.columnconfigure(0, weight=0) # Copyright label on left
    company_frame.columnconfigure(1, weight=1) # Spacer column
    company_frame.columnconfigure(2, weight=0) # ACE Micromatics label on right

    copyright_label = ttk.Label(company_frame, text="Copyright @2025 by Taurus", font=("Arial", 10), foreground="gray60")
    copyright_label.grid(row=0, column=0, sticky="w", padx=5, pady=(0, 5))

    company_label = ttk.Label(company_frame, text="ACE MICROMATICS", font=("Arial", 11, "bold"), foreground="gray60")
    company_label.grid(row=0, column=2, sticky="e", padx=5, pady=(0, 5))

    # Update model parameters
    update_model_parameters()
   
    # Start signal handler immediately when application launches
    # This ensures modbus frames are detected even before streaming is started
    global signal_handler
    signal_handler = SignalHandler(signal_callback=signal_handler_callback)
    signal_handler.start_detection()
    print("Modbus frame detection started - waiting for signals...")
    
    # Start loading the AI model in the background after UI is shown
    # This prevents blocking the UI during startup
    def delayed_model_loading():
        # Wait 2 seconds to ensure UI is responsive first
        time.sleep(2)
        print("Starting AI model loading in background...")
        # This will initiate the loading process in a background thread
        load_model_in_background()
    
    # Start model loading in a separate thread to avoid blocking UI
    threading.Thread(target=delayed_model_loading, daemon=True).start()
    
    # Start the mainloop
    root.mainloop()

# Entry point
if __name__ == "__main__":
    main()
