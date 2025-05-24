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
from utils import update_display, update_panel_image, update_clock
from camera_streams import RealSenseCamera, IPCamera, USBCamera, current_depth_image, frame_queue
from image_processing import process_frame, load_model_in_background
from wheel_measurements import classify_wheel_model
from settings_manager import load_settings, save_settings, load_realsense_calibration
from signal_handler import SignalHandler
from database import init_db, add_inspection
from reports_window import show_report_window
from settings_window import show_settings_window
from app_icon import set_app_icon

# Set the environment variable to avoid OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
SAVE_DIR = "captured_frames"
SIDE_PANEL_WIDTH = 640
SIDE_PANEL_HEIGHT = 480
TOP_PANEL_WIDTH = 640
TOP_PANEL_HEIGHT = 480

# UI Color Scheme
BG_COLOR = "#AFE1AF"
TEXT_COLOR = "#333333"
BUTTON_COLOR = "#023020"
HIGHLIGHT_COLOR = "#5c8cd5"
PASS_COLOR = "#4caf50"
FAIL_COLOR = "#f44336"
REALSENSE_URL = "realsense://"
TOP_CAMERA_URL = "tcp://192.168.100.50:8080"  # Adjust based on your camera

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

# Panel size constants
SIDE_PANEL_WIDTH = 640
SIDE_PANEL_HEIGHT = 480
TOP_PANEL_WIDTH = 640
TOP_PANEL_HEIGHT = 480

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

def take_photo():
    """
    Capture and process frames from cameras
    
    This function ensures the correct processing order:
    1. Side view first - to measure wheel height using depth data
    2. Top view second - to calculate diameter using the measured height
    """
    global photo_count, frame_top, frame_side
    
    if not streaming_active:
        messagebox.showerror("Error", "No streams available to process. Start the streams first!")
        return
    
    # Initialize frame variables
    has_top_camera = frame_top is not None
    has_side_camera = frame_side is not None
    
    if not has_top_camera and not has_side_camera:
        messagebox.showerror("Error", "No frames available to process. Start the streams first!")
        return
    
    # Track which camera scenario we're in
    camera_scenario = "both"  # Default: both cameras available
    
    # NEW CODE: Use the available frame for both if only one camera is working
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    side_measured_height = None
    
    # Process side view first to get wheel height
    measurements_side = {}
    status_label_main.config(text="Processing: Measuring wheel height from side view...")
    processed_side, measurements_side = process_frame(frame_side, is_top_view=False, 
                                                     camera_settings=current_settings["calibration"],
                                                     wheel_models=WHEEL_MODELS,
                                                     selected_model=current_settings["selected_model"])
    
    # Display processed side view
    if camera_scenario == "both" or camera_scenario == "side_only":
        # Display in side processed panel
        update_display(side_processed_panel, processed_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
        
        # If only side camera, also display in top processed panel
        if camera_scenario == "side_only":
            update_display(top_processed_panel, processed_side, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
            status_label_top.config(text="Using side camera image")
    
    if measurements_side.get("type") == "Side":
        # Extract height and depth information
        side_measured_height = measurements_side.get("height_mm", None)
        camera_to_wheel_distance = measurements_side.get("depth_mm", None)
        
        if side_measured_height is not None:
            measured_height_var.set(f"{side_measured_height:.1f} mm")
            current_settings["calibration"]["wheel_height"] = side_measured_height
            print(f"Updated wheel height for diameter calculation: {side_measured_height:.2f} mm")
        
        if camera_to_wheel_distance is not None:
            side_cam_distance_result.set(f"{camera_to_wheel_distance:.1f} mm")

    # Process top view using the height information from side view
    measurements_top = {}
    status_label_main.config(text="Processing: Calculating wheel diameter using height data...")
    processed_top, measurements_top = process_frame(frame_top, is_top_view=True,
                                                   camera_settings=current_settings["calibration"],
                                                   wheel_models=WHEEL_MODELS,
                                                   selected_model=current_settings["selected_model"],
                                                   wheel_height_mm=side_measured_height)
    
    # Display processed top view
    if camera_scenario == "both" or camera_scenario == "top_only":
        # Display in top processed panel
        update_display(top_processed_panel, processed_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
        
        # If only top camera, also display in side processed panel
        if camera_scenario == "top_only":
            update_display(side_processed_panel, processed_top, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            status_label_side.config(text="Using top camera image")
    
    # Update status based on camera scenario
    if camera_scenario == "both":
        status_label_main.config(text="Processing complete: Both height and diameter measured.")
    elif camera_scenario == "side_only":
        status_label_main.config(text="Processing complete: Using side camera for both views.")
        # Clear the "No Top Camera" message if it exists
        status_label_top.config(text="Using side camera image for top view")
    elif camera_scenario == "top_only":
        status_label_main.config(text="Processing complete: Using top camera for both views.")
        # Clear the "No Side Camera" message if it exists
        status_label_side.config(text="Using top camera image for side view")
    
    # Get measurements and update display
    if measurements_top.get("type") == "Round":
        diameter_mm = measurements_top.get("diameter_mm", 0)
        measured_dia_var.set(f"{diameter_mm:.1f} mm")
    else:
        measured_dia_var.set("No data")
        diameter_mm = 0

    # Update height measurement display
    if measurements_side.get("type") == "Side":
        height_mm = measurements_side.get("height_mm", 0)
        measured_height_var.set(f"{height_mm:.1f} mm")
    else:
        measured_height_var.set("No data")
        height_mm = 0
        
    # Update the current model display
    global current_model_display
    model_type = current_settings["selected_model"]
    if current_model_display:
        current_model_display.set(model_type)
        
    # No need for status updates since we're not checking tolerances anymore
    
    # Save to database
    part_no = f"INDIP {timestamp.split('_')[0]} {photo_count}"
    model_type = current_settings["selected_model"]
    diameter_mm = measurements_top.get("diameter_mm", 0)
    height_mm = measurements_side.get("height_mm", 0)
    
    # Update database with simplified parameters (no test_result or thickness_mm)
    add_inspection(
        part_no, 
        model_type, 
        diameter_mm, 
        height_mm, 
        filename_top if filename_top else '', 
        filename_side if filename_side else ''
    )
    
    # Update wheel counts
    update_wheel_counts()
    
    # Send measurement data via modbus frame after processing
    if signal_handler:
        try:
            signal_handler.send_measurement_data(model_type, diameter_mm, height_mm)
            status_text = f"Captured and processed frames {photo_count} at {timestamp}. Measurement data sent."
        except Exception as e:
            print(f"Error sending measurement data: {e}")
            status_text = f"Captured and processed frames {photo_count} at {timestamp}. Failed to send data."
    else:
        status_text = f"Captured and processed frames {photo_count} at {timestamp}. Signal handler not available."
    
    status_label_main.config(text=status_text)

# Function removed as we no longer track passing/failing status

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
            update_display(side_panel, frame_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            update_display(top_panel, frame_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
        elif has_realsense and not has_top:
            # Only side camera - display in both panels
            update_display(side_panel, frame_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            update_display(top_panel, frame_side, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
        elif has_top and not has_realsense:
            # Only top camera - display in both panels
            update_display(side_panel, frame_top, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            update_display(top_panel, frame_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
        
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
            utils.update_display(side_processed_panel, processed_image, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            
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

# Main function to create UI and start application
def main():
    global root, side_panel, top_panel, side_processed_panel, top_processed_panel
    global start_button, stop_button, photo_button, auto_photo_button
    global status_label_main, status_label_side, status_label_top
    global model_value, diameter_value, height_value, tolerance_value
    global top_cam_height_var, side_cam_height_var, side_cam_distance_result
    global measured_dia_var, measured_height_var, result_status_var
    global top_result_text, side_result_text
    global total_count_var, passed_count_var, faulty_count_var
    global result_status_label
    
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
    diameter_value = tk.StringVar(value="0.0 mm")
    height_value = tk.StringVar(value="0.0 mm")
    measured_dia_var = tk.StringVar(value="0.0 mm")
    measured_height_var = tk.StringVar(value="0.0 mm")
    
    # Initialize global current_model_display for result frame
    global current_model_display
    current_model_display = tk.StringVar(value=current_settings["selected_model"])
    
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
    for i in range(11):
        button_frame.columnconfigure(i, weight=1)
    
    start_button = ttk.Button(button_frame, text="Start Streaming", command=start_streaming)
    start_button.grid(row=0, column=0, padx=3, pady=5)
    
    stop_button = ttk.Button(button_frame, text="Stop Streaming", command=stop_streaming_func, state=tk.DISABLED)
    stop_button.grid(row=0, column=1, padx=3, pady=5)
    
    # photo_button = ttk.Button(button_frame, text="Take Photo", command=take_photo, state=tk.DISABLED)
    # photo_button.grid(row=0, column=2, padx=3, pady=5)
    
    # auto_photo_button = ttk.Button(button_frame, text="Start Auto Capture", command=auto_capture, state=tk.DISABLED)
    # auto_photo_button.grid(row=0, column=3, padx=3, pady=5)

    # detectObject_button = ttk.Button(button_frame, text="Detect Object", command=detect_object)
    # detectObject_button.grid(row=0, column=4, padx=3, pady=5)

    # upload_top_button = ttk.Button(button_frame, text="Upload Top View", command=lambda: upload_image(is_top_view=True))
    # upload_top_button.grid(row=0, column=4, padx=3, pady=5)
    
    # upload_side_button = ttk.Button(button_frame, text="Upload Side View", command=lambda: upload_image(is_top_view=False))
    # upload_side_button.grid(row=0, column=5, padx=3, pady=5)
    
    report_button = ttk.Button(button_frame, text="Generate Report", command=open_report_window)
    report_button.grid(row=0, column=5, padx=3, pady=5)
    
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
    # Use the global current_model_display that was initialized earlier
    ttk.Label(result_frame, textvariable=current_model_display, font=('Helvetica', 12)).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
    # Display diameter measurement
    ttk.Label(result_frame, text="Measured Diameter:", font=('Helvetica', 12)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=measured_dia_var, font=('Helvetica', 12)).grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    # Display height measurement
    ttk.Label(result_frame, text="Measured Height:", font=('Helvetica', 12)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=measured_height_var, font=('Helvetica', 12)).grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(result_frame, text="Side Camera Distance:", font=('Helvetica', 12)).grid(row=4, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=side_cam_distance_result, font=('Helvetica', 12)).grid(row=4, column=1, sticky="w", padx=5, pady=5)
    
    # Result status references removed
    
    # Wheel count panel - updated to show model counts
    count_frame = ttk.LabelFrame(info_frame, text="Wheel Count by Model", style="Info.TLabelframe")
    count_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=5)
    
    ttk.Label(count_frame, text="Total:", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(count_frame, textvariable=total_count_var, font=('Helvetica', 12, 'bold')).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
    # Add each model as a row in the count frame
    row = 1
    for model_name in sorted(WHEEL_MODELS.keys()):
        ttk.Label(count_frame, text=f"{model_name}:", font=('Helvetica', 11)).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(count_frame, textvariable=model_count_vars[model_name], font=('Helvetica', 11)).grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
    
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
    # side_panel_frame.pack_propagate(False)  # Prevent resizing
    
    # Create label inside the fixed frame
    side_panel = ttk.Label(side_panel_frame)
    side_panel.place(x=0, y=0, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)

    side_processed_frame = ttk.LabelFrame(side_container, text="Processed")
    side_processed_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    
    # Create a fixed size frame for side processed panel
    side_processed_panel_frame = ttk.Frame(side_processed_frame, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)
    side_processed_panel_frame.pack(padx=5, pady=5)
    # side_processed_panel_frame.pack_propagate(False)  # Prevent resizing
    
    # Create label inside the fixed frame
    side_processed_panel = ttk.Label(side_processed_panel_frame)
    side_processed_panel.place(x=0, y=0, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)

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
    top_panel.place(x=0, y=0, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)

    top_processed_frame = ttk.LabelFrame(top_container, text="Processed")
    top_processed_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    
    # Create a fixed size frame for top processed panel
    top_processed_panel_frame = ttk.Frame(top_processed_frame, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)
    top_processed_panel_frame.pack(padx=5, pady=5)
    top_processed_panel_frame.pack_propagate(False)  # Prevent resizing
    
    # Create label inside the fixed frame
    top_processed_panel = ttk.Label(top_processed_panel_frame)
    top_processed_panel.place(x=0, y=0, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)

    status_label_top = ttk.Label(top_frame, text="Top camera not connected", font=("Arial", 11, "bold"))
    status_label_top.grid(row=1, column=0, sticky="w", pady=(0, 5))
    
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
