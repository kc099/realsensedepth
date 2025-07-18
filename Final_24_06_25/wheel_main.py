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
import gc  # Add garbage collection import
import concurrent.futures  # Add for parallel processing
from PIL import Image, ImageTk  # Still needed for UI image handling
import pyrealsense2 as rs

# Thread pool for measurement processing (reuse threads for better performance)
measurement_thread_pool = None

# Import from modular files
from utils import update_display, update_panel_image, update_clock, resize_with_aspect_ratio
from camera_streams import RealSenseCamera, IPCamera, frame_queue
import image_processing
from signal_handler import SignalHandler
from database import init_db, add_inspection, get_daily_report, get_monthly_report, get_date_range_report
from reports_window import show_report_window
from settings_window import show_settings_window
from app_icon import set_app_icon
import torch

# Set the environment variable to avoid OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Debug mode flag - set to False for production to improve cycle time
DEBUG_MODE = False

# Import centralized debug system
from debug_utils import debug_print, enable_debug_mode, disable_debug_mode, toggle_debug_mode, debug_on, debug_off, debug_toggle

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
TOP_CAMERA_URL = "http://192.168.100.50:8080/stream-hd"  # Top view event camera
SIDE_EVENT_CAMERA_URL = "http://192.168.100.51:8080/stream-hd"  # Side view event camera

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
GLOBAL_INTRINSICS = {
    'top_camera': None,
    'side_camera': None,
    'realsense': None
}
MODEL_CACHE = None
SETTINGS_CACHE = None
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
side_event_camera = None # Event camera for side view (additional option)
signal_handler = None  # Will be initialized at startup

# Camera pre-warming system
camera_warmup_active = False
camera_warmup_thread = None

# Frame storage
frame_top = None
frame_side = None

# Global variables for camera timeout management
camera_timeout_timer = None
CAMERA_TIMEOUT_SECONDS = 30  # Turn off cameras after 30 seconds of inactivity
last_signal_time = 0
camera_auto_management = True  # Enable/disable automatic camera management

# Configuration flag to switch between camera management modes
USE_PER_SIGNAL_CAMERA_CONTROL = False  # Set True for maximum heat protection

# Camera heat management variables
camera_idle_mode = False
IDLE_FRAME_RATE = 5  # FPS when idle (vs normal 30 FPS)
NORMAL_FRAME_RATE = 30  # FPS when active

# ========================================
# CAMERA HEAT MANAGEMENT CONFIGURATION
# ========================================
# Choose ONE of the following options:

# OPTION 1: Smart Timeout System (RECOMMENDED - balanced performance/heat)
# Cameras auto-stop after 30 seconds of inactivity
# Cycle time impact: +50-100ms when cameras need restart
USE_SMART_TIMEOUT = True
CAMERA_TIMEOUT_SECONDS = 30  # Adjust timeout as needed

# OPTION 2: Per-Signal Control (MAXIMUM heat protection)
# Cameras start/stop for each signal
# Cycle time impact: +200-500ms per cycle
USE_PER_SIGNAL_CAMERA_CONTROL = False

# OPTION 3: Reduced Frame Rate (MINIMAL cycle time impact)
# Cameras stay on but use lower frame rate when idle
# Cycle time impact: +0-50ms
USE_REDUCED_FRAMERATE_MODE = False
IDLE_FRAME_RATE = 5  # FPS when idle
NORMAL_FRAME_RATE = 30  # FPS when active

# OPTION 4: Continuous Mode (ORIGINAL - no heat protection)
# Cameras always on at full frame rate
# Use this by setting all above options to False

def get_camera_heat_management_info():
    """Get current camera heat management configuration"""
    if USE_PER_SIGNAL_CAMERA_CONTROL:
        return "Per-Signal Control (Maximum Heat Protection)"
    elif USE_REDUCED_FRAMERATE_MODE:
        return "Reduced Frame Rate Mode (Minimal Cycle Time Impact)"
    elif USE_SMART_TIMEOUT:
        return f"Smart Timeout System ({CAMERA_TIMEOUT_SECONDS}s timeout)"
    else:
        return "Continuous Mode (No Heat Protection)"

# ========================================
# REALSENSE THERMAL OPTIMIZATION SETTINGS
# ========================================
# These settings can reduce RealSense heating

# Reduce laser power when not actively measuring (percentage of max)
REALSENSE_IDLE_LASER_POWER = 50  # 50% power when idle
REALSENSE_ACTIVE_LASER_POWER = 100  # 100% power when measuring

# Frame rate control for RealSense
REALSENSE_IDLE_FPS = 15  # Lower FPS when idle
REALSENSE_ACTIVE_FPS = 30  # Normal FPS when measuring

def configure_realsense_thermal_settings(camera, active=False):
    """Configure RealSense thermal settings based on mode"""
    if not camera or not hasattr(camera, 'pipeline'):
        return
    
    try:
        # Get device and depth sensor
        profile = getattr(camera, 'profile', None)
        if not profile:
            return
            
        device = profile.get_device()
        if not device:
            return
            
        depth_sensor = device.first_depth_sensor()
        if not depth_sensor:
            return
        
        # Adjust laser power based on mode
        if depth_sensor.supports(rs.option.laser_power):
            if active:
                max_power = depth_sensor.get_option_range(rs.option.laser_power).max
                target_power = max_power * (REALSENSE_ACTIVE_LASER_POWER / 100.0)
                debug_print(f"RealSense: Setting active laser power to {target_power:.1f}", "thermal")
            else:
                max_power = depth_sensor.get_option_range(rs.option.laser_power).max
                target_power = max_power * (REALSENSE_IDLE_LASER_POWER / 100.0)
                debug_print(f"RealSense: Setting idle laser power to {target_power:.1f}", "thermal")
            
            depth_sensor.set_option(rs.option.laser_power, target_power)
        
        # Note: Frame rate changes require pipeline restart, so we handle this
        # through the frame rate management system instead
        
    except Exception as e:
        debug_print(f"Error configuring RealSense thermal settings: {e}", "errors")

def optimize_realsense_for_mode(active=False):
    """Optimize RealSense settings for current mode"""
    global realsense_camera
    
    if realsense_camera and hasattr(realsense_camera, 'is_streaming') and realsense_camera.is_streaming:
        configure_realsense_thermal_settings(realsense_camera, active=active)

# Initialize global variables before GUI creation
def init_globals():
    """Initialize global variables and configurations - optimized with singleton pattern"""
    global current_settings, WHEEL_MODELS, realsense_camera, top_camera, side_camera, GLOBAL_INTRINSICS, config_manager   
    
    # Initialize singleton config manager first (loads everything once)
    from config_manager import ConfigManager
    config_manager = ConfigManager()
    
    # Get settings from cached config manager 
    current_settings = config_manager.settings
    WHEEL_MODELS = config_manager.wheel_models
    
    # Ensure required camera URL settings exist
    if "top_camera_url" not in current_settings:
        current_settings["top_camera_url"] = TOP_CAMERA_URL
    
    if "side_camera_url" not in current_settings:
        current_settings["side_camera_url"] = REALSENSE_URL
    
    # Add capture_interval if missing
    if "capture_interval" not in current_settings:
        if "auto_capture_interval" in current_settings:
            current_settings["capture_interval"] = current_settings["auto_capture_interval"]
        else:
            current_settings["capture_interval"] = 5.0
    
    # Ensure captured frames directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Load the AI model once at startup (this is the most time-consuming operation)
    debug_print("Loading AI model at startup...", "startup")
    model = image_processing.load_model()
    if model is not None:
        debug_print("AI model loaded successfully", "startup")
    else:
        debug_print("Warning: AI model failed to load", "errors")
    
    # Initialize camera objects with correct intrinsics flow: DEVICE FIRST, then fallback to cache
    try:
        # Get non-RealSense intrinsics (these are static/don't need device)
        top_camera_intrinsics = config_manager.get_intrinsics('top_camera')
        side_camera_intrinsics = config_manager.get_intrinsics('side_camera')
        
        # STEP 1: Try to get RealSense intrinsics from DEVICE FIRST (highest priority)
        debug_print("🎯 Acquiring RealSense intrinsics from device first...", "startup")
        device_intrinsics = None
        realsense_intrinsics = None
        
        # Initialize temporary RealSense camera to get device intrinsics (skip cache loading)
        temp_realsense = RealSenseCamera(skip_cache_loading=True)
        
        try:
            # Temporarily start RealSense to get device intrinsics
            if temp_realsense.start_realsense():
                device_intrinsics = temp_realsense.get_intrinsics()
                temp_realsense.stop_realsense()  # Stop immediately after getting intrinsics
                
                if device_intrinsics and device_intrinsics.get('fx', 0) > 0:
                    debug_print(f"✅ RealSense device intrinsics acquired: fx={device_intrinsics.get('fx', 0):.1f}, fy={device_intrinsics.get('fy', 0):.1f}", "startup")
                    
                    # Save fresh device intrinsics to both files
                    config_manager._save_realsense_intrinsics_to_both_files(device_intrinsics)
                    
                    # Cache the fresh device intrinsics for this session
                    config_manager._intrinsics['realsense'] = device_intrinsics
                    realsense_intrinsics = device_intrinsics
                    
                    debug_print("✅ Fresh RealSense intrinsics saved and cached for session", "startup")
                else:
                    debug_print("⚠️ Device started but invalid intrinsics returned", "errors")
                    device_intrinsics = None
            else:
                debug_print("⚠️ Could not start RealSense device", "errors")
                device_intrinsics = None
        except Exception as e:
            debug_print(f"⚠️ Error getting RealSense intrinsics from device: {e}", "errors")
            device_intrinsics = None
        
        # STEP 2: FALLBACK - If device failed, load from cache/file
        if device_intrinsics is None:
            debug_print("🔄 Device failed, falling back to cached RealSense intrinsics...", "startup")
            cached_intrinsics = config_manager.load_realsense_intrinsics_from_file()
            if cached_intrinsics and cached_intrinsics.get('fx', 0) > 0:
                realsense_intrinsics = cached_intrinsics
                # Success message already printed by config_manager method
            else:
                debug_print("❌ ERROR: No valid RealSense intrinsics available from device or cache!", "errors")
                realsense_intrinsics = None
        
        # STEP 3: Initialize RealSense camera WITH the correct intrinsics (device or fallback)
        debug_print("🔧 Initializing RealSense camera with determined intrinsics...", "startup")
        realsense_camera = RealSenseCamera(intrinsics=realsense_intrinsics)
        side_camera = realsense_camera
        
        # Initialize top camera (event camera)
        top_camera = IPCamera(TOP_CAMERA_URL, intrinsics=top_camera_intrinsics)
        
        # Initialize side event camera as additional option
        side_event_camera = IPCamera(SIDE_EVENT_CAMERA_URL, intrinsics=side_camera_intrinsics)
        
    except Exception as e:
        debug_print(f"Error initializing camera objects: {e}", "errors")
        
    # Use cached intrinsics (no repeated loading) - with latest RealSense data
    GLOBAL_INTRINSICS['top_camera'] = config_manager.get_intrinsics('top_camera')
    GLOBAL_INTRINSICS['side_camera'] = config_manager.get_intrinsics('side_camera')
    GLOBAL_INTRINSICS['realsense'] = config_manager.get_intrinsics('realsense')
    
    # Update current_settings with latest synced intrinsics (avoid redundant sync)
    if GLOBAL_INTRINSICS['realsense']:
        current_settings['calibration'].update({
            'fx': GLOBAL_INTRINSICS['realsense'].get('fx', 0),
            'fy': GLOBAL_INTRINSICS['realsense'].get('fy', 0),
            'cx': GLOBAL_INTRINSICS['realsense'].get('cx', 0),
            'cy': GLOBAL_INTRINSICS['realsense'].get('cy', 0)
        })
    
    # Save settings to ensure they persist (only if changes made)
    config_manager.save_settings()
    
    debug_print("Initialization complete - all intrinsics synced across camera_intrinsics.json and settings.json", "startup")

def take_photo_optimized():
    """Optimized photo processing with parallel image processing and memory management"""
    global measurement_thread_pool
    
    if not streaming_active:
        status_label_main.config(text="Error: Streams not active")
        return

    # Initialize thread pool if needed
    if measurement_thread_pool is None:
        measurement_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="measurement")

    # Submit measurement task to thread pool (reuse threads)
    measurement_thread_pool.submit(run_optimized_measurement_thread)

def run_optimized_measurement_thread():
    """Optimized measurement processing with parallel image processing and improved memory management"""
    global photo_count, frame_top, frame_side, signal_handler
    
    try:
        # Initialize status
        root.after(0, lambda: status_label_main.config(text="Starting optimized measurement..."))
        
        # Create local copies of frames to avoid threading issues
        local_top = frame_top.copy() if frame_top is not None else None
        local_side = frame_side.copy() if frame_side is not None else None
        
        # Check camera availability
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
            debug_print("[OPTIMIZED] Using top frame for both views", "processing")
        elif local_side is not None and local_top is None:
            local_top = local_side.copy()
            camera_scenario = "side_only"
            debug_print("[OPTIMIZED] Using side frame for both views", "processing")
        
        # Get aligned depth frame if available (cached access)
        depth_frame = None
        aligned_color_frame = None
        depth_intrinsics = get_cached_depth_intrinsics()  # Use cached version
        
        if realsense_camera and realsense_camera.is_streaming:
            try:
                if hasattr(realsense_camera, 'aligned_frames') and realsense_camera.aligned_frames:
                    aligned_depth_frame = realsense_camera.aligned_frames.get_depth_frame()
                    aligned_color_frame_rs = realsense_camera.aligned_frames.get_color_frame()
                    if aligned_depth_frame:
                        depth_frame = aligned_depth_frame
                        aligned_color_frame = np.asanyarray(aligned_color_frame_rs.get_data())  
                        local_side = aligned_color_frame.copy()  
                        debug_print("[OPTIMIZED] Using aligned depth frame", "processing")
                elif hasattr(realsense_camera, 'get_depth_frame'):
                    depth_frame = realsense_camera.get_depth_frame()
                    debug_print("[OPTIMIZED] Using unaligned depth frame", "processing")
            except Exception as e:
                debug_print(f"[OPTIMIZED] Depth frame error: {e}", "errors")
        
        # Display captured frames in main panels (non-blocking)
        root.after(0, lambda: update_display(side_panel, local_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit"))
        root.after(0, lambda: update_display(top_panel, local_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit"))
        
        # ===== PARALLEL PROCESSING OPTIMIZATION =====
        root.after(0, lambda: status_label_main.config(text="Processing both views in parallel..."))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both preprocessing tasks
            top_future = executor.submit(
                image_processing.run_model_and_mask,
                local_top,
                True  # is_top_view
            ) if local_top is not None else None
            
            side_future = executor.submit(
                image_processing.run_model_and_mask,
                local_side,
                False  # is_top_view
            ) if local_side is not None else None
            
            # Wait for both results
            top_mask_result = top_future.result() if top_future else None
            side_mask_result = side_future.result() if side_future else None
        
        # Get current model name (cached access)
        model_name = current_model_display.get() if current_model_display else "Unknown"
        
        debug_print(f"[OPTIMIZED] Processing with model: {model_name}", "processing")
        
        # ===== SEQUENTIAL MEASUREMENTS =====
        # Process side view first to get height
        side_result = image_processing.calculate_measurements(
            side_mask_result,
            current_settings,
            WHEEL_MODELS,
            model_name,
            depth_frame=depth_frame,
            depth_intrinsics=depth_intrinsics
        ) if side_mask_result else None
          
        if not side_result:
            root.after(0, lambda: status_label_main.config(text="Error: Side processing failed"))
            if signal_handler:
                signal_handler.send_measurement_data(None, 0.0, 0.0, WHEEL_MODELS)
            return
        
        # Get height from side view for top view processing
        wheel_height_mm = side_result['height_mm'] if side_result else None
        
        # Process top view using height from side view
        top_result = image_processing.calculate_measurements(
            top_mask_result,
            current_settings,
            WHEEL_MODELS,
            model_name,
            wheel_height_mm=wheel_height_mm
        ) if top_mask_result and wheel_height_mm else None

        if not top_result:
            root.after(0, lambda: status_label_main.config(text="Error: Top processing failed"))
            if signal_handler:
                signal_handler.send_measurement_data(None, 0.0, 0.0, WHEEL_MODELS)
            return
        
        # Update processed panels with visualizations (non-blocking)
        root.after(0, lambda: update_display(
            side_processed_panel, 
            side_result['visualization'], 
            SIDE_PANEL_WIDTH, 
            SIDE_PANEL_HEIGHT, 
            "fit"
        ))
        
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
        
        # Update all measurement displays in the UI (batched for efficiency)
        def update_ui_batch():
            measured_height_var.set(f"{side_result['height_mm']:.1f} mm")
            measured_dia_var.set(f"{top_result['diameter_mm']:.1f} mm")
            
            side_cam_distance_result.set(
                f"{side_result['side_camera_distance_m'] * 1000:.1f} mm" 
                if side_result.get('side_camera_distance_m') is not None 
                else "N/A"
            )
            
            result_model_code_var.set(
                str(signal_handler._get_model_code(model_name)) 
                if signal_handler 
                else "Err"
            )
            
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
            )
            
            update_wheel_counts()
            status_label_main.config(text="Optimized processing complete")
        
        root.after(0, update_ui_batch)
        
        # Save to database (without image paths) - use optimized connection
        try:
            add_inspection(
                part_no,
                model_name,
                top_result['diameter_mm'],
                side_result['height_mm'],
                None,  # No image path for top
                None   # No image path for side
            )
            debug_print("[OPTIMIZED] Saved to database", "database")
        except Exception as e:
            debug_print(f"[OPTIMIZED] Database error: {e}", "errors")
        
        # Send to Modbus
        if signal_handler:
            try:
                signal_handler.send_measurement_data(
                    model_name,
                    top_result['diameter_mm'],
                    side_result['height_mm'],
                    WHEEL_MODELS
                )
                debug_print("[OPTIMIZED] Sent to Modbus", "modbus")
            except Exception as e:
                debug_print(f"[OPTIMIZED] Modbus error: {e}", "errors")
        
        # Final garbage collection to clean up processed image data
        del local_top, local_side, side_result, top_result
        gc.collect()
        debug_print("[OPTIMIZED] Final cleanup completed", "processing")
        
    except Exception as e:
        debug_print(f"[OPTIMIZED ERROR] {str(e)}", "errors")
        traceback.print_exc()
        root.after(0, lambda: status_label_main.config(text="Error: Optimized processing failed"))
    finally:
        debug_print("[OPTIMIZED] Optimized measurement processing complete", "processing")

def log_performance_metrics(start_time, operation_name):
    """Log performance metrics for cycle time optimization"""
    end_time = time.time()
    duration = end_time - start_time
    debug_print(f"[PERFORMANCE] {operation_name}: {duration:.3f} seconds", "performance")
    
    # Optional: Save to file for analysis
    try:
        with open("performance_log.txt", "a") as f:
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            f.write(f"{timestamp}, {operation_name}, {duration:.3f}\n")
    except Exception as e:
        debug_print(f"Performance log error: {e}", "errors")

def start_streaming():
    """Start streaming from cameras - including both RealSense and event cameras"""
    global streaming_active, realsense_camera, top_camera, side_event_camera, stop_streaming, signal_handler
    
    if streaming_active:
        status_label_main.config(text="Streaming already active")
        return
    
    stop_streaming = False
    streaming_active = True
    status_label_main.config(text="Initializing cameras...")
    
    # Update UI
    # start_button.config(state=tk.DISABLED)
    # stop_button.config(state=tk.NORMAL)
    
    camera_status = []
    
    # Try to start RealSense camera first (for depth measurements)
    if realsense_camera:
        try:
            realsense_camera.start()
            camera_status.append("RealSense: Connected")
            debug_print("RealSense camera started successfully", "cameras")
        except Exception:
            camera_status.append("RealSense: Failed")
    
    # Start top camera (event camera)
    if top_camera:
        try:
            top_camera.start()
            camera_status.append("Top Event: Connected")
            debug_print("Top event camera started successfully", "cameras")
        except Exception:
            camera_status.append("Top Event: Failed")
    
    # Start side event camera if available
    if side_event_camera:
        try:
            side_event_camera.start()
            camera_status.append("Side Event: Connected")
            debug_print("Side event camera started successfully", "cameras")
        except Exception:
            camera_status.append("Side Event: Failed")
    
    # Update status labels
    status_label_side.config(text=" | ".join([s for s in camera_status if "Side" in s or "RealSense" in s]))
    status_label_top.config(text=" | ".join([s for s in camera_status if "Top" in s]))
    
    # Start frame update threads
    threading.Thread(target=update_frames, daemon=True).start()    
    
    status_label_main.config(text=f"Streaming started - {len([s for s in camera_status if 'Connected' in s])}/{len(camera_status)} cameras active")

def cleanup_measurement_cycle():
    """Complete memory cleanup after each cycle"""
    global frame_top, frame_side, depth_frame
    
    # Clear frame references
    frame_top = None
    frame_side = None
    depth_frame = None
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU memory if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def update_frames():
    """Update frames from cameras - with heat management via frame rate control"""
    global frame_top, frame_side, stop_streaming, camera_idle_mode
    
    while not stop_streaming:
        # Check camera availability
        has_realsense = realsense_camera and realsense_camera.is_streaming
        has_side_event = side_event_camera and side_event_camera.is_streaming
        has_top = top_camera and top_camera.is_streaming
        
        # Get frames from cameras - prioritize RealSense for side view (depth data)
        if has_realsense:
            frame_side = realsense_camera.get_frame()
        elif has_side_event:
            frame_side = side_event_camera.get_frame()
        
        if has_top:
            frame_top = top_camera.get_frame()
        
        # Update displays based on available cameras
        if frame_side is not None and frame_top is not None:
            # Both side and top cameras available - normal display
            update_display(side_panel, frame_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            update_display(top_panel, frame_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
        elif frame_side is not None and frame_top is None:
            # Only side camera - display in both panels
            update_display(side_panel, frame_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            update_display(top_panel, frame_side, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
        elif frame_top is not None and frame_side is None:
            # Only top camera - display in both panels
            update_display(side_panel, frame_top, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT, "fit")
            update_display(top_panel, frame_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT, "fit")
        
        # Adaptive frame rate for heat management
        if camera_idle_mode:
            time.sleep(1.0 / IDLE_FRAME_RATE)  # 5 FPS when idle
        else:
            time.sleep(1.0 / NORMAL_FRAME_RATE)  # 30 FPS when active

def stop_streaming_func():
    """Stop streaming from all cameras"""
    global streaming_active, stop_streaming, realsense_camera, top_camera, side_event_camera, signal_handler
    
    stop_streaming = True
    streaming_active = False
    
    # Stop all cameras without destroying objects, but only if they're streaming
    cameras_stopped = []
    
    if realsense_camera and hasattr(realsense_camera, 'is_streaming') and realsense_camera.is_streaming:
        try:
            realsense_camera.stop()
            cameras_stopped.append("RealSense")
            debug_print("RealSense camera streaming stopped", "cameras")
        except Exception as e:
            debug_print(f"Error stopping RealSense camera: {e}", "errors")
    
    if top_camera and hasattr(top_camera, 'is_streaming') and top_camera.is_streaming:
        try:
            top_camera.stop()
            cameras_stopped.append("Top Event")
            debug_print("Top camera streaming stopped", "cameras")
        except Exception as e:
            debug_print(f"Error stopping top camera: {e}", "errors")
    
    if side_event_camera and hasattr(side_event_camera, 'is_streaming') and side_event_camera.is_streaming:
        try:
            side_event_camera.stop()
            cameras_stopped.append("Side Event")
            debug_print("Side event camera streaming stopped", "cameras")
        except Exception as e:
            debug_print(f"Error stopping side event camera: {e}", "errors")
    
    # Update UI
    # start_button.config(state=tk.NORMAL)
    # stop_button.config(state=tk.DISABLED)
    
    status_label_main.config(text=f"Streaming stopped - {len(cameras_stopped)} cameras stopped")
    status_label_side.config(text="Cameras stopped")
    status_label_top.config(text="Cameras stopped")
    
    # Note: We don't stop the signal_handler as it needs to run continuously
    # to detect 24V signals even when streaming is stopped

def update_wheel_counts():
    """Update the wheel count display from database with counts by model (optimized)"""
    global wheel_model_counts, model_count_vars
    
    try:
        # Use context manager for automatic connection handling
        with sqlite3.connect('wheel_inspection.db') as conn:
            cursor = conn.cursor()
            
            # Single optimized query to get both total and model counts
            cursor.execute("""
                SELECT 
                    model_type, 
                    COUNT(*) as count,
                    SUM(COUNT(*)) OVER () as total_count
                FROM inspections 
                GROUP BY model_type
            """)
            
            results = cursor.fetchall()
            
            if results:
                # Get total count from first row
                total_count = results[0][2] if results else 0
                total_count_var.set(str(total_count))
                
                # Update wheel model count variables
                wheel_model_counts = {}
                for model, count, _ in results:
                    wheel_model_counts[model] = count
                    
                # Update UI elements if they exist, create new ones for new models
                for model in WHEEL_MODELS.keys():
                    # Create StringVar for new models if it doesn't exist
                    if model not in model_count_vars:
                        model_count_vars[model] = tk.StringVar(value='0')
                    
                    # Update the count
                    model_count_vars[model].set(str(wheel_model_counts.get(model, 0)))
            else:
                # No records found
                total_count_var.set('0')
                # Ensure all models have StringVar entries even if no data
                for model in WHEEL_MODELS.keys():
                    if model not in model_count_vars:
                        model_count_vars[model] = tk.StringVar(value='0')
                    else:
                        model_count_vars[model].set('0')
    except Exception as e:
        debug_print(f"Error updating wheel counts: {e}", "errors")
        # Don't crash if database update fails

def update_model_parameters():
    """Update UI with current model parameters using cached ConfigManager"""
    global current_settings, WHEEL_MODELS, current_model_display, config_manager
    
    # Use already cached settings (ConfigManager is singleton with auto-refresh)
    current_settings = config_manager.settings
    WHEEL_MODELS = config_manager.wheel_models
    
    model_name = current_settings.get("selected_model", "10-13")
    model_data = WHEEL_MODELS.get(model_name, {})
    
    # print(f"[DEBUG] Updating UI with model: {model_name}")
    # print(f"[DEBUG] Model data: {model_data}")
    
    # Update model info display
    model_value.set(model_name)
    
    # Update current model display in results frame
    if current_model_display is not None:
        current_model_display.set(model_name)
        debug_print(f"Updated model display to: {model_name}", "ui")
    
    # Show diameter range
    min_dia = model_data.get("min_dia", 0)
    max_dia = model_data.get("max_dia", 0)
    diameter_value.set(f"{min_dia}-{max_dia} inch")
    
    # Show expected height
    height = model_data.get("height", 0)
    height_value.set(f"{height} mm")
    
    # Update camera heights from cached settings
    top_cam_height = current_settings["calibration"]["base_height"]
    side_cam_height = current_settings["calibration"]["side_camera_height"]
    top_cam_height_var.set(f"{top_cam_height:.1f} mm")
    side_cam_height_var.set(f"{side_cam_height:.1f} mm")
    
    # Update wheel counts and count frame to reflect any model changes
    update_wheel_counts()
    update_count_frame()
    
    # print(f"[DEBUG] UI updated successfully with model: {model_name}, diameter: {min_dia}-{max_dia} inch, height: {height} mm")

def on_closing():
    """Handle window close event with proper cleanup"""
    global stop_streaming, signal_handler, realsense_camera, top_camera, side_event_camera, streaming_active
    
    # Set flag to stop all threads
    stop_streaming = True
    
    # Only attempt to stop cameras if they're still streaming
    if streaming_active:
        # This will call stop_streaming_func which properly stops all cameras
        stop_streaming_func()
    else:
        # If we're not streaming, we might still need to clean up camera objects
        cameras_to_stop = [
            ("RealSense", realsense_camera),
            ("Top", top_camera),
            ("Side Event", side_event_camera)
        ]
        
        for camera_name, camera_obj in cameras_to_stop:
            if camera_obj and hasattr(camera_obj, 'is_streaming') and camera_obj.is_streaming:
                try:
                    camera_obj.stop()
                    debug_print(f"{camera_name} camera stopped properly", "cameras")
                except Exception as e:
                    debug_print(f"Error stopping {camera_name} camera: {e}", "errors")
    
    # Always ensure streaming_active is set to False
    streaming_active = False
    
    # Make sure signal handler is stopped
    if signal_handler:
        try:
            signal_handler.stop_detection()
            debug_print("Signal handler stopped", "cameras")
        except Exception as e:
            debug_print(f"Error stopping signal handler: {e}", "errors")
    
    # Clean up database connections
    try:
        from database import cleanup_connections
        cleanup_connections()
        debug_print("Database connections cleaned up", "startup")
    except Exception as e:
        debug_print(f"Error cleaning up database connections: {e}", "errors")
    
    # Clear camera intrinsics cache
    try:
        from camera_utils import clear_intrinsics_cache
        clear_intrinsics_cache()
        debug_print("Camera intrinsics cache cleared", "startup")
    except Exception as e:
        debug_print(f"Error clearing camera intrinsics cache: {e}", "errors")
    
    # Clear GPU memory if using CUDA
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            debug_print("GPU memory cleared", "startup")
        except Exception as e:
            debug_print(f"Error clearing GPU memory: {e}", "errors")
    
    debug_print("Application closed properly", "startup")
    root.destroy()

def open_settings_window():
    """Open the settings window"""
    global settings_win, config_manager
    
    if settings_win is not None and settings_win.winfo_exists():
        settings_win.lift()  # Bring to front if exists
        settings_win.focus_force()
        return
        
    settings_win = show_settings_window(root, config_manager, update_model_parameters)
    
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
   
def reset_camera_timeout():
    """Reset the camera timeout timer"""
    global camera_timeout_timer, last_signal_time
    
    # Cancel existing timer
    if camera_timeout_timer is not None:
        camera_timeout_timer.cancel()
    
    # Update last signal time
    last_signal_time = time.time()
    
    # Start new timeout timer if auto management is enabled
    if camera_auto_management:
        camera_timeout_timer = threading.Timer(CAMERA_TIMEOUT_SECONDS, camera_timeout_handler)
        camera_timeout_timer.daemon = True
        camera_timeout_timer.start()
        debug_print(f"Camera timeout reset - cameras will auto-stop in {CAMERA_TIMEOUT_SECONDS} seconds", "thermal")

def camera_timeout_handler():
    """Handle camera timeout - stop cameras to prevent heating"""
    global streaming_active
    
    if streaming_active:
        debug_print("Camera timeout reached - stopping cameras to prevent heating", "thermal")
        root.after(0, stop_streaming_func)
        status_label_main.config(text="Cameras stopped due to inactivity (heat protection)")

def start_cameras_with_timeout():
    """Start cameras and initialize timeout system"""
    global streaming_active
    
    if not streaming_active:
        debug_print("Starting cameras with timeout protection", "thermal")
        start_streaming()
        
    # Reset timeout on any camera activity
    reset_camera_timeout()

def signal_handler_callback(signal_type=""):
    """Handle modbus frame with optimized cycle time"""
    global streaming_active
    
    if signal_type == "MODBUS_FRAME":
        debug_print("Processing modbus frame: Starting optimized measurement cycle", "modbus")
        
        # Reset camera timeout on signal
        reset_camera_timeout()
        
        # Cleanup at start
        cleanup_measurement_cycle()
        
        # Optimize RealSense for active measurement
        optimize_realsense_for_mode(active=True)
        
        # If cameras are off, start them with minimal delay
        if not streaming_active:
            start_streaming()
            # Reduced wait time for camera initialization
            def delayed_photo():
                take_photo_optimized()
                # Return to idle mode after measurement
                def return_to_idle():
                    optimize_realsense_for_mode(active=False)
                root.after(1500, return_to_idle)  # Reduced from 2000ms
            root.after(500, delayed_photo)  # Reduced from 800ms
        else:
            # Cameras already on, take photo immediately
            take_photo_optimized()
            # Return to idle mode after measurement
            def return_to_idle():
                optimize_realsense_for_mode(active=False)
            root.after(1500, return_to_idle)  # Reduced from 2000ms
        
        cleanup_measurement_cycle()

def signal_handler_callback_per_signal(signal_type=""):
    """Alternative callback: Start/stop cameras for each signal (maximum heat protection)"""
    global streaming_active
    
    if signal_type == "MODBUS_FRAME":
        debug_print("Processing modbus frame: Per-signal camera control", "modbus")
        
        # Cleanup at start
        cleanup_measurement_cycle()
        
        # Always start cameras fresh for each signal
        if streaming_active:
            stop_streaming_func()
            time.sleep(0.1)  # Brief pause
        
        # Start cameras
        start_streaming()
        
        def delayed_photo_and_stop():
            # Take photo
            take_photo_optimized()
            
            # Stop cameras immediately after processing
            def stop_cameras():
                stop_streaming_func()
                debug_print("Cameras stopped after processing - heat protection active", "thermal")
            
            # Stop cameras after processing complete
            root.after(1000, stop_cameras)
        
        # Wait for camera initialization then process
        root.after(1000, delayed_photo_and_stop)
        cleanup_measurement_cycle()

def signal_handler_callback_continuous(signal_type=""):
    """Original callback: Continuous camera operation (no heat protection)"""
    global streaming_active
    
    if signal_type == "MODBUS_FRAME":
        debug_print("Processing modbus frame: Continuous camera mode", "modbus")
        
         # Cleanup at start
        cleanup_measurement_cycle()
        
        # If already streaming, just take a photo
        if streaming_active:
            root.after(0, take_photo_optimized)
            return
        
        # Start the streaming to initialize cameras
        root.after(0, start_streaming)
        
        # Wait for camera initialization then process
        def delayed_photo():
            take_photo_optimized()
        
        root.after(500, delayed_photo)
        cleanup_measurement_cycle()

# Configuration flag to switch between camera management modes
USE_PER_SIGNAL_CAMERA_CONTROL = False  # Set True for maximum heat protection

def get_active_signal_callback():
    """Get the appropriate signal callback based on configuration"""
    if USE_PER_SIGNAL_CAMERA_CONTROL:
        return signal_handler_callback_per_signal
    elif USE_REDUCED_FRAMERATE_MODE:
        return signal_handler_callback_reduced_framerate
    elif USE_SMART_TIMEOUT:
        return signal_handler_callback  # Smart timeout system
    else:
        return signal_handler_callback_continuous  # Continuous mode

def update_count_frame():
    """Update the count frame with latest daily and monthly counts"""
    global count_frame
    
    if count_frame is None:
        debug_print("Count frame not initialized", "errors")
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

def set_camera_idle_mode(idle=True):
    """Set camera to idle mode (reduced frame rate) to manage heat"""
    global camera_idle_mode
    camera_idle_mode = idle
    
    if idle:
        debug_print("Cameras entering idle mode - reduced frame rate for heat management", "thermal")
    else:
        debug_print("Cameras entering active mode - normal frame rate", "thermal")

def signal_handler_callback_reduced_framerate(signal_type=""):
    """Alternative callback: Use reduced frame rate for heat management"""
    global streaming_active
    
    if signal_type == "MODBUS_FRAME":
        debug_print("Processing modbus frame: Reduced frame rate heat management", "modbus")
        
        # Cleanup at start
        cleanup_measurement_cycle()
        
        # Ensure cameras are active (not idle)
        set_camera_idle_mode(idle=False)
        
        # Start cameras if not already active
        if not streaming_active:
            start_streaming()
            # Wait brief moment for frame rate adjustment
            def delayed_photo():
                take_photo_optimized()
                # Return to idle mode after processing
                def return_to_idle():
                    set_camera_idle_mode(idle=True)
                root.after(2000, return_to_idle)  # Return to idle after 2 seconds
            root.after(500, delayed_photo)
        else:
            # Take photo immediately
            take_photo_optimized()
            # Return to idle mode after processing
            def return_to_idle():
                set_camera_idle_mode(idle=True)
            root.after(2000, return_to_idle)
        
        cleanup_measurement_cycle()

# Add frame rate management to configuration
USE_REDUCED_FRAMERATE_MODE = False  # Set True for frame rate heat management

# Cache RealSense depth intrinsics globally to avoid repeated calls
CACHED_DEPTH_INTRINSICS = None

def get_cached_depth_intrinsics():
    """Get cached RealSense depth intrinsics - avoids repeated device calls"""
    global CACHED_DEPTH_INTRINSICS, realsense_camera
    
    if CACHED_DEPTH_INTRINSICS is None:
        if realsense_camera and hasattr(realsense_camera, 'depth_intrin') and realsense_camera.depth_intrin:
            CACHED_DEPTH_INTRINSICS = realsense_camera.depth_intrin
            debug_print(f"Cached RealSense depth intrinsics: fx={CACHED_DEPTH_INTRINSICS.fx:.1f}", "processing")
        else:
            # Fallback to GLOBAL_INTRINSICS (but use correct 'realsense' key)
            fallback_intrinsics = GLOBAL_INTRINSICS.get('realsense')
            if fallback_intrinsics:
                # Create a simple object with the required attributes for compatibility
                class DepthIntrinsics:
                    def __init__(self, fx, fy, ppx, ppy, width, height):
                        self.fx = fx
                        self.fy = fy
                        self.ppx = ppx  # cx
                        self.ppy = ppy  # cy
                        self.width = width
                        self.height = height
                
                CACHED_DEPTH_INTRINSICS = DepthIntrinsics(
                    fx=fallback_intrinsics.get('fx', 0),
                    fy=fallback_intrinsics.get('fy', 0),
                    ppx=fallback_intrinsics.get('cx', 0),  # cx -> ppx
                    ppy=fallback_intrinsics.get('cy', 0),  # cy -> ppy
                    width=fallback_intrinsics.get('width', 1280),
                    height=fallback_intrinsics.get('height', 720)
                )
                debug_print(f"Using fallback depth intrinsics: fx={CACHED_DEPTH_INTRINSICS.fx:.1f}", "processing")
    
    return CACHED_DEPTH_INTRINSICS

def check_gpu_availability():
    """Check and report GPU availability for optimization"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available() if 'torch' in globals() else False,
        'device_count': 0,
        'device_name': 'None',
        'memory_total': 0,
        'optimization_potential': 'Unknown'
    }
    
    if gpu_info['cuda_available']:
        gpu_info['device_count'] = torch.cuda.device_count()
        gpu_info['device_name'] = torch.cuda.get_device_name(0)
        gpu_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # Estimate optimization potential based on GPU
        if 'RTX' in gpu_info['device_name'] or 'GTX 16' in gpu_info['device_name']:
            gpu_info['optimization_potential'] = 'High (60-75% cycle time reduction)'
        elif 'GTX' in gpu_info['device_name']:
            gpu_info['optimization_potential'] = 'Medium (40-60% cycle time reduction)'
        else:
            gpu_info['optimization_potential'] = 'Low-Medium (20-40% cycle time reduction)'
    
    return gpu_info

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
    
    # Check for model loading failure
    if image_processing.wheel_detection_model is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror("Fatal Error", "AI Model failed to load. The application cannot continue and will now exit.")
        root.destroy()
        return

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
    
    # Initialize database counts using existing database functions
    try:
        # Use existing database functions instead of direct SQL
        from database import get_daily_report
        today = datetime.now().strftime("%Y-%m-%d")
        total_count, model_counts, _ = get_daily_report(today)
        if not model_counts:
            model_counts = []
    except Exception as e:
        debug_print(f"Error getting initial counts: {e}", "errors")
        total_count = 0
        model_counts = []
    
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
        logo_path = os.path.join(os.path.dirname(__file__), "taurus.png")
        logo_img = Image.open(logo_path)
        logo_img = logo_img.resize((100, 100), Image.Resampling.LANCZOS)
        logo_imgtk = ImageTk.PhotoImage(image=logo_img)
        
        logo_label = ttk.Label(header_frame, image=logo_imgtk, background=BG_COLOR)
        logo_label.image = logo_imgtk
        logo_label.grid(row=0, column=0, sticky="w", padx=10)
    except Exception as e:
        debug_print(f"Error loading logo: {e}", "errors")
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
    
    # start_button = ttk.Button(button_frame, text="Start Streaming", command=start_streaming)
    # start_button.grid(row=0, column=0, padx=3, pady=5)
    
    # stop_button = ttk.Button(button_frame, text="Stop Streaming", command=stop_streaming_func, state=tk.DISABLED)
    # stop_button.grid(row=0, column=2, padx=3, pady=5)
    
    report_button = ttk.Button(button_frame, text="Generate Report", command=open_report_window)
    report_button.grid(row=0, column=2, padx=3, pady=5)
    
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
    signal_handler = SignalHandler(signal_callback=get_active_signal_callback())
    signal_handler.start_detection()
    
    # Display camera heat management configuration
    heat_mgmt_info = get_camera_heat_management_info()
    debug_print(f"Camera Heat Management: {heat_mgmt_info}", "startup")
    debug_print("Modbus frame detection started - waiting for signals...", "startup")

    # Display GPU info at startup
    gpu_status = check_gpu_availability()
    debug_print(f"GPU Status: {gpu_status}", "startup")

    # Start the mainloop
    root.mainloop()

# Entry point
if __name__ == "__main__":
    main()
