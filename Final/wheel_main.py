import cv2
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import numpy as np
import torch
from PIL import Image, ImageTk
import math
from datetime import datetime
import sqlite3
import queue
import traceback

# Import from modular files
from utils import update_display, update_panel_image, update_clock
from camera_streams import RealSenseCamera, IPCamera, USBCamera, current_depth_image, frame_queue
from image_processing import process_frame
from wheel_measurements import classify_wheel_model
from settings_manager import load_settings, save_settings, load_realsense_calibration
from signal_handler import SignalHandler
from database import init_db, add_inspection
from reports_window import show_report_window
from settings_window import show_settings_window

# Set the environment variable to avoid OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
SAVE_DIR = "captured_frames"
SIDE_PANEL_WIDTH = 640
SIDE_PANEL_HEIGHT = 480
TOP_PANEL_WIDTH = 640
TOP_PANEL_HEIGHT = 480

# UI Color Scheme
BG_COLOR = "#f0f0f0"
TEXT_COLOR = "#333333"
BUTTON_COLOR = "#4a7abc"
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

# Camera objects
realsense_camera = None
top_camera = None
side_camera = None

# Frame storage
frame_top = None
frame_side = None

# Initialize global variables before GUI creation
def init_globals():
    """Initialize global variables and configurations"""
    global current_settings, WHEEL_MODELS
    
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
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    photo_count += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize filename variables to None
    filename_side = None
    filename_top = None
    
    # Only save frames that actually exist
    if has_side_camera and frame_side is not None:
        filename_side = os.path.join(SAVE_DIR, f"side_view_{timestamp}.jpg")
        cv2.imwrite(filename_side, frame_side)
        print(f"Saved side view image to {filename_side}")
    
    if has_top_camera and frame_top is not None:
        filename_top = os.path.join(SAVE_DIR, f"top_view_{timestamp}.jpg")
        cv2.imwrite(filename_top, frame_top)
        print(f"Saved top view image to {filename_top}")
    
    # Process side view first to get wheel height
    # This will be used in top view calculation for more accurate dimensions
    side_measured_height = None
    
    # Process side view first to get wheel height (if available)
    # This is critical for accurate diameter calculation
    measurements_side = {}
    if has_side_camera:
        status_label_main.config(text="Processing: Measuring wheel height from side view...")
        processed_side, measurements_side = process_frame(frame_side, is_top_view=False, 
                                                         camera_settings=current_settings["calibration"],
                                                         wheel_models=WHEEL_MODELS,
                                                         selected_model=current_settings["selected_model"])
        update_display(side_processed_panel, processed_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
    else:
        # Clear side panel or show a "No Camera" message
        status_label_side.config(text="Side camera not connected")
        # Create a blank image with "No Camera" text
        no_camera_img = np.ones((SIDE_PANEL_HEIGHT, SIDE_PANEL_WIDTH, 3), dtype=np.uint8) * 240
        cv2.putText(no_camera_img, "No Side Camera", (int(SIDE_PANEL_WIDTH/4), int(SIDE_PANEL_HEIGHT/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    if measurements_side.get("type") == "Side":
        # Extract height and depth information
        side_measured_height = measurements_side.get("height_mm", None)
        camera_to_wheel_distance = measurements_side.get("depth_mm", None)
        
        if side_measured_height is not None:
            # Store measured height for display in result panel only
            # Do not update the model_frame which should display expected/reference values
            measured_height_var.set(f"{side_measured_height:.1f} mm")
            side_result_text.set(f"Side: Height={side_measured_height:.2f} mm")
            
            # Ensure the height is stored for diameter calculation
            # This is critical to maintain the correct measurement flow
            current_settings["calibration"]["wheel_height"] = side_measured_height
            print(f"Updated wheel height for diameter calculation: {side_measured_height:.2f} mm")
        
        if camera_to_wheel_distance is not None:
            # This is for display only, not updating model frame
            side_cam_distance_result.set(f"{camera_to_wheel_distance:.1f} mm")

    # Process top view (if available) using the height information from side view
    measurements_top = {}
    if has_top_camera:
        status_label_main.config(text="Processing: Calculating wheel diameter using height data...")
        processed_top, measurements_top = process_frame(frame_top, is_top_view=True,
                                                      camera_settings=current_settings["calibration"],
                                                      wheel_models=WHEEL_MODELS,
                                                      selected_model=current_settings["selected_model"])
        update_display(top_processed_panel, processed_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
    else:
        # Clear top panel or show a "No Camera" message
        status_label_top.config(text="Top camera not connected")
        # Create a blank image with "No Camera" text
        no_camera_img = np.ones((TOP_PANEL_HEIGHT, TOP_PANEL_WIDTH, 3), dtype=np.uint8) * 240
        cv2.putText(no_camera_img, "No Top Camera", (int(TOP_PANEL_WIDTH/4), int(TOP_PANEL_HEIGHT/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        update_display(top_processed_panel, no_camera_img, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
    
    # Update status based on which cameras were processed
    if has_top_camera and has_side_camera:
        status_label_main.config(text="Processing complete: Both height and diameter measured.")
    elif has_side_camera:
        status_label_main.config(text="Processing complete: Height measured (no top camera).")
    elif has_top_camera:
        status_label_main.config(text="Processing complete: Diameter measured (no side camera).")
    
    # Update result display for top view
    if measurements_top.get("type") == "Round":
        diameter_mm = measurements_top.get("diameter_mm", 0)
        measured_dia_var.set(f"{diameter_mm:.1f} mm")
        top_result_text.set(f"Top: Diameter={diameter_mm:.2f} mm")
    
    # Update overall result status based on both views
    is_ok_top = measurements_top.get("is_ok", False)
    is_ok_side = measurements_side.get("is_ok", False)
    overall_ok = is_ok_top and is_ok_side
    result_status_var.set("OK" if overall_ok else "NOT OK")
    
    # Update result frame style
    update_result_frame()
    
    # Save to database
    part_no = f"INDIP {timestamp.split('_')[0]} {photo_count}"
    model_type = current_settings["selected_model"]  # Use selected model, not detected
    diameter_mm = measurements_top.get("diameter_mm", 0)
    height_mm = measurements_side.get("height_mm", 0)
    camera_height_mm = float(top_cam_height_var.get().split()[0]) if top_cam_height_var.get() else current_settings["calibration"]["base_height"]
    test_result = "OK" if overall_ok else "NOT OK"
    
    # Update database
    # The order of parameters for add_inspection is:
    # (part_no, model_type, diameter_mm, thickness_mm, height_mm, test_result, image_path_top, image_path_side)
    add_inspection(
        part_no, 
        model_type, 
        diameter_mm, 
        camera_height_mm,  # Using camera height as thickness value
        height_mm, 
        test_result, 
        filename_top if has_top_camera else '', 
        filename_side if has_side_camera else ''
    )
    
    # Update wheel counts
    update_wheel_counts()
    
    status_label_main.config(text=f"Captured and processed frames {photo_count} at {timestamp}")

def update_result_frame():
    """Update the result panel appearance based on status"""
    status = result_status_var.get()
    if status == "OK":
        result_status_label.config(foreground=PASS_COLOR)
    else:
        result_status_label.config(foreground=FAIL_COLOR)

def start_streaming():
    """Start streaming from cameras"""
    global streaming_active, realsense_camera, top_camera, stop_streaming
    
    if streaming_active:
        status_label_main.config(text="Streaming already active")
        return
    
    stop_streaming = False
    streaming_active = True
    status_label_main.config(text="Initializing cameras...")
    
    # Update UI
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    photo_button.config(state=tk.NORMAL)
    auto_photo_button.config(state=tk.NORMAL)
    
    # Try to initialize RealSense camera first (for side view)
    realsense_camera = RealSenseCamera()
    realsense_camera.start()
    
    # Start top camera (event camera or fallback)
    try:
        top_camera = IPCamera(TOP_CAMERA_URL)
        top_camera.start()
        status_label_top.config(text="Top camera connected")
    except Exception as e:
        print(f"Error starting top camera: {e}")
        status_label_top.config(text="Top camera failed to connect")
    
    # Start frame update threads
    threading.Thread(target=update_frames, daemon=True).start()
    
    # Start signal detection for 24V signal
    signal_handler = SignalHandler(signal_callback=take_photo)
    signal_handler.start_detection()
    
    status_label_main.config(text="Streaming started")

def update_frames():
    """Update frames from cameras"""
    global frame_top, frame_side, stop_streaming
    
    while not stop_streaming:
        # Get frames from cameras
        if realsense_camera and realsense_camera.is_streaming:
            frame_side = realsense_camera.get_frame()
            update_display(side_camera_panel, frame_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
        
        if top_camera and top_camera.is_streaming:
            frame_top = top_camera.get_frame()
            update_display(top_camera_panel, frame_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
        
        time.sleep(0.03)  # ~30 FPS

def stop_streaming_func():
    """Stop streaming from cameras"""
    global streaming_active, stop_streaming, realsense_camera, top_camera
    
    stop_streaming = True
    streaming_active = False
    
    # Stop all cameras
    if realsense_camera:
        realsense_camera.stop()
        realsense_camera = None
    
    if top_camera:
        top_camera.stop()
        top_camera = None
    
    # Update UI
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    photo_button.config(state=tk.DISABLED)
    auto_photo_button.config(state=tk.DISABLED)
    
    status_label_main.config(text="Streaming stopped")

def auto_capture():
    """Toggle automatic photo capture"""
    global auto_capture_active
    
    if not streaming_active:
        messagebox.showerror("Error", "Start streaming first!")
        return
    
    auto_capture_active = not auto_capture_active
    
    if auto_capture_active:
        # Start auto capture
        auto_photo_button.config(text="Stop Auto Capture")
        interval = current_settings.get("auto_capture_interval", 5)  # Default 5 seconds
        threading.Thread(target=auto_capture_thread, args=(interval,), daemon=True).start()
    else:
        # Stop auto capture
        auto_photo_button.config(text="Start Auto Capture")

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
    """Update the wheel count statistics from database"""
    conn = sqlite3.connect('wheel_inspection.db')
    cursor = conn.cursor()
    
    try:
        # Get counts
        cursor.execute("SELECT COUNT(*) FROM inspections")
        total_count = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM inspections WHERE test_result = 'OK'")
        passed_count = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM inspections WHERE test_result = 'NOT OK'")
        faulty_count = cursor.fetchone()[0] or 0
        
        # Update UI
        total_count_var.set(str(total_count))
        passed_count_var.set(str(passed_count))
        faulty_count_var.set(str(faulty_count))
    except Exception as e:
        print(f"Error updating wheel counts: {e}")
    finally:
        conn.close()

def update_model_parameters():
    """Update UI with current model parameters"""
    model_name = current_settings["selected_model"]
    model_data = WHEEL_MODELS.get(model_name, {})
    
    # Update model info display
    model_value.set(model_name)
    
    # Show diameter range
    min_dia = model_data.get("min_dia", 0)
    max_dia = model_data.get("max_dia", 0)
    diameter_value.set(f"{min_dia}-{max_dia} mm")
    
    # Show expected height
    height = model_data.get("height", 0)
    height_value.set(f"{height} mm")
    
    # Show tolerance
    tolerance = model_data.get("tolerance", 0)
    tolerance_value.set(f"{tolerance} mm")
    
    # Update camera heights from settings, not measured values
    top_cam_height = current_settings["calibration"]["base_height"]
    side_cam_height = current_settings["calibration"]["side_camera_height"]
    top_cam_height_var.set(f"{top_cam_height:.1f} mm")
    side_cam_height_var.set(f"{side_cam_height:.1f} mm")

def on_closing():
    """Handle application closing"""
    global stop_streaming
    
    # Set flag to stop all threads
    stop_streaming = True
    
    # Stop cameras
    if realsense_camera:
        realsense_camera.stop()
    
    if top_camera:
        top_camera.stop()
    
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

# Main function to create UI and start application
def main():
    global root, side_camera_panel, top_camera_panel, side_processed_panel, top_processed_panel
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
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TFrame', background=BG_COLOR)
    style.configure('TLabelframe', background=BG_COLOR, foreground=TEXT_COLOR)
    style.configure('TLabelframe.Label', background=BG_COLOR, foreground=TEXT_COLOR)
    style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR)
    style.configure('TButton', background=BUTTON_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 12, 'bold'))
    style.map('TButton', background=[('active', HIGHLIGHT_COLOR)])
    
    # Initialize Tkinter variables
    height_adjustment_var = tk.StringVar(value=str(current_settings["calibration"]["base_height"]))
    model_value = tk.StringVar(value=current_settings["selected_model"])
    diameter_value = tk.StringVar(value="0.0 mm")
    thickness_value = tk.StringVar(value="0.0 mm")
    height_value = tk.StringVar(value="0.0 mm")
    tolerance_value = tk.StringVar(value="0.0 mm")
    status_value = tk.StringVar(value="Pending")
    top_result_text = tk.StringVar(value="No data")
    side_result_text = tk.StringVar(value="No data")
    measured_dia_var = tk.StringVar(value="0.0 mm")
    measured_height_var = tk.StringVar(value="0.0 mm")
    result_status_var = tk.StringVar(value="Pending")
    
    # Camera height variables
    top_cam_height_var = tk.StringVar(value=f"{current_settings['calibration']['base_height']:.1f} mm")
    side_cam_height_var = tk.StringVar(value=f"{current_settings['calibration']['side_camera_height']:.1f} mm")
    side_cam_distance_result = tk.StringVar(value="0.0 mm")  # For measured values
    
    # Initialize database counts
    conn = sqlite3.connect('wheel_inspection.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM inspections")
    total_count = cursor.fetchone()[0] or 0
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE test_result = 'OK'")
    passed_count = cursor.fetchone()[0] or 0
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE test_result = 'NOT OK'")
    faulty_count = cursor.fetchone()[0] or 0
    conn.close()
    
    total_count_var = tk.StringVar(value=str(total_count))
    passed_count_var = tk.StringVar(value=str(passed_count))
    faulty_count_var = tk.StringVar(value=str(faulty_count))
    
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
        anchor="center"
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
    
    photo_button = ttk.Button(button_frame, text="Take Photo", command=take_photo, state=tk.DISABLED)
    photo_button.grid(row=0, column=2, padx=3, pady=5)
    
    auto_photo_button = ttk.Button(button_frame, text="Start Auto Capture", command=auto_capture, state=tk.DISABLED)
    auto_photo_button.grid(row=0, column=3, padx=3, pady=5)
    
    # upload_top_button = ttk.Button(button_frame, text="Upload Top View", command=lambda: upload_image(is_top_view=True))
    # upload_top_button.grid(row=0, column=4, padx=3, pady=5)
    
    # upload_side_button = ttk.Button(button_frame, text="Upload Side View", command=lambda: upload_image(is_top_view=False))
    # upload_side_button.grid(row=0, column=5, padx=3, pady=5)
    
    report_button = ttk.Button(button_frame, text="Generate Report", command=open_report_window)
    report_button.grid(row=0, column=4, padx=3, pady=5)
    
    settings_button = ttk.Button(button_frame, text="Settings", command=open_settings_window)
    settings_button.grid(row=0, column=5, padx=3, pady=5)
    
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
    
    ttk.Label(model_frame, text="Tolerance:", font=('Helvetica', 12)).grid(row=3, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(model_frame, textvariable=tolerance_value, font=('Helvetica', 12)).grid(row=3, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(model_frame, text="Top Camera Height:", font=('Helvetica', 12)).grid(row=4, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(model_frame, textvariable=top_cam_height_var, font=('Helvetica', 12)).grid(row=4, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(model_frame, text="Side Camera Distance:", font=('Helvetica', 12)).grid(row=5, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(model_frame, textvariable=side_cam_height_var, font=('Helvetica', 12)).grid(row=5, column=1, sticky="w", padx=5, pady=5)
    
    # Result panel
    result_frame = ttk.LabelFrame(info_frame, text="Result", style="Info.TLabelframe")
    result_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    
    ttk.Label(result_frame, text="Top view:", font=('Helvetica', 12)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=top_result_text, font=('Helvetica', 12)).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(result_frame, text="Side view:", font=('Helvetica', 12)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=side_result_text, font=('Helvetica', 12)).grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(result_frame, text="Measured Diameter:", font=('Helvetica', 12)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=measured_dia_var, font=('Helvetica', 12)).grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(result_frame, text="Measured Height:", font=('Helvetica', 12)).grid(row=3, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=measured_height_var, font=('Helvetica', 12)).grid(row=3, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(result_frame, text="Side Camera Distance:", font=('Helvetica', 12)).grid(row=4, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(result_frame, textvariable=side_cam_distance_result, font=('Helvetica', 12)).grid(row=4, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(result_frame, text="Result:", font=('Helvetica', 12, 'bold')).grid(row=5, column=0, sticky="w", padx=5, pady=5)
    result_status_label = ttk.Label(result_frame, textvariable=result_status_var, font=('Helvetica', 12, 'bold'))
    result_status_label.grid(row=5, column=1, sticky="w", padx=5, pady=5)
    
    # Wheel count panel
    count_frame = ttk.LabelFrame(info_frame, text="Wheel Count", style="Info.TLabelframe")
    count_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=5)
    
    ttk.Label(count_frame, text="Total:", font=('Helvetica', 12)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(count_frame, textvariable=total_count_var, font=('Helvetica', 12)).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(count_frame, text="Pass:", font=('Helvetica', 12)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(count_frame, textvariable=passed_count_var, font=('Helvetica', 12), foreground=PASS_COLOR).grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    ttk.Label(count_frame, text="Fail:", font=('Helvetica', 12)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(count_frame, textvariable=faulty_count_var, font=('Helvetica', 12), foreground=FAIL_COLOR).grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    # Camera frames
    camera_frame = ttk.Frame(main_frame)
    camera_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
    camera_frame.columnconfigure(0, weight=1)
    camera_frame.columnconfigure(1, weight=1)
    camera_frame.rowconfigure(0, weight=0)  # Labels
    camera_frame.rowconfigure(1, weight=1)  # Live feeds
    camera_frame.rowconfigure(2, weight=0)  # Labels
    camera_frame.rowconfigure(3, weight=1)  # Processed
    
    # Labels for camera feeds
    status_label_side = ttk.Label(camera_frame, text="Side Camera (RealSense)", font=("Arial", 12))
    status_label_side.grid(row=0, column=0, sticky="w", padx=5)
    
    status_label_top = ttk.Label(camera_frame, text="Top Camera (Event Camera)", font=("Arial", 12))
    status_label_top.grid(row=0, column=1, sticky="w", padx=5)
    
    # Live camera feed panels
    side_camera_panel = ttk.Label(camera_frame)
    side_camera_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    
    top_camera_panel = ttk.Label(camera_frame)
    top_camera_panel.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
    
    # Labels for processed feeds
    ttk.Label(camera_frame, text="Processed Side View", font=("Arial", 12)).grid(row=2, column=0, sticky="w", padx=5)
    ttk.Label(camera_frame, text="Processed Top View", font=("Arial", 12)).grid(row=2, column=1, sticky="w", padx=5)
    
    # Processed camera feed panels
    side_processed_panel = ttk.Label(camera_frame)
    side_processed_panel.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    
    top_processed_panel = ttk.Label(camera_frame)
    top_processed_panel.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
    
    # Update model parameters
    update_model_parameters()
    
    # Start the mainloop
    root.mainloop()

# Entry point
if __name__ == "__main__":
    main()
