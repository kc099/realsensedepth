import cv2
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image, ImageTk
import math
from torchvision.transforms import functional as F
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from tkcalendar import DateEntry
from reports_window import show_report_window
from settings_window import show_settings_window
from database import init_db, add_inspection
import sys
import serial
import pystray

# Set the environment variable to avoid OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
SAVE_DIR = "captured_frames"
MODEL_PATH = "./maskrcnn_wheel_best.pth"
SCORE_THRESHOLD = 0.5
RATIO_THRESHOLD = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGO_PATH = "company_logo.png"
APP_NAME = "Wheel Inspection System"
DB_FILE = "wheel_inspection.db"

# UI Colors
BG_COLOR = "#96DED1"  # Dark blue-gray
PANEL_BG_COLOR = "#34495E"  # Slightly lighter blue-gray
HIGHLIGHT_COLOR = "#3498DB"  # Bright blue
BUTTON_COLOR = "#2980B9"  # Medium blue
TEXT_COLOR = "#000000"  # Off-white
ACCENT_COLOR = "#E74C3C"  # Red
OK_COLOR = "#2ECC71"
NOK_COLOR = "#E74C3C"

# Font sizes
TITLE_FONT = ('Helvetica', 18, 'bold')
SUBTITLE_FONT = ('Helvetica', 12)
BUTTON_FONT = ('Helvetica', 12, 'bold')
STATUS_FONT = ('Helvetica', 10)
INFO_FONT = ('Helvetica', 11)

# Panel dimensions
SIDE_PANEL_WIDTH = 400
SIDE_PANEL_HEIGHT = 200
TOP_PANEL_WIDTH = 400
TOP_PANEL_HEIGHT = 200

WHEEL_MODELS ={
    (10.0, 13.0): "Model1",
    (13.0, 16.0): "Model2",
    (16.0, 19.0): "Model3",
    (19.0, 22.0): "Model4"
}
# Global variables
settings_win = None
reports_win = None
stop_streaming = False
frame_top = None
frame_side = None
photo_count = 0
auto_capture_active = False
model = None
last_measurements = {"type": None, "radius": 0, "width": 0, "height": 0}

# Initialize settings
current_settings = {
    "selected_model": "",
    "tolerance": 3.0,
    "top_camera_url": "http://192.168.100.50:8080/stream-hd",
    "side_camera_url": "http://192.168.100.51:8080/stream-hd",
    "capture_interval": 5,
    "calibration": {
        "ref_diameter": 466.0,
        "ref_diameter_pixels": 632.62,
        "min_thickness": 77.0,
        "max_thickness": 115.0,
        "base_height": 1075.0,
        "side_camera_height": 800.0
    }
}

# Add this function definition before the main code execution
def detect_24v_signal():
    """Monitor for 24V signal from external device"""
    global stop_streaming, auto_capture_active
    
    if not SERIAL_AVAILABLE:
        print("24V signal detection not available (pyserial not installed)")
        return
        
    try:
        # Try to auto-detect Arduino port
        ports = list(serial.tools.list_ports.comports())
        arduino_port = None
        for port in ports:
            if 'Arduino' in port.description or 'USB' in port.description:
                arduino_port = port.device
                break
        
        if not arduino_port:
            print("No Arduino device found for 24V signal detection")
            return
            
        ser = serial.Serial(arduino_port, 9600, timeout=1)
        print(f"Connected to {arduino_port} for 24V signal detection")
        
        while not stop_streaming:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                if line == "24V_ON":
                    if not auto_capture_active:
                        auto_capture()
                elif line == "24V_OFF":
                    if auto_capture_active:
                        auto_capture()
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error in 24V detection: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

try:
    import serial
    from serial.tools import list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed, 24V signal detection disabled")

# Then when starting the detection thread:
if SERIAL_AVAILABLE:
    threading.Thread(target=detect_24v_signal, daemon=True).start()
else:
    print("24V signal detection not available")

def load_settings():
    global current_settings
    try:
        with open("settings.json", "r") as f:
            loaded_settings = json.load(f)
            current_settings.update(loaded_settings)
    except Exception as e:
        print(f"Error loading settings: {e}")
        save_settings()

def save_settings():
    try:
        with open("settings.json", "w") as f:
            json.dump(current_settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def classify_wheel_model(diameter_mm):
    """Classify wheel and check if it matches selected model with tolerance"""
    if diameter_mm is None or diameter_mm == 0:
        return "Unknown", False
        
    selected_model = current_settings["selected_model"]
    tolerance = current_settings["tolerance"]
    
    # Get model ranges from settings (convert inches to mm)
    model_ranges = {
        "Model 1": (10.0 * 25.4, 13.0 * 25.4),
        "Model 2": (13.0 * 25.4, 16.0 * 25.4),
        "Model 3": (16.0 * 25.4, 19.0 * 25.4),
        "Model 4": (19.0 * 25.4, 22.0 * 25.4)
    }
    
    if selected_model in model_ranges:
        min_diam, max_diam = model_ranges[selected_model]
        if (min_diam - tolerance) <= diameter_mm <= (max_diam + tolerance):
            return selected_model, True
        return selected_model, False
    
    return "Custom Size", False

def fit_circle_least_squares(points):
    pts = np.asarray(points, dtype=np.float64)
    A = np.column_stack((pts[:,0], pts[:,1], np.ones(len(pts))))
    b = (pts[:,0]**2 + pts[:,1]**2)
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx = c[0] / 2.0
    cy = c[1] / 2.0
    r = np.sqrt(cx**2 + cy**2 + c[2])
    return (cx, cy, r)

def correct_for_perspective(measurement_pixels, baseline_height, current_height, image_width_pixels):
    if baseline_height == current_height or baseline_height <= 0 or current_height <= 0:
        return measurement_pixels
    
    scaling_factor = baseline_height / current_height
    corrected_pixels = measurement_pixels * scaling_factor
    return corrected_pixels

def calculate_real_dimensions(measurements, is_top_view=True):
    real_dimensions = {}
    calib = current_settings["calibration"]

    ref_diameter_mm = calib["ref_diameter"]
    ref_diameter_pixels = calib["ref_diameter_pixels"]
    min_thickness = calib["min_thickness"]
    max_thickness = calib["max_thickness"]
    base_height = calib["base_height"]
    
    try:
        current_height = float(height_adjustment_var.get())
    except ValueError:
        current_height = base_height

    if measurements["type"] == "Round" and is_top_view:
        diameter_pixels = measurements["radius"] * 2
        
        if base_height > 0 and current_height > 0:
            diameter_pixels = correct_for_perspective(
                diameter_pixels, base_height, current_height, 1280)
        
        diameter_mm = ref_diameter_mm * (diameter_pixels / ref_diameter_pixels)
        radius_mm = diameter_mm / 2
        
        real_dimensions["diameter_mm"] = diameter_mm
        real_dimensions["radius_mm"] = radius_mm
        model, is_ok = classify_wheel_model(diameter_mm)
        real_dimensions["wheel_model"] = model
        real_dimensions["is_ok"] = is_ok
        
    elif measurements["type"] == "Side" and not is_top_view:
        width_pixels = measurements["width"]
        height_pixels = measurements["height"]
        
        side_cam_height = calib["side_camera_height"]
        if side_cam_height > 0 and current_height > 0:
            width_pixels = correct_for_perspective(width_pixels, side_cam_height, current_height, 1280)
            height_pixels = correct_for_perspective(height_pixels, side_cam_height, current_height, 1280)
        
        if width_pixels >= height_pixels:
            diameter_pixels = width_pixels
            thickness_pixels = height_pixels
        else:
            diameter_pixels = height_pixels
            thickness_pixels = width_pixels
            
        diameter_mm = ref_diameter_mm * (diameter_pixels / ref_diameter_pixels)
        raw_thickness = diameter_mm * (thickness_pixels / diameter_pixels)
        
        if raw_thickness < min_thickness:
            thickness_mm = min_thickness
        elif raw_thickness > max_thickness:
            thickness_mm = max_thickness
        else:
            thickness_mm = raw_thickness
            
        real_dimensions["diameter_mm"] = diameter_mm
        real_dimensions["thickness_mm"] = thickness_mm
        model, is_ok = classify_wheel_model(diameter_mm)
        real_dimensions["wheel_model"] = model
        real_dimensions["is_ok"] = is_ok
    
    return real_dimensions

def update_clock():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clock_label.config(text=now)
    root.after(1000, update_clock)

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
        
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    if dim[0] > w and dim[1] > h:
        return image
        
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def update_display(panel, frame, width, height):
    display_frame = resize_with_aspect_ratio(frame, width=width, height=height)
    cv2_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2_image)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

def process_frame(frame, is_top_view=True):
    global model, last_measurements
    
    if model is None:
        return frame, {"type": None, "radius": 0, "width": 0, "height": 0}
    
    img_height, img_width = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tensor = F.to_tensor(image_pil)
    
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor.to(DEVICE)])[0]
    
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    masks = prediction["masks"].cpu().numpy()
    
    keep = scores > SCORE_THRESHOLD
    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]
    
    output_image = image_rgb.copy()
    measurements = {"type": None, "radius": 0, "width": 0, "height": 0}
    real_dimensions = {}
    
    if len(masks) > 0:
        i = 0
        mask = masks[i]
        mask_binary = mask[0] > 0.5
        mask_uint8 = mask_binary.astype(np.uint8) * 255
        
        mask_colored = np.zeros_like(output_image, dtype=np.uint8)
        mask_colored[mask_binary] = [0, 255, 0]  # Green mask (BGR format)
        output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.5, 0)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            contour_points = np.squeeze(main_contour, axis=1)
            
            if not is_top_view:
                img_height = frame.shape[0]
                estimated_height = estimate_wheel_height(main_contour, img_height)
                height_adjustment_var.set(f"{estimated_height:.1f}")
            
            (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
            cx_ls, cy_ls, r_ls = fit_circle_least_squares(contour_points)
            
            circle_area = math.pi * (radius ** 2)
            contour_area = cv2.contourArea(main_contour)
            shape_ratio = contour_area / circle_area if circle_area > 0 else 0
            
            if shape_ratio > RATIO_THRESHOLD:
                cv2.circle(output_image, (int(cx_ls), int(cy_ls)), int(r_ls), (0, 255, 0), 2)
                cv2.circle(output_image, (int(cx_ls), int(cy_ls)), 2, (0, 0, 255), 3)
                measurements = {"type": "Round", "radius": r_ls, "width": 0, "height": 0}
                
                real_dimensions = calculate_real_dimensions(measurements, is_top_view)
                diameter_mm = real_dimensions.get("diameter_mm", 0)
                wheel_model = real_dimensions.get('wheel_model', 'Unknown')
                is_ok = real_dimensions.get('is_ok', False)
                
                status_color = (0, 255, 0) if is_ok else (0, 0, 255)
                text = f"Round: D={diameter_mm:.1f}mm, Model: {wheel_model}"
                cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(output_image, "OK" if is_ok else "NOT OK", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            else:
                rot_rect = cv2.minAreaRect(main_contour)
                box_pts = cv2.boxPoints(rot_rect)
                box_pts = np.array(box_pts, dtype=np.intp)
                cv2.drawContours(output_image, [box_pts], 0, (255, 0, 0), 2)
                
                (center_x, center_y), (w, h), angle = rot_rect
                measurements = {"type": "Side", "radius": 0, "width": w, "height": h}
                
                real_dimensions = calculate_real_dimensions(measurements, is_top_view)
                diameter_mm = real_dimensions.get("diameter_mm", 0)
                thickness_mm = real_dimensions.get("thickness_mm", 0)
                wheel_model = real_dimensions.get('wheel_model', 'Unknown')
                is_ok = real_dimensions.get('is_ok', False)
                
                status_color = (0, 255, 0) if is_ok else (0, 0, 255)
                text = f"Side: D={diameter_mm:.1f}mm, T={thickness_mm:.1f}mm, Model: {wheel_model}"
                cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(output_image, "OK" if is_ok else "NOT OK", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    combined_measurements = {**measurements, **real_dimensions}
    last_measurements = combined_measurements
    
    return output_image_bgr, combined_measurements

def stream_camera(url, camera_type, retry_interval=3):
    global stop_streaming, frame_top, frame_side
    
    cap = None
    status_text = f"Connecting to {camera_type} camera..."
    
    if camera_type == "Top":
        status_label_top.config(text=status_text)
    else:
        status_label_side.config(text=status_text)
    
    while not stop_streaming:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                status_text = f"Error: Could not open {camera_type} stream. Retrying..."
                if camera_type == "Top":
                    status_label_top.config(text=status_text)
                else:
                    status_label_side.config(text=status_text)
                time.sleep(retry_interval)
                continue
            
            status_text = f"{camera_type} camera connected!"
            if camera_type == "Top":
                status_label_top.config(text=status_text)
            else:
                status_label_side.config(text=status_text)

        ret, frame = cap.read()
        if not ret:
            status_text = f"Error: Lost {camera_type} stream. Reconnecting..."
            if camera_type == "Top":
                status_label_top.config(text=status_text)
            else:
                status_label_side.config(text=status_text)
            cap.release()
            cap = None
            time.sleep(retry_interval)
            continue

        if camera_type == "Top":
            frame_top = frame.copy()
            display_frame = resize_with_aspect_ratio(frame, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)
            update_display(top_panel, display_frame, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
        else:
            frame_side = frame.copy()
            display_frame = resize_with_aspect_ratio(frame, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)
            update_display(side_panel, display_frame, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_streaming = True
            status_label_top.config(text="Top camera streaming stopped.")
            status_label_side.config(text="Side camera streaming stopped.")
            status_label_main.config(text="Streaming stopped.")
            break

    if cap is not None:
        cap.release()
    
    status_text = f"{camera_type} stream closed."
    if camera_type == "Top":
        status_label_top.config(text=status_text)
    else:
        status_label_side.config(text=status_text)

def start_streaming():
    global stop_streaming
    stop_streaming = False
    
    top_url = current_settings["top_camera_url"]
    side_url = current_settings["side_camera_url"]
    
    if not top_url or not side_url:
        messagebox.showerror("Error", "Please configure camera URLs in Settings")
        return
    
    top_thread = threading.Thread(target=stream_camera, args=(top_url, "Top"))
    top_thread.daemon = True
    top_thread.start()
    
    side_thread = threading.Thread(target=stream_camera, args=(side_url, "Side"))
    side_thread.daemon = True
    side_thread.start()
    
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    photo_button.config(state=tk.NORMAL)
    auto_photo_button.config(state=tk.NORMAL)

def stop_streaming_func():
    global status_label_top, status_label_side, status_label_main
    global stop_streaming, auto_capture_active
    stop_streaming = True
    status_label_top.config(text="Top camera streaming stopped.")
    status_label_side.config(text="Side camera streaming stopped.")
    status_label_main.config(text="Streaming stopped.")
    auto_capture_active = False
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    photo_button.config(state=tk.DISABLED)
    auto_photo_button.config(state=tk.DISABLED)

def take_photo():
    global frame_top, frame_side, photo_count
    
    if frame_top is None or frame_side is None:
        messagebox.showerror("Error", "No frames available to process. Start the streams first!")
        return
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    photo_count += 1
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    filename_top = os.path.join(SAVE_DIR, f"top_{photo_count}_{timestamp}.png")
    filename_side = os.path.join(SAVE_DIR, f"side_{photo_count}_{timestamp}.png")
    cv2.imwrite(filename_top, frame_top)
    cv2.imwrite(filename_side, frame_side)
    
    # Process side view first to get height
    processed_side, measurements_side = process_frame(frame_side, is_top_view=False)
    update_display(side_processed_panel, processed_side, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
    
    # Then process top view with height adjustment
    processed_top, measurements_top = process_frame(frame_top, is_top_view=True)
    update_display(top_processed_panel, processed_top, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
    
    update_measurement_display(measurements_top, measurements_side)

    part_no = f"INDIP {timestamp.split('_')[0]} {photo_count}"
    model_type = measurements_top.get("wheel_model", "Unknown")
    diameter_mm = measurements_top.get("diameter_mm", 0)
    thickness_mm = measurements_side.get("thickness_mm", 0)
    height_mm = float(height_adjustment_var.get()) if height_adjustment_var.get() else 0
    test_result = "OK" if (measurements_top.get("is_ok", False) and 
                          measurements_side.get("is_ok", False)) else "NOT OK"
    
    add_inspection(part_no, model_type, diameter_mm, thickness_mm, 
                  height_mm, test_result, filename_top, filename_side)
    
    status_label_main.config(text=f"Captured and processed frames {photo_count} at {timestamp}")

def update_measurement_display(measurements_top, measurements_side):
    if measurements_top["type"] == "Round":
        top_diameter_mm = measurements_top.get("diameter_mm", 0)
        top_model = measurements_top.get("wheel_model", "Unknown")
        top_is_ok = measurements_top.get("is_ok", False)
        
        diameter_value.set(f"{top_diameter_mm:.2f} mm")
        model_value.set(top_model)
        status_value.set("OK" if top_is_ok else "NOT OK")
        status_label.config(foreground=OK_COLOR if top_is_ok else NOK_COLOR)
    
    if measurements_side["type"] == "Side":
        side_thickness_mm = measurements_side.get("thickness_mm", 0)
        height_mm = float(height_adjustment_var.get()) if height_adjustment_var.get() else 0
        
        thickness_value.set(f"{side_thickness_mm:.2f} mm")
        height_value.set(f"{height_mm:.1f} mm")
        
        tolerance = current_settings.get("tolerance", 3)
        tolerance_value.set(f"{tolerance} mm")

def auto_capture():
    global auto_capture_active
    
    if auto_capture_active:
        auto_capture_active = False
        auto_photo_button.config(text="Start Auto Capture")
        status_label_main.config(text="Auto capture stopped")
    else:
        try:
            interval = float(current_settings["capture_interval"])
            if interval <= 0:
                raise ValueError("Interval must be positive")
            
            auto_capture_active = True
            auto_photo_button.config(text="Stop Auto Capture")
            status_label_main.config(text=f"Auto capturing every {interval} seconds")
            
            threading.Thread(target=auto_capture_thread, args=(interval,), daemon=True).start()
        except ValueError as e:
            messagebox.showerror("Invalid Interval", f"Error: {e}")

def auto_capture_thread(interval):
    global auto_capture_active
    
    while auto_capture_active and not stop_streaming:
        take_photo()
        time.sleep(interval)

def estimate_wheel_height(contour, image_height):
    x, y, w, h = cv2.boundingRect(contour)
    center_y = y + h/2
    relative_pos = center_y / image_height
    
    base_height = current_settings["calibration"]["base_height"]
    side_cam_height = current_settings["calibration"]["side_camera_height"]
    max_height = side_cam_height * 1.2
    
    estimated_height = side_cam_height + (1 - relative_pos) * (max_height - side_cam_height)
    return estimated_height

def upload_image(is_top_view=True):
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
        
        if is_top_view:
            frame_top = frame.copy()
            display_frame = resize_with_aspect_ratio(frame, width=TOP_PANEL_WIDTH, height=TOP_PANEL_HEIGHT)
            update_display(top_panel, display_frame, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
            
            processed_frame, measurements = process_frame(frame, is_top_view=True)
            update_display(top_processed_panel, processed_frame, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
            
            update_measurement_display(measurements, {})
        else:
            frame_side = frame.copy()
            display_frame = resize_with_aspect_ratio(frame, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)
            update_display(side_panel, display_frame, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            
            processed_frame, measurements = process_frame(frame, is_top_view=False)
            update_display(side_processed_panel, processed_frame, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            
            update_measurement_display({}, measurements)
        
        status_label_main.config(text=f"Processed uploaded {'top' if is_top_view else 'side'} view image: {os.path.basename(file_path)}")
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error processing image: {e}")
def on_closing():
    global stop_streaming, auto_capture_active, ser
    
    # Set flags to stop all running processes
    stop_streaming = True
    status_label_top.config(text="Top camera streaming stopped.")
    status_label_side.config(text="Side camera streaming stopped.")
    status_label_main.config(text="Streaming stopped.")
    auto_capture_active = False
    
    # Close serial connection if it exists
    if 'ser' in globals() and ser is not None:
        try:
            ser.close()
            print("Serial port closed")
        except Exception as e:
            print(f"Error closing serial port: {e}")
    
    # Release camera resources if they exist
    if 'cap_top' in globals() and cap_top is not None:
        cap_top.release()
    if 'cap_side' in globals() and cap_side is not None:
        cap_side.release()
    
    # Destroy the window
    root.destroy()
    print("Application closed properly")

    # Set the close handler
    root.protocol("WM_DELETE_WINDOW", on_closing)
def update_model_parameters():
    selected_model = current_settings["selected_model"]
    model_value.set(selected_model)
    
    # Update expected values based on model
    if selected_model in WHEEL_MODELS:
        min_diam, max_diam = WHEEL_MODELS[selected_model]
        expected_diameter = f"{min_diam:.1f}-{max_diam:.1f} mm"
        diameter_value.set(expected_diameter)
        
        # Get thickness from calibration settings
        min_thickness = current_settings["calibration"].get("min_thickness", 0)
        max_thickness = current_settings["calibration"].get("max_thickness", 0)
        expected_thickness = f"{min_thickness:.1f}-{max_thickness:.1f} mm"
        height_value.set(expected_thickness)
    
    # Update height and tolerance
    height_adjustment_var.set(str(current_settings["calibration"]["base_height"]))
    tolerance_value.set(f"{current_settings['tolerance']} mm")

def open_settings_window():
    global settings_win
    if settings_win is None or not settings_win.winfo_exists():
        # settings_win = tk.Toplevel(root)
        show_settings_window(settings_win, current_settings, WHEEL_MODELS, update_model_parameters)
    else:
        settings_win.lift()

def open_report_window():
    global reports_win
    if reports_win is None or not reports_win.winfo_exists():
        # reports_win = tk.Toplevel(root)
        show_report_window(reports_win)
    else:
        reports_win.lift()

# Main application window
root = tk.Tk()
root.title("Wheel Inspection System")
root.geometry("1400x900")
root.configure(background=BG_COLOR)
root.protocol("WM_DELETE_WINDOW", on_closing)
# ref_height_var = tk.StringVar(value="0")       # Reference height from camera in mm
# height_adjustment_var = tk.StringVar(value="0") # Current height adjustment in mm

# Configure style
style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background=BG_COLOR)
style.configure('TLabelframe', background=BG_COLOR, foreground=TEXT_COLOR)
style.configure('TLabelframe.Label', background=BG_COLOR, foreground=TEXT_COLOR)
style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR)
style.configure('TButton', background=BUTTON_COLOR, foreground=TEXT_COLOR, font=BUTTON_FONT)
style.map('TButton', background=[('active', HIGHLIGHT_COLOR)])

# Initialize Tkinter variables AFTER root window is created
height_adjustment_var = tk.StringVar(value=str(current_settings["calibration"]["base_height"]))
model_value = tk.StringVar(value=current_settings["selected_model"])
diameter_value = tk.StringVar(value="0.0 mm")
thickness_value = tk.StringVar(value="0.0 mm")
height_value = tk.StringVar(value="0.0 mm")
tolerance_value = tk.StringVar(value=f"{current_settings['tolerance']} mm")
status_value = tk.StringVar(value="Pending")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
main_frame.columnconfigure(2, weight=1)
main_frame.rowconfigure(0, weight=0)  # Logo row
main_frame.rowconfigure(1, weight=0)  # Button row
main_frame.rowconfigure(2, weight=0)  # Status row
main_frame.rowconfigure(3, weight=0)  # Measurement/result row
main_frame.rowconfigure(4, weight=1)  # Image frame row - gets all extra space

logo_frame = ttk.Frame(main_frame)
logo_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
logo_frame.columnconfigure(0, weight=0)  # Logo
logo_frame.columnconfigure(1, weight=2)  # TAURUS
logo_frame.columnconfigure(2, weight=0)  # WHEEL DIMENSION ANALYZER
# logo_frame.columnconfigure(3, weight=0)  # Clock

logo_path = os.path.join(os.path.dirname(__file__), "Taurus_logo1.png")
logo_img = Image.open(logo_path)
logo_img = logo_img.resize((100, 100), Image.Resampling.LANCZOS)
logo_imgtk = ImageTk.PhotoImage(image=logo_img)

logo_label = ttk.Label(logo_frame, image=logo_imgtk, background=BG_COLOR)
logo_label.image = logo_imgtk
logo_label.grid(row=0, column=0, sticky="w", padx=10)

# Company name frame (TAURUS)
# company_frame = ttk.Frame(logo_frame, style='Header.TFrame')
# company_frame.grid(row=0, column=1, sticky="w")
# ttk.Label(
#     company_frame,
#     text="TAURUS",
#     font=("Arial", 30, "bold"),
#     foreground='red'
# ).grid(row=0, column=0, padx=5)

# App name (centered)
app_name_label = ttk.Label(
    logo_frame,
    text="WHEEL DIMENSION ANALYZER",
    font=("Arial", 26, "bold"),
    anchor="center"
)
app_name_label.grid(row=0, column=1, sticky="ew")

# Clock label (right)
clock_label = ttk.Label(
    logo_frame,
    font=('Helvetica', 22, 'bold'),  # Larger font    
    foreground='black'
)
clock_label.grid(row=0, column=2, sticky="e", padx=10)

# Call clock update
update_clock()

# # URL input frame
# url_frame = ttk.LabelFrame(main_frame, text="Stream Settings")
# url_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
# url_frame.columnconfigure(1, weight=1)

# url_label = ttk.Label(url_frame, text="Stream URL:")
# url_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
# url_entry = ttk.Entry(url_frame, width=30)
# url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
# url_entry.insert(0, "http://192.168.100.50:8080/stream-hd")

# interval_label = ttk.Label(url_frame, text="Capture Interval (s):")
# interval_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
# interval_entry = ttk.Entry(url_frame, width=30)
# interval_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
# interval_entry.insert(0, "5")

# Buttons frame
button_frame = ttk.Frame(main_frame)
button_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)
button_frame.columnconfigure(2, weight=1)
button_frame.columnconfigure(3, weight=1)
button_frame.columnconfigure(4, weight=1)
button_frame.columnconfigure(5, weight=1)
button_frame.columnconfigure(6, weight=1)
button_frame.columnconfigure(7, weight=1)


start_button = ttk.Button(button_frame, text="Start Streaming", command=start_streaming, style='TButton')
start_button.grid(row=0, column=0, padx=5, pady=5)

stop_button = ttk.Button(button_frame, text="Stop Streaming", command=stop_streaming_func, state=tk.DISABLED, style='TButton')
stop_button.grid(row=0, column=1, padx=5, pady=5)

photo_button = ttk.Button(button_frame, text="Take Photo", command=take_photo, state=tk.DISABLED, style='TButton')
photo_button.grid(row=0, column=2, padx=5, pady=5)

auto_photo_button = ttk.Button(button_frame, text="Start Auto Capture", command=auto_capture, state=tk.DISABLED, style='TButton')
auto_photo_button.grid(row=0, column=3, padx=5, pady=5)

# upload_button = ttk.Button(button_frame, text="Upload Image", command=upload_image, style='Accent.TButton')
# upload_button.grid(row=0, column=4, padx=5, pady=5)

upload_top_button = ttk.Button(button_frame, text="Upload Top View", command=lambda: upload_image(is_top_view=True), style='TButton')
upload_top_button.grid(row=0, column=4, padx=5, pady=5)

upload_side_button = ttk.Button(button_frame, text="Upload Side View", command=lambda: upload_image(is_top_view=False), style='TButton')
upload_side_button.grid(row=0, column=5, padx=5, pady=5)

report_button = ttk.Button(button_frame, text="Generate Report", command=open_report_window, style='TButton')
report_button.grid(row=0, column=6, padx=5, pady=5)

settings_button = ttk.Button(button_frame, text="Settings", command=open_settings_window, style='TButton')
settings_button.grid(row=0, column=7, padx=5, pady=5)

# Status label
status_label = ttk.Label(main_frame, text="Ready. Click 'Start Streaming' or 'Upload Image' to begin.", font=("Arial", 14, "italic"))
status_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)

# Status label
status_label_main = ttk.Label(main_frame, text="Status: Ready", font=("Arial", 14, "italic"))
status_label_main.grid(row=2, column=1,  sticky=tk.W, padx=5, pady=5)

# ttk.Label(main_frame, text="Height (mm):").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
# height_entry = ttk.Entry(main_frame, textvariable=height_adjustment_var, width=10)
# height_entry.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)

# Measurement display
style.configure("Custom.TLabelframe.Label", font=('Helvetica', 13, 'bold'))
measurement_frame = ttk.LabelFrame(main_frame, text="Model Data", style="Custom.TLabelframe")
measurement_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(50, 25), pady=5)

model_info_frame = ttk.Frame(measurement_frame)
model_info_frame.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(model_info_frame, text="Model:", font=('Helvetica', 12)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(model_info_frame, textvariable=model_value, font=('Helvetica', 12)).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

height_frame = ttk.Frame(measurement_frame)
height_frame.grid(row=1, columnspan=2, sticky=tk.W, padx=5)
ttk.Label(height_frame, text="Height:", font=('Helvetica', 12)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(height_frame, textvariable=height_value, font=('Helvetica', 12)).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

diameter_frame = ttk.Frame(measurement_frame)
diameter_frame.grid(row=2, columnspan=2, sticky=tk.W, padx=5)
ttk.Label(diameter_frame, text="Diameter:", font=('Helvetica', 12)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(diameter_frame, textvariable=diameter_value, font=('Helvetica', 12)).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

# thickness_frame = ttk.Frame(measurement_frame)
# thickness_frame.grid(row=3, columnspan=2, sticky=tk.W, padx=5)
# ttk.Label(thickness_frame, text="Thickness:", font=('Helvetica', 12)).grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
# ttk.Label(thickness_frame, textvariable=thickness_value, font=('Helvetica', 12)).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

tolerance_frame = ttk.Frame(measurement_frame)
tolerance_frame.grid(row=3, columnspan=2, sticky=tk.W, padx=5)
ttk.Label(tolerance_frame, text="Tolerance:", font=('Helvetica', 12)).grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(tolerance_frame, textvariable=tolerance_value, font=('Helvetica', 12)).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

height_frame1 = ttk.Frame(measurement_frame)
height_frame1.grid(row=4, columnspan=2, sticky=tk.W, padx=5)
ttk.Label(height_frame1, text="Top Camera to Base height: (mm):", font=('Helvetica', 12)).grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(height_frame1, textvariable=height_adjustment_var, font=('Helvetica', 12)).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
# status_frame = ttk.Frame(measurement_frame)
# status_frame.grid(row=5, columnspan=2, sticky=tk.W, padx=5)
# ttk.Label(status_frame, text="Status:", font=('Helvetica', 12, 'bold')).grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
# status_label = ttk.Label(status_frame, textvariable=status_value, font=('Helvetica', 12, 'bold'))
# status_label.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)

# Get counts from database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM inspections")
total_count = cursor.fetchone()[0] or 0
cursor.execute("SELECT COUNT(*) FROM inspections WHERE test_result = 'OK'")
passed_count = cursor.fetchone()[0] or 0
cursor.execute("SELECT COUNT(*) FROM inspections WHERE test_result = 'NOT OK'")
faulty_count = cursor.fetchone()[0] or 0
conn.close()

def update_model_parameters():
    selected_model = current_settings["selected_model"]
    model_value.set(selected_model)
    
    # Update expected values based on model
    if selected_model in WHEEL_MODELS:
        min_diam, max_diam = WHEEL_MODELS[selected_model]
        expected_diameter = f"{min_diam:.1f}-{max_diam:.1f} mm"
        diameter_value.set(expected_diameter)
        
        # Get thickness from calibration settings
        min_thickness = current_settings["calibration"].get("min_thickness", 0)
        max_thickness = current_settings["calibration"].get("max_thickness", 0)
        expected_thickness = f"{min_thickness:.1f}-{max_thickness:.1f} mm"
        height_value.set(expected_thickness)
    
    # Update height and tolerance
    height_adjustment_var.set(str(current_settings["calibration"]["base_height"]))
    tolerance_value.set(f"{current_settings['tolerance']} mm")

# Call this function after loading settings
load_settings()
update_model_parameters()
# totalcount_frame = ttk.Frame(measurement_frame)
# totalcount_frame.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
# ttk.Label(totalcount_frame, text="Total count:", font=('Helvetica', 12)).grid(row=0, column=0, sticky=tk.W, padx=5)
# total_count_var = tk.StringVar(value=str(total_count))
# ttk.Label(totalcount_frame, textvariable=total_count_var, font=('Helvetica', 12)).grid(row=0, column=1, sticky=tk.W, padx=5)

# passcount_frame = ttk.Frame(measurement_frame)
# passcount_frame.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
# ttk.Label(passcount_frame, text="Passed:", font=('Helvetica', 12), foreground=OK_COLOR).grid(row=0, column=0, sticky=tk.W, padx=5)
# passed_count_var = tk.StringVar(value=str(passed_count))
# ttk.Label(passcount_frame, textvariable=passed_count_var, font=('Helvetica', 12), foreground=OK_COLOR).grid(row=0, column=1, sticky=tk.W, padx=5)

# faultcount_frame = ttk.Frame(measurement_frame)
# faultcount_frame.grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)
# ttk.Label(faultcount_frame, text="Faulty:", font=('Helvetica', 12), foreground=NOK_COLOR).grid(row=0, column=0, sticky=tk.W, padx=5)
# faulty_count_var = tk.StringVar(value=str(faulty_count))
# ttk.Label(faultcount_frame, textvariable=faulty_count_var, font=('Helvetica', 12), foreground=NOK_COLOR).grid(row=0, column=1, sticky=tk.W, padx=5)


# Right panel - Result
style.configure("Result.TLabelframe.Label", font=('Helvetica', 13, 'bold'))
result_frame = ttk.LabelFrame(main_frame, text="Result", style="Result.TLabelframe")
result_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(25, 25), pady=5)

# Result variables
result_status_var = tk.StringVar(value="Pending")
result_color_var = tk.StringVar(value=BG_COLOR)  # Default background

# Top view result
top_result_frame = ttk.Frame(result_frame)
top_result_frame.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

ttk.Label(top_result_frame, text="Top view:", font=('Helvetica', 12)).grid(row=1, column=0, sticky=tk.W, padx=5)
top_result_text = tk.StringVar(value="No data")
ttk.Label(top_result_frame, textvariable=top_result_text, font=('Helvetica', 12)).grid(row=1, column=1, sticky=tk.W, padx=5)

# Diameter result
dia_result_frame = ttk.Frame(result_frame)
dia_result_frame.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)

ttk.Label(dia_result_frame, text="Diameter:", font=('Helvetica', 12)).grid(row=3, column=0, sticky=tk.W, padx=5)
measured_dia_var = tk.StringVar(value="0.0 mm")
ttk.Label(dia_result_frame, textvariable=measured_dia_var, font=('Helvetica', 12)).grid(row=3, column=1, sticky=tk.W, padx=5)

# Side view result
side_result_frame = ttk.Frame(result_frame)
side_result_frame.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)

ttk.Label(side_result_frame, text="Side view:", font=('Helvetica', 12)).grid(row=5, column=0, sticky=tk.W, padx=5)
side_result_text = tk.StringVar(value="No data")
ttk.Label(side_result_frame, textvariable=side_result_text, font=('Helvetica', 12)).grid(row=5, column=1, sticky=tk.W, padx=5)

# Height result
height_result_frame = ttk.Frame(result_frame)
height_result_frame.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)

ttk.Label(height_result_frame, text="Height:", font=('Helvetica', 12)).grid(row=7, column=0, sticky=tk.W, padx=5)
measured_height_var = tk.StringVar(value="0.0 mm")
ttk.Label(height_result_frame, textvariable=measured_height_var, font=('Helvetica', 12)).grid(row=7, column=1, sticky=tk.W, padx=5)

# Status
status_result_frame = ttk.Frame(result_frame)
status_result_frame.grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)

ttk.Label(status_result_frame, text="Status:", font=('Helvetica', 14, 'bold')).grid(row=9, column=0, sticky=tk.W, padx=5)
result_status_label = ttk.Label(status_result_frame, textvariable=result_status_var, font=('Helvetica', 14, 'bold'))
result_status_label.grid(row=9, column=1, sticky=tk.W, padx=5)

# Configure result frame background based on status
def update_result_frame():
    if result_status_var.get() == "OK":
        result_frame.configure(style='ResultOK.TLabelframe')
        result_status_label.configure(foreground='white')
    else:
        result_frame.configure(style='ResultNOK.TLabelframe')
        result_status_label.configure(foreground='white')

style.configure('ResultOK.TLabelframe', background=OK_COLOR, foreground='white')
style.configure('ResultNOK.TLabelframe', background=NOK_COLOR, foreground='white')

style.configure("Result.TLabelframe.Label", font=('Helvetica', 13, 'bold'))
wheel_frame = ttk.LabelFrame(main_frame, text="Wheel Count", style="Result.TLabelframe")
wheel_frame.grid(row=3, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(25, 50), pady=5)

totalcount_frame = ttk.Frame(wheel_frame)
totalcount_frame.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(totalcount_frame, text="Total count:", font=('Helvetica', 12)).grid(row=0, column=0, sticky=tk.W, padx=5)
total_count_var = tk.StringVar(value=str(total_count))
ttk.Label(totalcount_frame, textvariable=total_count_var, font=('Helvetica', 12)).grid(row=0, column=1, sticky=tk.W, padx=5)

passcount_frame = ttk.Frame(wheel_frame)
passcount_frame.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(passcount_frame, text="Passed:", font=('Helvetica', 12), foreground=OK_COLOR).grid(row=0, column=0, sticky=tk.W, padx=5)
passed_count_var = tk.StringVar(value=str(passed_count))
ttk.Label(passcount_frame, textvariable=passed_count_var, font=('Helvetica', 12), foreground=OK_COLOR).grid(row=0, column=1, sticky=tk.W, padx=5)

faultcount_frame = ttk.Frame(wheel_frame)
faultcount_frame.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
ttk.Label(faultcount_frame, text="Faulty:", font=('Helvetica', 12), foreground=NOK_COLOR).grid(row=0, column=0, sticky=tk.W, padx=5)
faulty_count_var = tk.StringVar(value=str(faulty_count))
ttk.Label(faultcount_frame, textvariable=faulty_count_var, font=('Helvetica', 12), foreground=NOK_COLOR).grid(row=0, column=1, sticky=tk.W, padx=5)

# Main content area - image panels occupy most of the screen
image_frame = ttk.Frame(main_frame)
image_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
image_frame.columnconfigure(0, weight=1)  # Side view column
image_frame.columnconfigure(1, weight=1)  # Top view column
image_frame.rowconfigure(0, weight=1)     # Main content row

# Side view panel (left side)
style.configure("image.TLabelframe.Label", font=('Helvetica', 13, 'bold'))
side_camera_frame = ttk.LabelFrame(image_frame, text="Side View", style="image.TLabelframe")
side_camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
side_camera_frame.columnconfigure(0, weight=1)
side_camera_frame.rowconfigure(0, weight=1)  # Container frame
side_camera_frame.rowconfigure(1, weight=0)  # Status label

side_view_container = ttk.Frame(side_camera_frame)
side_view_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
side_view_container.columnconfigure(0, weight=1)  # Original frame
side_view_container.columnconfigure(1, weight=1)  # Processed frame
side_view_container.rowconfigure(0, weight=1)

# Original side view
style.configure("image.TLabelframe.Label", font=('Helvetica', 13, 'bold'))
side_original_frame = ttk.LabelFrame(side_view_container, text="Original")
side_original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2)
side_original_frame.config(width=200, height=400)
side_original_frame.grid_propagate(False)
side_panel = ttk.Label(side_original_frame)
side_panel.pack(fill=tk.BOTH, expand=True)

# Processed side view
side_processed_frame = ttk.LabelFrame(side_view_container, text="Processed")
side_processed_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2)
side_processed_frame.config(width=200, height=400)
side_processed_frame.grid_propagate(False)
side_processed_panel = ttk.Label(side_processed_frame)
side_processed_panel.pack(fill=tk.BOTH, expand=True)

# Side view status
status_label_side = ttk.Label(side_camera_frame, text="Side camera not connected", 
                             font=("Arial", 11, "bold"))
status_label_side.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

# Top view panel (right side)
top_camera_frame = ttk.LabelFrame(image_frame, text="Top View", style="image.TLabelframe")
top_camera_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
top_camera_frame.columnconfigure(0, weight=1)
top_camera_frame.rowconfigure(0, weight=1)  # Container frame
top_camera_frame.rowconfigure(1, weight=0)  # Status label

top_view_container = ttk.Frame(top_camera_frame)
top_view_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
top_view_container.columnconfigure(0, weight=1)  # Original frame
top_view_container.columnconfigure(1, weight=1)  # Processed frame
top_view_container.rowconfigure(0, weight=1)

# Original top view
top_original_frame = ttk.LabelFrame(top_view_container, text="Original")
top_original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2)
top_original_frame.config(width=200, height=400)
top_original_frame.grid_propagate(False)
top_panel = ttk.Label(top_original_frame)
top_panel.pack(fill=tk.BOTH, expand=True)

# Processed top view
top_processed_frame = ttk.LabelFrame(top_view_container, text="Processed")
top_processed_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2)
top_processed_frame.config(width=200, height=400)
top_processed_frame.grid_propagate(False)
top_processed_panel = ttk.Label(top_processed_frame)
top_processed_panel.pack(fill=tk.BOTH, expand=True)

# Top view status
status_label_top = ttk.Label(top_camera_frame, text="Top camera not connected", 
                           font=("Arial", 11, "bold"))
status_label_top.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

# Initialize database
init_db()

# Load settings
load_settings()

# Load model
def load_model_thread():
    global model
    model = get_model_instance_segmentation(num_classes=2)
    model.to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        status_label_main.config(text="Model loaded successfully")
    except Exception as e:
        messagebox.showerror("Model Error", f"Failed to load model: {e}")
        status_label_main.config(text="Error loading model")

threading.Thread(target=load_model_thread, daemon=True).start()

# Run the application
root.mainloop()
class WheelInspector:
    def __init__(self):
        # Initialize cameras and USB
        self.top_cam_url = "http://192.168.100.50:8080/stream-hd"
        self.side_cam_url = "http://192.168.100.51:8080/stream-hd"
        self.serial_port = self.detect_serial_port()
        self.baudrate = 9600
        self.running = False
        if not self.serial_port:
            print("Warning: No serial port detected for 24V signal input")
        # Tray icon setup
        self.icon = pystray.Icon(
            "Wheel Inspector",
            icon=self.create_tray_icon(),
            menu=pystray.Menu(
                pystray.MenuItem("Exit", self.exit_app)
            )
        )

    def create_tray_icon(self):
        # Generate a simple tray icon (replace with your logo)
        img = Image.new('RGB', (64, 64), color='blue')
        return img

    def check_cameras(self):
        """Verify both cameras are streaming"""
        cap_top = cv2.VideoCapture(self.top_cam_url)
        cap_side = cv2.VideoCapture(self.side_cam_url)
        ret_top, _ = cap_top.read()
        ret_side, _ = cap_side.read()
        cap_top.release()
        cap_side.release()
        return ret_top and ret_side

    def capture_wheel(self):
        """Capture and process wheel images"""
        print("24V Signal received - Capturing wheel...")
        cap_top = cv2.VideoCapture(self.top_cam_url)
        cap_side = cv2.VideoCapture(self.side_cam_url)
        
        ret_top, frame_top = cap_top.read()
        ret_side, frame_side = cap_side.read()
        
        if ret_top and ret_side:
            # Save/process images (your existing logic)
            cv2.imwrite(f"top_{time.time()}.png", frame_top)
            cv2.imwrite(f"side_{time.time()}.png", frame_side)
            print("Capture successful!")
        
        cap_top.release()
        cap_side.release()

    def detect_serial_port(self):
        """Try to automatically detect the correct serial port"""
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if 'Arduino' in port.description or 'USB' in port.description:
                    return port.device
        except Exception as e:
            print(f"Error detecting serial ports: {e}")
        return None
        
    def monitor_usb(self):
        """Monitor USB for 24V signals with robust error handling"""
        while self.running:
            if not self.serial_port:
                time.sleep(5)
                continue
                
            try:
                with serial.Serial(self.serial_port, self.baudrate, timeout=1) as ser:
                    print(f"Connected to {self.serial_port}")
                    while self.running:
                        try:
                            if ser.in_waiting > 0:
                                signal = ser.readline().decode().strip()
                                if signal == "WHEEL_PLACED":
                                    if self.check_cameras():
                                        self.capture_wheel()
                        except UnicodeDecodeError:
                            continue  # Skip malformed data
                        time.sleep(0.1)
            except serial.SerialException as e:
                print(f"Serial error: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(5)
    def run(self):
        """Start the application"""
        self.running = True
        # Start USB monitoring in background
        usb_thread = threading.Thread(target=self.monitor_usb, daemon=True)
        usb_thread.start()
        # Start tray icon
        self.icon.run()

    def exit_app(self):
        """Cleanup on exit"""
        self.running = False
        self.icon.stop()
        sys.exit(0)

if __name__ == "__main__":
    inspector = WheelInspector()
    inspector.run()
