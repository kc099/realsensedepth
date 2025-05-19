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
from datetime import datetime
import glob
import csv
from reports_window import show_report_window
from settings_window import show_settings_window
from database import init_db, add_inspection
import sys
import queue
import traceback

# Set the environment variable to avoid OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
SAVE_DIR = "captured_frames"
MODEL_PATH = "./maskrcnn_wheel_best.pth"
SCORE_THRESHOLD = 0.5
RATIO_THRESHOLD = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DB_FILE = "wheel_inspection.db"
frame_queue = queue.Queue(maxsize=2)
streaming_active = threading.Event()

# UI Colors
BG_COLOR = "#96DED1"  # Light blue-green
HIGHLIGHT_COLOR = "#3498DB"  # Bright blue
BUTTON_COLOR = "#2980B9"  # Medium blue
TEXT_COLOR = "#000000"  # Black
OK_COLOR = "#2ECC71"  # Green
NOK_COLOR = "#E74C3C"  # Red

# Panel dimensions
SIDE_PANEL_WIDTH = 400
SIDE_PANEL_HEIGHT = 300
TOP_PANEL_WIDTH = 400
TOP_PANEL_HEIGHT = 300

# Wheel models data - enhanced to include tolerance per model
WHEEL_MODELS = {
    "10-13": {"min_dia": 10, "max_dia": 13, "height": 70, "tolerance": 3.0},
    "13-16": {"min_dia": 13, "max_dia": 16, "height": 75, "tolerance": 3.0},
    "16-19": {"min_dia": 16, "max_dia": 19, "height": 77, "tolerance": 3.0},
    "19-22": {"min_dia": 19, "max_dia": 22, "height": 115, "tolerance": 3.0}
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
depth_data = None
realsense_pipeline = None
realsense_align = None
rgb_folder_path = None
depth_folder_path = None
current_depth_image = None
metadata_file_path = None

# Initialize settings with enhanced structure
current_settings = {
    "selected_model": "10-13",
    "tolerance": 3.0,  # Default tolerance, can be overridden by model-specific
    "top_camera_url": "http://192.168.100.50:8080/stream-hd",
    "side_camera_url": "http://192.168.100.51:8080/stream-hd",
    "capture_interval": 5,
    "calibration": {
        "ref_diameter": 466.0,
        "ref_diameter_pixels": 632.62,
        "base_height": 1075.0,
        "side_camera_height": 800.0,
        "side_ref_pixels": 500.0,
        "wheel_height": 75.0,  # Standard wheel height in mm
        "fx": 640.268494,  # Default from metadata
        "fy": 640.268494,  # Default from metadata
        "cx": 642.991272,  # Default from metadata
        "cy": 364.303680,  # Default from metadata
        "depth_scale": 0.001,  # Default depth scale in meters (1mm)
        "depth_units": 1000.0,  # Default depth units in mm
        "depth_min": 200.0,  # Min depth in mm
        "depth_max": 3000.0   # Max depth in mm
    },
    "wheel_models": WHEEL_MODELS
}

# Check for RealSense library
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("RealSense library found")
except ImportError:
    REALSENSE_AVAILABLE = False
    print("RealSense library not found. RealSense camera support disabled.")

# Check for Serial library for 24V signal detection
try:
    import serial
    from serial.tools import list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed, 24V signal detection disabled")

def load_metadata(file_path):
    """Load camera intrinsics from metadata file"""
    if not file_path or not os.path.exists(file_path):
        return None
        
    try:
        metadata = {}
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 2:
                    key, value = row[0], row[1]
                    try:
                        # Convert numeric values
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = value
        
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        print(traceback.format_exc())
        return None

def load_settings():
    """Enhanced settings loading with proper model structure handling"""
    global current_settings, WHEEL_MODELS
    
    default_settings = {
        "selected_model": "10-13",
        "tolerance": 3.0,
        "top_camera_url": "http://192.168.100.50:8080/stream-hd",
        "side_camera_url": "http://192.168.100.51:8080/stream-hd",
        "capture_interval": 5,
        "calibration": {
            "ref_diameter": 466.0,
            "ref_diameter_pixels": 632.62,
            "base_height": 1075.0,
            "side_camera_height": 800.0,
            "side_ref_pixels": 500.0,
            "wheel_height": 75.0,
            "fx": 640.268494,  # Default from metadata
            "fy": 640.268494,  # Default from metadata
            "cx": 642.991272,  # Default from metadata
            "cy": 364.303680,  # Default from metadata
            "depth_scale": 0.001,  # Default depth scale in meters (1mm)
            "depth_units": 1000.0,  # Default depth units in mm
            "depth_min": 200.0,  # Min depth in mm
            "depth_max": 3000.0   # Max depth in mm
        },
        "wheel_models": WHEEL_MODELS
    }
    
    try:
        with open("settings.json", "r") as f:
            loaded_settings = json.load(f)
            # Use deep copy to ensure all nested dictionaries are properly loaded
            current_settings = default_settings.copy()
            current_settings.update(loaded_settings)
            
            # Ensure calibration dictionary is properly loaded
            if "calibration" in loaded_settings:
                current_settings["calibration"].update(loaded_settings["calibration"])
            
            # Restore wheel models if saved
            if "wheel_models" in loaded_settings:
                WHEEL_MODELS.clear()
                for model_name, model_data in loaded_settings["wheel_models"].items():
                    if isinstance(model_data, dict):
                        # New format with all parameters
                        WHEEL_MODELS[model_name] = model_data
                    elif isinstance(model_data, (list, tuple)) and len(model_data) >= 3:
                        # Old format - convert to new format
                        WHEEL_MODELS[model_name] = {
                            "min_dia": model_data[0],
                            "max_dia": model_data[1],
                            "height": model_data[2],
                            "tolerance": current_settings.get("tolerance", 3.0)
                        }
                
                # Update default models if they don't exist
                default_models = {
                    "10-13": {"min_dia": 10, "max_dia": 13, "height": 70, "tolerance": 3.0},
                    "13-16": {"min_dia": 13, "max_dia": 16, "height": 75, "tolerance": 3.0},
                    "16-19": {"min_dia": 16, "max_dia": 19, "height": 77, "tolerance": 3.0},
                    "19-22": {"min_dia": 19, "max_dia": 22, "height": 115, "tolerance": 3.0}
                }
                
                for model_name, model_data in default_models.items():
                    if model_name not in WHEEL_MODELS:
                        WHEEL_MODELS[model_name] = model_data
            
        print(f"Loaded settings with model: {current_settings['selected_model']}")
        print(f"Loaded calibration: {current_settings['calibration']}")
        print(f"Loaded models: {WHEEL_MODELS}")
    except Exception as e:
        print(f"Error loading settings: {e}")
        print(traceback.format_exc())
        current_settings = default_settings
        save_settings()

def save_settings():
    """Save all settings including models"""
    try:
        # Create a deep copy to ensure all nested dictionaries are saved properly
        settings_to_save = {}
        settings_to_save.update(current_settings)
        settings_to_save["wheel_models"] = WHEEL_MODELS
        
        with open("settings.json", "w") as f:
            json.dump(settings_to_save, f, indent=4)
        
        print(f"Saved settings: {settings_to_save.get('calibration', 'No calibration')}")
    except Exception as e:
        print(f"Error saving settings: {e}")
        print(traceback.format_exc())

def get_model_instance_segmentation(num_classes):
    """Create and configure the object detection model"""
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def classify_wheel_model(diameter_mm, height_mm=None):
    """Classify wheel and check if it matches selected model with model-specific tolerance"""
    if diameter_mm is None or diameter_mm == 0:
        return "Unknown", False
        
    selected_model = current_settings["selected_model"]
    
    if selected_model in WHEEL_MODELS:
        model_data = WHEEL_MODELS[selected_model]
        
        # Get model parameters
        if isinstance(model_data, dict):
            min_diam = model_data.get("min_dia", 0)
            max_diam = model_data.get("max_dia", 0)
            expected_height = model_data.get("height", 0)
            tolerance = model_data.get("tolerance", current_settings.get("tolerance", 3.0))
        else:
            # Old format support
            min_diam, max_diam, expected_height = model_data[:3]
            tolerance = current_settings.get("tolerance", 3.0)
        
        # Convert to mm for comparison (assuming WHEEL_MODELS diameters are in inches)
        min_diam_mm = min_diam * 25.4
        max_diam_mm = max_diam * 25.4
        
        # Check diameter
        diameter_ok = (min_diam_mm - tolerance) <= diameter_mm <= (max_diam_mm + tolerance)
        
        # Check height if provided
        height_ok = True
        if height_mm is not None and expected_height > 0:
            height_ok = (expected_height - tolerance) <= height_mm <= (expected_height + tolerance)
        
        is_ok = diameter_ok and height_ok
        return selected_model, is_ok
    
    return "Custom Size", False

def fit_circle_least_squares(points):
    """Fit a circle to points using least squares method"""
    pts = np.asarray(points, dtype=np.float64)
    A = np.column_stack((pts[:,0], pts[:,1], np.ones(len(pts))))
    b = (pts[:,0]**2 + pts[:,1]**2)
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx = c[0] / 2.0
    cy = c[1] / 2.0
    r = np.sqrt(cx**2 + cy**2 + c[2])
    return (cx, cy, r)

def correct_for_perspective(measurement_pixels, baseline_height, current_height, image_width_pixels):
    """Correct measurements for perspective based on height"""
    if baseline_height == current_height or baseline_height <= 0 or current_height <= 0:
        return measurement_pixels
    
    scaling_factor = baseline_height / current_height
    corrected_pixels = measurement_pixels * scaling_factor
    return corrected_pixels

def calculate_distance_from_depth(depth_image, mask, x, y, w, h):
    """Calculate distance using depth data and segmentation mask"""
    try:
        if depth_image is None:
            return None
        
        # Extract region of interest from depth image
        depth_roi = depth_image[y:y+h, x:x+w]
        
        # If mask provided, use it to extract only depth values within the wheel contour
        if mask is not None:
            # Make sure mask is properly sized to match ROI
            mask_roi = mask[y:y+h, x:x+w] if y+h <= mask.shape[0] and x+w <= mask.shape[1] else None
            
            if mask_roi is not None:
                # Get depth values only where mask is non-zero (the wheel)
                valid_depths = depth_roi[mask_roi > 0]
            else:
                # Use all values if mask ROI is invalid
                valid_depths = depth_roi.flatten()
        else:
            # No mask provided, use all values
            valid_depths = depth_roi.flatten()
        
        # Filter out zeros and extreme outliers
        valid_depths = valid_depths[valid_depths > 0]
        
        if len(valid_depths) == 0:
            print("No valid depth values found in ROI")
            return None
        
        # Use median for robustness against outliers
        distance_mm = np.median(valid_depths)
        
        # Print some debug info
        print(f"Depth statistics: min={np.min(valid_depths):.1f}, max={np.max(valid_depths):.1f}, median={distance_mm:.1f}")
        
        # Apply depth scale if using RealSense
        if realsense_pipeline is not None:
            try:
                depth_scale = realsense_pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
                distance_mm = distance_mm * depth_scale * 1000  # Convert to mm
            except Exception as e:
                print(f"Error getting depth scale: {e}")
        
        return float(distance_mm)
    except Exception as e:
        print(f"Error calculating distance: {e}")
        print(traceback.format_exc())
        return None
def measure_wheel_height_from_depth(depth_image, color_image, mask_binary, box):
    """
    Measure wheel height using RealSense depth data and wheel mask
    Returns height in mm
    """
    if depth_image is None or mask_binary is None:
        print("No depth image or mask available for height measurement")
        return None
        
    # Extract box coordinates
    x1, y1, x2, y2 = [int(val) for val in box]
    
    # Find wheel points from mask
    wheel_points = np.where(mask_binary > 0)
    if len(wheel_points[0]) == 0:
        print("No wheel points found in mask")
        return None
        
    # Get min and max y coordinates (top and bottom points)
    min_y = np.min(wheel_points[0])
    max_y = np.max(wheel_points[0])
    
    # Get horizontal center of the wheel
    center_x = (x1 + x2) // 2
    
    # Set the measurement points
    center_top_x, center_top_y = center_x, min_y
    center_bottom_x, center_bottom_y = center_x, max_y
    center_y = (min_y + max_y) // 2
    
    # Ensure coordinates are within bounds
    h_img, w_img = depth_image.shape[:2]
    center_x = min(max(center_x, 0), w_img-1)
    center_y = min(max(center_y, 0), h_img-1)
    center_top_x = min(max(center_top_x, 0), w_img-1)
    center_top_y = min(max(center_top_y, 0), h_img-1)
    center_bottom_x = min(max(center_bottom_x, 0), w_img-1)
    center_bottom_y = min(max(center_bottom_y, 0), h_img-1)
    
    # Helper function to get valid depth in a window
    def get_valid_depth_in_window(x, y, depth_img, window_size=5):
        half_window = window_size // 2
        
        # Handle different shapes of depth image
        if len(depth_img.shape) == 2:
            h, w = depth_img.shape
        else:
            h, w = depth_img.shape[:2]
        
        # Ensure window is within image bounds
        x_start = max(x - half_window, 0)
        x_end = min(x + half_window + 1, w)
        y_start = max(y - half_window, 0)
        y_end = min(y + half_window + 1, h)
        
        # Extract window
        window = depth_img[y_start:y_end, x_start:x_end]
        
        # If window has multiple channels, convert to single channel
        if len(window.shape) > 2:
            window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        
        # Get valid depths (non-zero)
        valid_depths = window[window > 0]
        
        if len(valid_depths) > 0:
            # Return median of valid depths
            return np.median(valid_depths)
        else:
            return 0
    
    # Get depth values with larger windows and searching along horizontal lines
    center_depth_value = get_valid_depth_in_window(center_x, center_y, depth_image, 15)
    
    # For top point - search along top row of the wheel mask
    top_depth_value = 0
    window_size = 25  # Larger window for more robust depth sampling
    search_width = 100  # Search this many pixels left and right of center
    
    for offset in range(-search_width, search_width+1, 10):  # Sample every 10 pixels
        x_pos = center_x + offset
        if 0 <= x_pos < depth_image.shape[1]:
            # Check if this point is within the mask
            y_range = 10  # Check a few rows from the top
            for y_offset in range(y_range):
                check_y = min_y + y_offset
                if 0 <= check_y < depth_image.shape[0] and mask_binary[check_y, x_pos] > 0:
                    depth = get_valid_depth_in_window(x_pos, check_y, depth_image, window_size)
                    if depth > 0:
                        top_depth_value = depth
                        # Update the position of the top point for visualization
                        center_top_x, center_top_y = x_pos, check_y
                        break
            if top_depth_value > 0:
                break
    
    # For bottom point - search along bottom row of the wheel mask 
    bottom_depth_value = 0
    for offset in range(-search_width, search_width+1, 10):
        x_pos = center_x + offset
        if 0 <= x_pos < depth_image.shape[1]:
            # Check if this point is within the mask
            y_range = 10  # Check a few rows from the bottom
            for y_offset in range(y_range):
                check_y = max_y - y_offset
                if 0 <= check_y < depth_image.shape[0] and mask_binary[check_y, x_pos] > 0:
                    depth = get_valid_depth_in_window(x_pos, check_y, depth_image, window_size)
                    if depth > 0:
                        bottom_depth_value = depth
                        # Update the position of the bottom point for visualization
                        center_bottom_x, center_bottom_y = x_pos, check_y
                        break
            if bottom_depth_value > 0:
                break
    
    # If we still don't have valid depths, try using the center depth for all points
    if center_depth_value > 0 and (top_depth_value == 0 or bottom_depth_value == 0):
        print("Using center depth for top/bottom points since they have invalid depths")
        if top_depth_value == 0:
            top_depth_value = center_depth_value
        if bottom_depth_value == 0:
            bottom_depth_value = center_depth_value
    
    # Convert raw depth values to meters
    depth_scale = current_settings["calibration"]["depth_scale"]
    center_dist = center_depth_value * depth_scale
    top_dist = top_depth_value * depth_scale
    bottom_dist = bottom_depth_value * depth_scale
    
    # Mark measurement points on the color image for visualization
    display_image = color_image.copy()
    cv2.circle(display_image, (center_top_x, center_top_y), 5, (255, 0, 0), -1)  # Blue for top
    cv2.circle(display_image, (center_bottom_x, center_bottom_y), 5, (0, 0, 255), -1)  # Red for bottom
    cv2.circle(display_image, (center_x, center_y), 5, (0, 255, 255), -1)  # Yellow for center
    
    # Draw line connecting center top and center bottom
    cv2.line(display_image, (center_top_x, center_top_y), 
             (center_bottom_x, center_bottom_y), (0, 255, 0), 2)
    
    # Calculate wheel height using depth and camera intrinsics
    if all(d > 0 for d in [center_dist, top_dist, bottom_dist]):
        # Create intrinsics object for calculation
        fx = current_settings["calibration"]["fx"]
        fy = current_settings["calibration"]["fy"]
        cx = current_settings["calibration"]["cx"]
        cy = current_settings["calibration"]["cy"]
        
        # Setup intrinsics (similar to RealSense's rs.intrinsics)
        depth_intrin = {
            "fx": fx,
            "fy": fy,
            "ppx": cx,
            "ppy": cy,
            "width": color_image.shape[1],
            "height": color_image.shape[0]
        }
        
        # Deproject 2D points to 3D
        def deproject_pixel_to_point(intrin, pixel, depth):
            x = (pixel[0] - intrin["ppx"]) / intrin["fx"] * depth
            y = (pixel[1] - intrin["ppy"]) / intrin["fy"] * depth
            z = depth
            return [x, y, z]
            
        top_point = deproject_pixel_to_point(depth_intrin, [center_top_x, center_top_y], top_dist)
        bottom_point = deproject_pixel_to_point(depth_intrin, [center_bottom_x, center_bottom_y], bottom_dist)
        
        # Calculate height in 3D space
        wheel_height_meters = np.sqrt(
            (top_point[0] - bottom_point[0])**2 + 
            (top_point[1] - bottom_point[1])**2 + 
            (top_point[2] - bottom_point[2])**2)
        
        wheel_height_mm = wheel_height_meters * 1000
        
        # Add height text to display on the image
        cv2.putText(display_image, f"WHEEL HEIGHT: {wheel_height_mm:.1f} mm", 
                  (x1, max_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return wheel_height_mm, display_image
    else:
        # Fall back to pixel-based estimation
        pixel_height = max_y - min_y
        dist_to_use = center_dist if center_dist > 0 else 0.5  # Default 0.5m if no valid depth
        
        # Estimate angular height using camera FOV and focal length
        vertical_fov_radians = 2 * np.arctan2(color_image.shape[0] / 2, current_settings["calibration"]["fy"])
        angular_height = (pixel_height / color_image.shape[0]) * vertical_fov_radians
        
        # Estimate real height using simple trigonometry
        wheel_height_meters = 2 * dist_to_use * np.tan(angular_height / 2)
        wheel_height_mm = wheel_height_meters * 1000
        
        # Add estimated height text to display on the image
        cv2.putText(display_image, f"EST. HEIGHT: {wheel_height_mm:.1f} mm", 
                  (x1, max_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        return wheel_height_mm, display_image

def calculate_real_dimensions(measurements, is_top_view=True, mask=None):
    """Calculate real-world dimensions from pixel measurements using camera intrinsics"""
    if not measurements:
        return measurements
    
    # Add measurement type
    measurements["type"] = "Top" if is_top_view else "Side"
    
    # Copy mask to measurements for reuse
    if mask is not None:
        measurements["mask"] = mask
        
    # For top view calculations that need wheel height from side view
    if is_top_view:
        # Get the wheel height from side view (measured by RealSense depth camera)
        # This value is continuously updated by the side view processing in stream_realsense()
        wheel_height_mm = current_settings["calibration"]["wheel_height"]
            
        # Get camera base height (fixed calibration value)
        base_height_mm = current_settings["calibration"]["base_height"]
            
        # Log that we're using the latest height from depth camera
        print(f"Using wheel height from depth camera: {wheel_height_mm:.2f} mm for diameter calculation")
            
        if "diameter_pixels" in measurements:
            # Calculate real distance between camera and wheel top surface
            # by subtracting the measured wheel height from the camera's base height
            # This distance is crucial for accurate diameter calculation due to perspective effects
            distance_mm = base_height_mm - wheel_height_mm
            if distance_mm <= 0:
                print("Warning: Invalid height difference calculation - wheel height greater than base height")
                distance_mm = base_height_mm  # Fallback to base height as a safety measure
                    
            ref_diameter = current_settings["calibration"]["ref_diameter"]
            ref_diameter_pixels = current_settings["calibration"]["ref_diameter_pixels"]
                
            # Calculate scale based on reference diameter and distance
            scale_factor = ref_diameter / ref_diameter_pixels
            
            # Adjust scale factor based on distance from camera
            # (Objects appear smaller with distance)
            base_distance = 1000.0  # Reference distance in mm
            distance_ratio = base_distance / distance_mm
            
            # Calculate diameter in mm
            diameter_mm = measurements["diameter_pixels"] * scale_factor * distance_ratio
            measurements["diameter_mm"] = diameter_mm
            
            print(f"Top view real dimensions calculation:")
            print(f"  Base height: {base_height_mm}mm")
            print(f"  Wheel height: {wheel_height_mm}mm")
            print(f"  Distance: {distance_mm}mm")
            print(f"  Diameter pixels: {measurements['diameter_pixels']:.2f}")
            print(f"  Reference diameter: {ref_diameter}mm at {ref_diameter_pixels}px")
            print(f"  Calculated diameter: {diameter_mm:.2f}mm")
            
            # Check if diameter is within the specified model range
            wheel_model = current_settings["selected_model"]
            if wheel_model in WHEEL_MODELS:
                model_data = WHEEL_MODELS[wheel_model]
                if isinstance(model_data, dict):
                    min_dia = model_data.get("min_dia", 0) * 25.4  # Convert inches to mm
                    max_dia = model_data.get("max_dia", 0) * 25.4  # Convert inches to mm
                    tolerance = model_data.get("tolerance", 3.0)  # Tolerance in mm
                else:
                    # Handle old format if still in use
                    min_dia, max_dia, _ = model_data
                    min_dia *= 25.4  # Convert inches to mm
                    max_dia *= 25.4  # Convert inches to mm
                    tolerance = current_settings.get("tolerance", 3.0)
                
                # Check if diameter is within range including tolerance
                within_range = min_dia - tolerance <= diameter_mm <= max_dia + tolerance
                measurements["within_range"] = within_range
                measurements["min_dia_mm"] = min_dia
                measurements["max_dia_mm"] = max_dia
                measurements["tolerance_mm"] = tolerance
    
    # For side view calculations
    else:
        # If we have a depth measurement, use it
        if "depth_mm" in measurements:
            # We already have the depth measurement
            pass
        elif current_depth_image is not None and "contour" in measurements:
            # Calculate depth from the depth image using the wheel contour
            if mask is not None:
                depth_mm = calculate_distance_from_depth(
                    current_depth_image, 
                    mask,
                    int(measurements.get("center_x", 0)),
                    int(measurements.get("center_y", 0)),
                    int(measurements.get("width", 100)),
                    int(measurements.get("height", 100))
                )
                if depth_mm > 0:
                    measurements["depth_mm"] = depth_mm
        
        # Calculate wheel height if not already provided by the depth processing
        if "height_mm" not in measurements and "height_pixels" in measurements:
            # Fallback to simple calculation using side camera calibration
            side_ref_pixels = current_settings["calibration"]["side_ref_pixels"]
            ref_height = current_settings["calibration"]["wheel_height"]
            
            height_mm = (measurements["height_pixels"] / side_ref_pixels) * ref_height
            measurements["height_mm"] = height_mm
    
    return measurements
def load_realsense_calibration(calibration_file):
    """Load RealSense calibration data from JSON file"""
    try:
        with open(calibration_file, 'r') as f:
            data = json.load(f)
        
        # Extract camera matrix values
        fx = data["camera_matrix"][0][0]  # 1004.9319760981635
        fy = data["camera_matrix"][1][1]  # 1007.6079264638521
        ppx = data["camera_matrix"][0][2]  # 685.1355896680004
        ppy = data["camera_matrix"][1][2]  # 232.6670955056262
        
        # Create a calibration dict
        calibration = {
            "fx": fx,
            "fy": fy,
            "ppx": ppx,
            "ppy": ppy,
            "image_width": data.get("image_width", 1280),
            "image_height": data.get("image_height", 720)
        }
        
        return calibration
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None
        
def update_clock():
    """Update the clock in the UI"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clock_label.config(text=now)
    root.after(1000, update_clock)

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image while maintaining aspect ratio"""
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
    """Update a panel with an image frame - threaded safe"""
    try:
        if frame is None:
            return
            
        display_frame = resize_with_aspect_ratio(frame, width=width, height=height)
        cv2_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Use after() to update in main thread
        root.after(0, lambda: update_panel_image(panel, imgtk))
    except Exception as e:
        print(f"Error updating display: {e}")
        print(traceback.format_exc())

def update_panel_image(panel, imgtk):
    """Safely update panel image in main thread"""
    try:
        panel.imgtk = imgtk
        panel.config(image=imgtk)
    except Exception as e:
        print(f"Error updating panel image: {e}")

def process_frame(frame, is_top_view=True):
    """Process a frame through the detection model and calculate real-world dimensions
    
    Parameters:
        frame: Image frame to process
        is_top_view: If True, processes as top view (diameter calculation)
                    If False, processes as side view (height calculation)
                    
    Order of operations is important:
    1. The side view (is_top_view=False) should be processed first to measure wheel height
    2. The top view (is_top_view=True) should be processed second to use that height for 
       accurate diameter calculation that accounts for the wheel's distance from the camera
       
    The wheel height measured from the depth camera is stored in:
    current_settings["calibration"]["wheel_height"]
    """
    """Process a frame through the detection model"""
    global model
    
    if model is None:
        return frame, {"type": None, "radius": 0, "width": 0, "height": 0, "bbox": (0, 0, 0, 0)}
    
    try:
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
        measurements = {"type": None, "radius": 0, "width": 0, "height": 0, "bbox": (0, 0, 0, 0)}
        real_dimensions = {}
        
        if len(masks) > 0:
            i = 0
            mask = masks[i]
            mask_binary = mask[0] > 0.5
            mask_uint8 = mask_binary.astype(np.uint8) * 255
            
            mask_colored = np.zeros_like(output_image, dtype=np.uint8)
            mask_colored[mask_binary] = [0, 255, 0]  # Green mask
            output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.5, 0)
            
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # In process_frame function, when extracting measurements
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                
                # Extract contour points - THIS LINE WAS MISSING
                contour_points = np.squeeze(main_contour, axis=1)
                
                # Create a mask from the contour
                contour_mask = np.zeros_like(mask_uint8)
                cv2.drawContours(contour_mask, [main_contour], 0, 255, -1)
                
                x, y, w, h = cv2.boundingRect(main_contour)
                measurements["bbox"] = (x, y, w, h)
                
                # Process depth with contour mask
                depth_distance = None
                if depth_data is not None:
                    depth_distance = calculate_distance_from_depth(depth_data, contour_mask, x, y, w, h)
                    if depth_distance is not None:
                        # Store the depth value without updating UI directly
                        depth_distance_mm = depth_distance
                elif current_depth_image is not None:
                    depth_distance = calculate_distance_from_depth(current_depth_image, contour_mask, x, y, w, h)
                    if depth_distance is not None:
                        depth_distance_mm = depth_distance
                
                if not is_top_view:
                    img_height = frame.shape[0]
                    estimated_height = estimate_wheel_height(main_contour, img_height)
                
                (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
                cx_ls, cy_ls, r_ls = fit_circle_least_squares(contour_points)
                
                circle_area = math.pi * (radius ** 2)
                contour_area = cv2.contourArea(main_contour)
                shape_ratio = contour_area / circle_area if circle_area > 0 else 0
                
                if shape_ratio > RATIO_THRESHOLD:
                    cv2.circle(output_image, (int(cx_ls), int(cy_ls)), int(r_ls), (0, 255, 0), 2)
                    cv2.circle(output_image, (int(cx_ls), int(cy_ls)), 2, (0, 0, 255), 3)
                    measurements = {"type": "Round", "radius": r_ls, "width": 0, "height": 0, "bbox": (x, y, w, h)}
                    
                    real_dimensions = calculate_real_dimensions(measurements, is_top_view, contour_mask)
                    diameter_mm = real_dimensions.get("diameter_mm", 0)
                    wheel_model = real_dimensions.get('wheel_model', 'Unknown')
                    is_ok = real_dimensions.get('is_ok', False)
                    
                    status_color = (0, 255, 0) if is_ok else (0, 0, 255)
                    text = f"Round: D={diameter_mm:.1f}mm, Model: {wheel_model}"
                    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(output_image, "OK" if is_ok else "NOT OK", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                                
                    # Add depth info to display
                    if depth_distance is not None:
                        cv2.putText(output_image, f"Depth: {depth_distance:.1f}mm", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    rot_rect = cv2.minAreaRect(main_contour)
                    box_pts = cv2.boxPoints(rot_rect)
                    box_pts = np.array(box_pts, dtype=np.intp)
                    cv2.drawContours(output_image, [box_pts], 0, (255, 0, 0), 2)
                    
                    (center_x, center_y), (w_rect, h_rect), angle = rot_rect
                    measurements = {"type": "Side", "radius": 0, "width": w_rect, "height": h_rect, "bbox": (x, y, w, h)}
                    
                    real_dimensions = calculate_real_dimensions(measurements, is_top_view, contour_mask)
                    diameter_mm = real_dimensions.get("diameter_mm", 0)
                    height_mm = real_dimensions.get("height_mm", 0)
                    wheel_model = real_dimensions.get('wheel_model', 'Unknown')
                    is_ok = real_dimensions.get('is_ok', False)
                    
                    status_color = (0, 255, 0) if is_ok else (0, 0, 255)
                    text = f"Side: D={diameter_mm:.1f}mm, H={height_mm:.1f}mm, Model: {wheel_model}"
                    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(output_image, "OK" if is_ok else "NOT OK", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Add depth info to display
                    if depth_distance is not None:
                        cv2.putText(output_image, f"Depth: {depth_distance:.1f}mm", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        combined_measurements = {**measurements, **real_dimensions}
        
        return output_image_bgr, combined_measurements
    except Exception as e:
        print(f"Error processing frame: {e}")
        print(traceback.format_exc())
        return frame, {"type": None, "radius": 0, "width": 0, "height": 0, "bbox": (0, 0, 0, 0)}

def start_realsense():
    """Initialize and start RealSense camera"""
    global realsense_pipeline, realsense_align
    
    if not REALSENSE_AVAILABLE:
        print("RealSense library not available")
        return False
        
    try:
        # Create pipeline
        realsense_pipeline = rs.pipeline()
        
        # Create a config
        config = rs.config()
        
        # Configure streams
        # Set resolution to 1280x720 as required
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        
        # Load calibration data
        try:
            with open("realsense_calib_327122076353.json", 'r') as f:
                calib_data = json.load(f)
                
            # Update calibration settings
            current_settings["calibration"]["fx"] = calib_data["camera_matrix"][0][0]
            current_settings["calibration"]["fy"] = calib_data["camera_matrix"][1][1]
            current_settings["calibration"]["cx"] = calib_data["camera_matrix"][0][2]
            current_settings["calibration"]["cy"] = calib_data["camera_matrix"][1][2]
            
            print(f"Loaded calibration from file: fx={current_settings['calibration']['fx']}, " +
                  f"fy={current_settings['calibration']['fy']}, " +
                  f"cx={current_settings['calibration']['cx']}, " +
                  f"cy={current_settings['calibration']['cy']}")
        except Exception as e:
            print(f"Warning: Could not load calibration data: {e}")
            # Continue with default calibration
        
        # Start streaming
        profile = realsense_pipeline.start(config)
        
        # Get depth scale (meters per unit)
        depth_sensor = profile.get_device().first_depth_sensor()
        current_settings["calibration"]["depth_scale"] = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {current_settings['calibration']['depth_scale']}")
        
        # Create alignment object
        realsense_align = rs.align(rs.stream.color)
        
        print("RealSense camera started successfully")
        return True
        
    except Exception as e:
        print(f"Error starting RealSense camera: {e}")
        print(traceback.format_exc())
        return False

def stream_top_camera(url):
    """Stream from top camera (event camera) and process frames for wheel diameter calculation
    This function works in coordination with the RealSense side camera to calculate accurate wheel dimensions
    using the measured wheel height from the depth camera."""
    """Optimized streaming from top camera"""
    global stop_streaming, frame_top, streaming_active
    
    cap = None
    last_update_time = 0
    update_interval = 0.033  # ~30 FPS
    
    try:
        # Initialize camera with optimized settings
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG if url.startswith('http') else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Set smaller resolution if possible to improve performance
        if url.startswith('http'):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            root.after(0, lambda: status_label_top.config(text="Error: Could not open top camera"))
            return
            
        root.after(0, lambda: status_label_top.config(text="Top camera connected!"))
        streaming_active.set()
        
        while not stop_streaming:
            current_time = time.time()
            if current_time - last_update_time < update_interval:
                time.sleep(0.001)  # Short sleep to prevent CPU hogging
                continue
                
            ret, frame = cap.read()
            if not ret:
                root.after(0, lambda: status_label_top.config(text="Error: Lost top camera stream"))
                break
                
            frame_top = frame.copy()
            update_display(top_panel, frame, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
            
            last_update_time = current_time
            
    except Exception as e:
        print(f"Error in top camera stream: {e}")
        print(traceback.format_exc())
        root.after(0, lambda: status_label_top.config(text=f"Error: {str(e)}"))
    finally:
        streaming_active.clear()
        if cap is not None:
            cap.release()
        root.after(0, lambda: status_label_top.config(text="Top camera streaming stopped."))

def stream_side_camera(url):
    """Optimized streaming from side camera"""
    global stop_streaming, frame_side, streaming_active
    
    cap = None
    last_update_time = 0
    update_interval = 0.033  # ~30 FPS
    
    root.after(0, lambda: status_label_side.config(text="Connecting to side camera..."))
    
    while not stop_streaming:
        if cap is None or not cap.isOpened():
            try:
                # Try to open camera with optimized settings
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Set smaller resolution if possible
                if url.startswith('http'):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                if not cap.isOpened():
                    root.after(0, lambda: status_label_side.config(text="Error: Could not open side camera. Retrying..."))
                    time.sleep(3)
                    continue
                
                root.after(0, lambda: status_label_side.config(text="Side camera connected!"))
                streaming_active.set()
            except Exception as e:
                root.after(0, lambda: status_label_side.config(text=f"Error: {str(e)}"))
                time.sleep(3)
                continue
        
        try:
            current_time = time.time()
            if current_time - last_update_time < update_interval:
                time.sleep(0.001)
                continue
                
            ret, frame = cap.read()
            if not ret:
                root.after(0, lambda: status_label_side.config(text="Error: Lost side camera stream. Reconnecting..."))
                cap.release()
                cap = None
                time.sleep(3)
                continue
            
            frame_side = frame.copy()
            update_display(side_panel, frame, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            
            last_update_time = current_time
        except Exception as e:
            print(f"Error in side camera stream: {e}")
            print(traceback.format_exc())
            root.after(0, lambda: status_label_side.config(text=f"Error: {str(e)}"))
            time.sleep(3)

    # Clean up when stopped
    streaming_active.clear()
    if cap is not None:
        cap.release()
    
    root.after(0, lambda: status_label_side.config(text="Side camera streaming stopped."))

def stream_realsense():
    """Stream from RealSense camera"""
    global frame_side, frame_queue, current_depth_image, streaming_active
    
    try:
        if realsense_pipeline is None:
            print("RealSense pipeline not started")
            root.after(0, lambda: status_label_side.config(text="RealSense camera not initialized"))
            return
            
        # Signal that streaming is now active
        streaming_active.set()
        root.after(0, lambda: status_label_side.config(text="RealSense camera streaming active"))
            
        while not stop_streaming:
            try:
                # Wait for frames with timeout to avoid blocking
                frames = realsense_pipeline.wait_for_frames(timeout_ms=1000)
                
                # Align depth to color frame
                aligned_frames = realsense_align.process(frames)
                
                # Get color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    time.sleep(0.01)
                    continue
                    
                # Convert frames to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Store current depth image for later processing
                current_depth_image = depth_image.copy()
                
                # Update the side view with the color image
                frame_side = color_image.copy()
                
                # Directly update side panel with color image first
                update_display(side_panel, color_image, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
                
                # Process the frame as a side view (do this after updating display to prevent delay)
                # processed_frame, measurements = process_frame(color_image, is_top_view=False)
                
                # If wheel detected, measure height using depth data
                if measurements and "mask" in measurements and measurements["mask"] is not None:
                    mask_binary = measurements["mask"]
                    if "box" in measurements and measurements["box"] is not None:
                        try:
                            # Calculate wheel height from depth data
                            wheel_height_mm, height_visualization = measure_wheel_height_from_depth(
                                depth_image, color_image, mask_binary, measurements["box"])
                                
                            if wheel_height_mm is not None:
                                # Update measurements with the actual height
                                measurements["height_mm"] = wheel_height_mm
                                
                                # Store height for top view calculations - this is crucial for accurate diameter calculation
                                # in top view which requires the current wheel height to calculate proper distance
                                current_settings["calibration"]["wheel_height"] = wheel_height_mm
                                
                                # Update display with height visualization
                                update_display(side_processed_panel, height_visualization, 
                                              SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
                                
                                # Log the updated height for debugging
                                print(f"Updated wheel height for diameter calculation: {wheel_height_mm:.2f} mm")
                        except Exception as e:
                            print(f"Error measuring wheel height: {e}")
                            # Continue with normal processing
                            update_display(side_processed_panel, processed_frame, 
                                          SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
                    else:
                        # No box detected, use regular processing
                        update_display(side_processed_panel, processed_frame, 
                                      SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
                else:
                    # No wheel detected, use regular processing
                    update_display(side_processed_panel, processed_frame, 
                                  SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
                
                # Update measurements display
                if measurements and "type" in measurements and measurements["type"] == "Side":
                    side_result_text.set(f"Side: Height={measurements.get('height_mm', 0):.2f} mm")
                    measured_height_var.set(f"{measurements.get('height_mm', 0):.2f} mm")
                    
                    # Update side camera distance result if available
                    if "depth_mm" in measurements:
                        side_cam_distance_result.set(f"{measurements.get('depth_mm'):.1f} mm")
            
            except Exception as e:
                print(f"Error processing RealSense frame: {e}")
                time.sleep(0.1)  # Add a short delay to prevent rapid error messages
                
            # Small sleep to prevent CPU overuse
            time.sleep(0.01)
            
    except Exception as e:
        print(f"Error in RealSense streaming: {e}")
        print(traceback.format_exc())
        
    # Clear the streaming active flag and update status when done
    streaming_active.clear()
    root.after(0, lambda: status_label_side.config(text="RealSense camera streaming stopped"))
    
    # Try to stop the pipeline gracefully if it's running
    if realsense_pipeline:
        try:
            realsense_pipeline.stop()
            print("RealSense pipeline stopped")
        except Exception as e:
            print(f"Error stopping RealSense pipeline: {e}")

def start_streaming():
    """Start streaming from cameras"""
    global stop_streaming, depth_data, streaming_active
    
    # Set the stop_streaming flag to False to indicate streaming should be active
    stop_streaming = False
    depth_data = None
    
    # Clear the streaming_active event to ensure proper initialization
    streaming_active.clear()
    
    # Update UI to indicate streaming is starting
    status_label_main.config(text="Initializing cameras...")
    
    # Try to start top camera
    top_camera_url = current_settings["top_camera_url"]
    
    # Create and start the top camera thread
    top_thread = threading.Thread(target=stream_top_camera, args=(top_camera_url,), daemon=True)
    top_thread.start()
    
    # Try to start side camera (with priority to RealSense if available)
    if REALSENSE_AVAILABLE:
        try:
            if start_realsense():
                status_label_side.config(text="Starting RealSense camera...")
                side_thread = threading.Thread(target=stream_realsense, daemon=True)
                side_thread.start()
                # Give some time for the RealSense camera to initialize
                time.sleep(0.5)
            else:
                status_label_side.config(text="RealSense initialization failed")
        except Exception as e:
            print(f"Error starting RealSense: {e}")
            status_label_side.config(text="RealSense error: {str(e)[:50]}")
    
    # If RealSense not available or failed, try IP or USB cameras
    if not REALSENSE_AVAILABLE or realsense_pipeline is None:
        side_camera_url = current_settings["side_camera_url"]
        
        # Try IP-based side camera
        try:
            side_cap = cv2.VideoCapture(side_camera_url)
            if side_cap.isOpened():
                side_cap.release()
                status_label_side.config(text="Starting IP side camera...")
                side_thread = threading.Thread(target=stream_side_camera, args=(side_camera_url,), daemon=True)
                side_thread.start()
            else:
                # Try USB webcam as last resort
                try:
                    side_cap = cv2.VideoCapture(1)  # Index 1 (second camera)
                    if side_cap.isOpened():
                        side_cap.release()
                        status_label_side.config(text="Starting USB side camera...")
                        side_thread = threading.Thread(target=stream_side_camera, args=(1,), daemon=True)
                        side_thread.start()
                    else:
                        status_label_side.config(text="No side camera available")
                except Exception as e:
                    print(f"Error with USB camera: {e}")
                    status_label_side.config(text="USB camera error")
        except Exception as e:
            print(f"Error with IP camera: {e}")
            status_label_side.config(text="IP camera error")
    
    # Update button states
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    photo_button.config(state=tk.NORMAL)
    auto_photo_button.config(state=tk.NORMAL)
    
    # Set streaming_active after everything is initialized
    streaming_active.set()
    status_label_main.config(text="Streaming started")

def stop_streaming_func():
    """Stop streaming from cameras"""
    global stop_streaming, realsense_pipeline, streaming_active
    
    # Set flag to stop all streaming threads
    stop_streaming = True
    
    # Clear the streaming active flag
    streaming_active.clear()
    
    # Update UI to indicate streaming is stopping
    status_label_main.config(text="Stopping streams...")
    
    # Update button states immediately
    start_button.config(state=tk.DISABLED)  # Temporarily disable until fully stopped
    stop_button.config(state=tk.DISABLED)
    photo_button.config(state=tk.DISABLED)
    auto_photo_button.config(state=tk.DISABLED)

def update_wheel_counts():
    """Update the wheel count statistics from database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM inspections")
    total_count = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE test_result = 'OK'")
    passed_count = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE test_result = 'NOT OK'")
    faulty_count = cursor.fetchone()[0] or 0
    
    conn.close()
    
    # Update UI variables
    total_count_var.set(str(total_count))
    passed_count_var.set(str(passed_count))
    faulty_count_var.set(str(faulty_count))
    
    # Update Result panel
    update_result_frame()

def take_photo():
    """Capture and process frames from cameras
    This function ensures the correct processing order:
    1. Side view first - to measure wheel height using depth data
    2. Top view second - to calculate diameter using the measured height
    """
    global frame_top, frame_side, photo_count, depth_data, current_depth_image
    
    # Check if at least one camera is available
    if frame_top is None and frame_side is None:
        messagebox.showerror("Error", "No frames available to process. Start the streams first!")
        return
    
    # Check which cameras are available and process only those
    has_top_camera = frame_top is not None
    has_side_camera = frame_side is not None
    
    # We should avoid copying frames between cameras to prevent showing
    # processed images in both panels when only one camera is available
    
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
        processed_side, measurements_side = process_frame(frame_side, is_top_view=False)
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
        processed_top, measurements_top = process_frame(frame_top, is_top_view=True)
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
        measured_dia_var.set(f"{diameter_mm:.2f} mm")
        top_result_text.set(f"Round: {diameter_mm:.2f} mm")
    
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

def update_measurement_display(measurements_top, measurements_side):
    """Update the measurement result display with detection results
    NOTE: This should NOT update the model_frame which displays reference values"""
    
    # Update ONLY the results displays with measurements, not model parameters
    if measurements_top.get("type") == "Round":
        top_diameter_mm = measurements_top.get("diameter_mm", 0)
        top_is_ok = measurements_top.get("is_ok", False)
        
        # Update result variables only, not model parameters
        measured_dia_var.set(f"{top_diameter_mm:.2f} mm")
        top_result_text.set(f"Round: {top_diameter_mm:.2f} mm")
        
        # Update only the result status
        result_status_var.set("OK" if top_is_ok else "NOT OK")
    
    if measurements_side.get("type") == "Side":
        side_height_mm = measurements_side.get("height_mm", 0)
        camera_height_mm = measurements_side.get("depth_mm", 0)
        
        # Update result variables only, not model parameters
        measured_height_var.set(f"{side_height_mm:.2f} mm")
        side_result_text.set(f"Side: Height={side_height_mm:.2f} mm")
        
        # Update side camera distance result only
        if camera_height_mm:
            side_cam_distance_result.set(f"{camera_height_mm:.1f} mm")

def auto_capture():
    """Toggle automatic photo capture"""
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
    """Thread for automatic photo capture
    This function runs a loop that calls take_photo() on a regular interval
    The take_photo() function ensures the correct processing order:
    1. Side view first - for height measurement
    2. Top view second - for diameter calculation using the height
    """
    global auto_capture_active, stop_streaming
    
    while auto_capture_active and not stop_streaming:
        try:
            # Use after() to call take_photo in main thread
            root.after(0, take_photo)
            time.sleep(interval)
        except Exception as e:
            print(f"Error in auto capture: {e}")
            print(traceback.format_exc())

def estimate_wheel_height(contour, image_height):
    """Estimate wheel height based on contour position in image"""
    x, y, w, h = cv2.boundingRect(contour)
    center_y = y + h/2
    relative_pos = center_y / image_height
    
    base_height = current_settings["calibration"]["base_height"]
    side_cam_height = current_settings["calibration"]["side_camera_height"]
    max_height = side_cam_height * 1.2
    
    estimated_height = side_cam_height + (1 - relative_pos) * (max_height - side_cam_height)
    return estimated_height

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
            update_display(top_panel, display_frame, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
            
            processed_frame, measurements = process_frame(frame, is_top_view=True)
            update_display(top_processed_panel, processed_frame, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
            
            # Update result display only, not model parameters
            if measurements.get("type") == "Round":
                top_result_text.set(f"Round: {measurements.get('diameter_mm', 0):.2f} mm")
                measured_dia_var.set(f"{measurements.get('diameter_mm', 0):.2f} mm")
                result_status_var.set("OK" if measurements.get("is_ok", False) else "NOT OK")
                update_result_frame()
        else:
            global frame_side
            frame_side = frame.copy()
            display_frame = resize_with_aspect_ratio(frame, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)
            update_display(side_panel, display_frame, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            
            processed_frame, measurements = process_frame(frame, is_top_view=False)
            update_display(side_processed_panel, processed_frame, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
            
            # Update result display only, not model parameters
            if measurements.get("type") == "Side":
                side_result_text.set(f"Side: Height={measurements.get('height_mm', 0):.2f} mm")
                measured_height_var.set(f"{measurements.get('height_mm', 0):.2f} mm")
                
                # Update side camera distance result
                if measurements.get("depth_mm"):
                    side_cam_distance_result.set(f"{measurements.get('depth_mm'):.1f} mm")
        
        status_label_main.config(text=f"Processed uploaded {'top' if is_top_view else 'side'} view image: {os.path.basename(file_path)}")
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error processing image: {e}")
        print(traceback.format_exc())

def upload_depth_image():
    """Upload and load a depth image for processing"""
    global current_depth_image, metadata_file_path
    
    file_path = filedialog.askopenfilename(
        title="Select Depth Image", 
        filetypes=[("PNG files", "*.png"), ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    
    if not file_path:
        return
    
    try:
        # Read depth image - use cv2.IMREAD_UNCHANGED to preserve bit depth
        depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if depth_image is None:
            messagebox.showerror("Error", "Failed to read depth image file.")
            return
        
        # Store depth image for processing
        current_depth_image = depth_image
        
        # Check for associated metadata file
        base_name = os.path.splitext(file_path)[0]
        meta_file = f"{base_name}_metadata.csv"
        if os.path.exists(meta_file):
            metadata_file_path = meta_file
            print(f"Using metadata from: {meta_file}")
            
            # Load metadata
            metadata = load_metadata(meta_file)
            if metadata:
                # Update calibration with metadata values
                if "fx" in metadata:
                    current_settings["calibration"]["fx"] = metadata["fx"]
                if "fy" in metadata:
                    current_settings["calibration"]["fy"] = metadata["fy"]
                if "cx" in metadata:
                    current_settings["calibration"]["cx"] = metadata["cx"]
                if "cy" in metadata:
                    current_settings["calibration"]["cy"] = metadata["cy"]
        
        messagebox.showinfo("Success", f"Depth image loaded: {os.path.basename(file_path)}")
        status_label_main.config(text=f"Depth image loaded: {os.path.basename(file_path)}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error loading depth image: {e}")
        print(traceback.format_exc())


def on_closing():
    """Handle application closing"""
    global stop_streaming, auto_capture_active
    
    # Stop all processes
    stop_streaming = True
    auto_capture_active = False
    
    # Wait a moment for threads to cleanup
    time.sleep(0.5)
    
    # Clean up RealSense resources
    if realsense_pipeline is not None:
        try:
            realsense_pipeline.stop()
        except Exception as e:
            print(f"Error stopping RealSense pipeline: {e}")
            print(traceback.format_exc())
    
    # Close OpenCV windows
    cv2.destroyAllWindows()
    
    # Destroy the window
    root.destroy()
    print("Application closed properly")

def update_model_parameters():
    """Update UI with current model parameters
    This function ensures model parameters are displayed, not measurements"""
    global current_settings
    
    selected_model = current_settings["selected_model"]
    model_value.set(selected_model)
    
    # Update expected values based on model
    if selected_model in WHEEL_MODELS:
        model_data = WHEEL_MODELS[selected_model]
        
        if isinstance(model_data, dict):
            min_diam = model_data.get("min_dia", 0)
            max_diam = model_data.get("max_dia", 0)
            expected_height = model_data.get("height", 0)
            tolerance = model_data.get("tolerance", 3.0)
            
            expected_diameter = f"{min_diam:.1f}-{max_diam:.1f} inches ({min_diam*25.4:.1f}-{max_diam*25.4:.1f} mm)"
            expected_height_text = f"{expected_height:.0f} mm"
            tolerance_text = f"{tolerance} mm"
        else:
            # Old format support
            min_diam, max_diam, expected_height = model_data
            expected_diameter = f"{min_diam:.1f}-{max_diam:.1f} inches ({min_diam*25.4:.1f}-{max_diam*25.4:.1f} mm)"
            expected_height_text = f"{expected_height:.0f} mm"
            tolerance_text = f"{current_settings.get('tolerance', 3.0)} mm"
        
        diameter_value.set(expected_diameter)
        height_value.set(expected_height_text)
        tolerance_value.set(tolerance_text)
    else:
        # For custom models, display current values
        diameter_value.set("Custom")
        height_value.set("Custom")
        tolerance_value.set(f"{current_settings.get('tolerance', 3.0)} mm")
    
    # Update camera heights from settings, not measured values
    top_cam_height = current_settings["calibration"]["base_height"]
    side_cam_height = current_settings["calibration"]["side_camera_height"]
    top_cam_height_var.set(f"{top_cam_height:.1f} mm")
    side_cam_height_var.set(f"{side_cam_height:.1f} mm")

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
                    # When 24V signal is received, take a single photo rather than starting auto-capture
                    print("24V signal received - taking a photo")
                    root.after(0, take_photo)
                # No need to handle 24V_OFF as we're not toggling auto_capture anymore
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error in 24V detection: {e}")
        print(traceback.format_exc())
    finally:
        if 'ser' in locals():
            ser.close()


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

# Initialize database
init_db()

# Load settings before creating UI elements
load_settings()

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
# New variables for camera heights/distances
top_cam_height_var = tk.StringVar(value=f"{current_settings['calibration']['base_height']:.1f} mm")
side_cam_height_var = tk.StringVar(value=f"{current_settings['calibration']['side_camera_height']:.1f} mm")
side_cam_distance_result = tk.StringVar(value="0.0 mm")  # For measured values

# Initialize database counts
conn = sqlite3.connect(DB_FILE)
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
update_clock()  # Start clock update

# Buttons row - Enhanced with new buttons
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

upload_top_button = ttk.Button(button_frame, text="Upload Top View", command=lambda: upload_image(is_top_view=True))
upload_top_button.grid(row=0, column=4, padx=3, pady=5)

upload_side_button = ttk.Button(button_frame, text="Upload Side View", command=lambda: upload_image(is_top_view=False))
upload_side_button.grid(row=0, column=5, padx=3, pady=5)

report_button = ttk.Button(button_frame, text="Generate Report", command=open_report_window)
report_button.grid(row=0, column=6, padx=3, pady=5)

settings_button = ttk.Button(button_frame, text="Settings", command=open_settings_window)
settings_button.grid(row=0, column=7, padx=3, pady=5)

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

# Model Data panel - Enhanced to show model-specific tolerance and separate camera heights
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

# Result panel - With separate fields for measurements vs. model parameters
result_frame = ttk.LabelFrame(info_frame, text="Result", style="Info.TLabelframe")
result_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

ttk.Label(result_frame, text="Top view:", font=('Helvetica', 12)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
ttk.Label(result_frame, textvariable=top_result_text, font=('Helvetica', 12)).grid(row=0, column=1, sticky="w", padx=5, pady=5)

ttk.Label(result_frame, text="Measured Diameter:", font=('Helvetica', 12)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
ttk.Label(result_frame, textvariable=measured_dia_var, font=('Helvetica', 12)).grid(row=1, column=1, sticky="w", padx=5, pady=5)

ttk.Label(result_frame, text="Side view:", font=('Helvetica', 12)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
ttk.Label(result_frame, textvariable=side_result_text, font=('Helvetica', 12)).grid(row=2, column=1, sticky="w", padx=5, pady=5)

ttk.Label(result_frame, text="Measured Height:", font=('Helvetica', 12)).grid(row=3, column=0, sticky="w", padx=5, pady=5)
ttk.Label(result_frame, textvariable=measured_height_var, font=('Helvetica', 12)).grid(row=3, column=1, sticky="w", padx=5, pady=5)

ttk.Label(result_frame, text="Camera Distance:", font=('Helvetica', 12)).grid(row=4, column=0, sticky="w", padx=5, pady=5)
ttk.Label(result_frame, textvariable=side_cam_distance_result, font=('Helvetica', 12)).grid(row=4, column=1, sticky="w", padx=5, pady=5)

ttk.Label(result_frame, text="Status:", font=('Helvetica', 14, 'bold')).grid(row=5, column=0, sticky="w", padx=5, pady=5)
result_status_label = ttk.Label(result_frame, textvariable=result_status_var, font=('Helvetica', 14, 'bold'))
result_status_label.grid(row=5, column=1, sticky="w", padx=5, pady=5)

# Wheel Count panel
wheel_frame = ttk.LabelFrame(info_frame, text="Wheel Count", style="Info.TLabelframe")
wheel_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=5)

ttk.Label(wheel_frame, text="Total count:", font=('Helvetica', 12)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
ttk.Label(wheel_frame, textvariable=total_count_var, font=('Helvetica', 12)).grid(row=0, column=1, sticky="w", padx=5, pady=5)

ttk.Label(wheel_frame, text="Passed:", font=('Helvetica', 12), foreground=OK_COLOR).grid(row=1, column=0, sticky="w", padx=5, pady=5)
ttk.Label(wheel_frame, textvariable=passed_count_var, font=('Helvetica', 12), foreground=OK_COLOR).grid(row=1, column=1, sticky="w", padx=5, pady=5)

ttk.Label(wheel_frame, text="Faulty:", font=('Helvetica', 12), foreground=NOK_COLOR).grid(row=2, column=0, sticky="w", padx=5, pady=5)
ttk.Label(wheel_frame, textvariable=faulty_count_var, font=('Helvetica', 12), foreground=NOK_COLOR).grid(row=2, column=1, sticky="w", padx=5, pady=5)

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
side_panel = ttk.Label(side_original)
side_panel.pack(fill=tk.BOTH, expand=True)

side_processed_frame = ttk.LabelFrame(side_container, text="Processed")
side_processed_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
side_processed_panel = ttk.Label(side_processed_frame)
side_processed_panel.pack(fill=tk.BOTH, expand=True)

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
top_panel = ttk.Label(top_original)
top_panel.pack(fill=tk.BOTH, expand=True)

top_processed_frame = ttk.LabelFrame(top_container, text="Processed")
top_processed_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
top_processed_panel = ttk.Label(top_processed_frame)
top_processed_panel.pack(fill=tk.BOTH, expand=True)

status_label_top = ttk.Label(top_frame, text="Top camera not connected", font=("Arial", 11, "bold"))
status_label_top.grid(row=1, column=0, sticky="w", pady=(0, 5))

# Function to update result frame style
def update_result_frame():
    """Update the result panel appearance based on status"""
    if result_status_var.get() == "OK":
        result_status_label.configure(foreground=OK_COLOR)
    else:
        result_status_label.configure(foreground=NOK_COLOR)

# Update model parameters with loaded settings
update_model_parameters()

# Start 24V signal detection thread if available
if SERIAL_AVAILABLE:
    threading.Thread(target=detect_24v_signal, daemon=True).start()

# Load model in background
def load_model_thread():
    global model
    try:
        model = get_model_instance_segmentation(num_classes=2)
        model.to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        root.after(0, lambda: status_label_main.config(text="Model loaded successfully"))
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(traceback.format_exc())
        root.after(0, lambda: status_label_main.config(text=f"Error loading model: {str(e)}"))

threading.Thread(target=load_model_thread, daemon=True).start()

# Start the main loop
if __name__ == "__main__":
    root.mainloop()