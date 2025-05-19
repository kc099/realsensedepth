import cv2
import time
import os
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
import traceback

# Image processing utilities
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image while maintaining aspect ratio"""
    if image is None:
        return None

    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
        
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# UI update utilities
def update_display(panel, frame, width, height):
    """Update a panel with an image frame - threaded safe"""
    if frame is None:
        return
    
    try:
        # Ensure we have a color image
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Calculate target size while maintaining aspect ratio
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate scale factors for width and height
        scale_w = width / frame_w
        scale_h = height / frame_h
        
        # Use the smaller scale to ensure the image fits
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        
        # Resize the image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a canvas with the exact target size (filled with gray)
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 240  # light gray background
        
        # Calculate position to center the image
        y_offset = (height - new_h) // 2
        x_offset = (width - new_w) // 2
        
        # Place the resized image onto the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert to PIL format and then to ImageTk
        image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=image)
        
        # Update in main thread
        panel.after(0, lambda: update_panel_image(panel, imgtk))
    except Exception as e:
        print(f"Error updating display: {e}")
        print(traceback.format_exc())

def update_panel_image(panel, imgtk):
    """Safely update panel image in main thread"""
    try:
        panel.configure(image=imgtk)
        panel.image = imgtk  # Keep reference to prevent garbage collection
    except Exception as e:
        print(f"Error in panel update: {e}")

def update_clock(clock_label):
    """Update the clock in the UI"""
    time_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    clock_label.config(text=time_string)
    clock_label.after(1000, lambda: update_clock(clock_label))  # Update every second

def fit_circle_least_squares(points):
    """Fit a circle to points using least squares method"""
    # Convert points to numpy array
    points = np.array(points)
    
    # Get x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Center all points
    u = x - x_mean
    v = y - y_mean
    
    # Linear system
    Suv = np.sum(u * v)
    Suu = np.sum(u * u)
    Svv = np.sum(v * v)
    Suuv = np.sum(u * u * v)
    Suvv = np.sum(u * v * v)
    Suuu = np.sum(u * u * u)
    Svvv = np.sum(v * v * v)
    
    # Solving for the center and radius
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    
    try:
        center = np.linalg.solve(A, b)
        center[0] += x_mean
        center[1] += y_mean
        r = np.sqrt(np.mean((x - center[0])**2 + (y - center[1])**2))
        return center[0], center[1], r
    except np.linalg.LinAlgError:
        print("Error in circle fitting: Singular matrix")
        return None, None, None

def correct_for_perspective(measurement_pixels, baseline_height, current_height, image_width_pixels):
    """Correct measurements for perspective based on height"""
    if baseline_height == current_height or baseline_height == 0:
        return measurement_pixels
    
    # Linear correction based on relative heights
    correction_factor = baseline_height / current_height
    corrected_measurement = measurement_pixels * correction_factor
    
    return corrected_measurement

def estimate_wheel_height(contour, image_height):
    """Estimate wheel height based on contour position in image"""
    if contour is None or len(contour) == 0:
        return None
    
    _, y, _, h = cv2.boundingRect(contour)
    
    # Normalize y position relative to image height
    normalized_y = y / image_height
    
    # Simple linear mapping - this would need calibration
    estimated_height = (1.0 - normalized_y) * 100  # Map to 0-100mm range
    
    return estimated_height
