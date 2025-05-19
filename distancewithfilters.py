import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os
from datetime import datetime
from scipy import ndimage

# Create directory to save captures
if not os.path.exists("captures"):
    os.makedirs("captures")

# Initialize RealSense pipeline with post-processing
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Depth processing filters
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

# Start pipeline
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Alignment setup
align_to = rs.stream.color
align = rs.align(align_to)

# UI Controls
auto_capture = False
last_capture_time = time.time()
detection_active = True

def save_frame(color_frame, depth_frame, distance_info):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"captures/color_{timestamp}.png", color_frame)
    cv2.imwrite(f"captures/depth_{timestamp}.png", depth_frame)
    with open(f"captures/distance_{timestamp}.txt", "w") as f:
        f.write(distance_info)

def add_ui_overlay(image, status_text):
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (300, 120), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    y_offset = 30
    for line in status_text.split('\n'):
        cv2.putText(image, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    return image

def get_accurate_distance(depth_frame, bbox):
    """Calculate median distance within bounding box with validation"""
    x1, y1, x2, y2 = bbox
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # Extract ROI and convert to meters
    depth_roi = depth_image[y1:y2, x1:x2].astype(float) * depth_scale
    
    # Apply median filter to remove outliers
    depth_roi = ndimage.median_filter(depth_roi, size=3)
    
    # Remove zeros (invalid measurements)
    valid_depths = depth_roi[depth_roi > 0]
    
    if len(valid_depths) == 0:
        return 0.0
    
    # Return median distance
    return np.median(valid_depths)

def process_frame(color_image, depth_frame):
    global auto_capture, last_capture_time
    
    distance_info = ""
    
    # Simple motion detection using frame difference
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if not hasattr(process_frame, "prev_gray"):
        process_frame.prev_gray = gray
        return color_image, distance_info
    
    frame_diff = cv2.absdiff(process_frame.prev_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Minimum object size
            continue
            
        (x, y, w, h) = cv2.boundingRect(contour)
        dist = get_accurate_distance(depth_frame, (x, y, x+w, y+h))
        
        if 0.1 < dist < 10.0:  # Valid distance range (0.1m to 10m)
            label = f"Object: {dist:.3f}m"
            distance_info += f"{label}\n"
            cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            print(f"Detected: {label} at {datetime.now().strftime('%H:%M:%S')}")
    
    process_frame.prev_gray = gray
    
    # Auto-capture logic
    current_time = time.time()
    if auto_capture and (current_time - last_capture_time >= 5):
        last_capture_time = current_time
        if distance_info:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), 
                cv2.COLORMAP_JET)
            save_frame(color_image, depth_colormap, distance_info)
            print(f"Auto-captured: {distance_info.strip()}")
    
    return color_image, distance_info

# Main window setup
cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RealSense", 1280, 520)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
               # Wait for frames and align
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Apply filters to the depth frame only
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        if not depth_frame or not color_frame:
            continue

        # Convert frames
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), 
            cv2.COLORMAP_JET)

        # Process frame
        processed_color, distance_info = process_frame(color_image.copy(), depth_frame)

        # Create UI status
        status_text = f"Controls:\n[C] Capture  [A] Auto-capture: {'ON' if auto_capture else 'OFF'}\n"
        status_text += f"[Q] Quit\n\nDetected:\n{distance_info if distance_info else 'No objects'}"

        # Add UI overlay
        processed_color = add_ui_overlay(processed_color, status_text)

        # Ensure both images have the same height before stacking
        if depth_colormap.shape[0] != processed_color.shape[0]:
            depth_colormap = cv2.resize(depth_colormap, (processed_color.shape[1], processed_color.shape[0]))


        # Display
        combined = np.hstack((processed_color, depth_colormap))
        cv2.imshow("RealSense", combined)

        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            if distance_info:
                save_frame(processed_color, depth_colormap, distance_info)
                print(f"Manual capture: {distance_info.strip()}")
        elif key == ord('a'):
            auto_capture = not auto_capture
            print(f"Auto-capture {'enabled' if auto_capture else 'disabled'}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()