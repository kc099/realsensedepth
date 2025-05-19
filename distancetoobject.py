import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os
from datetime import datetime

# Create directory to save captures
if not os.path.exists("captures"):
    os.makedirs("captures")

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Load MobileNetSSD model
prototxt = "./MobileNetSSD_deploy.prototxt"
model = "./MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = ["aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "mobile", "cellphone"]

# CLASSES = ["object"]

# Alignment setup
align_to = rs.stream.color
align = rs.align(align_to)

# Start pipeline
pipeline.start(config)

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
    """Add UI panel to the image"""
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (300, 120), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    y_offset = 30
    for line in status_text.split('\n'):
        cv2.putText(image, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    return image

def process_frame(color_image, depth_image):
    global auto_capture, last_capture_time
    
    distance_info = ""
    if detection_active:
        # Object detection
        (h, w) = color_image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                #idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Calculate distance
                depth_roi = depth_image[startY:endY, startX:endX].astype(float)
                depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
                depth_roi = depth_roi * depth_scale
                dist = np.mean(depth_roi)

                label = f"Object: {dist:.3f}m"
                #label = f"{CLASSES[idx]}: {dist:.2f}m"
                distance_info += f"{label}\n"
                
                # Draw detection
                cv2.rectangle(color_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(color_image, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Print to terminal
                print(f"Detected: {label} at {datetime.now().strftime('%H:%M:%S')}")

        # Auto-capture logic
        current_time = time.time()
        if auto_capture and (current_time - last_capture_time >= 5):
            last_capture_time = current_time
            if distance_info:
                save_frame(color_image, depth_colormap, distance_info)
                print(f"Auto-captured: {distance_info.strip()}")

    return color_image, distance_info

# Main window setup
cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RealSense", 1280, 520)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Process frame
        processed_color, distance_info = process_frame(color_image.copy(), depth_image)

        # Create UI status
        status_text = f"Controls:\n[C] Capture  [A] Auto-capture: {'ON' if auto_capture else 'OFF'}\n"
        status_text += f"[D] Detection: {'ON' if detection_active else 'OFF'}\n"
        status_text += f"[Q] Quit\n\nDetected:\n{distance_info if distance_info else 'No objects'}"

        # Add UI overlay
        processed_color = add_ui_overlay(processed_color, status_text)

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
        elif key == ord('d'):
            detection_active = not detection_active
            print(f"Detection {'enabled' if detection_active else 'disabled'}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()