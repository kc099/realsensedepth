import cv2
import numpy as np
import torch
import math
import os
import pyrealsense2 as rs
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

from wheel_measurements import calculate_real_dimensions, measure_wheel_height_from_depth
from utils import fit_circle_least_squares

# Global model variable
wheel_detection_model = None

def get_model_instance_segmentation(num_classes):
    """
    Create and configure the object detection model
    
    Args:
        num_classes (int): Number of classes to detect (usually 2: background and wheel)
        
    Returns:
        model: PyTorch MaskRCNN model
    """
    # Load pre-trained model
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    
    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def load_model(device="cpu"):
    """
    Load the wheel detection model
    
    Args:
        device (str): Device to run the model on (cpu/cuda)
        
    Returns:
        model: PyTorch model
    """
    global wheel_detection_model
    
    if wheel_detection_model is None:
        try:
            # Initialize model
            model = get_model_instance_segmentation(num_classes=2)  # Background and wheel
            
            # Load weights if available
            model_path = "maskrcnn_wheel_best.pth"
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Loaded model from", model_path)
            except Exception as e:
                print(f"Could not load model weights: {e}. Using pre-trained weights.")
            
            # Set model to evaluation mode
            model.eval()
            wheel_detection_model = model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    return wheel_detection_model

def process_frame(frame, is_top_view=True, camera_settings=None, wheel_models=None, selected_model=None):
    """
    Process a frame through the detection model and calculate real-world dimensions
    
    Parameters:
        frame: Image frame to process
        is_top_view: If True, processes as top view (diameter calculation)
                     If False, processes as side view (height calculation)
        camera_settings: Camera calibration settings
        wheel_models: Dictionary of wheel model specifications
        selected_model: Currently selected wheel model name
                     
    Returns:
        tuple: (processed_image, measurements_dict)
    """
    if frame is None:
        return None, {}
    
    # Make a copy to avoid modifying the original
    visual_frame = frame.copy()
    
    # Load model if not already loaded
    model = load_model()
    if model is None:
        # If model fails to load, fallback to basic OpenCV processing
        return fallback_processing(frame, is_top_view, camera_settings)
    
    try:
        # Prepare image for model
        image = frame.copy()
        
        # Convert to RGB for PyTorch model (which expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PyTorch tensor
        image_tensor = F.to_tensor(image_rgb)
        
        # Make sure dimensions are NCHW (batch, channels, height, width)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Process predictions
        if len(predictions) > 0 and len(predictions[0]['boxes']) > 0:
            # Get the prediction with highest score
            scores = predictions[0]['scores'].cpu().numpy()
            high_scores_idxs = np.where(scores > 0.7)[0].tolist()
            
            if len(high_scores_idxs) > 0:
                # Use the highest scoring prediction
                idx = high_scores_idxs[0]
                
                # Get bounding box
                box = predictions[0]['boxes'][idx].cpu().numpy().astype(np.int32)
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2-x1, y2-y1
                
                # Get mask
                mask = predictions[0]['masks'][idx, 0].cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8) * 255
                
                # Different processing based on view
                if is_top_view:
                    # For top view, find the circle diameter
                    processed_img, measurements = process_top_view(
                        visual_frame, mask, (x, y, w, h), camera_settings, wheel_models, selected_model
                    )
                else:
                    # For side view, calculate height from depth or conventional methods
                    processed_img, measurements = process_side_view(
                        visual_frame, mask, (x, y, w, h), camera_settings, wheel_models, selected_model
                    )
                
                return processed_img, measurements
    
    except Exception as e:
        print(f"Error processing frame: {e}")
    
    # Fallback to basic processing if model fails
    return fallback_processing(frame, is_top_view, camera_settings)

def fallback_processing(frame, is_top_view, camera_settings):
    """
    Basic fallback processing when the model fails
    
    Args:
        frame: Image frame
        is_top_view: True if processing top view
        camera_settings: Camera calibration settings
        
    Returns:
        tuple: (processed_image, measurements_dict)
    """
    print("Using fallback image processing")
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a visual copy
    visual_frame = frame.copy()
    
    # Initialize empty measurements
    measurements = {}
    
    if contours:
        # Find the largest contour (assume it's the wheel)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw contour and bounding box
        cv2.drawContours(visual_frame, [largest_contour], -1, (0, 255, 0), 2)
        cv2.rectangle(visual_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Create a mask from the contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        if is_top_view:
            # For top view, try to fit a circle to the contour points
            try:
                # Get contour points
                contour_points = largest_contour.reshape(-1, 2)
                
                # Fit circle
                cx, cy, radius = fit_circle_least_squares(contour_points)
                
                if radius is not None:
                    # Draw the detected circle
                    cv2.circle(visual_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
                    
                    # Add measurement data
                    measurements["circle_center"] = (cx, cy)
                    measurements["radius_pixels"] = radius
                    
                    # Add text with radius info
                    cv2.putText(visual_frame, f"Radius: {radius:.1f}px", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except:
                print("Circle fitting failed in fallback processing")
        else:
            # For side view, use the bounding box height
            measurements["height_pixels"] = h
            
            # Add text with height info
            cv2.putText(visual_frame, f"Height: {h}px", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Calculate real-world dimensions if we have camera settings
    if camera_settings and measurements:
        if is_top_view and "radius_pixels" in measurements:
            measurements = calculate_real_dimensions(measurements, True, camera_settings)
        elif not is_top_view and "height_pixels" in measurements:
            measurements = calculate_real_dimensions(measurements, False, camera_settings)
    
    return visual_frame, measurements

def process_top_view(frame, mask, box, camera_settings, wheel_models, selected_model):
    """
    Process top view image to detect wheel diameter
    
    Args:
        frame: Image frame
        mask: Binary mask of the wheel
        box: Bounding box (x, y, w, h)
        camera_settings: Camera calibration settings
        wheel_models: Dictionary of wheel models
        selected_model: Currently selected model name
        
    Returns:
        tuple: (processed_image, measurements_dict)
    """
    x, y, w, h = box
    visual_frame = frame.copy()
    
    # Draw bounding box
    cv2.rectangle(visual_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Find contour from mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    measurements = {}
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw contour
        cv2.drawContours(visual_frame, [largest_contour], -1, (0, 255, 0), 2)
        
        # Try to fit a circle to the contour
        try:
            # Get contour points
            contour_points = largest_contour.reshape(-1, 2)
            
            # Fit circle
            cx, cy, radius = fit_circle_least_squares(contour_points)
            
            if radius is not None:
                # Draw the detected circle
                cv2.circle(visual_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
                
                # Add circle center marker
                cv2.drawMarker(visual_frame, (int(cx), int(cy)), (0, 0, 255), 
                              markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                
                # Add measurement data
                measurements["circle_center"] = (cx, cy)
                measurements["radius_pixels"] = radius
                
                # Get model data for the selected model
                if wheel_models and selected_model and selected_model in wheel_models:
                    measurements["model_data"] = wheel_models[selected_model]
                
                # Calculate real-world dimensions
                if camera_settings:
                    real_measurements = calculate_real_dimensions(measurements, True, camera_settings)
                    
                    # Update with real-world measurements
                    measurements.update(real_measurements)
                    
                    # Add diameter text
                    if "diameter_mm" in real_measurements:
                        diameter_mm = real_measurements["diameter_mm"]
                        cv2.putText(visual_frame, f"Diameter: {diameter_mm:.1f}mm", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Add pass/fail indicator based on model tolerance
                        if "is_ok" in real_measurements:
                            is_ok = real_measurements["is_ok"]
                            status_text = "PASS" if is_ok else "FAIL"
                            status_color = (0, 255, 0) if is_ok else (0, 0, 255)
                            
                            cv2.putText(visual_frame, status_text, 
                                       (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        except Exception as e:
            print(f"Error in circle fitting: {e}")
    
    return visual_frame, measurements

def detect_object(color_frame, depth_frame):
    """
    Detect object in the center of the frame and calculate its distance and height
    using morphology-based contour detection and depth thresholding
    
    Args:
        color_frame: RGB image from the camera
        depth_frame: Depth frame from RealSense camera
        
    Returns:
        tuple: (processed_image, depth_image, measurements)
    """
    try:
        # Create colorizer for depth visualization
        colorizer = rs.colorizer()
        
        # Convert to numpy arrays if not already
        if not isinstance(color_frame, np.ndarray):
            color_image = np.asanyarray(color_frame.get_data())
        else:
            color_image = color_frame.copy()
            
        # Get the depth image as numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Create a copy for drawing on
        display_image = color_image.copy()
        
        # Get image dimensions
        img_height, img_width = color_image.shape[:2]
        
        # Mark center of the image
        center_x, center_y = img_width // 2, img_height // 2
        
        # Draw crosshair at center
        cv2.line(display_image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(display_image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
        
        # Get intrinsics for point deprojection
        color_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Get depth scale for accurate distance measurement
        try:
            depth_sensor = depth_frame.get_sensor()
            depth_scale = depth_sensor.get_depth_scale()
        except:
            # Use default depth scale if unable to get from sensor
            depth_scale = 0.001  # Default scale for RealSense cameras
            print(f"Using default depth scale: {depth_scale}")
            
        # Get distance at center point
        depth_value = depth_frame.get_distance(center_x, center_y)
        
        # Deproject to 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(color_intrin, [center_x, center_y], depth_value)
        
        # Calculate distance
        distance = math.sqrt(point_3d[0]**2 + point_3d[1]**2 + point_3d[2]**2)
        
        # Convert depth image to mm for thresholding
        depth_mm = depth_image * depth_scale * 1000  # Convert to mm
        
        # Define depth thresholds based on center point depth
        # Get depth at center in mm
        center_depth_mm = depth_value * 1000
        
        # Set depth range to detect objects (+-500mm from center point depth)
        depth_min = max(center_depth_mm - 500, 0)
        depth_max = center_depth_mm + 500
        
        # Create normalized depth image for visualization
        depth_normalized = np.clip(depth_mm, depth_min, depth_max)
        valid_depth_indices = (depth_normalized > depth_min) & (depth_normalized < depth_max)
        if np.any(valid_depth_indices):
            min_val = np.min(depth_normalized[valid_depth_indices])
            max_val = np.max(depth_normalized[valid_depth_indices])
            if max_val > min_val:
                depth_normalized = ((depth_normalized - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_normalized, dtype=np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_normalized, dtype=np.uint8)
        
        # Helper function to get valid depth in a window
        def get_valid_depth_in_window(x, y, depth_img, window_size=5):
            half_window = window_size // 2
            h, w = depth_img.shape
            
            # Ensure window is within image bounds
            x_start = max(x - half_window, 0)
            x_end = min(x + half_window + 1, w)
            y_start = max(y - half_window, 0)
            y_end = min(y + half_window + 1, h)
            
            # Extract window
            window = depth_img[y_start:y_end, x_start:x_end]
            
            # Get valid depths (non-zero)
            valid_depths = window[window > 0]
            
            if len(valid_depths) > 0:
                # Return median of valid depths
                return np.median(valid_depths)
            else:
                return 0
        
        # Create binary thresholded image for object detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Use Otsu's thresholding for more reliable segmentation
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up binary image with morphological operations
        kernel = np.ones((7, 7), np.uint8)  # Larger kernel for better noise removal
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Create depth threshold mask - filter out pixels outside the desired range
        depth_mask = np.zeros_like(binary)
        
        # Narrow the depth range for better focus on objects in front
        depth_min = max(center_depth_mm - 300, 0)  # More focused range (was 500)
        depth_max = center_depth_mm + 300           # More focused range (was 500)
        
        valid_depth = (depth_mm >= depth_min) & (depth_mm <= depth_max)
        depth_mask[valid_depth] = 255
        
        # Apply distance-based mask: focus on objects near the center point
        center_mask = np.zeros_like(binary)
        cv2.circle(center_mask, (center_x, center_y), img_width//4, 255, -1)  # Create circular mask
        
        # Ensure all masks have the same shape
        if depth_mask.shape != binary.shape:
            depth_mask = cv2.resize(depth_mask, (binary.shape[1], binary.shape[0]))
        if center_mask.shape != binary.shape:
            center_mask = cv2.resize(center_mask, (binary.shape[1], binary.shape[0]))
            
        # Combine all masks for better object detection
        binary = binary.astype(np.uint8)
        depth_mask = depth_mask.astype(np.uint8)
        center_mask = center_mask.astype(np.uint8)
        
        # Perform bitwise operations to combine masks
        # First combine depth mask with binary mask
        depth_binary_mask = cv2.bitwise_and(binary, depth_mask)
        # Then combine with center mask to focus on objects in front
        combined_mask = cv2.bitwise_and(depth_binary_mask, center_mask)
        
        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw all the contours on the image for debugging
        cv2.drawContours(display_image, contours, -1, (0, 255, 0), 2)  # Thicker green outline
        
        # Initialize variables
        obj_height = 0.0
        obj_class = "unknown"
        obj_confidence = 0.0
        obj_bbox = None
        has_detection = False
        
        # Process contours if any were found
        if contours:
            # Filter contours by area to remove noise
            min_area = 2000  # Increased minimum area for better detection
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            # Find the contour closest to the center
            closest_contour = None
            min_distance = float('inf')
            
            for cnt in valid_contours:
                # Get centroid of contour
                M = cv2.moments(cnt)
                if M['m00'] != 0:  # Prevent division by zero
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Calculate distance to center
                    dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    if dist < min_distance:
                        min_distance = dist
                        closest_contour = cnt
            
            if closest_contour is not None:
                # Draw the selected contour prominently
                cv2.drawContours(display_image, [closest_contour], 0, (0, 255, 255), 3)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(closest_contour)
                obj_bbox = (x, y, x + w, y + h)
                
                # Draw prominent bounding box
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                
                # Calculate measurement points more accurately
                # Use the actual contour for height measurement instead of just bounding box
                # Find the topmost and bottommost points of the contour
                topmost = tuple(closest_contour[closest_contour[:, :, 1].argmin()][0])
                bottommost = tuple(closest_contour[closest_contour[:, :, 1].argmax()][0])
                
                center_top_x, center_top_y = topmost
                center_bottom_x, center_bottom_y = bottommost
                
                # Draw points clearly
                cv2.circle(display_image, (center_top_x, center_top_y), 7, (255, 0, 0), -1)  # Blue for top
                cv2.circle(display_image, (center_bottom_x, center_bottom_y), 7, (0, 0, 255), -1)  # Red for bottom
                
                # Draw center point of object
                M = cv2.moments(closest_contour)
                if M['m00'] != 0:  # Prevent division by zero
                    bbox_center_x = int(M['m10'] / M['m00'])
                    bbox_center_y = int(M['m01'] / M['m00'])
                    cv2.circle(display_image, (bbox_center_x, bbox_center_y), 7, (0, 255, 255), -1)  # Yellow for center
                else:
                    bbox_center_x, bbox_center_y = x + w//2, y + h//2
                    cv2.circle(display_image, (bbox_center_x, bbox_center_y), 7, (0, 255, 255), -1)  # Yellow for center
                
                # Draw line connecting top and bottom points
                cv2.line(display_image, (center_top_x, center_top_y), 
                         (center_bottom_x, center_bottom_y), (0, 255, 0), 3)
                
                # Get depth values with window-based approach for robustness
                center_depth_value = get_valid_depth_in_window(bbox_center_x, bbox_center_y, depth_image, 9)
                top_depth_value = get_valid_depth_in_window(center_top_x, center_top_y, depth_image, 9)
                bottom_depth_value = get_valid_depth_in_window(center_bottom_x, center_bottom_y, depth_image, 9)
                
                # Convert raw depth values to meters
                center_dist = center_depth_value * depth_scale
                top_dist = top_depth_value * depth_scale
                bottom_dist = bottom_depth_value * depth_scale
                
                # Check if depth values are valid
                if all(d > 0 for d in [center_dist, top_dist, bottom_dist]):
                    # Convert 2D points to 3D using depth
                    top_point = rs.rs2_deproject_pixel_to_point(
                        color_intrin, [center_top_x, center_top_y], top_dist)
                    bottom_point = rs.rs2_deproject_pixel_to_point(
                        color_intrin, [center_bottom_x, center_bottom_y], bottom_dist)
                    
                    # Calculate height in 3D space
                    obj_height = np.sqrt(
                        (top_point[0] - bottom_point[0])**2 + 
                        (top_point[1] - bottom_point[1])**2 + 
                        (top_point[2] - bottom_point[2])**2)
                    
                    # Set detection flag
                    has_detection = True
                    obj_class = "object"
                    obj_confidence = 1.0
        
        # If no object detected using contours, calculate height from center to bottom of frame
        if not has_detection:
            # Bottom point is 20 pixels from bottom of frame
            bottom_y = img_height - 20
            bottom_x = center_x
            
            # Get valid depth at bottom point using window approach
            bottom_depth_value = get_valid_depth_in_window(bottom_x, bottom_y, depth_image, 9)
            bottom_dist = bottom_depth_value * depth_scale
            
            if bottom_dist > 0:
                # Convert 2D bottom point to 3D
                bottom_point = rs.rs2_deproject_pixel_to_point(color_intrin, [bottom_x, bottom_y], bottom_dist)
                
                # Calculate height as difference between center and bottom
                obj_height = abs(point_3d[1] - bottom_point[1])
                
                # Draw line from center to bottom
                cv2.line(display_image, (center_x, center_y), (bottom_x, bottom_y), (0, 255, 255), 2)
                
                # Mark manually measured points
                cv2.circle(display_image, (center_x, center_y), 5, (0, 255, 255), -1)
                cv2.circle(display_image, (bottom_x, bottom_y), 5, (0, 255, 255), -1)
        
        # Add measurement text with background for better visibility
        distance_text = f"Distance: {distance:.3f} m"
        depth_text = f"Depth: {depth_value:.3f} m"
        height_text = f"Height: {obj_height*100:.1f} cm"
        
        # Function to add text with background
        def put_text_with_background(img, text, position, font, scale, text_color, thickness):
            # Get text size
            text_size, _ = cv2.getTextSize(text, font, scale, thickness)
            # Create background rectangle
            cv2.rectangle(img, 
                          (position[0]-5, position[1]-text_size[1]-5), 
                          (position[0]+text_size[0]+5, position[1]+5), 
                          (0, 0, 0), -1)  # Black background
            # Add text
            cv2.putText(img, text, position, font, scale, text_color, thickness)
            
        # Add text with background
        put_text_with_background(display_image, distance_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        put_text_with_background(display_image, depth_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        put_text_with_background(display_image, height_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if has_detection:
            put_text_with_background(display_image, "Object Detected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Create a colorized depth image for visualization
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        
        # Draw bounding box on depth image if object detected
        if has_detection and obj_bbox:
            cv2.rectangle(colorized_depth, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), (255, 255, 255), 2)
        
        measurements = {
            "distance": distance,
            "depth": depth_value,
            "height": obj_height * 100,  # Convert to cm
            "class": obj_class,
            "confidence": obj_confidence
        }
        
        # Return processed color image with overlays, depth image, and measurements
        return display_image, colorized_depth, measurements
        
    except Exception as e:
        print(f"Error in detect_object: {e}")
        import traceback
        traceback.print_exc()
        return None, None, {"error": str(e)}

def process_side_view(frame, mask, box, camera_settings, wheel_models, selected_model, depth_image=None):
    """
    Process side view image to detect wheel height
    
    Args:
        frame: Image frame
        mask: Binary mask of the wheel
        box: Bounding box (x, y, w, h)
        camera_settings: Camera calibration settings
        wheel_models: Dictionary of wheel models
        selected_model: Currently selected model name
        depth_image: Optional depth image from RealSense camera
        
    Returns:
        tuple: (processed_image, measurements_dict)
    """
    x, y, w, h = box
    visual_frame = frame.copy()
    
    # Draw bounding box
    cv2.rectangle(visual_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Initialize measurements
    measurements = {
        "height_pixels": h,
        "width_pixels": w
    }
    
    # Get model data for the selected model
    if wheel_models and selected_model and selected_model in wheel_models:
        measurements["model_data"] = wheel_models[selected_model]
    
    # If we have depth data, use it for more accurate measurements
    if depth_image is not None:
        return measure_wheel_height_from_depth(depth_image, frame, mask, box, camera_settings)
    
    # Otherwise use conventional processing
    # Draw the measurement line
    mid_x = x + w // 2
    cv2.line(visual_frame, (mid_x, y), (mid_x, y + h), (0, 255, 0), 2)
    
    # Calculate real-world dimensions
    if camera_settings:
        real_measurements = calculate_real_dimensions(measurements, False, camera_settings)
        
        # Update with real-world measurements
        measurements.update(real_measurements)
        
        # Add height text
        if "height_mm" in real_measurements:
            height_mm = real_measurements["height_mm"]
            cv2.putText(visual_frame, f"Height: {height_mm:.1f}mm", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add pass/fail indicator based on model tolerance
            if "is_ok" in real_measurements:
                is_ok = real_measurements["is_ok"]
                status_text = "PASS" if is_ok else "FAIL"
                status_color = (0, 255, 0) if is_ok else (0, 0, 255)
                
                cv2.putText(visual_frame, status_text, 
                           (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    
    return visual_frame, measurements
