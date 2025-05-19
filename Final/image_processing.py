import cv2
import numpy as np
import torch
import time
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

from wheel_measurements import calculate_real_dimensions, measure_wheel_height_from_depth
from utils import fit_circle_least_squares

# Global model variables
wheel_detection_model = None
bottle_detection_model = None

# MobileNetSSD class labels
MOBILENET_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                  "sofa", "train", "tvmonitor"]

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
