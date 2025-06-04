import cv2
import numpy as np
import math
import os
import threading
import time
import traceback

# Import PyTorch dependencies directly (no lazy loading)
try:
    import torch
    from torchvision.transforms import functional as F
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    TORCH_AVAILABLE = True
    print("PyTorch dependencies loaded successfully")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch dependencies not available: {e}")

from wheel_measurements import calculate_real_dimensions, measure_wheel_height_from_depth
from utils import fit_circle_least_squares
from camera_utils import load_camera_intrinsics, project_point_to_3d, calculate_3d_distance, get_valid_depth_in_window

# Configuration
MODEL_PATH = "./maskrcnn_wheel_best.pth"
SCORE_THRESHOLD = 0.5
RATIO_THRESHOLD = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None

# Global model variable
wheel_detection_model = None

def get_model_instance_segmentation(num_classes):
    """Create and configure the object detection model"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot create model")
        return None
        
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

def load_model():
    """Load the wheel detection model synchronously"""
    global wheel_detection_model
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot load model")
        return None
        
    if wheel_detection_model is not None:
        print("Model already loaded")
        return wheel_detection_model
     
    try:
        print("Loading wheel detection model...")
        model = get_model_instance_segmentation(num_classes=2)  # Background and wheel
        
        if model is None:
            print("Failed to create model instance")
            return None
        
        # Move to device
        if DEVICE is not None:
            model.to(DEVICE)
            print(f"Model moved to device: {DEVICE}")
        
        # Check for model weights file
        model_exists = os.path.exists(MODEL_PATH)
        print(f"Model file exists at {MODEL_PATH}: {model_exists}")
        
        # Load weights if available
        if model_exists:
            print(f"Loading model weights from {MODEL_PATH}")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model weights loaded successfully")
        else:
            print(f"Model file not found at {MODEL_PATH}. Using pre-trained COCO weights.")
            print("Note: This may not work well for wheel detection without fine-tuning.")
        
        # Set model to evaluation mode
        model.eval()
        wheel_detection_model = model
        print("Model loaded and ready for inference")
        return wheel_detection_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def load_model_in_background():
    """Start loading the model in a background thread"""
    def _load_model_thread():
        try:
            result = load_model()
            if result is not None:
                print("Background model loading completed successfully")
            else:
                print("Background model loading failed")
        except Exception as e:
            print(f"Error in background model loading: {e}")
            traceback.print_exc()
    
    # Start loading in background thread
    threading.Thread(target=_load_model_thread, daemon=True).start()
    print("Started background model loading thread")

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

def process_frame(frame, is_top_view=True, camera_settings=None, wheel_models=None, selected_model=None, wheel_height_mm=None, depth_frame=None):
    """
    Process a frame through the detection model and calculate real-world dimensions
    
    Parameters:
        frame: Image frame to process
        is_top_view: If True, processes as top view (diameter calculation)
                     If False, processes as side view (height calculation)
        camera_settings: Camera calibration settings
        wheel_models: Dictionary of wheel model specifications
        selected_model: Currently selected wheel model name
        wheel_height_mm: Height of wheel measured from side view (in mm),
                         used for accurate diameter calculation in top view
        depth_frame: Depth image from RealSense camera (required for side view)
    """
    global wheel_detection_model
    
    print(f"Processing frame - is_top_view: {is_top_view}, model loaded: {wheel_detection_model is not None}")
    
    # Load model if not loaded yet
    if wheel_detection_model is None:
        print("Model not loaded yet, attempting to load...")
        wheel_detection_model = load_model()
        
        if wheel_detection_model is None:
            print("Failed to load model - using fallback processing")
            return fallback_processing(frame, is_top_view, camera_settings, wheel_models, selected_model, wheel_height_mm, depth_frame)
    
    try:
        print("Converting frame for model input...")
        # Convert frame to RGB for model input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PyTorch tensor and normalize
        image_tensor = F.to_tensor(frame_rgb)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        if DEVICE is not None:
            image_tensor = image_tensor.to(DEVICE)
        
        print("Running model inference...")
        # Get model predictions
        with torch.no_grad():
            predictions = wheel_detection_model(image_tensor)
        
        # Get the first (and only) prediction
        pred = predictions[0]
        
        # Get masks and boxes
        masks = pred['masks'].cpu().numpy()
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        print(f"Model predictions: {len(scores)} detections found")
        print(f"Scores: {scores}")
        
        # Filter predictions by confidence
        valid_detections = scores > SCORE_THRESHOLD
        
        print(f"Valid detections (confidence > {SCORE_THRESHOLD}): {np.sum(valid_detections)}")
        
        if not np.any(valid_detections):
            print("No wheel detected with sufficient confidence - using fallback processing")
            return fallback_processing(frame, is_top_view, camera_settings, wheel_models, selected_model, wheel_height_mm, depth_frame)
        
        # Get the best detection
        best_idx = np.argmax(scores[valid_detections])
        mask = masks[valid_detections][best_idx][0]  # Get binary mask
        box = boxes[valid_detections][best_idx]      # Get bounding box
        best_score = scores[valid_detections][best_idx]
        
        print(f"Using detection with confidence: {best_score:.3f}")
        
        # Convert mask to binary
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        print(f"Mask shape: {mask_binary.shape}, non-zero pixels: {np.sum(mask_binary > 0)}")
        
        # Process based on view type
        if is_top_view:
            return process_top_view(frame, mask_binary, box, camera_settings, wheel_models, selected_model, wheel_height_mm)
        else:
            return process_side_view(frame, mask_binary, box, camera_settings, wheel_models, selected_model, depth_frame)
            
    except Exception as e:
        print(f"Error processing frame with model: {e}")
        traceback.print_exc()
        print("Falling back to basic processing...")
        return fallback_processing(frame, is_top_view, camera_settings, wheel_models, selected_model, wheel_height_mm, depth_frame)

def fallback_processing(frame, is_top_view, camera_settings, wheel_models, selected_model, wheel_height_mm, depth_frame):
    """
    Fallback processing when MaskRCNN model is not available or fails
    Uses basic image processing techniques to estimate wheel boundaries
    """
    print("Using fallback processing (no AI model)")
    
    try:
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to create binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found in fallback processing")
            return None
        
        # Find the largest contour (assume it's the wheel)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask from largest contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        box = [x, y, x + w, y + h]
        
        print(f"Fallback processing found contour with area: {cv2.contourArea(largest_contour)}")
        
        # Process based on view type
        if is_top_view:
            if wheel_height_mm is None:
                print("Warning: No wheel height provided for top view processing")
                return None
            return process_top_view(frame, mask, box, camera_settings, wheel_models, selected_model, wheel_height_mm)
        else:
            return process_side_view(frame, mask, box, camera_settings, wheel_models, selected_model, depth_frame)
            
    except Exception as e:
        print(f"Error in fallback processing: {e}")
        traceback.print_exc()
        return None

def process_top_view(frame, mask_binary, box, camera_settings, wheel_models, selected_model, wheel_height_mm):
    """
    Process top view frame to calculate wheel diameter
    
    Parameters:
        frame: Color image frame
        mask_binary: Binary mask of detected wheel
        box: Bounding box of detected wheel
        camera_settings: Camera calibration settings
        wheel_models: Dictionary of wheel model specifications
        selected_model: Currently selected wheel model name
        wheel_height_mm: Height of wheel measured from side view (in mm)
    """
    try:
        print("Processing top view for diameter calculation...")
        
        if wheel_height_mm is None:
            print("Warning: No wheel height provided for top view processing")
            return None
            
        # Get camera intrinsics
        intrinsics = load_camera_intrinsics("top_camera")
        if intrinsics is None:
            print("Could not load camera intrinsics")
            return None
            
        # Get camera height from base (in mm)
        camera_height = camera_settings.get('camera_height_mm', 1000) if camera_settings else 1000
        
        # Calculate distance to wheel center
        distance_to_wheel = camera_height - wheel_height_mm
        
        # Find contours from mask
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found in mask")
            return None
            
        # Get largest contour (should be the wheel)
        main_contour = max(contours, key=cv2.contourArea)
        contour_points = np.squeeze(main_contour, axis=1)
        
        # Fit circle to contour points
        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        cx_ls, cy_ls, r_ls = fit_circle_least_squares(contour_points)
        
        # Use least squares circle if valid
        if cx_ls is not None and cy_ls is not None and r_ls is not None:
            cx, cy, radius = cx_ls, cy_ls, r_ls
        
        # Calculate pixel diameter
        pixel_diameter = radius * 2
        
        # Convert to real-world diameter using camera intrinsics and distance
        focal_length = intrinsics['fx']  # Use x focal length
        diameter_mm = (pixel_diameter * distance_to_wheel) / focal_length
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw mask outline
        cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 2)
        
        # Draw fitted circle
        cv2.circle(vis_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
        cv2.circle(vis_frame, (int(cx), int(cy)), 2, (255, 0, 0), 3)
        
        # Add measurement text
        text = f"Diameter: {diameter_mm:.1f}mm"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add debug info
        debug_text = f"Distance: {distance_to_wheel:.1f}mm, Radius: {radius:.1f}px"
        cv2.putText(vis_frame, debug_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        print(f"Top view processing complete - diameter: {diameter_mm:.1f}mm")
        
        return {
            'diameter_mm': diameter_mm,
            'visualization': vis_frame,
            'mask': mask_binary,
            'box': box,
            'center': (cx, cy),
            'radius': radius
        }
        
    except Exception as e:
        print(f"Error in top view processing: {e}")
        traceback.print_exc()
        return None

def process_side_view(frame, mask_binary, box, camera_settings, wheel_models, selected_model, depth_frame=None):
    """
    Process side view frame to calculate wheel height
    
    Parameters:
        frame: Color image frame
        mask_binary: Binary mask of detected wheel
        box: Bounding box of detected wheel
        camera_settings: Camera calibration settings
        wheel_models: Dictionary of wheel model specifications
        selected_model: Currently selected wheel model name
        depth_frame: Depth image from RealSense camera (None for uploaded images)
    """
    try:
        print(f"Processing side view - mask shape: {mask_binary.shape}, depth available: {depth_frame is not None}")
        
        # Find top and bottom center points of the mask
        mask_points = np.where(mask_binary > 0)
        if len(mask_points[0]) == 0:
            print("No valid points in mask")
            return None
            
        print(f"Found {len(mask_points[0])} mask pixels")
            
        # Get vertical center line
        center_x = int(np.mean(mask_points[1]))
        top_y = np.min(mask_points[0])
        bottom_y = np.max(mask_points[0])
        
        print(f"Mask bounds - center_x: {center_x}, top_y: {top_y}, bottom_y: {bottom_y}")
        
        height_mm = None
        
        if depth_frame is not None:
            print("Attempting depth-based height calculation...")
            # Use depth data for accurate height calculation
            intrinsics = load_camera_intrinsics("side_camera")
            if intrinsics is None:
                print("Could not load camera intrinsics")
            else:
                print(f"Loaded intrinsics: fx={intrinsics['fx']}, fy={intrinsics['fy']}")
                
                # Get depth scale from camera settings
                depth_scale = camera_settings.get('depth_scale', 0.001) if camera_settings else 0.001
                print(f"Using depth scale: {depth_scale}")
                
                # Get depth values at top and bottom points
                top_depth = get_valid_depth_in_window(center_x, top_y, depth_frame)
                bottom_depth = get_valid_depth_in_window(center_x, bottom_y, depth_frame)
                
                print(f"Depth values - top: {top_depth}, bottom: {bottom_depth}")
                
                if top_depth is not None and bottom_depth is not None:
                    # Convert depth values to real-world coordinates
                    top_point_3d = project_point_to_3d(center_x, top_y, top_depth, intrinsics, depth_scale)
                    bottom_point_3d = project_point_to_3d(center_x, bottom_y, bottom_depth, intrinsics, depth_scale)
                    
                    if top_point_3d is not None and bottom_point_3d is not None:
                        # Calculate height in mm
                        height_mm = calculate_3d_distance(top_point_3d, bottom_point_3d)
                        print(f"Calculated depth-based height: {height_mm:.1f}mm")
                else:
                    print("Could not get valid depth values")
        
        if height_mm is None:
            # Fallback: estimate height using pixel measurements and known scale
            print("Using pixel-based height estimation (less accurate)")
            pixel_height = bottom_y - top_y
            print(f"Pixel height: {pixel_height} pixels")
            
            # Try to get estimated scale from camera settings or use default
            pixels_per_mm = camera_settings.get('pixels_per_mm', 2.0) if camera_settings else 2.0
            height_mm = pixel_height / pixels_per_mm
            print(f"Initial height estimate: {height_mm:.1f}mm (using {pixels_per_mm} pixels/mm)")
            
            # If we have a selected model, we can adjust based on expected height
            if selected_model and wheel_models and selected_model in wheel_models:
                expected_height = wheel_models[selected_model].get('height_mm', height_mm)
                # Use expected height as a reference for scaling
                if expected_height and pixel_height > 0:
                    pixels_per_mm = pixel_height / expected_height
                    height_mm = expected_height
                    print(f"Using model reference height: {height_mm:.1f}mm")
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw mask outline
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 2)
        
        # Draw measurement line
        cv2.line(vis_frame, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)
        
        # Add measurement points
        cv2.circle(vis_frame, (center_x, top_y), 5, (255, 0, 0), -1)
        cv2.circle(vis_frame, (center_x, bottom_y), 5, (255, 0, 0), -1)
        
        # Add measurement text
        method = "depth-based" if depth_frame is not None and height_mm else "pixel-based"
        text = f"Height: {height_mm:.1f}mm ({method})"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add pixel dimensions for debugging
        pixel_height = bottom_y - top_y
        debug_text = f"Pixels: {pixel_height}px"
        cv2.putText(vis_frame, debug_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        print(f"Side view processing complete - height: {height_mm:.1f}mm")
        
        return {
            'height_mm': height_mm,
            'visualization': vis_frame,
            'mask': mask_binary,
            'box': box
        }
        
    except Exception as e:
        print(f"Error in side view processing: {e}")
        traceback.print_exc()
        return None
