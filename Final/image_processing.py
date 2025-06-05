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
SCORE_THRESHOLD = 0.85
SCORE_THRESHOLD1 = 0.5 #dont delete this
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
            
        # Set model to evaluation mode (always do this)
        model.eval()
        wheel_detection_model = model
        print("Model loaded and ready for inference")
        return wheel_detection_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def load_model_in_background():
    """Start loading the model in a background thread (only if not already loaded)"""
    global wheel_detection_model
    
    # Check if model is already loaded
    if wheel_detection_model is not None:
        print("Model already loaded - skipping background loading")
        return
    
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

def find_vertical_intersections(contour, center_x, image_shape):
    """
    Find the top and bottom intersection points of a vertical line with a contour
    
    Args:
        contour: OpenCV contour points
        center_x: X coordinate of the vertical line
        image_shape: Shape of the image (height, width)
        
    Returns:
        tuple: (top_y, bottom_y) intersection points or (None, None) if not found
    """
    try:
        # Extract contour points and reshape
        contour_points = contour.reshape(-1, 2)
        
        # Find all contour points that are close to our vertical center line
        # Allow a small tolerance (±2 pixels) to account for discrete pixels
        tolerance = 2
        near_center = contour_points[
            np.abs(contour_points[:, 0] - center_x) <= tolerance
        ]
        
        if len(near_center) == 0:
            print(f"No contour points found near center_x={center_x} with tolerance={tolerance}")
            
            # Fallback: interpolate intersections by checking each Y coordinate
            height = image_shape[0]
            intersections_y = []
            
            # Create a more detailed search by checking every Y coordinate
            for y in range(height):
                # Check if the vertical line at center_x intersects the contour at this y
                if cv2.pointPolygonTest(contour, (float(center_x), float(y)), False) >= 0:
                    intersections_y.append(y)
            
            if len(intersections_y) >= 2:
                top_y = min(intersections_y)
                bottom_y = max(intersections_y)
                print(f"Found intersections via polygon test: top_y={top_y}, bottom_y={bottom_y}")
                return top_y, bottom_y
            else:
                print(f"Polygon test found only {len(intersections_y)} intersections")
                return None, None
        else:
            # Found points near the center line, get the extremes
            y_coordinates = near_center[:, 1]
            top_y = int(np.min(y_coordinates))
            bottom_y = int(np.max(y_coordinates))
            
            print(f"Found {len(near_center)} contour points near center line")
            print(f"Top intersection: y={top_y}, Bottom intersection: y={bottom_y}")
            
            return top_y, bottom_y
            
    except Exception as e:
        print(f"Error finding vertical intersections: {e}")
        return None, None

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
            print("Failed to load model - cannot process frame")
            return None
    
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
            print("No wheel detected with sufficient confidence - processing failed")
            return None
        
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
        
        # Check if the fitted circle represents the complete wheel
        frame_height, frame_width = frame.shape[:2]
        circle_extends_beyond = (int(cx - radius) < 0 or int(cx + radius) >= frame_width or 
                               int(cy - radius) < 0 or int(cy + radius) >= frame_height)
        
        if circle_extends_beyond:
            print("Warning: Fitted circle extends beyond image boundaries")
            print("Diameter measurement may be inaccurate - ensure full wheel is visible")
        
        # Convert to real-world diameter using camera intrinsics and distance
        focal_length = intrinsics['fx']  # Use x focal length
        diameter_mm = (pixel_diameter * distance_to_wheel) / focal_length
        
        # Add accuracy warning to results if circle extends beyond boundaries
        accuracy_note = " (partial wheel)" if circle_extends_beyond else ""
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw mask outline
        cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 2)
        
        # Draw fitted circle with boundary checking
        frame_height, frame_width = vis_frame.shape[:2]
        
        # Check if circle extends beyond image boundaries
        circle_left = int(cx - radius)
        circle_right = int(cx + radius)
        circle_top = int(cy - radius)
        circle_bottom = int(cy + radius)
        
        circle_fits = (circle_left >= 0 and circle_right < frame_width and 
                      circle_top >= 0 and circle_bottom < frame_height)
        
        if circle_fits:
            # Draw full circle if it fits within image
            cv2.circle(vis_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            print(f"Full circle drawn - radius: {radius:.1f}px")
        else:
            # Draw partial circle using arc or just show center with text warning
            cv2.circle(vis_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # Center point
            
            # Draw lines to show the detected radius at image boundaries
            if circle_left < 0:
                cv2.line(vis_frame, (int(cx), int(cy)), (0, int(cy)), (0, 0, 255), 2)
            if circle_right >= frame_width:
                cv2.line(vis_frame, (int(cx), int(cy)), (frame_width-1, int(cy)), (0, 0, 255), 2)
            if circle_top < 0:
                cv2.line(vis_frame, (int(cx), int(cy)), (int(cx), 0), (0, 0, 255), 2)
            if circle_bottom >= frame_height:
                cv2.line(vis_frame, (int(cx), int(cy)), (int(cx), frame_height-1), (0, 0, 255), 2)
                
            print(f"Circle extends beyond image - radius: {radius:.1f}px, showing partial visualization")
            
            # Add warning text
            cv2.putText(vis_frame, "Circle extends beyond image", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Always draw center point
        cv2.circle(vis_frame, (int(cx), int(cy)), 2, (255, 0, 0), 3)
        
        # Add measurement text with accuracy note
        text = f"Diameter: {diameter_mm:.1f}mm{accuracy_note}"
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
        
        # Find contours from mask for precise measurement
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found in mask")
            return None
            
        # Get the largest contour (should be the wheel)
        main_contour = max(contours, key=cv2.contourArea)
        print(f"Found main contour with {len(main_contour)} points")
        
        # Calculate the center X of the bounding rectangle for accurate vertical line
        x, y, w, h = cv2.boundingRect(main_contour)
        center_x = x + w // 2
        print(f"Calculated center_x: {center_x} from bounding rect")
        
        # Find intersection points of vertical center line with contour
        top_y, bottom_y = find_vertical_intersections(main_contour, center_x, mask_binary.shape)
        
        if top_y is None or bottom_y is None:
            print("Could not find valid top/bottom intersections with contour")
            return None
            
        print(f"Contour intersections - center_x: {center_x}, top_y: {top_y}, bottom_y: {bottom_y}")
        print(f"Wheel height in pixels: {bottom_y - top_y}")
        
        height_mm = None
        
        if depth_frame is not None:
            print("Attempting depth-based height calculation using RealSense factory methods...")
            
            # Get intrinsics directly from the RealSense aligned depth frame
            try:
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                print(f"RealSense intrinsics: fx={depth_intrin.fx:.1f}, fy={depth_intrin.fy:.1f}")
                print(f"Principal point: cx={depth_intrin.ppx:.1f}, cy={depth_intrin.ppy:.1f}")
                
                # Get depth scale from the RealSense device
                depth_scale = camera_settings.get('depth_scale', 0.001) if camera_settings else 0.001
                print(f"Using depth scale: {depth_scale}")
                
                # Helper function to get valid depth in a window (from simple_measurement.py)
                def get_valid_depth_in_window_rs(x, y, depth_img, window_size=9):
                    """Get valid depth value in a window around the specified point"""
                    # Convert RealSense depth frame to numpy array if needed
                    if hasattr(depth_img, 'get_data'):
                        depth_array = np.asanyarray(depth_img.get_data(), dtype=np.uint16)
                    else:
                        depth_array = depth_img
                    
                    half_window = window_size // 2
                    h, w = depth_array.shape
                    
                    # Ensure coordinates are within bounds
                    x = min(max(x, 0), w-1)
                    y = min(max(y, 0), h-1)
                    
                    # Define window bounds
                    x_start = max(x - half_window, 0)
                    x_end = min(x + half_window + 1, w)
                    y_start = max(y - half_window, 0)
                    y_end = min(y + half_window + 1, h)
                    
                    # Extract window
                    window = depth_array[y_start:y_end, x_start:x_end]
                    
                    # Get valid depths (non-zero)
                    valid_depths = window[window > 0]
                    
                    if len(valid_depths) > 0:
                        # Return median of valid depths for robustness
                        return np.median(valid_depths)
                    else:
                        return 0
                
                # Try multiple points along the vertical center line to find valid depth
                top_depth_raw = None
                bottom_depth_raw = None
                actual_top_y = None
                actual_bottom_y = None
                
                # Search for valid top depth, moving down from the extreme top
                for offset in range(0, 20):  # Search up to 20 pixels down from top
                    test_y = top_y + offset
                    if test_y < bottom_y:  # Don't go past the bottom
                        depth_val = get_valid_depth_in_window_rs(center_x, test_y, depth_frame, 9)
                        if depth_val > 0:
                            top_depth_raw = depth_val
                            actual_top_y = test_y
                            print(f"Found valid top depth at y={test_y} (offset +{offset}): {depth_val}")
                            break
                
                # Search for valid bottom depth, moving up from the extreme bottom  
                for offset in range(0, 20):  # Search up to 20 pixels up from bottom
                    test_y = bottom_y - offset
                    if test_y > top_y:  # Don't go past the top
                        depth_val = get_valid_depth_in_window_rs(center_x, test_y, depth_frame, 9)
                        if depth_val > 0:
                            bottom_depth_raw = depth_val
                            actual_bottom_y = test_y
                            print(f"Found valid bottom depth at y={test_y} (offset -{offset}): {depth_val}")
                            break
                
                print(f"Raw depth values - top: {top_depth_raw}, bottom: {bottom_depth_raw}")
                
                if top_depth_raw is not None and bottom_depth_raw is not None and top_depth_raw > 0 and bottom_depth_raw > 0:
                    # Convert raw depth values to meters
                    top_dist = top_depth_raw * depth_scale
                    bottom_dist = bottom_depth_raw * depth_scale
                    
                    print(f"Distances - top: {top_dist:.3f}m, bottom: {bottom_dist:.3f}m")
                    
                    # Use RealSense factory function to convert 2D points to 3D
                    import pyrealsense2 as rs
                    top_point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [center_x, actual_top_y], top_dist)
                    bottom_point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [center_x, actual_bottom_y], bottom_dist)
                    
                    print(f"3D points - top: {top_point_3d}, bottom: {bottom_point_3d}")
                    
                    # Calculate height using 3D Euclidean distance
                    height_meters = np.sqrt(
                        (top_point_3d[0] - bottom_point_3d[0])**2 + 
                        (top_point_3d[1] - bottom_point_3d[1])**2 + 
                        (top_point_3d[2] - bottom_point_3d[2])**2)
                    
                    # Convert to millimeters
                    height_mm = height_meters * 1000
                    print(f"Calculated 3D height: {height_mm:.1f}mm using RealSense factory methods")
                else:
                    print("Could not get valid depth values")
                    
            except Exception as e:
                print(f"Error using RealSense intrinsics: {e}")
                print("Falling back to manual method...")
                # Fallback to the old method if RealSense intrinsics fail
                intrinsics = load_camera_intrinsics("side_camera")
                if intrinsics is None:
                    print("Could not load camera intrinsics from file either")
                else:
                    # Original depth calculation code as fallback
                    print("Using fallback method with JSON intrinsics")
        
        if height_mm is None:
            print("Failed to calculate height - depth data required for accurate measurement")
            return None
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw contour outline
        cv2.drawContours(vis_frame, [main_contour], -1, (0, 255, 0), 2)
        
        # Draw vertical center line (full height for reference)
        cv2.line(vis_frame, (center_x, 0), (center_x, vis_frame.shape[0]), (255, 255, 0), 1)
        
        # Use actual measurement points if they were found, otherwise use contour extremes
        measurement_top_y = actual_top_y if 'actual_top_y' in locals() and actual_top_y is not None else top_y
        measurement_bottom_y = actual_bottom_y if 'actual_bottom_y' in locals() and actual_bottom_y is not None else bottom_y
        
        # Draw measurement line (actual measurement)
        cv2.line(vis_frame, (center_x, measurement_top_y), (center_x, measurement_bottom_y), (0, 0, 255), 3)
        
        # Add measurement points with larger circles
        cv2.circle(vis_frame, (center_x, measurement_top_y), 8, (255, 0, 0), -1)
        cv2.circle(vis_frame, (center_x, measurement_bottom_y), 8, (255, 0, 0), -1)
        
        # Add smaller circles for contour extremes if different from measurement points
        if measurement_top_y != top_y:
            cv2.circle(vis_frame, (center_x, top_y), 4, (255, 255, 0), -1)  # Yellow for contour extreme
        if measurement_bottom_y != bottom_y:
            cv2.circle(vis_frame, (center_x, bottom_y), 4, (255, 255, 0), -1)  # Yellow for contour extreme
        
        # Add center point (using actual measurement points)
        cv2.circle(vis_frame, (center_x, (measurement_top_y + measurement_bottom_y) // 2), 5, (0, 255, 255), -1)
        
        # Add measurement text
        text = f"Height: {height_mm:.1f}mm (depth-based)"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add pixel dimensions for debugging
        contour_pixel_height = bottom_y - top_y
        measurement_pixel_height = measurement_bottom_y - measurement_top_y
        debug_text = f"Contour: {contour_pixel_height}px, Measured: {measurement_pixel_height}px"
        cv2.putText(vis_frame, debug_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
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
