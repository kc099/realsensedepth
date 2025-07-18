import cv2
import numpy as np
import math
import os
import threading
import time
import traceback

# Import centralized debug system
from debug_utils import debug_print

# Import PyTorch dependencies directly (no lazy loading)
try:
    import torch
    from torchvision.transforms import functional as F
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    TORCH_AVAILABLE = True
    debug_print("PyTorch dependencies loaded successfully", "startup")
except ImportError as e:
    TORCH_AVAILABLE = False
    debug_print(f"PyTorch dependencies not available: {e}", "errors")

from utils import fit_circle_least_squares
from camera_utils import load_camera_intrinsics

# Import PyRealSense2 for depth processing
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    debug_print("PyRealSense2 loaded successfully", "startup")
except ImportError as e:
    REALSENSE_AVAILABLE = False
    debug_print(f"PyRealSense2 not available: {e}", "errors")
# Configuration
MODEL_PATH = "./maskrcnn_wheel_best.pth"
SCORE_THRESHOLD = 0.85
RATIO_THRESHOLD = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None

# Global model variable
wheel_detection_model = None

def get_model_instance_segmentation(num_classes):
    """Create and configure the object detection model"""
    if not TORCH_AVAILABLE:
        debug_print("PyTorch not available - cannot create model", "errors")
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
        debug_print("PyTorch not available - cannot load model", "errors")
        return None
        
    if wheel_detection_model is not None:
        debug_print("Model already loaded", "startup")
        return wheel_detection_model
     
    try:
        debug_print("Loading wheel detection model...", "startup")
        model = get_model_instance_segmentation(num_classes=2)  # Background and wheel
        
        if model is None:
            debug_print("Failed to create model instance", "errors")
            return None
        
        # Move to device
        if DEVICE is not None:
            model.to(DEVICE)
            debug_print(f"Model moved to device: {DEVICE}", "startup")
        
        # Check for model weights file
        model_exists = os.path.exists(MODEL_PATH)
        debug_print(f"Model file exists at {MODEL_PATH}: {model_exists}", "startup")
            
        # Load weights if available
        if model_exists:
            debug_print(f"Loading model weights from {MODEL_PATH}", "startup")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            debug_print("Model weights loaded successfully", "startup")
        else:
            debug_print(f"Model file not found at {MODEL_PATH}. Using pre-trained COCO weights.", "errors")
            debug_print("Note: This may not work well for wheel detection without fine-tuning.", "errors")
            
        # Set model to evaluation mode (always do this)
        model.eval()
        wheel_detection_model = model
        debug_print("Model loaded and ready for inference", "startup")
        return wheel_detection_model
        
    except Exception as e:
        debug_print(f"Error loading model: {e}", "errors")
        traceback.print_exc()
        return None

def load_model_in_background():
    """Start loading the model in a background thread (only if not already loaded)"""
    global wheel_detection_model
    
    # Check if model is already loaded
    if wheel_detection_model is not None:
        debug_print("Model already loaded - skipping background loading", "startup")
        return
    
    def _load_model_thread():
        try:
            result = load_model()
            if result is not None:
                debug_print("Background model loading completed successfully", "startup")
            else:
                debug_print("Background model loading failed", "errors")
        except Exception as e:
            debug_print(f"Error in background model loading: {e}", "errors")
            traceback.print_exc()
    
    # Start loading in background thread
    threading.Thread(target=_load_model_thread, daemon=True).start()
    debug_print("Started background model loading thread", "startup")

# Circle fitting function moved to utils.py - import from there
# from utils import fit_circle_least_squares
def process_frame(frame, is_top_view=True, camera_settings=None, wheel_models=None, 
                 selected_model=None, wheel_height_mm=None, depth_frame=None):
    """Main processing function (now just a wrapper)"""
    # Run model and get mask
    mask_result = run_model_and_mask(frame, is_top_view)
    
    # Calculate measurements
    return calculate_measurements(
        mask_result,
        camera_settings,
        wheel_models,
        selected_model,
        depth_frame=depth_frame,
        wheel_height_mm=wheel_height_mm
    )

def run_model_and_mask(frame, is_top_view=False):
    """Run Mask R-CNN model and generate mask (optimized with better error handling)"""
    global wheel_detection_model
    
    # Ensure model is loaded (only once)
    if wheel_detection_model is None:
        wheel_detection_model = load_model()
        if wheel_detection_model is None:
            debug_print("Failed to load model", "errors")
            return None
    
    try:
        # Convert frame to RGB for model input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(frame_rgb).unsqueeze(0)
        if DEVICE is not None:
            image_tensor = image_tensor.to(DEVICE)
        
        # Run model inference
        with torch.no_grad():
            predictions = wheel_detection_model(image_tensor)
        
        # Process predictions
        pred = predictions[0]
        masks = pred['masks'].cpu().numpy()
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Filter by confidence
        valid_detections = scores > SCORE_THRESHOLD
        if not np.any(valid_detections):
            valid_detections = scores > SCORE_THRESHOLD  # Try lower threshold
            if not np.any(valid_detections):
                return None
        
        # Get best detection
        best_idx = np.argmax(scores[valid_detections])
        mask = masks[valid_detections][best_idx][0]
        box = boxes[valid_detections][best_idx]
        
        # Convert mask to binary
        mask_binary = (mask > 0.5).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return {
            'frame': frame,
            'mask': mask_binary,
            'box': box,
            'is_top_view': is_top_view
        }
        # if frame is None or frame.size == 0:
        #     debug_print("ERROR: Invalid top frame!", "errors")
        #     return
        # if frame is None or frame.size == 0:
        #     debug_print("ERROR: Invalid side frame!", "errors")
        #     return
        
        # debug_print(f"Top frame shape: {local_top.shape}", "processing")
        # debug_print(f"Side frame shape: {local_side.shape}", "processing")
    except Exception as e:
        debug_print(f"Error in model/mask processing: {e}", "errors")
        return None

def calculate_measurements(mask_result, camera_settings, wheel_models, selected_model, 
                         depth_frame=None, depth_intrinsics=None, wheel_height_mm=None):
    """Calculate measurements from mask (sequential step)"""
    if mask_result is None:
        return None
    
    if mask_result['is_top_view']:
        return process_top_view(
            mask_result['frame'],
            mask_result['mask'],
            mask_result['box'],
            camera_settings,
            wheel_models,
            selected_model,
            wheel_height_mm
        )
    else:
        return process_side_view(
            mask_result['frame'],
            mask_result['mask'],
            mask_result['box'],
            camera_settings,
            wheel_models,
            selected_model,
            depth_frame,
            depth_intrinsics           
        )

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
        debug_print("Processing top view for diameter calculation...", "processing")
        
        if wheel_height_mm is None:
            debug_print("Warning: No wheel height provided for top view processing", "errors")
            return None
        from wheel_main import GLOBAL_INTRINSICS    
        # Get camera intrinsics
        debug_print(f"Available intrinsics keys: {GLOBAL_INTRINSICS.keys()}", "processing")
        debug_print(f"Top camera intrinsics: {GLOBAL_INTRINSICS['top_camera']}", "processing")
        intrinsics = load_camera_intrinsics('top_camera')
        if intrinsics is None:
            debug_print("Could not load camera intrinsics", "errors")
            intrinsics = {
                'fx': 1000.0,  # Default focal length
                'fy': 690.0,
                'cx': 640.0,
                'cy': 360.0,
                'width': 1280,
                'height': 720
            }
            
        # Get camera height from base (in mm)
        camera_height = camera_settings.get('camera_height_mm', 1000) if camera_settings else 1000
        
        # Calculate distance to wheel center
        distance_to_wheel = camera_height - wheel_height_mm
        
        # Find contours from mask
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            debug_print("No contours found in mask", "processing")
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
            debug_print("Warning: Fitted circle extends beyond image boundaries", "processing")
            debug_print("Diameter measurement may be inaccurate - ensure full wheel is visible", "processing")
        
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
            debug_print(f"Full circle drawn - radius: {radius:.1f}px", "processing")
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
                
            debug_print(f"Circle extends beyond image - radius: {radius:.1f}px, showing partial visualization", "processing")
            
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
        
        debug_print(f"Top view processing complete - diameter: {diameter_mm:.1f}mm", "processing")
        
        return {
            'diameter_mm': diameter_mm,
            'visualization': vis_frame,
            'mask': mask_binary,
            'box': box,
            'center': (cx, cy),
            'radius': radius
        }
        
    except Exception as e:
        debug_print(f"Error in top view processing: {e}", "errors")
        traceback.print_exc()
        return None

def process_side_view(frame, mask_binary, box, camera_settings, wheel_models, selected_model, depth_frame=None, depth_intrinsics=None):
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
        depth_intrinsics: Depth camera intrinsics (None for uploaded images)
    """
    try:
        debug_print(f"Processing side view - mask shape: {mask_binary.shape}, depth available: {depth_frame is not None}", "processing")
          # Add depth threshold (350mm)
        
        # Find contours from mask for precise measurement
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            debug_print("No contours found in mask", "processing")
            return None
            
        # Get the largest contour (should be the wheel)
        main_contour = max(contours, key=cv2.contourArea)
        # print(f"Found main contour with {len(main_contour)} points")
        
        # Calculate the center X of the bounding rectangle for accurate vertical line
        x, y, w, h = cv2.boundingRect(main_contour)
        center_x = x + w // 2
        center_y = y + h // 2
        # print(f"Calculated center_x: {center_x} from bounding rect")
        
        # Find the global topmost and bottommost mask pixel (not just center)
        # ys, xs = np.where(mask_binary > 0)
        # if len(ys) > 0:
        #     top_y = int(np.min(ys))
        #     bottom_y = int(np.max(ys))
        #     # For depth, use the median x at those ys
        #     top_xs = xs[ys == top_y]
        #     bottom_xs = xs[ys == bottom_y]
        #     top_x = int(np.median(top_xs)) if len(top_xs) > 0 else center_x
        #     bottom_x = int(np.median(bottom_xs)) if len(bottom_xs) > 0 else center_x
        #     print(f"Global mask top_y: {top_y}, bottom_y: {bottom_y}, top_x: {top_x}, bottom_x: {bottom_x}")
        # else:
        #     print("No mask pixels found for height calculation")
        #     return None
        # print(f"Wheel height in pixels (global mask): {bottom_y - top_y}")
        # NEW: Get only center 30% of the mask for height calculation
        center_start_x = x + int(w * 0.35)
        center_end_x = x + int(w * 0.65)
        
        # Get all mask pixels in the center region
        ys, xs = np.where(mask_binary[y:y+h, center_start_x:center_end_x] > 0)
        if len(ys) == 0:
            debug_print("No mask pixels found in center region for height calculation", "processing")
            return None
            
        # Adjust x coordinates to full image coordinates
        xs = xs + center_start_x
        ys = ys + y
        
        # Create dictionaries to store top/bottom for each x column in center region
        x_to_ys = {}
        for x_val, y_val in zip(xs, ys):
            if x_val not in x_to_ys:
                x_to_ys[x_val] = []
            x_to_ys[x_val].append(y_val)
        
        # Get all top and bottom points (min and max y for each x in center region)
        all_tops = []
        all_bottoms = []
        for x_val, y_vals in x_to_ys.items():
            valid_ys = [y for y in y_vals if not np.isnan(y)]  # Filter out NaN values
            if valid_ys:  # Only proceed if we have valid y values
                all_tops.append(min(valid_ys))
                all_bottoms.append(max(valid_ys))
        
        if not all_tops or not all_bottoms:
            debug_print("No valid top/bottom points found in center region", "processing")
            return None
            
        # Calculate mean of all top and bottom points (filtering out NaN/inf)
        top_y = int(np.nanmean(all_tops)) if all_tops else center_y
        bottom_y = int(np.nanmean(all_bottoms)) if all_bottoms else center_y
        
        debug_print(f"Center region top_y: {top_y}, bottom_y: {bottom_y}", "processing")
        # print(f"Wheel height in pixels (center region): {bottom_y - top_y}")

        height_mm = None
        avg_side_distance_m = None  # Initialize side camera distance
        
        if depth_frame is not None:
            # print("Attempting depth-based height calculation using RealSense factory methods...")
            
            # Get intrinsics directly from the RealSense aligned depth frame
            try:
                # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                # print(f"RealSense intrinsics: fx={depth_intrin.fx:.1f}, fy={depth_intrin.fy:.1f}")
                # print(f"Principal point: cx={depth_intrin.ppx:.1f}, cy={depth_intrin.ppy:.1f}")
                
                # Get depth scale from the RealSense device
                depth_scale = camera_settings.get('depth_scale', 0.001) if camera_settings else 0.001
                # print(f"Using depth scale: {depth_scale}")
                if depth_intrinsics is None:
                    debug_print("Warning: No depth intrinsics provided, trying fallback methods", "errors")
                    try:
                        # Try to get from frame first
                        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                        debug_print("Successfully got intrinsics from depth frame", "processing")
                    except Exception as e:
                        debug_print(f"Error getting intrinsics from frame: {e}", "errors")
                        # Import here to avoid circular imports
                        from wheel_main import GLOBAL_INTRINSICS
                        fallback_intrinsics = GLOBAL_INTRINSICS.get('realsense')
                        if fallback_intrinsics:
                            # Create compatible intrinsics object
                            class FrameIntrinsics:
                                def __init__(self, fx, fy, ppx, ppy):
                                    self.fx = fx
                                    self.fy = fy
                                    self.ppx = ppx
                                    self.ppy = ppy
                            
                            depth_intrin = FrameIntrinsics(
                                fx=fallback_intrinsics.get('fx', 0),
                                fy=fallback_intrinsics.get('fy', 0),
                                ppx=fallback_intrinsics.get('cx', 0),
                                ppy=fallback_intrinsics.get('cy', 0)
                            )
                            debug_print("Using fallback GLOBAL_INTRINSICS for depth processing", "processing")
                        else:
                            depth_intrin = None
                else:
                    depth_intrin = depth_intrinsics
                
                if depth_intrin is None:
                    debug_print("No depth intrinsics available - cannot calculate height", "errors")
                    return None
                
                # print(f"Using intrinsics - fx: {depth_intrin.fx:.1f}, fy: {depth_intrin.fy:.1f}")
                # Helper function to get valid depth in a window (from simple_measurement.py)
                def get_valid_depth_in_window_rs(x, y, depth_img, window_size=30):
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
                    valid_depths = window[(window > 0) & np.isfinite(window) & (window < 65535)]
                    
                    if len(valid_depths) > 0:
                        median_depth = np.median(valid_depths)
                        # depth_meters = median_depth * depth_scale
                        # # Check against depth threshold
                        # if depth_meters * 1000 > DEPTH_THRESHOLD_MM:  # Compare in mm
                        #     print(f"Depth {depth_meters*1000:.1f}mm exceeds threshold {DEPTH_THRESHOLD_MM}mm - ignoring")
                        #     return 0
                        return median_depth
                    else:
                        return 0
                
                # Try multiple points along the vertical center line to find valid depth
                center_depth_raw = None
                
                # Search for valid top depth, moving down from the extreme top
                # For depth, use the median x at center of mask for reference depth
                center_depth_raw = get_valid_depth_in_window_rs(center_x, center_y, depth_frame, 30)
                search_radius = 30  # pixels
                step = 5  # pixel step for spiral search
                # If center depth is 0, check nearby points in a spiral pattern
                if center_depth_raw == 0:
                     debug_print("Center depth is 0, searching neighboring points...", "processing")
                                                     
                # Spiral search pattern
                for r in range(step, search_radius + step, step):
                    for angle in np.linspace(0, 2*np.pi, 16):  # 16 points around the circle
                        check_x = int(center_x + r * np.cos(angle))
                        check_y = int(center_y + r * np.sin(angle))
                        
                        # Get depth at this point
                        temp_depth = get_valid_depth_in_window_rs(check_x, check_y, depth_frame, 15)
                        if temp_depth > 0:
                            center_depth_raw = temp_depth
                        #   print(f"Found valid depth at offset ({r}, {angle:.1f}): {center_depth_raw}")
                            break
                    if center_depth_raw > 0:
                        break


                # print(f"Raw depth values - center: {center_depth_raw}")
                
                if center_depth_raw is not None and center_depth_raw > 0:
                    # Calculate depth threshold as 110% of center depth (raw depth units)
                    depth_threshold_raw = center_depth_raw
                    # print(f"Depth threshold (raw): {depth_threshold_raw}")
                else:
                    # Fallback to fixed threshold (350mm converted to raw depth units)
                    depth_scale = camera_settings.get('depth_scale', 0.001) if camera_settings else 0.001
                    depth_threshold_raw = (350 / 1000) / depth_scale  # Convert 350mm to raw depth units
                    center_depth_raw = depth_threshold_raw
                    # print(f"Using fallback depth threshold (350mm): {depth_threshold_raw} raw units")
                    
                # Apply threshold mask to depth frame
                depth_array = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
                depth_frame_threshold = depth_array.copy()
                # Set pixels beyond threshold to 0 (invalid)
                depth_frame_threshold[depth_array > depth_threshold_raw] = 0
                # print(f"Applied depth threshold - valid pixels: {np.sum(depth_frame_threshold > 0)}")
                
                # Helper function to get valid depth from numpy array
                def get_valid_depth_from_array(x, y, depth_array, window_size=9):
                    """Get valid depth value from numpy array"""
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
                    valid_depths = window[(window > 0) & np.isfinite(window)]
                    
                    if len(valid_depths) > 0:
                        return np.mean(valid_depths)
                    else:
                        return 0
                
                # Get depth values at actual top and bottom points using thresholded depth
                actual_top_depth = get_valid_depth_from_array(center_x, top_y, depth_frame_threshold, 9)
                actual_bottom_depth = get_valid_depth_from_array(center_x, bottom_y, depth_frame_threshold, 9)
                
                # print(f"Thresholded depth values - top: {actual_top_depth}, bottom: {actual_bottom_depth}")
                            # Adjust depth values with fallback logic
                if actual_top_depth > 0 and actual_bottom_depth == 0:
                    debug_print("Bottom depth missing — using top depth value", "processing")
                    actual_bottom_depth = actual_top_depth
                elif actual_bottom_depth > 0 and actual_top_depth == 0:
                    debug_print("Top depth missing — using bottom depth value", "processing")
                    actual_top_depth = actual_bottom_depth
                elif actual_top_depth == 0 and actual_bottom_depth == 0:
                    debug_print("Both top and bottom depths missing — using center depth", "processing")
                    actual_top_depth = actual_bottom_depth = center_depth_raw

                # Final top/bottom depths
                top_depth = actual_top_depth
                bottom_depth = actual_bottom_depth

                
                # Convert raw depth values to meters
                top_dist = top_depth * depth_scale
                bottom_dist = bottom_depth * depth_scale
                
                # print(f"Distances - top: {top_dist:.3f}m, bottom: {bottom_dist:.3f}m")
                avg_side_distance_m = (top_dist + bottom_dist) / 2.0
                # print(f"Average side distance: {avg_side_distance_m:.3f}m")
                debug_print(f"depth_intrin: {depth_intrin}", "processing")
                # Use RealSense factory function to convert 2D points to 3D
                import pyrealsense2 as rs
                top_point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, [center_x, top_y], top_dist)
                bottom_point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, [center_x, bottom_y], bottom_dist)
                #actual top and bottom y values from rgb segmentation mask not depth frame.
                
                # print(f"3D points - top: {top_point_3d}, bottom: {bottom_point_3d}")
                
                # Calculate height using 3D Euclidean distance
                height_meters = np.sqrt(
                    (top_point_3d[0] - bottom_point_3d[0])**2 + 
                    (top_point_3d[1] - bottom_point_3d[1])**2 + 
                    (top_point_3d[2] - bottom_point_3d[2])**2)
                
                # Convert to millimeters
                height_mm = height_meters * 1000
                debug_print(f"Calculated 3D height: {height_mm:.1f}mm using RealSense factory methods", "processing")
                            
            except Exception as e:
                debug_print(f"Error using RealSense intrinsics: {e}", "errors")
                debug_print("Falling back to manual method...", "processing")
                # Fallback to the old method if RealSense intrinsics fail
                intrinsics = load_camera_intrinsics("side_camera")
                if intrinsics is None:
                    debug_print("Could not load camera intrinsics from file either", "errors")
                else:
                    # Original depth calculation code as fallback
                    debug_print("Using fallback method with JSON intrinsics", "processing")
        
        if height_mm is None:
            debug_print("Failed to calculate height - depth data required for accurate measurement", "errors")
            return None
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw contour outline
        cv2.drawContours(vis_frame, [main_contour], -1, (0, 255, 0), 2)
        
        # Draw vertical center line (full height for reference)
        cv2.line(vis_frame, (center_x, 0), (center_x, vis_frame.shape[0]), (255, 255, 0), 1)
        
        # Use the top and bottom Y coordinates from the mask (these are the actual measurement points)
        measurement_top_y = top_y
        measurement_bottom_y = bottom_y
        
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
        
        debug_print(f"Side view processing complete - height: {height_mm:.1f}mm", "processing")
        
        return {
            'height_mm': height_mm,
            'visualization': vis_frame,
            'mask': mask_binary,
            'box': box,
            'side_camera_distance_m': avg_side_distance_m
        }
        
    except Exception as e:
        debug_print(f"Error in side view processing: {e}", "errors")
        traceback.print_exc()
        return None


