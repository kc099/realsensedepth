import cv2
import numpy as np
import math
from utils import correct_for_perspective

def calculate_real_dimensions(measurements, is_top_view=True, camera_settings=None, mask=None):
    """
    Calculate real-world dimensions from pixel measurements using camera intrinsics
    
    Args:
        measurements (dict): Pixel measurements
        is_top_view (bool): If True, process as top view (diameter calculation)
                           If False, process as side view (height calculation)
        camera_settings (dict): Camera calibration settings
        mask (numpy.ndarray): Binary mask for the detected wheel
        
    Returns:
        dict: Measurements in real-world units (mm)
    """
    if camera_settings is None:
        print("Error: No camera settings provided for real dimension calculation")
        return {}
        
    if not measurements:
        return {}
    
    result = {}
    result["type"] = "Round" if is_top_view else "Side"
    
    if is_top_view:
        # For top view, calculate diameter from the detected circle
        if "radius_pixels" in measurements:
            # Convert radius from pixels to mm based on calibration
            ref_diameter_mm = camera_settings["ref_diameter"]
            ref_diameter_pixels = camera_settings["ref_diameter_pixels"]
            wheel_height = camera_settings.get("wheel_height", 75.0)  # Default if not provided
            base_height = camera_settings["base_height"]
            
            # Get the scale factor (mm per pixel)
            scale_factor = ref_diameter_mm / ref_diameter_pixels
            
            # Apply height-based correction if wheel height is different from calibration height
            diameter_pixels = measurements["radius_pixels"] * 2
            corrected_diameter_pixels = correct_for_perspective(
                diameter_pixels,
                base_height, 
                base_height - wheel_height,  # Current camera-to-wheel distance
                diameter_pixels  # Original measurement
            )
            
            # Calculate final diameter in mm
            diameter_mm = corrected_diameter_pixels * scale_factor
            
            # Store results
            result["diameter_mm"] = diameter_mm
            result["scale_factor"] = scale_factor
            
            # Check if the diameter is within expected range
            model_data = measurements.get("model_data", {})
            min_dia = model_data.get("min_dia", 0)
            max_dia = model_data.get("max_dia", 100)
            tolerance = model_data.get("tolerance", 3.0)
            
            # Determine if the measurement is within tolerance
            is_in_range = min_dia <= diameter_mm <= max_dia
            is_within_tolerance = abs(diameter_mm - ((min_dia + max_dia) / 2)) <= tolerance
            result["is_ok"] = is_in_range and is_within_tolerance
            
            # Add raw data for debugging
            result["uncorrected_diameter_pixels"] = diameter_pixels
            result["corrected_diameter_pixels"] = corrected_diameter_pixels
            
    else:
        # For side view, calculate height from the depth data
        if "height_pixels" in measurements:
            # Convert height from pixels to mm
            side_ref_pixels = camera_settings["side_ref_pixels"]
            side_camera_height = camera_settings["side_camera_height"]
            
            # Calculate height based on ratio of detected height to reference height
            height_scale = side_camera_height / side_ref_pixels
            height_mm = measurements["height_pixels"] * height_scale
            
            # Store results
            result["height_mm"] = height_mm
            
            # If we have depth data, use it for actual measurement
            if "depth_mm" in measurements:
                result["depth_mm"] = measurements["depth_mm"]
                # The depth value can be used later for exact 3D positioning
            
            # Check if height is within expected range for the model
            model_data = measurements.get("model_data", {})
            expected_height = model_data.get("height", 0)
            tolerance = model_data.get("tolerance", 3.0)
            
            # Determine if the measurement is within tolerance
            is_within_tolerance = abs(height_mm - expected_height) <= tolerance
            result["is_ok"] = is_within_tolerance
            
            # Store the height for use in top view calculations
            camera_settings["wheel_height"] = height_mm
    
    return result

def calculate_distance_from_depth(depth_image, mask, x, y, w, h):
    """
    Calculate distance using depth data and segmentation mask
    
    Args:
        depth_image: RealSense depth image
        mask: Binary mask of the wheel
        x, y, w, h: Bounding box coordinates
        
    Returns:
        tuple: (mean_depth, height_mm, min_depth, max_depth, depth_points)
    """
    if depth_image is None or mask is None:
        return None, None, None, None, []
    
    # Ensure mask is binary and the same size as depth image
    if mask.shape[:2] != depth_image.shape[:2]:
        mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]))
    
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Take first channel if it's a 3-channel mask
    
    # Convert mask to binary
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Get region of interest in the depth image
    roi = depth_image[y:y+h, x:x+w]
    mask_roi = mask_binary[y:y+h, x:x+w]
    
    # Use the mask to get depth values only for the wheel
    depth_values = roi[mask_roi > 0]
    
    # Filter out zero values (no depth data)
    depth_values = depth_values[depth_values > 0]
    
    if len(depth_values) == 0:
        return None, None, None, None, []
    
    # Calculate statistics
    mean_depth = np.mean(depth_values)
    min_depth = np.min(depth_values)
    max_depth = np.max(depth_values)
    
    # For height, use the difference between lowest and highest points
    height_mm = max_depth - min_depth
    
    # Sample some points for visualization
    depth_points = []
    step = max(1, mask_roi.shape[0] // 10)  # Sample about 10 points vertically
    
    for i in range(0, mask_roi.shape[0], step):
        row = mask_roi[i, :]
        if np.any(row > 0):
            # Find the middle of the wheel in this row
            points = np.where(row > 0)[0]
            mid_point = points[len(points) // 2]
            
            # Get depth at this point
            depth_val = roi[i, mid_point]
            if depth_val > 0:
                # Store point coordinates and depth
                world_x = x + mid_point
                world_y = y + i
                depth_points.append((world_x, world_y, depth_val))
    
    return mean_depth, height_mm, min_depth, max_depth, depth_points

def measure_wheel_height_from_depth(depth_image, color_image, mask_binary, box, camera_settings=None):
    """
    Measure wheel height using RealSense depth data and wheel mask
    
    Args:
        depth_image: Depth image from RealSense camera
        color_image: Color image corresponding to depth image
        mask_binary: Binary mask of the wheel
        box: Bounding box (x, y, w, h)
        camera_settings: Camera calibration settings
        
    Returns:
        tuple: (processed_image, measurements_dict)
    """
    if depth_image is None or color_image is None:
        return color_image, {}
    
    if box is None:
        x, y, w, h = 0, 0, depth_image.shape[1], depth_image.shape[0]
    else:
        x, y, w, h = box
    
    # Calculate depth statistics for the wheel mask
    mean_depth, height_mm, min_depth, max_depth, depth_points = calculate_distance_from_depth(
        depth_image, mask_binary, x, y, w, h
    )
    
    # Create a copy of the color image for visualization
    visual_image = color_image.copy()
    
    # Add bounding box
    cv2.rectangle(visual_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Initialize measurements dictionary
    measurements = {
        "height_pixels": h,
        "width_pixels": w,
        "model_data": {},  # Will be populated by caller
        "depth_mm": mean_depth if mean_depth is not None else 0
    }
    
    # Add depth visualization if we have valid data
    if mean_depth is not None:
        # Add text with depth information
        text_offset = 30
        cv2.putText(visual_image, f"Depth: {mean_depth:.1f}mm", 
                   (x, y - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(visual_image, f"Height: {height_mm:.1f}mm", 
                   (x, y - text_offset - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw depth points
        for point in depth_points:
            px, py, d = point
            # Color code based on depth (red for closer, blue for farther)
            color_val = int(255 * (d - min_depth) / (max_depth - min_depth + 0.001))
            color = (color_val, 0, 255 - color_val)
            cv2.circle(visual_image, (px, py), 3, color, -1)
            
        # Add the measured height
        measurements["height_mm_from_depth"] = height_mm
    
    return visual_image, measurements

def classify_wheel_model(diameter_mm, height_mm=None, wheel_models=None):
    """
    Classify wheel and check if it matches selected model with model-specific tolerance
    
    Args:
        diameter_mm (float): Measured diameter in mm
        height_mm (float): Measured height in mm, optional
        wheel_models (dict): Dictionary of wheel model specifications
        
    Returns:
        tuple: (model_name, is_within_tolerance)
    """
    if wheel_models is None or len(wheel_models) == 0:
        return None, False
    
    matched_model = None
    closest_model = None
    min_diameter_diff = float('inf')
    
    for model_name, specs in wheel_models.items():
        min_dia = specs.get("min_dia", 0)
        max_dia = specs.get("max_dia", 100)
        expected_height = specs.get("height", 0)
        tolerance = specs.get("tolerance", 3.0)
        
        # Check if diameter is within model range
        if min_dia <= diameter_mm <= max_dia:
            matched_model = model_name
            
            # Further check height if provided
            if height_mm is not None and expected_height > 0:
                height_diff = abs(height_mm - expected_height)
                if height_diff > tolerance:
                    # Height outside tolerance, not a perfect match
                    matched_model = None
        
        # Keep track of closest model for fallback
        center_dia = (min_dia + max_dia) / 2
        dia_diff = abs(diameter_mm - center_dia)
        
        if dia_diff < min_diameter_diff:
            min_diameter_diff = dia_diff
            closest_model = model_name
    
    # If no exact match, return closest model but mark as not within tolerance
    if matched_model is None:
        return closest_model, False
    
    return matched_model, True
