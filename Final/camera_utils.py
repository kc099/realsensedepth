import cv2
import numpy as np
import json
import os
import pyrealsense2 as rs

def load_camera_intrinsics(camera_type="side_camera"):
    """
    Load camera intrinsics from calibration file
    
    Args:
        camera_type (str): Type of camera ("top_camera" or "side_camera")
        
    Returns:
        dict: Camera intrinsics including fx, fy, cx, cy values, or None if failed to load
    """
    intrinsics_file = os.path.join(os.path.dirname(__file__), "camera_intrinsics.json")
    
    if not os.path.exists(intrinsics_file):
        print(f"Error: Camera intrinsics file not found at {intrinsics_file}")
        print("Cannot proceed without camera calibration data")
        return None
    
    try:
        with open(intrinsics_file, 'r') as f:
            intrinsics_data = json.load(f)
        
        if camera_type not in intrinsics_data:
            print(f"Error: Camera type '{camera_type}' not found in intrinsics file")
            print(f"Available camera types: {list(intrinsics_data.keys())}")
            return None
        
        camera_data = intrinsics_data[camera_type]
        
        # Validate required fields
        if "camera_matrix" not in camera_data:
            print(f"Error: 'camera_matrix' not found for camera type '{camera_type}'")
            return None
            
        camera_matrix = camera_data["camera_matrix"]
        
        # Validate camera matrix structure
        if (not isinstance(camera_matrix, list) or 
            len(camera_matrix) != 3 or 
            len(camera_matrix[0]) != 3):
            print(f"Error: Invalid camera matrix format for camera type '{camera_type}'")
            return None
        
        # Extract intrinsic parameters from JSON data only
        intrinsics = {
            "fx": camera_matrix[0][0],
            "fy": camera_matrix[1][1], 
            "cx": camera_matrix[0][2],
            "cy": camera_matrix[1][2]
        }
        
        # Add optional parameters if they exist
        if "width" in camera_data:
            intrinsics["width"] = camera_data["width"]
        if "height" in camera_data:
            intrinsics["height"] = camera_data["height"]
        if "dist_coeffs" in camera_data:
            intrinsics["dist_coeffs"] = camera_data["dist_coeffs"]
            
        print(f"Successfully loaded intrinsics for {camera_type}: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
        return intrinsics
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in camera intrinsics file: {e}")
        return None
    except Exception as e:
        print(f"Error loading camera intrinsics: {e}")
        return None

def project_point_to_3d(x, y, depth, intrinsics, depth_scale):
    """
    Project a 2D pixel point to 3D space using camera intrinsics
    
    Args:
        x (int): X pixel coordinate
        y (int): Y pixel coordinate  
        depth (float): Depth value in depth units
        intrinsics (dict): Camera intrinsics with fx, fy, cx, cy
        depth_scale (float): Scale factor to convert depth units to mm
        
    Returns:
        numpy.ndarray: 3D point in camera coordinates [X, Y, Z] in mm
    """
    # Extract intrinsic parameters
    fx = intrinsics['fx']
    fy = intrinsics['fy'] 
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    # Convert depth to real-world units
    depth_mm = depth * depth_scale * 1000  # Convert to mm
    
    # Calculate 3D coordinates (perspective projection)
    X = (x - cx) * depth_mm / fx
    Y = (y - cy) * depth_mm / fy
    Z = depth_mm
    
    return np.array([X, Y, Z])

def calculate_3d_distance(point1, point2):
    """
    Calculate Euclidean distance between two 3D points
    
    Args:
        point1 (numpy.ndarray): First 3D point [X, Y, Z]
        point2 (numpy.ndarray): Second 3D point [X, Y, Z]
        
    Returns:
        float: Euclidean distance in mm
    """
    return np.linalg.norm(point1 - point2)

def get_valid_depth_in_window(x, y, depth_frame, window_size=5):
    """
    Get a valid depth value in a window around the specified point
    
    Args:
        x (int): X pixel coordinate
        y (int): Y pixel coordinate
        depth_frame: Depth frame (can be pyrealsense2.depth_frame or numpy array)
        window_size (int): Size of window to search for valid depth
        
    Returns:
        float: Valid depth value or None if no valid depth found
    """
    if depth_frame is None:
        return None
    
    # Convert RealSense depth frame to numpy array if needed
    if hasattr(depth_frame, 'get_data'):
        # This is a pyrealsense2.depth_frame object
        print("Converting RealSense depth frame to numpy array")
        try:
            # Convert to numpy array
            depth_img = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            print(f"Converted depth frame shape: {depth_img.shape}")
        except Exception as e:
            print(f"Error converting depth frame: {e}")
            return None
    elif isinstance(depth_frame, np.ndarray):
        # Already a numpy array
        depth_img = depth_frame
        print(f"Using numpy depth array shape: {depth_img.shape}")
    else:
        print(f"Unknown depth frame type: {type(depth_frame)}")
        return None
        
    h, w = depth_img.shape[:2]
    half_window = window_size // 2
    
    # Define window bounds
    x_start = max(0, x - half_window)
    x_end = min(w, x + half_window + 1)
    y_start = max(0, y - half_window) 
    y_end = min(h, y + half_window + 1)
    
    # Extract window region
    window = depth_img[y_start:y_end, x_start:x_end]
    
    # Find valid (non-zero) depth values
    valid_depths = window[window > 0]
    
    if len(valid_depths) > 0:
        # Return median of valid depths for robustness
        depth_value = np.median(valid_depths)
        print(f"Found valid depth at ({x}, {y}): {depth_value}")
        return depth_value
    else:
        print(f"No valid depth found at ({x}, {y}) in window")
        return None
