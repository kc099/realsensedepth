import cv2
import numpy as np
import json
import os

def load_camera_intrinsics(camera_type="top_camera"):
    """
    Load camera intrinsics from calibration file
    
    Args:
        camera_type (str): Type of camera ("top_camera" or "side_camera")
        
    Returns:
        dict: Camera intrinsics including camera matrix and distortion coefficients
    """
    intrinsics_file = os.path.join(os.path.dirname(__file__), "camera_intrinsics.json")
    
    if not os.path.exists(intrinsics_file):
        print(f"Warning: Camera intrinsics file not found at {intrinsics_file}")
        # Return default values
        return {
            "camera_matrix": np.array([
                [900.0, 0.0, 640.0],
                [0.0, 900.0, 360.0],
                [0.0, 0.0, 1.0]
            ]),
            "dist_coeffs": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "width": 1280,
            "height": 720
        }
    
    try:
        with open(intrinsics_file, 'r') as f:
            intrinsics_data = json.load(f)
        
        camera_data = intrinsics_data.get(camera_type, {})
        
        return {
            "camera_matrix": np.array(camera_data.get("camera_matrix", [
                [900.0, 0.0, 640.0],
                [0.0, 900.0, 360.0],
                [0.0, 0.0, 1.0]
            ])),
            "dist_coeffs": np.array(camera_data.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])),
            "width": camera_data.get("width", 1280),
            "height": camera_data.get("height", 720)
        }
    except Exception as e:
        print(f"Error loading camera intrinsics: {e}")
        # Return default values
        return {
            "camera_matrix": np.array([
                [900.0, 0.0, 640.0],
                [0.0, 900.0, 360.0],
                [0.0, 0.0, 1.0]
            ]),
            "dist_coeffs": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "width": 1280,
            "height": 720
        }

def project_point_to_3d(pixel_point, depth, camera_matrix):
    """
    Project a 2D pixel point to 3D space using camera intrinsics
    
    Args:
        pixel_point (tuple): (x, y) pixel coordinates
        depth (float): Distance from camera to point in mm
        camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
        
    Returns:
        numpy.ndarray: 3D point in camera coordinates [X, Y, Z] in mm
    """
    # Extract intrinsic parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Pixel coordinates
    x, y = pixel_point
    
    # Calculate 3D coordinates (perspective projection)
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    
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
