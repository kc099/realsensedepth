import cv2
import numpy as np
import json
import os
import pyrealsense2 as rs

# Import centralized debug system
from debug_utils import debug_print

def load_camera_intrinsics(camera_type="side_camera"):
    """
    Load camera intrinsics from unified calibration file
    
    Args:
        camera_type (str): Type of camera ("top_camera", "side_camera", or "realsense")
        
    Returns:
        dict: Camera intrinsics including fx, fy, cx, cy values, or None if failed to load
    """
    intrinsics_file = os.path.join(os.path.dirname(__file__), "camera_intrinsics.json")
    
    if not os.path.exists(intrinsics_file):
        debug_print(f"Error: Camera intrinsics file not found at {intrinsics_file}", "errors")
        debug_print("Cannot proceed without camera calibration data", "errors")
        return None
    
    try:
        with open(intrinsics_file, 'r') as f:
            intrinsics_data = json.load(f)
        
        if camera_type not in intrinsics_data:
            debug_print(f"Error: Camera type '{camera_type}' not found in intrinsics file", "errors")
            debug_print(f"Available camera types: {list(intrinsics_data.keys())}", "errors")
            return None
        
        camera_data = intrinsics_data[camera_type]
        intrinsics = {}
        
        # Extract intrinsic parameters - now supporting direct fx, fy, cx, cy fields
        if "fx" in camera_data and "fy" in camera_data:
            # Direct intrinsic parameters available
            intrinsics = {
                "fx": camera_data["fx"],
                "fy": camera_data["fy"], 
                "cx": camera_data["cx"],
                "cy": camera_data["cy"]
            }
        elif "camera_matrix" in camera_data:
            # Extract from camera matrix
            camera_matrix = camera_data["camera_matrix"]
            
            # Validate camera matrix structure
            if (not isinstance(camera_matrix, list) or 
                len(camera_matrix) != 3 or 
                len(camera_matrix[0]) != 3):
                debug_print(f"Error: Invalid camera matrix format for camera type '{camera_type}'", "errors")
                return None
            
            intrinsics = {
                "fx": camera_matrix[0][0],
                "fy": camera_matrix[1][1], 
                "cx": camera_matrix[0][2],
                "cy": camera_matrix[1][2]
            }
        else:
            debug_print(f"Error: No intrinsic parameters found for camera type '{camera_type}'", "errors")
            return None
        
        # Add optional parameters if they exist
        if "width" in camera_data:
            intrinsics["width"] = camera_data["width"]
        if "height" in camera_data:
            intrinsics["height"] = camera_data["height"]
        if "dist_coeffs" in camera_data:
            intrinsics["dist_coeffs"] = camera_data["dist_coeffs"]
        if "device_serial" in camera_data:
            intrinsics["device_serial"] = camera_data["device_serial"]
        if "device_product_line" in camera_data:
            intrinsics["device_product_line"] = camera_data["device_product_line"]
            
        # Only log success message for RealSense when first loaded to avoid spam
        if camera_type == "realsense":
            debug_print(f"RealSense intrinsics loaded from cache: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}", "startup")
        else:
            debug_print(f"Successfully loaded intrinsics for {camera_type}: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}", "startup")
        return intrinsics
        
    except json.JSONDecodeError as e:
        debug_print(f"Error: Invalid JSON format in camera intrinsics file: {e}", "errors")
        return None
    except Exception as e:
        debug_print(f"Error loading camera intrinsics: {e}", "errors")
        return None

def save_camera_intrinsics(camera_type, intrinsics_data):
    """
    Save camera intrinsics to unified calibration file
    
    Args:
        camera_type (str): Type of camera ("top_camera", "side_camera", or "realsense")
        intrinsics_data (dict): Intrinsics data to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    intrinsics_file = os.path.join(os.path.dirname(__file__), "camera_intrinsics.json")
    
    try:
        # Load existing data
        existing_data = {}
        if os.path.exists(intrinsics_file):
            with open(intrinsics_file, 'r') as f:
                existing_data = json.load(f)
        
        # Update with new data
        existing_data[camera_type] = intrinsics_data
        
        # Save back to file
        with open(intrinsics_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        debug_print(f"Successfully saved intrinsics for {camera_type}", "startup")
        return True
        
    except Exception as e:
        debug_print(f"Error saving camera intrinsics for {camera_type}: {e}", "errors")
        return False
