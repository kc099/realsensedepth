import pyrealsense2 as rs
import numpy as np
import json
import os

def initialize_camera(width=1280, height=720, fps=30):
    """Initialize RealSense camera with optimal settings"""
    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get device info
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    device_serial = str(device.get_info(rs.camera_info.serial_number))
    
    # Set to high accuracy preset if available
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy preset
    
    # Create alignment object
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline, align, device_product_line, device_serial

def create_filters():
    """Create RealSense depth post-processing filters"""
    depth_to_disparity = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    
    temporal = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter()
    
    filters = [depth_to_disparity, spatial, temporal, disparity_to_depth, hole_filling]
    return filters

def apply_filters(depth_frame, filters):
    """Apply post-processing filters to depth frame"""
    filtered = depth_frame
    for filter in filters:
        filtered = filter.process(filtered)
    return filtered

def load_calibration(filename=None):
    """Load camera calibration from file"""
    if filename is None:
        # Find the most recent calibration file
        calib_files = [f for f in os.listdir('.') if f.startswith('realsense_calib_') and f.endswith('.json')]
        if not calib_files:
            print("No calibration files found.")
            return None, None
        filename = sorted(calib_files)[-1]  # Get the latest file
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data["camera_matrix"])
        dist_coeffs = np.array(data["dist_coeffs"])
        print(f"Loaded calibration from {filename}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None, None

def save_calibration(camera_matrix, dist_coeffs, img_size, error, device_product_line, device_serial):
    """Save calibration parameters to file"""
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_width": img_size[0],
        "image_height": img_size[1],
        "reprojection_error": error,
        "device_product_line": device_product_line,
        "device_serial": device_serial,
        "width": width,
        "height": height
    }
    
    filename = f"realsense_calib_{device_serial}.json"
    with open(filename, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    return filename