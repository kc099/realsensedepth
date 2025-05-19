import json
import os
import time

# Default settings and paths
SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "selected_model": "10-13",
    "auto_capture_interval": 5,
    "calibration": {
        "ref_diameter": 466.0,  # Reference diameter in mm
        "ref_diameter_pixels": 632.62,  # Corresponding pixels in image
        "base_height": 1075.0,  # Top camera height in mm
        "side_camera_height": 800.0,  # Side camera height in mm
        "side_ref_pixels": 500.0,  # Reference height in pixels (side view)
        "wheel_height": 75.0,  # Current wheel height (will be updated during processing)
        "fx": 640.268494,  # Default camera intrinsics
        "fy": 640.268494,
        "cx": 642.991272,
        "cy": 364.30368,
        "depth_scale": 0.001,
        "depth_units": 1000.0,
        "depth_min": 200.0,
        "depth_max": 3000.0
    }
}

# Default wheel models
DEFAULT_WHEEL_MODELS = {
    "10-13": {"min_dia": 10, "max_dia": 13, "height": 70, "tolerance": 3.0},
    "13-16": {"min_dia": 13, "max_dia": 16, "height": 75, "tolerance": 3.0},
    "16-19": {"min_dia": 16, "max_dia": 19, "height": 77, "tolerance": 3.0},
    "19-22": {"min_dia": 19, "max_dia": 22, "height": 115, "tolerance": 3.0}
}

def load_settings():
    """
    Load application settings from file or create defaults if not found
    
    Returns:
        tuple: (settings dict, wheel models dict)
    """
    settings = DEFAULT_SETTINGS.copy()
    wheel_models = DEFAULT_WHEEL_MODELS.copy()
    
    # Try to load existing settings
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
                
            if "settings" in data:
                # Update existing settings
                for key, value in data["settings"].items():
                    if key in settings:
                        if key == "calibration" and isinstance(value, dict):
                            # Merge calibration settings
                            for cal_key, cal_value in value.items():
                                settings["calibration"][cal_key] = cal_value
                        else:
                            settings[key] = value
            
            if "wheel_models" in data:
                wheel_models = data["wheel_models"]
                
            # print(f"Loaded settings with model: {settings['selected_model']}")
            # print(f"Loaded calibration: {settings['calibration']}")
            # print(f"Loaded models: {wheel_models}")
                
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
    else:
        # Create default settings file
        save_settings(settings, wheel_models)
    
    return settings, wheel_models

def save_settings(settings, wheel_models):
    """
    Save application settings to file
    
    Args:
        settings (dict): Application settings
        wheel_models (dict): Wheel model definitions
    """
    try:
        data = {
            "settings": settings,
            "wheel_models": wheel_models,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
            
        print("Settings saved successfully")
    except Exception as e:
        print(f"Error saving settings: {e}")

def load_realsense_calibration(calibration_file):
    """
    Load RealSense calibration data from JSON file
    
    Args:
        calibration_file (str): Path to calibration file
        
    Returns:
        dict: Calibration parameters
    """
    calibration = {}
    
    try:
        if os.path.exists(calibration_file):
            with open(calibration_file, "r") as f:
                calib_data = json.load(f)
                
            # Extract camera intrinsics
            if "intrinsics" in calib_data:
                calibration["fx"] = calib_data["intrinsics"]["fx"]
                calibration["fy"] = calib_data["intrinsics"]["fy"]
                calibration["cx"] = calib_data["intrinsics"]["ppx"]
                calibration["cy"] = calib_data["intrinsics"]["ppy"]
                
                print(f"Loaded calibration from file: fx={calibration['fx']}, fy={calibration['fy']}, " +
                      f"cx={calibration['cx']}, cy={calibration['cy']}")
    except Exception as e:
        print(f"Error loading RealSense calibration: {e}")
        
    return calibration
