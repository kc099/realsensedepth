import json
import os
import time
from functools import lru_cache
from typing import Dict, Any

# Import centralized debug system
from debug_utils import debug_print

# Try importing RealSense library to check availability
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# Default settings and paths, moved from settings_manager.py
SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "selected_model": "10-13",
    "auto_capture_interval": 5,
    "top_camera_url": "http://192.168.100.50:8080/stream-hd",
    "side_camera_url": "http://192.168.100.51:8080/stream-hd",
    "com_port": None,
    "baud_rate": 19200,
    "modbus_slave_id": 1,
    "calibration": {
        "ref_diameter": 466.0,
        "ref_diameter_pixels": 632.62,
        "base_height": 1075.0,
        "side_camera_height": 800.0,
        "side_ref_pixels": 500.0,
        "wheel_height": 75.0,
        "fx": 640.268494,
        "fy": 640.268494,
        "cx": 642.991272,
        "cy": 364.30368,
        "depth_scale": 0.001,
        "depth_units": 1000.0,
        "depth_min": 200.0,
        "depth_max": 3000.0
    }
}
DEFAULT_WHEEL_MODELS = {
    "10-13": {"min_dia": 10, "max_dia": 13, "height": 70, "tolerance": 3.0},
    "13-16": {"min_dia": 13, "max_dia": 16, "height": 75, "tolerance": 3.0},
    "16-19": {"min_dia": 16, "max_dia": 19, "height": 77, "tolerance": 3.0},
    "19-22": {"min_dia": 19, "max_dia": 22, "height": 115, "tolerance": 3.0}
}

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance
    
    def _init_manager(self):
        self._settings = {}
        self._wheel_models = {}
        self._intrinsics = {
            'top_camera': None,
            'side_camera': None,
            'realsense': None
        }
        self._last_load_time = 0
        self.load_all_configs()
        self._cache_intrinsics()

    def _load_settings(self):
        """Load application settings from file, using defaults if necessary."""
        self._settings = DEFAULT_SETTINGS.copy()
        self._wheel_models = DEFAULT_WHEEL_MODELS.copy()
        
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                    
                if "settings" in data:
                    for key, value in data["settings"].items():
                        if key == "calibration" and isinstance(value, dict):
                            self._settings["calibration"].update(value)
                        elif key in self._settings:
                            self._settings[key] = value

                if "wheel_models" in data:
                    self._wheel_models = data["wheel_models"]
                    
                if "selected_model" in data:
                    self._settings["selected_model"] = data["selected_model"]
                    
            except Exception as e:
                debug_print(f"Error loading settings: {e}. Using defaults.", "errors")
        else:
            # If no settings file, save the defaults
            self._save_settings(silent=True)

    def _save_settings(self, silent=False):
        """Save settings and invalidate cache."""
        try:
            if "com_port" in self._settings:
                self._settings["com_port"] = None
                
            data = {
                "settings": self._settings,
                "wheel_models": self._wheel_models,
                "selected_model": self._settings.get("selected_model", "10-13"),
                "last_updated": time.strftime("%d-%m-%Y %H:%M:%S")
            }
            
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=4)
                
            if not silent:
                debug_print("Settings saved and cache invalidated", "startup")
            return True
        except Exception as e:
            debug_print(f"Error saving settings: {e}", "errors")
            return False

    def _cache_intrinsics(self):
        """Cache all intrinsics at startup - load once and store"""
        from camera_utils import load_camera_intrinsics
    
        # Load once and store from unified intrinsics file
        debug_print("Loading camera intrinsics at startup...", "startup")
        self._intrinsics['top_camera'] = load_camera_intrinsics("top_camera")
        self._intrinsics['side_camera'] = load_camera_intrinsics("side_camera") 
        
        # DO NOT pre-load RealSense intrinsics - wheel_main will handle device-first loading
        # This will be populated by wheel_main after device acquisition (or fallback to file)
        self._intrinsics['realsense'] = None
        
        # RealSense intrinsics are handled by wheel_main.py device-first flow
        # DO NOT load RealSense intrinsics here - wheel_main will populate this after device acquisition
        
        debug_print("Camera intrinsics cached successfully", "startup")

    def _get_realsense_intrinsics_from_device(self):
        """Get RealSense intrinsics directly from device"""
        try:
            from camera_streams import RealSenseCamera
            debug_print("Attempting to get RealSense intrinsics from device...", "startup")
            
            # Create temporary camera instance
            temp_camera = RealSenseCamera()
            
            # Start camera to get intrinsics
            if temp_camera.start_realsense():
                intrinsics = temp_camera.get_intrinsics()
                temp_camera.stop_realsense()  # Stop immediately after getting intrinsics
                debug_print("Successfully acquired RealSense intrinsics from device", "startup")
                return intrinsics
            else:
                debug_print("Failed to start RealSense camera for intrinsics", "errors")
                return None
                
        except Exception as e:
            debug_print(f"Error getting RealSense intrinsics from device: {e}", "errors")
            return None
    
    def _save_realsense_intrinsics_to_both_files(self, intrinsics):
        """Save RealSense intrinsics to both settings.json and unified intrinsics file"""
        try:
            # Save to settings.json
            if 'calibration' not in self._settings:
                self._settings['calibration'] = {}
            
            # Update calibration section with RealSense intrinsics
            self._settings['calibration'].update({
                'fx': intrinsics.get('fx', 0),
                'fy': intrinsics.get('fy', 0),
                'cx': intrinsics.get('cx', 0),
                'cy': intrinsics.get('cy', 0)
            })
            
            # Save to settings.json
            self.save_settings()
            
            # Save to unified intrinsics file
            from camera_utils import save_camera_intrinsics
            
            # Prepare complete intrinsics data for unified file
            unified_intrinsics = {
                "camera_matrix": [
                    [intrinsics.get('fx', 0), 0.0, intrinsics.get('cx', 0)],
                    [0.0, intrinsics.get('fy', 0), intrinsics.get('cy', 0)],
                    [0.0, 0.0, 1.0]
                ],
                "dist_coeffs": intrinsics.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0]),
                "width": intrinsics.get('width', 1280),
                "height": intrinsics.get('height', 720),
                "fx": intrinsics.get('fx', 0),
                "fy": intrinsics.get('fy', 0),
                "cx": intrinsics.get('cx', 0),
                "cy": intrinsics.get('cy', 0)
            }
            
            # Add optional fields if available
            if 'device_serial' in intrinsics:
                unified_intrinsics['device_serial'] = intrinsics['device_serial']
            if 'device_product_line' in intrinsics:
                unified_intrinsics['device_product_line'] = intrinsics['device_product_line']
                
            save_camera_intrinsics('realsense', unified_intrinsics)
            debug_print("RealSense intrinsics saved to both settings.json and unified intrinsics file", "startup")
            
        except Exception as e:
            debug_print(f"Error saving RealSense intrinsics to both files: {e}", "errors")

    def get_intrinsics(self, camera_type):
        """Get cached intrinsics without reloading"""
        return self._intrinsics.get(camera_type)
    
    def load_realsense_intrinsics_from_file(self):
        """Load RealSense intrinsics from file as fallback when device fails"""
        from camera_utils import load_camera_intrinsics
        cached_intrinsics = load_camera_intrinsics("realsense")
        if cached_intrinsics:
            self._intrinsics['realsense'] = cached_intrinsics
            debug_print(f"✅ Fallback: Using cached RealSense intrinsics (fx={cached_intrinsics.get('fx', 0):.1f}, fy={cached_intrinsics.get('fy', 0):.1f})", "startup")
            return cached_intrinsics
        else:
            debug_print("❌ No valid RealSense intrinsics found in cache files", "errors")
            return None
    
    @property
    def settings(self) -> Dict[str, Any]:
        """Get current settings"""
        if time.time() - self._last_load_time > 5:  # 5 second cache
            self.load_all_configs()
        return self._settings
    
    @property
    def wheel_models(self) -> Dict[str, Any]:
        """Get current wheel models"""
        if time.time() - self._last_load_time > 5:
            self.load_all_configs()
        return self._wheel_models
    
    @property
    def intrinsics(self) -> Dict[str, Any]:
        """Get camera intrinsics"""
        return self._intrinsics
    
    def load_all_configs(self):
        """Load all configuration files"""
        self._load_settings()
        self._last_load_time = time.time()
    
    def sync_calibration_values(self):
        """Sync calibration values between settings.json and unified intrinsics file"""
        try:
            from camera_utils import save_camera_intrinsics
            
            # Get current calibration from settings
            calib = self._settings.get('calibration', {})
            
            # Sync RealSense intrinsics if available in settings
            if calib.get('fx') and calib.get('fy'):
                # Prepare unified intrinsics data
                unified_data = {
                    "camera_matrix": [
                        [calib.get('fx', 0), 0.0, calib.get('cx', 0)],
                        [0.0, calib.get('fy', 0), calib.get('cy', 0)],
                        [0.0, 0.0, 1.0]
                    ],
                    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "width": 1280,
                    "height": 720,
                    "fx": calib.get('fx', 0),
                    "fy": calib.get('fy', 0),
                    "cx": calib.get('cx', 0),
                    "cy": calib.get('cy', 0)
                }
                
                # Save to unified file
                save_camera_intrinsics("realsense", unified_data)
                # Update cached intrinsics
                self._intrinsics['realsense'] = unified_data
                
            # Don't reload intrinsics from file since we already have them cached
            # This prevents the repeated "Successfully loaded intrinsics" messages
            
            # Silent sync - only print if there was an actual change
            # print("Calibration values synced between settings.json and unified intrinsics file")
            
        except Exception as e:
            debug_print(f"Error syncing calibration values: {e}", "errors")
    
    def load_intrinsics_to_settings(self):
        """Load intrinsics from camera_intrinsics.json into settings.json calibration section"""
        try:
            # Ensure calibration section exists
            if 'calibration' not in self._settings:
                self._settings['calibration'] = {}
                
            # Use cached RealSense intrinsics if available to avoid repeated file loading
            realsense_intrinsics = None
            if 'realsense' in self._intrinsics and self._intrinsics['realsense']:
                realsense_intrinsics = self._intrinsics['realsense']
            else:
                # Fallback to loading from file if not cached
                from camera_utils import load_camera_intrinsics
                realsense_intrinsics = load_camera_intrinsics("realsense")
                if realsense_intrinsics:
                    self._intrinsics['realsense'] = realsense_intrinsics
            
            if realsense_intrinsics:
                # Only update if values are different to avoid unnecessary operations
                current_fx = self._settings['calibration'].get('fx', 0)
                if abs(current_fx - realsense_intrinsics.get('fx', 0)) > 0.1:  # Only update if significantly different
                    self._settings['calibration'].update({
                        'fx': realsense_intrinsics.get('fx', 0),
                        'fy': realsense_intrinsics.get('fy', 0),
                        'cx': realsense_intrinsics.get('cx', 0),
                        'cy': realsense_intrinsics.get('cy', 0)
                    })
                return True
                
        except Exception as e:
            debug_print(f"Error loading intrinsics to settings: {e}", "errors")
            return False
    
    def update_calibration_value(self, key, value):
        """Update a single calibration value and sync to both files"""
        try:
            # Ensure calibration section exists
            if 'calibration' not in self._settings:
                self._settings['calibration'] = {}
                
            # Update in settings
            self._settings['calibration'][key] = value
            
            # Save settings and sync
            if self.save_settings(silent=True):
                debug_print(f"Calibration value {key} updated to {value} and synced", "startup")
                return True
            return False
            
        except Exception as e:
            debug_print(f"Error updating calibration value {key}: {e}", "errors")
            return False
    
    def get_all_intrinsics_for_display(self):
        """Get all intrinsics formatted for display in settings window"""
        try:
            # Ensure we have the latest data
            self.load_intrinsics_to_settings()
            
            # Return calibration data from settings for display
            return self._settings.get('calibration', {})
            
        except Exception as e:
            debug_print(f"Error getting intrinsics for display: {e}", "errors")
            return {}
    
    def get_camera_intrinsics_for_display(self, camera_type):
        """Get specific camera intrinsics for display in settings window"""
        try:
            # Use cached intrinsics if available to avoid repeated file loading
            if camera_type in self._intrinsics and self._intrinsics[camera_type]:
                intrinsics = self._intrinsics[camera_type]
                return {
                    'fx': intrinsics.get('fx', 0),
                    'fy': intrinsics.get('fy', 0),
                    'cx': intrinsics.get('cx', 0),
                    'cy': intrinsics.get('cy', 0),
                    'width': intrinsics.get('width', 0),
                    'height': intrinsics.get('height', 0)
                }
            
            # Fallback to loading from file if not cached (but this shouldn't happen normally)
            from camera_utils import load_camera_intrinsics
            intrinsics = load_camera_intrinsics(camera_type)
            
            if intrinsics:
                # Cache the loaded intrinsics
                self._intrinsics[camera_type] = intrinsics
                return {
                    'fx': intrinsics.get('fx', 0),
                    'fy': intrinsics.get('fy', 0),
                    'cx': intrinsics.get('cx', 0),
                    'cy': intrinsics.get('cy', 0),
                    'width': intrinsics.get('width', 0),
                    'height': intrinsics.get('height', 0)
                }
            return {}
            
        except Exception as e:
            debug_print(f"Error getting {camera_type} intrinsics for display: {e}", "errors")
            return {}
    
    def update_camera_intrinsics(self, camera_type, intrinsics_data):
        """Update specific camera intrinsics in camera_intrinsics.json"""
        try:
            from camera_utils import save_camera_intrinsics
            
            # Prepare complete intrinsics data
            complete_data = {
                "camera_matrix": [
                    [intrinsics_data.get('fx', 0), 0.0, intrinsics_data.get('cx', 0)],
                    [0.0, intrinsics_data.get('fy', 0), intrinsics_data.get('cy', 0)],
                    [0.0, 0.0, 1.0]
                ],
                "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                "width": intrinsics_data.get('width', 1280),
                "height": intrinsics_data.get('height', 720),
                "fx": intrinsics_data.get('fx', 0),
                "fy": intrinsics_data.get('fy', 0),
                "cx": intrinsics_data.get('cx', 0),
                "cy": intrinsics_data.get('cy', 0)
            }
            
            # Save to unified file
            if save_camera_intrinsics(camera_type, complete_data):
                # Update cached intrinsics
                self._intrinsics[camera_type] = complete_data
                debug_print(f"{camera_type} intrinsics updated and cached", "startup")
                return True
            return False
            
        except Exception as e:
            debug_print(f"Error updating {camera_type} intrinsics: {e}", "errors")
            return False
    
    def save_settings(self, silent=False):
        """Save settings back to file and sync calibration"""
        if self._save_settings(silent=silent):
            self._last_load_time = time.time()
            # Sync calibration values after saving settings
            self.sync_calibration_values()
            return True
        return False