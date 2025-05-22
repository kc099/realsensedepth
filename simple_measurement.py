import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import json


def load_calibration(filename=None):
    """Load camera calibration from file"""
    # Find the most recent calibration file if none specified
    if filename is None:
        calib_files = [f for f in os.listdir('.') if f.startswith('realsense_calib_') and f.endswith('.json')]
        if not calib_files:
            print("No calibration files found. Running uncalibrated.")
            return None, None
        filename = sorted(calib_files)[-1]  # Get the latest file
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data["camera_matrix"])
        dist_coeffs = np.array(data["dist_coeffs"])
        print(f"Loaded calibration from {filename}")
        print(f"Camera matrix: \n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None, None

def main():
    # Load calibration data
    camera_matrix, dist_coeffs = load_calibration()
    
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams at the same resolution for both depth and color
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")
    
    # Try to enable high accuracy preset if available
    if depth_sensor.supports(rs.option.visual_preset):
        print("Setting high accuracy depth preset")
        try:
            depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy preset
        except Exception as e:
            print(f"Failed to set visual preset: {e}")
    
    # Try to set higher laser power if available
    if depth_sensor.supports(rs.option.laser_power):
        print("Setting maximum laser power")
        try:
            max_power = depth_sensor.get_option_range(rs.option.laser_power).max
            depth_sensor.set_option(rs.option.laser_power, max_power)
            print(f"Laser power set to: {max_power}")
        except Exception as e:
            print(f"Failed to set laser power: {e}")
    
    # Get device info
    device = profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    device_serial = str(device.get_info(rs.camera_info.serial_number))
    print(f"Using device: {device_product_line}, Serial: {device_serial}")
    
    # Create alignment object to align depth frames to color frames
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Create trackbars for object detection tuning
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Threshold', 'Controls', 120, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Controls', 500, 20000, lambda x: None)
    cv2.createTrackbar('Depth Alpha', 'Controls', 30, 100, lambda x: None)
    
    # Create trackbars for depth thresholds (in mm)
    cv2.createTrackbar('Depth Min (mm)', 'Controls', 200, 2000, lambda x: None)
    cv2.createTrackbar('Depth Max (mm)', 'Controls', 800, 4000, lambda x: None)
    
    # For temporal filtering of measurements
    distance_history = []
    height_history = []
    history_size = 5
    
    # Variables for measurement timing
    last_measurement_time = 0
    measurement_interval = 0.5  # 2 times per second
    
    # Current measurement values to display
    current_distance = 0
    current_height = 0
    
    # Simple mouse callback to check depth at clicked point
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if param['depth_image'] is not None:
                depth_value = param['depth_image'][y, x]
                distance = depth_value * depth_scale
                print(f"Clicked at ({x}, {y}) - Depth: {depth_value}, Distance: {distance:.3f}m")
    
    # Create dict to pass depth image to callback
    callback_params = {'depth_image': None}
    
    # Register the callback
    cv2.namedWindow('RealSense Measurement')
    cv2.setMouseCallback('RealSense Measurement', mouse_callback, callback_params)
    
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
                
            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Update depth image for mouse callback
            callback_params['depth_image'] = depth_image
            
            # Apply calibration to undistort the color image
            if camera_matrix is not None and dist_coeffs is not None:
                undistorted = cv2.undistort(color_image, camera_matrix, dist_coeffs)
                calibration_text = "CALIBRATED"
            else:
                undistorted = color_image.copy()
                calibration_text = "UNCALIBRATED"
            
            # Get parameters from trackbars
            thresh_val = cv2.getTrackbarPos('Threshold', 'Controls')
            min_area = cv2.getTrackbarPos('Min Area', 'Controls')
            depth_alpha = cv2.getTrackbarPos('Depth Alpha', 'Controls') / 1000.0
            depth_min = cv2.getTrackbarPos('Depth Min (mm)', 'Controls')
            depth_max = cv2.getTrackbarPos('Depth Max (mm)', 'Controls')
            
            # Make a copy for display
            display_image = undistorted.copy()
            
            # Create depth visualization with adjustable range
            depth_colormap = np.zeros_like(color_image)
            
            # Convert depth image to mm for thresholding
            depth_mm = depth_image * depth_scale * 1000  # Convert to mm
            
            # Create normalized depth image for visualization
            depth_normalized = np.clip(depth_mm, depth_min, depth_max)
            depth_normalized = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
            # Create color-mapped depth visualization
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Create binary thresholded image for object detection
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up binary image with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Create depth threshold mask - filter out pixels outside the desired range
            # Make sure both images have the same dimensions
            depth_mask = np.zeros_like(binary)
            valid_depth = (depth_mm >= depth_min) & (depth_mm <= depth_max)
            depth_mask[valid_depth] = 255
            
            # Ensure depth_mask and binary have the same shape
            if depth_mask.shape != binary.shape:
                depth_mask = cv2.resize(depth_mask, (binary.shape[1], binary.shape[0]))
                
            # Combine color-based and depth-based segmentation
            # Ensure both are the same type
            binary = binary.astype(np.uint8)
            depth_mask = depth_mask.astype(np.uint8)
            
            # Now perform the bitwise operation
            combined_mask = cv2.bitwise_and(binary, depth_mask)
            
            # Create debug images
            debug_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            debug_depth = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)
            debug_combined = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            
            # Find contours in the combined mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the contours on the debug image
            cv2.drawContours(debug_combined, contours, -1, (0, 255, 0), 2)
            cv2.drawContours(display_image, contours, -1, (0, 255, 0), 1)  # Thin line on main display
            
            # Flag to track if we found an object
            object_detected = False
            
            # Check if it's time to update measurements (twice per second)
            current_time = time.time()
            update_measurement = current_time - last_measurement_time >= measurement_interval
            
            # Process largest contour if any were found
            if contours:
                # Find the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Only process if contour is large enough
                if area > min_area:
                    # Draw contour on main display
                    cv2.drawContours(display_image, [largest_contour], 0, (0, 255, 0), 2)
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Draw bounding box
                    cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Calculate center top and center bottom points
                    center_top_x = x + w // 2
                    center_top_y = y
                    center_bottom_x = x + w // 2
                    center_bottom_y = y + h
                    
                    # Draw points
                    cv2.circle(display_image, (center_top_x, center_top_y), 5, (255, 0, 0), -1)  # Blue for top
                    cv2.circle(display_image, (center_bottom_x, center_bottom_y), 5, (0, 0, 255), -1)  # Red for bottom
                    
                    # Draw center point of bounding box
                    center_x, center_y = x + w//2, y + h//2
                    cv2.circle(display_image, (center_x, center_y), 5, (0, 255, 255), -1)  # Yellow for center
                    
                    # Draw line connecting center top and center bottom
                    cv2.line(display_image, (center_top_x, center_top_y), 
                             (center_bottom_x, center_bottom_y), (0, 255, 0), 2)
                    
                    # Only update measurements at specified intervals
                    if update_measurement:
                        # Get depth intrinsics for 3D point calculation
                        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                        
                        # Ensure coordinates are within bounds
                        h, w = depth_image.shape
                        
                        center_x = min(max(center_x, 0), w-1)
                        center_y = min(max(center_y, 0), h-1)
                        
                        center_top_x = min(max(center_top_x, 0), w-1)
                        center_top_y = min(max(center_top_y, 0), h-1)
                        
                        center_bottom_x = min(max(center_bottom_x, 0), w-1)
                        center_bottom_y = min(max(center_bottom_y, 0), h-1)
                        
                        # Helper function to get valid depth in a window
                        def get_valid_depth_in_window(x, y, depth_img, window_size=5):
                            half_window = window_size // 2
                            h, w = depth_img.shape
                            
                            # Ensure window is within image bounds
                            x_start = max(x - half_window, 0)
                            x_end = min(x + half_window + 1, w)
                            y_start = max(y - half_window, 0)
                            y_end = min(y + half_window + 1, h)
                            
                            # Extract window
                            window = depth_img[y_start:y_end, x_start:x_end]
                            
                            # Get valid depths (non-zero)
                            valid_depths = window[window > 0]
                            
                            if len(valid_depths) > 0:
                                # Return median of valid depths
                                return np.median(valid_depths)
                            else:
                                return 0
                        
                        # Get depth values with window-based approach for robustness
                        center_depth_value = get_valid_depth_in_window(center_x, center_y, depth_image, 9)
                        top_depth_value = get_valid_depth_in_window(center_top_x, center_top_y, depth_image, 9)
                        bottom_depth_value = get_valid_depth_in_window(center_bottom_x, center_bottom_y, depth_image, 9)
                        
                        # Convert raw depth values to meters
                        center_dist = center_depth_value * depth_scale
                        top_dist = top_depth_value * depth_scale
                        bottom_dist = bottom_depth_value * depth_scale
                        
                        # Log the depths for debugging
                        print(f"Time: {current_time:.2f}, Interval: {current_time - last_measurement_time:.2f}")
                        print(f"Raw depths - Center: {center_depth_value}, Top: {top_depth_value}, Bottom: {bottom_depth_value}")
                        print(f"Distances - Center: {center_dist:.3f}m, Top: {top_dist:.3f}m, Bottom: {bottom_dist:.3f}m")
                        
                        # Check if depth values are valid
                        if all(d > 0 for d in [center_dist, top_dist, bottom_dist]):
                            # Convert 2D points to 3D using depth
                            top_point = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [center_top_x, center_top_y], top_dist)
                            bottom_point = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [center_bottom_x, center_bottom_y], bottom_dist)
                            
                            # Calculate height in 3D space
                            height_meters = np.sqrt(
                                (top_point[0] - bottom_point[0])**2 + 
                                (top_point[1] - bottom_point[1])**2 + 
                                (top_point[2] - bottom_point[2])**2)
                            
                            # Add to history for temporal filtering
                            distance_history.append(center_dist)
                            height_history.append(height_meters)
                            
                            # Keep history at fixed size
                            if len(distance_history) > history_size:
                                distance_history.pop(0)
                            if len(height_history) > history_size:
                                height_history.pop(0)
                            
                            # Calculate filtered values (median for robustness)
                            current_distance = np.median(distance_history)
                            current_height = np.median(height_history)
                            
                            # Update last measurement time
                            last_measurement_time = current_time
                            
                            # Print measurements
                            print(f"Updated measurements - Distance: {current_distance:.3f}m, Height: {current_height*100:.1f}cm")
                    
                    # Display measurements on image (using current values)
                    # Draw semi-transparent background for text
                    overlay = display_image.copy()
                    cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
                    
                    # Display measurements prominently
                    cv2.putText(display_image, f"Distance: {current_distance:.3f} m", 
                               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    cv2.putText(display_image, f"Height: {current_height*100:.1f} cm", 
                               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # Mark that we found an object
                    object_detected = True
            
            # Display "No object detected" if no object found
            if not object_detected:
                # Reset measurements if no object is detected for a while
                if current_time - last_measurement_time > 1.0:
                    current_distance = 0
                    current_height = 0
                    distance_history = []
                    height_history = []
                
                # Draw semi-transparent background for text
                overlay = display_image.copy()
                cv2.rectangle(overlay, (10, 10), (250, 45), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
                
                # Display message
                cv2.putText(display_image, "No object detected", 
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add calibration status to images
            cv2.putText(display_image, calibration_text, (display_image.shape[1] - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if camera_matrix is not None else (0, 0, 255), 2)
            cv2.putText(depth_colormap, calibration_text, (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add depth range info to depth colormap
            cv2.putText(depth_colormap, f"BLUE = {depth_min}mm, RED = {depth_max}mm", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add measurement rate info
            cv2.putText(display_image, f"Measuring 2x/sec", (20, display_image.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stack the images horizontally for row 1
            images_row1 = np.hstack((display_image, depth_colormap))
            
            # Stack the debug images for row 2
            debug_row = np.hstack((debug_color, debug_depth))
            images_row2 = np.hstack((debug_combined, np.zeros_like(debug_combined)))
            
            # Ensure both rows have the same width
            h1, w1 = images_row1.shape[:2]
            h2, w2 = images_row2.shape[:2]
            
            if w1 != w2:
                # Resize second row to match first row width
                images_row2 = cv2.resize(images_row2, (w1, int(h2 * (w1 / w2))))
            
            # Stack the rows vertically
            combined_image = np.vstack((images_row1, images_row2))
            
            # Show the combined image
            cv2.imshow('RealSense Measurement', combined_image)
            
            # Save screenshot with 's' key
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                # Create screenshots directory if it doesn't exist
                if not os.path.exists("screenshots"):
                    os.makedirs("screenshots")
                
                # Generate filename with timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"screenshots/measurement_{timestamp}.png"
                
                # Save the combined image
                cv2.imwrite(filename, combined_image)
                print(f"Screenshot saved: {filename}")
            
            # Break loop with 'q' key or ESC
            if key & 0xFF == ord('q') or key == 27:
                break
                
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()