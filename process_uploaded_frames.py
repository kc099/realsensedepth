import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
from maskrcnn import process_frame
from PIL import Image

def get_valid_depth_in_window(x, y, depth_image, window_size=5):
    """
    Get a valid depth value in a window around the specified coordinates.
    Returns the median of valid depth values (non-zero) in the window.
    If no valid depth values are found, returns 0.
    
    Args:
        x: X-coordinate
        y: Y-coordinate
        depth_image: Depth image array
        window_size: Size of window (e.g., 5 means 5x5 window)
    
    Returns:
        Valid depth value or 0 if no valid value is found
    """
    # Ensure coordinates are integers
    x, y = int(x), int(y)
    
    # Calculate window bounds
    half_size = window_size // 2
    min_x = max(0, x - half_size)
    max_x = min(depth_image.shape[1] - 1, x + half_size)
    min_y = max(0, y - half_size)
    max_y = min(depth_image.shape[0] - 1, y + half_size)
    
    # Extract the window
    window = depth_image[min_y:max_y+1, min_x:max_x+1]
    
    # Ensure window is 2D
    if len(window.shape) > 2:
        window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    
    # Get valid depth values (non-zero)
    valid_depths = window[window > 0]
    
    if len(valid_depths) > 0:
        # Return median of valid depths (more robust than mean)
        return np.median(valid_depths)
    else:
        # No valid depth found in window
        return 0

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

def process_uploaded_frames(rgb_image_path, depth_image_path, calibration_file=None):
    """Process uploaded RGB and depth frames"""
    # Load calibration data
    camera_matrix, dist_coeffs = load_calibration(calibration_file)
    
    # Load uploaded images
    color_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Load as-is to preserve depth values
    
    # Ensure depth image is single-channel
    if depth_image is not None and len(depth_image.shape) > 2:
        # If depth image has multiple channels, convert to grayscale
        print(f"Converting {depth_image.shape} depth image to single channel")
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    
    if color_image is None or depth_image is None:
        print("Error loading images")
        return
    
    # Ensure both images have the same dimensions
    if color_image.shape[:2] != depth_image.shape[:2]:
        print(f"Warning: RGB image shape {color_image.shape[:2]} doesn't match depth image shape {depth_image.shape[:2]}")
        # Resize depth to match color if needed
        depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Set up RealSense pipeline (needed for rs.rs2_deproject_pixel_to_point)
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Start streaming with a dummy configuration (we won't use the actual streams)
    profile = pipeline.start(config)
    
    # Create a custom intrinsics object with the provided values
    depth_intrin = rs.intrinsics()
    depth_intrin.width = 640
    depth_intrin.height = 480
    depth_intrin.ppx = 330.815    # Principal point x
    depth_intrin.ppy = 244.086    # Principal point y
    depth_intrin.fx = 606.365     # Focal length x
    depth_intrin.fy = 605.852     # Focal length y
    depth_intrin.model = rs.distortion.inverse_brown_conrady
    depth_intrin.coeffs = [0, 0, 0, 0, 0]
    
    print(f"Using depth intrinsics: {depth_intrin}")
    
    # Depth scale (meters per unit)
    depth_scale = 0.001  # 1mm = 0.001 meters for most RealSense depth data
    
    # Make copies for display
    display_image = color_image.copy()
    
    # Run MaskRCNN detection for wheels
    result = process_frame(color_image)
    boxes, masks, scores, labels = result["boxes"], result["masks"], result["scores"], result["labels"]
    
    # Log detection results
    print(f"Detected wheels: {len(boxes) if boxes is not None else 0}")
    if boxes is not None and len(boxes) > 0:
        print(f"Wheel bounding boxes: {boxes}")
        print(f"Wheel confidence scores: {scores}")
        print(f"Mask shape: {masks.shape if masks is not None else None}")
        print("\n====== WHEEL DETECTION SUCCEEDED ======")
    else:
        print("\n!!! NO WHEELS DETECTED - Check your image or model !!!")
    
    # Create trackbars for object detection tuning
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Threshold', 'Controls', 120, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Controls', 500, 20000, lambda x: None)
    cv2.createTrackbar('Depth Min (mm)', 'Controls', 200, 2000, lambda x: None)
    cv2.createTrackbar('Depth Max (mm)', 'Controls', 800, 4000, lambda x: None)
    
    # Simple mouse callback to check depth at clicked point
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Handle different depth image shapes
            if len(depth_image.shape) == 2:
                depth_value = depth_image[y, x]
            else:
                # For multi-channel depth images, use the first channel or convert to grayscale
                if depth_image.shape[2] == 1:
                    depth_value = depth_image[y, x, 0]
                else:
                    # Use average of channels or convert to grayscale first
                    depth_value = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)[y, x]
            
            distance = depth_value * depth_scale
            print(f"Clicked at ({x}, {y}) - Depth: {depth_value}, Distance: {distance:.3f}m")
    
    # Register the callback
    cv2.namedWindow('RealSense Measurement')
    cv2.setMouseCallback('RealSense Measurement', mouse_callback)
    
    # For simplicity, we'll process once rather than in a loop
    try:
        # Apply calibration to undistort the color image if available
        if camera_matrix is not None and dist_coeffs is not None:
            undistorted = cv2.undistort(color_image, camera_matrix, dist_coeffs)
            calibration_text = "CALIBRATED"
        else:
            undistorted = color_image.copy()
            calibration_text = "UNCALIBRATED"
        
        # Get parameters from trackbars
        thresh_val = cv2.getTrackbarPos('Threshold', 'Controls')
        min_area = cv2.getTrackbarPos('Min Area', 'Controls')
        depth_min = cv2.getTrackbarPos('Depth Min (mm)', 'Controls')
        depth_max = cv2.getTrackbarPos('Depth Max (mm)', 'Controls')
        
        # Make a copy for display
        display_image = undistorted.copy()
        
        # Create depth visualization with adjustable range
        depth_colormap = np.zeros_like(color_image)
        
        # Adjust scale based on your depth image format
        # For 16-bit depth images from RealSense, they typically need to be scaled
        depth_mm = depth_image * depth_scale * 1000  # Convert to mm
        
        # Ensure depth_mm is single-channel
        if len(depth_mm.shape) > 2:
            print(f"Warning: depth_mm has shape {depth_mm.shape}, converting to single channel")
            depth_mm = cv2.cvtColor(depth_mm.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(depth_mm.dtype)
        
        # Create normalized depth image for visualization
        depth_normalized = np.clip(depth_mm, depth_min, depth_max)
        depth_normalized = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Ensure depth_normalized is single-channel for colormap
        if len(depth_normalized.shape) > 2:
            depth_normalized = cv2.cvtColor(depth_normalized, cv2.COLOR_BGR2GRAY)
        
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
        depth_mask = np.zeros_like(binary)
        valid_depth = (depth_mm >= depth_min) & (depth_mm <= depth_max)
        depth_mask[valid_depth] = 255
        
        # Ensure depth_mask and binary have the same shape
        if depth_mask.shape != binary.shape:
            depth_mask = cv2.resize(depth_mask, (binary.shape[1], binary.shape[0]))
            
        # Combine color-based and depth-based segmentation
        binary = binary.astype(np.uint8)
        depth_mask = depth_mask.astype(np.uint8)
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
        
        # Process wheel detections from MaskRCNN
        if boxes is not None and len(boxes) > 0 and masks is not None:
            # Create a visualization mask for detected wheels
            wheel_mask = np.zeros_like(color_image)
            
            # Process the highest confidence wheel
            i = 0  # First wheel (highest confidence after our filtering)
            box = boxes[i]
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box for the wheel
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Add label and score
            label_text = f"Wheel: {scores[i]:.2f}"
            cv2.putText(display_image, label_text, 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Get the binary mask for the wheel
            mask = masks[i, 0]
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Ensure mask has correct dimensions
            if mask_binary.shape[:2] != color_image.shape[:2]:
                mask_binary = cv2.resize(mask_binary, (color_image.shape[1], color_image.shape[0]))
            
            # Create colored mask (green with transparency)
            wheel_mask[:, :, 1] = mask_binary * 255  # Green channel
            
            # Apply the mask with transparency
            display_image = cv2.addWeighted(display_image, 1, wheel_mask, 0.5, 0)
            

            # Find top and bottom points of the wheel mask
            wheel_points = np.where(mask_binary > 0)
            if len(wheel_points[0]) > 0:
                # Get min and max y coordinates (top and bottom points)
                min_y = np.min(wheel_points[0])
                max_y = np.max(wheel_points[0])
                
                # Get horizontal center of the wheel
                center_x = (x1 + x2) // 2
                center_top_y = min_y
                center_bottom_y = max_y
                
                # Update the points for measurement
                center_top_x, center_top_y = center_x, center_top_y
                center_bottom_x, center_bottom_y = center_x, center_bottom_y
                
                # Mark these points on the image
                cv2.circle(display_image, (center_top_x, center_top_y), 5, (255, 0, 0), -1)  # Blue for top
                cv2.circle(display_image, (center_bottom_x, center_bottom_y), 5, (0, 0, 255), -1)  # Red for bottom
                
                # Draw line connecting center top and center bottom
                cv2.line(display_image, (center_top_x, center_top_y), 
                        (center_bottom_x, center_bottom_y), (0, 255, 0), 2)
                
                # Get depth values for the wheel points
                center_depth_value = get_valid_depth_in_window(center_x, (min_y + max_y) // 2, depth_image, 9)
                top_depth_value = get_valid_depth_in_window(center_top_x, center_top_y, depth_image, 9)
                bottom_depth_value = get_valid_depth_in_window(center_bottom_x, center_bottom_y, depth_image, 9)
                
                # Convert raw depth values to meters
                center_dist = center_depth_value * depth_scale
                top_dist = top_depth_value * depth_scale
                bottom_dist = bottom_depth_value * depth_scale
                
                # Log the depths for debugging
                print(f"Wheel Raw depths - Center: {center_depth_value}, Top: {top_depth_value}, Bottom: {bottom_depth_value}")
                print(f"Wheel Distances - Center: {center_dist:.3f}m, Top: {top_dist:.3f}m, Bottom: {bottom_dist:.3f}m")
                
                # Check if depth values are valid
                if all(d > 0 for d in [center_dist, top_dist, bottom_dist]):
                    # Convert 2D points to 3D using depth and our custom intrinsics
                    top_point = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [center_top_x, center_top_y], top_dist)
                    bottom_point = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [center_bottom_x, center_bottom_y], bottom_dist)
                    
                    # Calculate height in 3D space
                    height_meters = np.sqrt(
                        (top_point[0] - bottom_point[0])**2 + 
                        (top_point[1] - bottom_point[1])**2 + 
                        (top_point[2] - bottom_point[2])**2)
                    height_cm = height_meters*100
                    
                    # Enhanced debugging output to terminal with more detail
                    print("\n====== WHEEL HEIGHT MEASUREMENT ======")
                    print(f"Wheel Measured height: {height_cm:.1f} cm")
                    print(f"3D Points - Top: ({top_point[0]:.3f}, {top_point[1]:.3f}, {top_point[2]:.3f}) m")
                    print(f"3D Points - Bottom: ({bottom_point[0]:.3f}, {bottom_point[1]:.3f}, {bottom_point[2]:.3f}) m")
                    print(f"Pixel coords - Top: ({center_top_x}, {center_top_y}), Bottom: ({center_bottom_x}, {center_bottom_y})")
                    print("====================================")
                    
                    # Add height text to display
                    cv2.putText(display_image, f"Wheel Height: {height_cm:.1f} cm", 
                                (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Distance: {center_dist:.3f} m", 
                                (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add height text overlay directly on the segmented area
                    # Calculate position for text in the middle of the wheel mask
                    mask_center_x = (x1 + x2) // 2
                    mask_center_y = (min_y + max_y) // 2
                    
                    # Draw a semi-transparent background for better text visibility
                    text_size = cv2.getTextSize(f"{height_cm:.1f} cm", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    cv2.rectangle(display_image, 
                                (mask_center_x - text_size[0]//2 - 10, mask_center_y - text_size[1]//2 - 10),
                                (mask_center_x + text_size[0]//2 + 10, mask_center_y + text_size[1]//2 + 10),
                                (0, 0, 0), -1)
                    
                    # Draw height text with large bold font in the center of the wheel mask
                    cv2.putText(display_image, f"{height_cm:.1f} cm", 
                              (mask_center_x - text_size[0]//2, mask_center_y + text_size[1]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Display the results
        cv2.putText(display_image, calibration_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show images
        cv2.imshow('RealSense Measurement', display_image)
        cv2.imshow('Depth Colormap', depth_colormap)
        cv2.imshow('Debug: Combined', debug_combined)
        
        print("\nPress any key to close the windows.")
        cv2.waitKey(0)
        
    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()

def create_gui():
    """Create a GUI for selecting RGB and depth images"""
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk
    import os
    
    root = tk.Tk()
    root.title("RealSense Frame Processor")
    root.geometry("800x600")
    
    # Variables to store file paths
    rgb_path = tk.StringVar()
    depth_path = tk.StringVar()
    calib_path = tk.StringVar()
    
    # Set default calibration path if available
    calib_files = [f for f in os.listdir('.') if f.startswith('realsense_calib_') and f.endswith('.json')]
    if calib_files:
        default_calib = os.path.abspath(sorted(calib_files)[-1])  # Get the latest file
        calib_path.set(default_calib)
    
    # Image preview frames
    preview_frame = tk.Frame(root)
    preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    rgb_preview_frame = tk.LabelFrame(preview_frame, text="RGB Image Preview", width=350, height=250)
    rgb_preview_frame.grid(row=0, column=0, padx=10, pady=10)
    rgb_preview_frame.grid_propagate(False)
    
    depth_preview_frame = tk.LabelFrame(preview_frame, text="Depth Image Preview", width=350, height=250)
    depth_preview_frame.grid(row=0, column=1, padx=10, pady=10)
    depth_preview_frame.grid_propagate(False)
    
    rgb_preview = tk.Label(rgb_preview_frame)
    rgb_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    depth_preview = tk.Label(depth_preview_frame)
    depth_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # File selection frame
    file_frame = tk.Frame(root)
    file_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # RGB image selection
    tk.Label(file_frame, text="RGB Image:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    tk.Entry(file_frame, textvariable=rgb_path, width=50).grid(row=0, column=1, padx=5, pady=5)
    
    # Depth image selection
    tk.Label(file_frame, text="Depth Image:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    tk.Entry(file_frame, textvariable=depth_path, width=50).grid(row=1, column=1, padx=5, pady=5)
    
    # Calibration file selection
    tk.Label(file_frame, text="Calibration File:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    tk.Entry(file_frame, textvariable=calib_path, width=50).grid(row=2, column=1, padx=5, pady=5)
    
    # Function to preview images
    def update_preview(path_var, preview_label, is_depth=False):
        path = path_var.get()
        if path and os.path.exists(path):
            try:
                # Load image
                if is_depth:
                    # If depth images, normalize for preview
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        return
                    
                    # Ensure single channel for depth visualization
                    if len(img.shape) > 2:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Normalize for display
                    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                    img = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.imread(path)
                    if img is None:
                        return
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize for preview
                h, w = img.shape[:2]
                max_size = 340  # Max dimension for preview
                scale = min(max_size/w, max_size/h)
                new_w, new_h = int(w*scale), int(h*scale)
                img = cv2.resize(img, (new_w, new_h))
                
                # Convert to PhotoImage
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Update label
                preview_label.configure(image=img_tk)
                preview_label.image = img_tk  # Keep reference
            except Exception as e:
                messagebox.showerror("Preview Error", f"Error loading image: {e}")
    
    # Functions for browsing files
    def browse_rgb():
        path = filedialog.askopenfilename(title="Select RGB Image", 
                                         filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            rgb_path.set(os.path.abspath(path))
            update_preview(rgb_path, rgb_preview, False)
    
    def browse_depth():
        path = filedialog.askopenfilename(title="Select Depth Image", 
                                         filetypes=[("Image files", "*.png *.tiff *.exr *.pgm")])
        if path:
            depth_path.set(os.path.abspath(path))
            update_preview(depth_path, depth_preview, True)
    
    def browse_calib():
        path = filedialog.askopenfilename(title="Select Calibration File", 
                                         filetypes=[("JSON files", "*.json")])
        if path:
            calib_path.set(os.path.abspath(path))
    
    # Browse buttons
    tk.Button(file_frame, text="Browse...", command=browse_rgb).grid(row=0, column=2, padx=5, pady=5)
    tk.Button(file_frame, text="Browse...", command=browse_depth).grid(row=1, column=2, padx=5, pady=5)
    tk.Button(file_frame, text="Browse...", command=browse_calib).grid(row=2, column=2, padx=5, pady=5)
    
    # Function to process images
    def process_images():
        rgb = rgb_path.get()
        depth = depth_path.get()
        calib = calib_path.get() if calib_path.get() else None
        
        if not rgb or not os.path.exists(rgb):
            messagebox.showerror("Error", "Please select a valid RGB image file")
            return
        
        if not depth or not os.path.exists(depth):
            messagebox.showerror("Error", "Please select a valid depth image file")
            return
        
        if calib and not os.path.exists(calib):
            messagebox.showerror("Error", "Calibration file not found")
            return
        
        root.destroy()  # Close the GUI
        process_uploaded_frames(rgb, depth, calib)  # Process images
    
    # Buttons frame
    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    tk.Button(button_frame, text="Process Images", command=process_images, 
              bg="#4CAF50", fg="white", height=2, width=20).pack(pady=10)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    create_gui()
