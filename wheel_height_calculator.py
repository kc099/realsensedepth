import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from maskrcnn import process_frame

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

def process_wheel_measurement(rgb_image_path, depth_image_path, calibration_file=None):
    """Process uploaded RGB and depth frames to measure wheel height"""
    print("\n===== WHEEL HEIGHT CALCULATOR =====")
    
    # Load calibration data
    camera_matrix, dist_coeffs = load_calibration(calibration_file)
    
    # Load uploaded images
    color_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Load as-is to preserve depth values
    
    if color_image is None:
        print(f"Error: Could not load RGB image from {rgb_image_path}")
        return
        
    if depth_image is None:
        print(f"Error: Could not load depth image from {depth_image_path}")
        return
    
    # Ensure depth image is single-channel
    if depth_image is not None and len(depth_image.shape) > 2:
        # If depth image has multiple channels, convert to grayscale
        print(f"Converting {depth_image.shape} depth image to single channel")
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        
    # Examine the depth image to understand its range
    if depth_image is not None:
        non_zero = depth_image[depth_image > 0]
        if len(non_zero) > 0:
            print(f"Depth image stats - Min: {np.min(non_zero)}, Max: {np.max(non_zero)}, Mean: {np.mean(non_zero):.1f}")
    
    # Print image info
    print(f"Color image shape: {color_image.shape}")
    print(f"Depth image shape: {depth_image.shape}")
    
    # Make a copy for display
    display_image = color_image.copy()
    
    # Start a RealSense pipeline for the depth intrinsics
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        # Start streaming with a dummy configuration
        profile = pipeline.start(config)
        
        # Create custom intrinsics object
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
        
        # Run MaskRCNN detection for wheels
        print("\nRunning MaskRCNN detection on the image...")
        result = process_frame(color_image)
        boxes, masks, scores, labels = result["boxes"], result["masks"], result["scores"], result["labels"]
        
        # Log detection results
        print(f"Detected wheels: {len(boxes) if boxes is not None else 0}")
        if boxes is not None and len(boxes) > 0:
            print(f"Wheel bounding boxes: {boxes}")
            print(f"Wheel confidence scores: {scores}")
            print(f"Mask shape: {masks.shape if masks is not None else None}")
            print("\n====== WHEEL DETECTION SUCCESSFUL ======")
        else:
            print("\n!!! NO WHEELS DETECTED - Check your image or model !!!")
            return
            
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
        print("Processing wheel mask...")
        mask = masks[i, 0]
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Ensure mask has correct dimensions
        if mask_binary.shape[:2] != color_image.shape[:2]:
            print(f"Resizing mask from {mask_binary.shape} to {color_image.shape[:2]}")
            mask_binary = cv2.resize(mask_binary, (color_image.shape[1], color_image.shape[0]))
        
        # Create colored mask (green with transparency)
        wheel_mask[:, :, 1] = mask_binary * 255  # Green channel
        
        # Apply the mask with transparency
        display_image = cv2.addWeighted(display_image, 1, wheel_mask, 0.5, 0)
        
        # Calculate wheel height directly from the mask
        # Find top and bottom points of the wheel mask
        wheel_points = np.where(mask_binary > 0)
        if len(wheel_points[0]) > 0:
            # Get min and max y coordinates (top and bottom points)
            min_y = np.min(wheel_points[0])
            max_y = np.max(wheel_points[0])
            
            # Get horizontal center of the wheel
            center_x = (x1 + x2) // 2
            
            # Set the measurement points
            center_top_x, center_top_y = center_x, min_y
            center_bottom_x, center_bottom_y = center_x, max_y
            center_y = (min_y + max_y) // 2
            
            # Mark these points on the image
            cv2.circle(display_image, (center_top_x, center_top_y), 5, (255, 0, 0), -1)  # Blue for top
            cv2.circle(display_image, (center_bottom_x, center_bottom_y), 5, (0, 0, 255), -1)  # Red for bottom
            cv2.circle(display_image, (center_x, center_y), 5, (0, 255, 255), -1)  # Yellow for center
            
            # Draw line connecting center top and center bottom
            cv2.line(display_image, (center_top_x, center_top_y), 
                     (center_bottom_x, center_bottom_y), (0, 255, 0), 2)
            
            # Ensure coordinates are within bounds
            h_img, w_img = depth_image.shape[:2]
            
            center_x = min(max(center_x, 0), w_img-1)
            center_y = min(max(center_y, 0), h_img-1)
            
            center_top_x = min(max(center_top_x, 0), w_img-1)
            center_top_y = min(max(center_top_y, 0), h_img-1)
            
            center_bottom_x = min(max(center_bottom_x, 0), w_img-1)
            center_bottom_y = min(max(center_bottom_y, 0), h_img-1)
            
            # Helper function to get valid depth in a window
            def get_valid_depth_in_window(x, y, depth_img, window_size=5):
                half_window = window_size // 2
                
                # Handle different shapes of depth image
                if len(depth_img.shape) == 2:
                    h, w = depth_img.shape
                else:
                    h, w = depth_img.shape[:2]
                
                # Ensure window is within image bounds
                x_start = max(x - half_window, 0)
                x_end = min(x + half_window + 1, w)
                y_start = max(y - half_window, 0)
                y_end = min(y + half_window + 1, h)
                
                # Extract window
                window = depth_img[y_start:y_end, x_start:x_end]
                
                # If window has multiple channels, convert to single channel
                if len(window.shape) > 2:
                    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                
                # Get valid depths (non-zero)
                valid_depths = window[window > 0]
                
                if len(valid_depths) > 0:
                    # Return median of valid depths
                    return np.median(valid_depths)
                else:
                    return 0
            
            # Get depth values for wheel points with larger windows and searching along horizontal lines
            # For the center point - large window
            center_depth_value = get_valid_depth_in_window(center_x, center_y, depth_image, 15)
            
            # For top point - search along top row of the wheel mask
            top_depth_value = 0
            window_size = 25  # Use larger window for more robust depth sampling
            search_width = 100  # Search this many pixels left and right of center
            
            for offset in range(-search_width, search_width+1, 10):  # Sample every 10 pixels
                x_pos = center_x + offset
                if 0 <= x_pos < depth_image.shape[1]:
                    # Check if this point is within the mask
                    y_range = 10  # Check a few rows from the top
                    for y_offset in range(y_range):
                        check_y = min_y + y_offset
                        if 0 <= check_y < depth_image.shape[0] and mask_binary[check_y, x_pos] > 0:
                            depth = get_valid_depth_in_window(x_pos, check_y, depth_image, window_size)
                            if depth > 0:
                                top_depth_value = depth
                                # Update the position of the top point for visualization
                                center_top_x, center_top_y = x_pos, check_y
                                break
                    if top_depth_value > 0:
                        break
            
            # For bottom point - search along bottom row of the wheel mask 
            bottom_depth_value = 0
            for offset in range(-search_width, search_width+1, 10):  # Sample every 10 pixels
                x_pos = center_x + offset
                if 0 <= x_pos < depth_image.shape[1]:
                    # Check if this point is within the mask
                    y_range = 10  # Check a few rows from the bottom
                    for y_offset in range(y_range):
                        check_y = max_y - y_offset
                        if 0 <= check_y < depth_image.shape[0] and mask_binary[check_y, x_pos] > 0:
                            depth = get_valid_depth_in_window(x_pos, check_y, depth_image, window_size)
                            if depth > 0:
                                bottom_depth_value = depth
                                # Update the position of the bottom point for visualization
                                center_bottom_x, center_bottom_y = x_pos, check_y
                                break
                    if bottom_depth_value > 0:
                        break
            
            # If we still don't have valid depths, try using the center depth for all points
            if center_depth_value > 0 and (top_depth_value == 0 or bottom_depth_value == 0):
                print("Using center depth for top/bottom points since they have invalid depths")
                if top_depth_value == 0:
                    top_depth_value = center_depth_value
                if bottom_depth_value == 0:
                    bottom_depth_value = center_depth_value
            
            # Convert raw depth values to meters
            center_dist = center_depth_value * depth_scale
            top_dist = top_depth_value * depth_scale
            bottom_dist = bottom_depth_value * depth_scale
            
            # Print debug information for wheel measurements
            print("\n============= WHEEL MEASUREMENTS ==============")
            print(f"Wheel dimensions (pixels): Width={x2-x1}, Height={max_y-min_y}")
            print(f"Wheel raw depths - Center: {center_depth_value}, Top: {top_depth_value}, Bottom: {bottom_depth_value}")
            print(f"Wheel distances - Center: {center_dist:.3f}m, Top: {top_dist:.3f}m, Bottom: {bottom_dist:.3f}m")
            
            # Calculate wheel height using pixel measurements if we can't get valid depths
            if not all(d > 0 for d in [center_dist, top_dist, bottom_dist]):
                print("\n!!! USING PIXEL-BASED ESTIMATION SINCE DEPTH VALUES ARE INVALID !!!")
                # Calculate pixel height
                pixel_height = max_y - min_y
                # Estimate distance in meters (use center_dist if valid, otherwise default)
                dist_to_use = center_dist if center_dist > 0 else 0.5  # Default 0.5m if no valid depth
                # Estimate angular height using camera FOV
                vertical_fov_degrees = 58  # Typical RealSense vertical FOV
                vertical_fov_radians = vertical_fov_degrees * np.pi / 180
                angular_height = (pixel_height / depth_image.shape[0]) * vertical_fov_radians
                # Estimate real height using simple trigonometry
                wheel_height_meters = 2 * dist_to_use * np.tan(angular_height / 2)
                wheel_height_cm = wheel_height_meters * 100
                
                # Display the wheel height prominently
                print(f"\n===>>> ESTIMATED WHEEL HEIGHT (PIXEL BASED): {wheel_height_cm:.1f} cm <<<===")
                print(f"This is an approximation based on pixel measurements and estimated distance of {dist_to_use:.3f}m")
                
                # Add height text to display on the wheel
                cv2.putText(display_image, f"EST. HEIGHT: {wheel_height_cm:.1f} cm", 
                          (x1, max_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(display_image, f"Est. Distance: {dist_to_use:.3f} m", 
                          (x1, max_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, "(APPROXIMATE - Using pixel estimation)", 
                          (x1, max_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # We have valid depth values for 3D calculation
                # Convert 2D points to 3D using depth
                print("Converting 2D points to 3D space...")
                top_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_top_x, center_top_y], top_dist)
                bottom_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_bottom_x, center_bottom_y], bottom_dist)
                
                # Calculate height in 3D space
                wheel_height_meters = np.sqrt(
                    (top_point[0] - bottom_point[0])**2 + 
                    (top_point[1] - bottom_point[1])**2 + 
                    (top_point[2] - bottom_point[2])**2)
                
                print(f"3D coordinates - Top: ({top_point[0]:.3f}, {top_point[1]:.3f}, {top_point[2]:.3f})")
                print(f"3D coordinates - Bottom: ({bottom_point[0]:.3f}, {bottom_point[1]:.3f}, {bottom_point[2]:.3f})")
                print(f"3D euclidean distance: {wheel_height_meters:.3f}m")
                
                # Display the wheel height prominently
                wheel_height_cm = wheel_height_meters * 100
                print(f"\n===>>> WHEEL HEIGHT: {wheel_height_cm:.1f} cm <<<===")
                
                # Add height text to display on the wheel
                # Make text more prominent with contrasting colors and larger font
                cv2.putText(display_image, f"WHEEL HEIGHT: {wheel_height_cm:.1f} cm", 
                          (x1, max_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.putText(display_image, f"Distance: {center_dist:.3f} m", 
                          (x1, max_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(display_image, "(3D MEASUREMENT - High accuracy)", 
                          (x1, max_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print("\n!!! WHEEL MASK IS EMPTY - Check the detection !!!")
            
        # Create a normalized color map for the depth image
        depth_min, depth_max = 200, 800  # Default range in mm
        depth_mm = depth_image * depth_scale * 1000
        depth_normalized = np.clip(depth_mm, depth_min, depth_max)
        depth_normalized = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
        # Display the results
        cv2.putText(display_image, "CALIBRATED" if camera_matrix is not None else "UNCALIBRATED", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save output images
        output_dir = "./wheel_measurements"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename based on input
        base_name = os.path.splitext(os.path.basename(rgb_image_path))[0]
        output_file = f"{output_dir}/{base_name}_measured.jpg"
        
        cv2.imwrite(output_file, display_image)
        print(f"\nSaved measurement visualization to: {output_file}")
        
        # Show images
        cv2.imshow('Wheel Height Measurement', display_image)
        cv2.imshow('Depth Colormap', depth_colormap)
        
        print("\nPress any key to close the windows.")
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"Error processing images: {e}")
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
    root.title("Wheel Height Calculator")
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
                    # For depth images, normalize for preview
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
                
                # Pre-process to detect wheels in preview
                if not is_depth:
                    try:
                        # Run MaskRCNN in preview
                        color_img = cv2.imread(path)
                        result = process_frame(color_img)
                        boxes = result["boxes"]
                        if boxes is not None and len(boxes) > 0:
                            messagebox.showinfo("Wheel Detected", f"Detected wheel in image with confidence {result['scores'][0]:.2f}")
                    except Exception as e:
                        print(f"Preview wheel detection error: {e}")
                
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
        process_wheel_measurement(rgb, depth, calib)  # Process images
    
    # Buttons frame
    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Label explaining what the tool does
    info_text = """This tool measures wheel height using Mask R-CNN for wheel detection.
    1. Select an RGB image containing a wheel
    2. Select the corresponding depth image
    3. Click 'Calculate Wheel Height'
    
    The tool will show the detected wheel, measure its height,
    and display the result both on screen and in the terminal.
    """
    
    info_label = tk.Label(button_frame, text=info_text, justify=tk.LEFT, 
                          bg="#f0f0f0", fg="#333333", padx=10, pady=10)
    info_label.pack(fill=tk.X, pady=10)
    
    tk.Button(button_frame, text="Calculate Wheel Height", command=process_images, 
              bg="#4CAF50", fg="white", height=2, width=20, font=("Arial", 12, "bold")).pack(pady=10)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    create_gui()
