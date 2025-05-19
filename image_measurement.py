import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
import pandas as pd
import pyrealsense2 as rs
import math

class ImageMeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Distance and Height Measurement")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.color_image = None
        self.depth_image = None
        self.depth_scale = 0.001  # Default value, will be updated if available in metadata
        self.depth_intrinsics = None
        
        # Variables for manual contour drawing
        self.drawing = False
        self.contour_points = []
        self.current_contour = None
        self.selected_contour = None
        
        # Variable to track current mode
        self.mode = "auto"  # Modes: "auto", "manual"
        
        # Calibration data
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Image display scaling
        self.display_scale = 1.0
        
        # Create UI
        self.create_widgets()
        
        # Try to load calibration data
        self.load_calibration()
    
    def create_widgets(self):
        # Top Frame for controls
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create buttons for loading images
        ttk.Button(top_frame, text="Load Color Image", command=self.load_color_image).grid(row=0, column=0, padx=5)
        ttk.Button(top_frame, text="Load Depth Image", command=self.load_depth_image).grid(row=0, column=1, padx=5)
        ttk.Button(top_frame, text="Load Metadata", command=self.load_metadata).grid(row=0, column=2, padx=5)
        
        # Create mode switches
        self.mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(top_frame, text="Auto Detection", variable=self.mode_var, value="auto", 
                       command=self.change_mode).grid(row=0, column=3, padx=5)
        ttk.Radiobutton(top_frame, text="Manual Contour", variable=self.mode_var, value="manual", 
                       command=self.change_mode).grid(row=0, column=4, padx=5)
        
        # Threshold controls for auto mode
        ttk.Label(top_frame, text="Threshold:").grid(row=0, column=5, padx=5)
        self.threshold_var = tk.IntVar(value=128)
        threshold_scale = ttk.Scale(top_frame, from_=0, to=255, variable=self.threshold_var, 
                                  orient=tk.HORIZONTAL, length=100, command=self.update_threshold)
        threshold_scale.grid(row=0, column=6, padx=5)
        ttk.Label(top_frame, textvariable=self.threshold_var).grid(row=0, column=7, padx=5)
        
        # Min area control for auto mode
        ttk.Label(top_frame, text="Min Area:").grid(row=0, column=8, padx=5)
        self.min_area_var = tk.IntVar(value=1000)
        min_area_scale = ttk.Scale(top_frame, from_=100, to=10000, variable=self.min_area_var,
                                 orient=tk.HORIZONTAL, length=100, command=self.update_min_area)
        min_area_scale.grid(row=0, column=9, padx=5)
        ttk.Label(top_frame, textvariable=self.min_area_var).grid(row=0, column=10, padx=5)
        
        # Process button
        ttk.Button(top_frame, text="Process Images", command=self.process_images).grid(row=0, column=11, padx=5)
        
        # Reset button
        ttk.Button(top_frame, text="Reset", command=self.reset).grid(row=0, column=12, padx=5)
        
        # Save results button
        ttk.Button(top_frame, text="Save Results", command=self.save_results).grid(row=0, column=13, padx=5)
        
        # Middle Frame for images
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Color image canvas
        self.color_canvas = tk.Canvas(middle_frame, bg="light gray", width=640, height=480)
        self.color_canvas.grid(row=0, column=0, padx=5, pady=5)
        self.color_canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.color_canvas.bind("<B1-Motion>", self.draw)
        self.color_canvas.bind("<ButtonRelease-1>", self.end_drawing)
        
        # Depth image canvas
        self.depth_canvas = tk.Canvas(middle_frame, bg="light gray", width=640, height=480)
        self.depth_canvas.grid(row=0, column=1, padx=5, pady=5)
        
        # Bottom Frame for results
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Results display
        ttk.Label(bottom_frame, text="Results:").grid(row=0, column=0, padx=5, sticky=tk.W)
        
        # Distance display
        ttk.Label(bottom_frame, text="Distance:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.distance_var = tk.StringVar(value="-- m")
        ttk.Label(bottom_frame, textvariable=self.distance_var, font=("Arial", 12, "bold")).grid(row=1, column=1, padx=5, sticky=tk.W)
        
        # Height display
        ttk.Label(bottom_frame, text="Height:").grid(row=2, column=0, padx=5, sticky=tk.W)
        self.height_var = tk.StringVar(value="-- cm")
        ttk.Label(bottom_frame, textvariable=self.height_var, font=("Arial", 12, "bold")).grid(row=2, column=1, padx=5, sticky=tk.W)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load color and depth images to begin.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, padx=10, pady=5)

    def change_mode(self):
        """Change between auto and manual modes"""
        self.mode = self.mode_var.get()
        if self.mode == "auto":
            self.status_var.set("Auto detection mode. Use threshold controls to adjust detection.")
        else:
            self.status_var.set("Manual mode. Draw contour on color image.")
            # Reset any previously drawn contour
            self.contour_points = []
            self.current_contour = None
            self.selected_contour = None
            self.update_color_display()
    
    def update_threshold(self, *args):
        """Update the threshold value and reprocess if in auto mode"""
        if self.mode == "auto" and self.color_image is not None and self.depth_image is not None:
            self.process_images()
    
    def update_min_area(self, *args):
        """Update the minimum area value and reprocess if in auto mode"""
        if self.mode == "auto" and self.color_image is not None and self.depth_image is not None:
            self.process_images()
    
    def load_color_image(self):
        """Load a color image from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load the image using OpenCV
                self.color_image = cv2.imread(file_path)
                if self.color_image is None:
                    raise ValueError("Failed to load color image")
                
                # Convert from BGR to RGB for display
                rgb_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                
                # Store image dimensions
                self.image_height, self.image_width = self.color_image.shape[:2]
                
                # Update the display
                self.update_color_display()
                
                self.status_var.set(f"Color image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load color image: {e}")
    
    def load_depth_image(self):
        """Load a depth image from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.tiff *.exr"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # The depth image could be in different formats
                # Try loading as a normal image first
                self.depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                
                if self.depth_image is None:
                    raise ValueError("Failed to load depth image")
                
                # Check if it's a 16-bit image (common for depth)
                if self.depth_image.dtype == np.uint16:
                    # Already loaded as 16-bit
                    pass
                elif len(self.depth_image.shape) == 3:
                    # If it's a 3-channel image, convert to single channel
                    self.depth_image = cv2.cvtColor(self.depth_image, cv2.COLOR_BGR2GRAY)
                
                # Create a visualization of the depth image
                self.update_depth_display()
                
                self.status_var.set(f"Depth image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load depth image: {e}")
    
    def load_metadata(self):
        """Load metadata CSV file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load the CSV file
                metadata = pd.read_csv(file_path)
                
                # Look for depth scale in metadata
                if 'depth_scale' in metadata.columns:
                    self.depth_scale = float(metadata['depth_scale'][0])
                    self.status_var.set(f"Metadata loaded. Depth scale: {self.depth_scale}")
                
                # Look for intrinsics in metadata
                try:
                    if all(col in metadata.columns for col in ['fx', 'fy', 'ppx', 'ppy']):
                        fx = float(metadata['fx'][0])
                        fy = float(metadata['fy'][0])
                        ppx = float(metadata['ppx'][0])
                        ppy = float(metadata['ppy'][0])
                        
                        # Create depth intrinsics object
                        self.depth_intrinsics = rs.intrinsics()
                        self.depth_intrinsics.width = self.image_width
                        self.depth_intrinsics.height = self.image_height
                        self.depth_intrinsics.ppx = ppx
                        self.depth_intrinsics.ppy = ppy
                        self.depth_intrinsics.fx = fx
                        self.depth_intrinsics.fy = fy
                        self.depth_intrinsics.model = rs.distortion.none
                        self.depth_intrinsics.coeffs = [0, 0, 0, 0, 0]
                        
                        print(f"Loaded intrinsics: fx={fx}, fy={fy}, ppx={ppx}, ppy={ppy}")
                except Exception as e:
                    print(f"Failed to load intrinsics: {e}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load metadata: {e}")
    
    def load_calibration(self, filename=None):
        """Load camera calibration from file"""
        # Find the most recent calibration file if none specified
        if filename is None:
            calib_files = [f for f in os.listdir('.') if f.startswith('realsense_calib_') and f.endswith('.json')]
            if not calib_files:
                print("No calibration files found.")
                return
            filename = sorted(calib_files)[-1]  # Get the latest file
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data["camera_matrix"])
            self.dist_coeffs = np.array(data["dist_coeffs"])
            
            # Extract intrinsics if not already set
            if self.depth_intrinsics is None and self.camera_matrix is not None:
                try:
                    # Create depth intrinsics object from camera matrix
                    self.depth_intrinsics = rs.intrinsics()
                    self.depth_intrinsics.width = int(data["image_width"])
                    self.depth_intrinsics.height = int(data["image_height"])
                    self.depth_intrinsics.ppx = self.camera_matrix[0, 2]
                    self.depth_intrinsics.ppy = self.camera_matrix[1, 2]
                    self.depth_intrinsics.fx = self.camera_matrix[0, 0]
                    self.depth_intrinsics.fy = self.camera_matrix[1, 1]
                    self.depth_intrinsics.model = rs.distortion.none
                    self.depth_intrinsics.coeffs = [0, 0, 0, 0, 0]
                    
                    print(f"Loaded intrinsics from calibration: fx={self.depth_intrinsics.fx}, fy={self.depth_intrinsics.fy}")
                except Exception as e:
                    print(f"Failed to create intrinsics from calibration: {e}")
            
            self.status_var.set(f"Loaded calibration: {filename}")
            print(f"Loaded calibration from {filename}")
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
    
    def update_color_display(self):
        """Update the color image display"""
        if self.color_image is not None:
            # Convert the image to RGB for display
            rgb_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
            
            # Create a copy for display with any contours
            display_image = rgb_image.copy()
            
            # Draw the contour if in manual mode and we have points
            if self.mode == "manual" and self.contour_points:
                # Draw the current partial contour
                if len(self.contour_points) > 1:
                    pts = np.array(self.contour_points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_image, [pts], False, (0, 255, 0), 2)
                
                # Draw the points
                for pt in self.contour_points:
                    cv2.circle(display_image, pt, 3, (255, 0, 0), -1)
            
            # If we have a completed contour, draw it
            if self.current_contour is not None:
                cv2.drawContours(display_image, [self.current_contour], 0, (0, 255, 0), 2)
            
            # If we have a selected contour from auto mode, draw it
            if self.selected_contour is not None:
                cv2.drawContours(display_image, [self.selected_contour], 0, (0, 255, 0), 2)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(self.selected_contour)
                
                # Calculate top and bottom center points
                center_top = (x + w // 2, y)
                center_bottom = (x + w // 2, y + h)
                
                # Draw points and line
                cv2.circle(display_image, center_top, 5, (255, 0, 0), -1)  # Blue for top
                cv2.circle(display_image, center_bottom, 5, (0, 0, 255), -1)  # Red for bottom
                cv2.line(display_image, center_top, center_bottom, (0, 255, 0), 2)
                
                # Add current measurements to display
                if hasattr(self, 'current_distance') and hasattr(self, 'current_height'):
                    # Draw semi-transparent background for text
                    overlay = display_image.copy()
                    cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
                    
                    # Add measurement text
                    cv2.putText(display_image, f"Distance: {self.current_distance:.3f} m", 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(display_image, f"Height: {self.current_height*100:.1f} cm", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Resize if necessary
            img_height, img_width = display_image.shape[:2]
            canvas_width = self.color_canvas.winfo_width()
            canvas_height = self.color_canvas.winfo_height()
            
            # Calculate scale factor to fit the canvas
            width_scale = canvas_width / img_width if canvas_width > 0 else 1.0
            height_scale = canvas_height / img_height if canvas_height > 0 else 1.0
            self.display_scale = min(width_scale, height_scale)
            
            # Apply scaling if needed
            if self.display_scale < 1.0:
                display_image = cv2.resize(display_image, (0, 0), fx=self.display_scale, fy=self.display_scale)
            
            # Convert to PhotoImage for display in canvas
            img = Image.fromarray(display_image)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Display the image
            self.color_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.color_canvas.image = img_tk  # Keep a reference
    
    def update_depth_display(self):
        """Update the depth image display"""
        if self.depth_image is not None:
            # Create a visualization of the depth image
            # Normalize to 0-255 range for display
            depth_min = np.min(self.depth_image)
            depth_max = np.max(self.depth_image)
            
            if depth_max > depth_min:
                normalized_depth = ((self.depth_image - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                normalized_depth = np.zeros_like(self.depth_image, dtype=np.uint8)
            
            # Apply colormap for better visualization
            depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
            
            # Resize if necessary
            img_height, img_width = depth_colormap.shape[:2]
            canvas_width = self.depth_canvas.winfo_width()
            canvas_height = self.depth_canvas.winfo_height()
            
            # Calculate scale factor to fit the canvas
            width_scale = canvas_width / img_width if canvas_width > 0 else 1.0
            height_scale = canvas_height / img_height if canvas_height > 0 else 1.0
            display_scale = min(width_scale, height_scale)
            
            # Apply scaling if needed
            if display_scale < 1.0:
                depth_colormap = cv2.resize(depth_colormap, (0, 0), fx=display_scale, fy=display_scale)
            
            # Convert to RGB and then to PhotoImage
            rgb_depth = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_depth)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Display the image
            self.depth_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.depth_canvas.image = img_tk  # Keep a reference
    
    def start_drawing(self, event):
        """Start drawing a contour"""
        if self.mode == "manual" and self.color_image is not None:
            self.drawing = True
            # Convert from canvas coordinates to image coordinates
            x, y = int(event.x / self.display_scale), int(event.y / self.display_scale)
            self.contour_points = [(x, y)]
            self.current_contour = None  # Reset current contour
            self.selected_contour = None  # Reset selected contour
            self.update_color_display()
    
    def draw(self, event):
        """Continue drawing a contour"""
        if self.mode == "manual" and self.drawing and self.color_image is not None:
            # Convert from canvas coordinates to image coordinates
            x, y = int(event.x / self.display_scale), int(event.y / self.display_scale)
            self.contour_points.append((x, y))
            self.update_color_display()
    
    def end_drawing(self, event):
        """End drawing a contour"""
        if self.mode == "manual" and self.drawing and self.color_image is not None:
            self.drawing = False
            
            # Close the contour if we have enough points
            if len(self.contour_points) > 2:
                # Close the contour by adding the first point again
                self.contour_points.append(self.contour_points[0])
                
                # Convert to numpy array for OpenCV
                self.current_contour = np.array(self.contour_points, dtype=np.int32).reshape((-1, 1, 2))
                self.selected_contour = self.current_contour
                
                # Process the contour
                self.process_images()
            else:
                self.contour_points = []
                self.current_contour = None
                self.selected_contour = None
            
            self.update_color_display()
    
    def process_images(self):
        """Process the loaded images to detect objects and measure distance/height"""
        if self.color_image is None or self.depth_image is None:
            messagebox.showwarning("Warning", "Please load both color and depth images first.")
            return
        
        try:
            # Check if color and depth images have the same dimensions
            if self.color_image.shape[:2] != self.depth_image.shape[:2]:
                messagebox.showerror("Error", "Color and depth images must have the same dimensions.")
                return
            
            # Handle different modes
            if self.mode == "auto":
                # Auto object detection mode
                # Convert to grayscale
                gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold
                _, binary = cv2.threshold(gray, self.threshold_var.get(), 255, cv2.THRESH_BINARY_INV)
                
                # Clean up binary image
                kernel = np.ones((5, 5), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Check if we have contours
                if not contours:
                    self.status_var.set("No objects detected. Try adjusting the threshold.")
                    self.selected_contour = None
                    self.update_color_display()
                    return
                
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Check if the area is large enough
                if area < self.min_area_var.get():
                    self.status_var.set(f"Object too small ({area} px²). Increase min area or adjust threshold.")
                    self.selected_contour = None
                    self.update_color_display()
                    return
                
                # Use the largest contour
                self.selected_contour = largest_contour
            
            # At this point, we should have a contour either from auto detection or manual drawing
            if self.selected_contour is None:
                self.status_var.set("No valid contour selected. Draw a contour or use auto detection.")
                return
            
            # Calculate measurements using the selected contour
            self.calculate_measurements()
            
            # Update displays
            self.update_color_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing images: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_measurements(self):
        """Calculate distance and height from the selected contour"""
        try:
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(self.selected_contour)
            
            # Calculate center top and center bottom points
            center_top = (x + w // 2, y)
            center_bottom = (x + w // 2, y + h)
            
            # Get object center point
            center_x, center_y = x + w // 2, y + h // 2
            
            # Helper function to get valid depth in a window
            def get_valid_depth_in_window(x, y, depth_img, window_size=9):
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
            
            # Get depth values at key points
            center_depth_value = get_valid_depth_in_window(center_x, center_y, self.depth_image)
            top_depth_value = get_valid_depth_in_window(center_top[0], center_top[1], self.depth_image)
            bottom_depth_value = get_valid_depth_in_window(center_bottom[0], center_bottom[1], self.depth_image)
            
            # Convert raw depth values to meters
            center_dist = float(center_depth_value) * self.depth_scale
            top_dist = float(top_depth_value) * self.depth_scale
            bottom_dist = float(bottom_depth_value) * self.depth_scale
            
            print(f"Raw depths - Center: {center_depth_value}, Top: {top_depth_value}, Bottom: {bottom_depth_value}")
            print(f"Distances - Center: {center_dist:.3f}m, Top: {top_dist:.3f}m, Bottom: {bottom_dist:.3f}m")
            
            # Check if depth values are valid
            if all(d > 0 for d in [center_dist, top_dist, bottom_dist]):
                if self.depth_intrinsics is not None:
                    # Convert 2D points to 3D space
                    top_point = rs.rs2_deproject_pixel_to_point(
                        self.depth_intrinsics, [float(center_top[0]), float(center_top[1])], top_dist)
                    bottom_point = rs.rs2_deproject_pixel_to_point(
                        self.depth_intrinsics, [float(center_bottom[0]), float(center_bottom[1])], bottom_dist)
                    
                    # Calculate height in 3D space
                    height_meters = math.sqrt(
                        (top_point[0] - bottom_point[0])**2 + 
                        (top_point[1] - bottom_point[1])**2 + 
                        (top_point[2] - bottom_point[2])**2)
                else:
                    # Fallback using pixel-based height estimation
                    # This is less accurate but works without intrinsics
                    # Simple pinhole camera model approximation using similar triangles
                    focal_length = 1000  # approximation, adjust based on camera
                    object_height_pixels = h
                    height_meters = (object_height_pixels * center_dist) / focal_length
                
                # Store current measurements as instance variables
                self.current_distance = center_dist
                self.current_height = height_meters
                
                # Update UI
                self.distance_var.set(f"{center_dist:.3f} m")
                self.height_var.set(f"{height_meters*100:.1f} cm")
                
                self.status_var.set(f"Measurements calculated. Distance: {center_dist:.3f}m, Height: {height_meters*100:.1f}cm")
                
                # Update the display with the new measurements
                self.update_color_display()
            else:
                self.status_var.set("Unable to get valid depth values for the selected points.")
                self.distance_var.set("-- m")
                self.height_var.set("-- cm")
                
        except Exception as e:
            self.status_var.set(f"Error calculating measurements: {e}")
            import traceback
            traceback.print_exc()
    
    def reset(self):
        """Reset the application state"""
        self.color_image = None
        self.depth_image = None
        self.contour_points = []
        self.current_contour = None
        self.selected_contour = None
        self.distance_var.set("-- m")
        self.height_var.set("-- cm")
        
        # Clear instance variables for measurements
        if hasattr(self, 'current_distance'):
            del self.current_distance
        if hasattr(self, 'current_height'):
            del self.current_height
        
        # Clear canvases
        self.color_canvas.delete("all")
        self.depth_canvas.delete("all")
        
        self.status_var.set("Reset complete. Load color and depth images to begin.")
    
    def save_results(self):
        """Save the processed image with measurements"""
        if self.color_image is None or self.selected_contour is None:
            messagebox.showwarning("Warning", "No processed image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Create a copy of the color image
                result_image = self.color_image.copy()
                
                # Draw the contour
                cv2.drawContours(result_image, [self.selected_contour], 0, (0, 255, 0), 2)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(self.selected_contour)
                
                # Calculate center top and center bottom points
                center_top = (x + w // 2, y)
                center_bottom = (x + w // 2, y + h)
                
                # Draw points and line
                cv2.circle(result_image, center_top, 5, (255, 0, 0), -1)  # Blue for top
                cv2.circle(result_image, center_bottom, 5, (0, 0, 255), -1)  # Red for bottom
                cv2.line(result_image, center_top, center_bottom, (0, 255, 0), 2)
                
                # Ensure we have measurements
                if hasattr(self, 'current_distance') and hasattr(self, 'current_height'):
                    # Draw semi-transparent background for text
                    overlay = result_image.copy()
                    cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
                    
                    # Format measurements with proper precision
                    distance_str = f"{self.current_distance:.3f} m"
                    height_str = f"{self.current_height*100:.1f} cm"
                    
                    # Add measurement text
                    cv2.putText(result_image, f"Distance: {distance_str}", (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(result_image, f"Height: {height_str}", (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Save the image
                cv2.imwrite(file_path, result_image)
                
                self.status_var.set(f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")

def main():
    # Create the main window
    root = tk.Tk()
    app = ImageMeasurementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()