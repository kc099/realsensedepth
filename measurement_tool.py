import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import threading
from PIL import Image, ImageTk
import realsense_utils as utils
import os

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Measurement Tool")
        self.root.geometry("1340x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize variables
        self.running = False
        self.pipeline = None
        self.align = None
        self.filters = None
        self.pc = rs.pointcloud()
        
        # Calibration data
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_file = None
        
        # Temporal filtering buffers
        self.distance_history = []
        self.height_history = []
        self.history_size = 5
        
        # Create GUI elements
        self.create_widgets()
        
        # Load last calibration automatically
        self.load_last_calibration()
        
    def create_widgets(self):
        # Main frame for camera views
        self.camera_frame = ttk.Frame(self.root)
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Color view
        self.color_view = ttk.Label(self.camera_frame)
        self.color_view.grid(row=0, column=0, padx=5, pady=5)
        
        # Depth view
        self.depth_view = ttk.Label(self.camera_frame)
        self.depth_view.grid(row=0, column=1, padx=5, pady=5)
        
        # Control panel
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Distance display
        self.distance_var = tk.StringVar(value="Distance: -- m")
        self.distance_label = ttk.Label(self.control_frame, textvariable=self.distance_var, font=("Arial", 12, "bold"))
        self.distance_label.grid(row=0, column=0, padx=20, pady=5)
        
        # Height display
        self.height_var = tk.StringVar(value="Height: -- cm")
        self.height_label = ttk.Label(self.control_frame, textvariable=self.height_var, font=("Arial", 12, "bold"))
        self.height_label.grid(row=0, column=1, padx=20, pady=5)
        
        # Calibration selector
        self.calib_label = ttk.Label(self.control_frame, text="Calibration:")
        self.calib_label.grid(row=0, column=2, padx=5, pady=5)
        
        self.calib_var = tk.StringVar()
        self.calib_dropdown = ttk.Combobox(self.control_frame, textvariable=self.calib_var, state="readonly", width=30)
        self.calib_dropdown.grid(row=0, column=3, padx=5, pady=5)
        self.update_calibration_list()
        self.calib_dropdown.bind("<<ComboboxSelected>>", self.on_calibration_selected)
        
        # Buttons
        self.start_button = ttk.Button(self.control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=0, column=4, padx=5, pady=5)
        
        self.screenshot_button = ttk.Button(self.control_frame, text="Screenshot", command=self.take_screenshot, state=tk.DISABLED)
        self.screenshot_button.grid(row=0, column=5, padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready. Press 'Start Camera' to begin.", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Key bindings
        self.root.bind("<Escape>", lambda event: self.on_close())
        self.root.bind("<s>", lambda event: self.take_screenshot())
        
    def update_calibration_list(self):
        """Update the calibration file dropdown"""
        # Find all calibration files
        calib_files = [f for f in os.listdir('.') if f.startswith('realsense_calib_') and f.endswith('.json')]
        
        if calib_files:
            self.calib_dropdown['values'] = calib_files
            self.calib_dropdown.current(0)  # Select first file by default
        else:
            self.calib_dropdown['values'] = ["No calibration files found"]
            self.calib_dropdown.current(0)
            messagebox.showwarning("No Calibration", "No calibration files found. Accuracy may be reduced.")
    
    def load_last_calibration(self):
        """Load the most recent calibration file automatically"""
        if self.calib_dropdown['values'] and self.calib_dropdown['values'][0] != "No calibration files found":
            self.calibration_file = self.calib_dropdown['values'][0]
            self.camera_matrix, self.dist_coeffs = utils.load_calibration(self.calibration_file)
            
            if self.camera_matrix is not None:
                self.status_label.config(text=f"Loaded calibration: {self.calibration_file}")
    
    def on_calibration_selected(self, event):
        """Handle calibration file selection"""
        selected = self.calib_var.get()
        if selected != "No calibration files found":
            self.calibration_file = selected
            self.camera_matrix, self.dist_coeffs = utils.load_calibration(self.calibration_file)
            
            if self.camera_matrix is not None:
                self.status_label.config(text=f"Loaded calibration: {self.calibration_file}")
    
    def start_camera(self):
        if not self.running:
            try:
                # Initialize camera
                self.pipeline, self.align, device_product_line, device_serial = utils.initialize_camera()
                self.filters = utils.create_filters()
                self.depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
                print(f"Depth scale: {self.depth_scale}")
                # Display device info
                self.status_label.config(text=f"Camera initialized: {device_product_line}, SN: {device_serial}")
                
                # Update UI
                self.start_button.config(text="Stop Camera")
                self.screenshot_button.config(state=tk.NORMAL)
                
                # Start camera thread
                self.running = True
                self.camera_thread = threading.Thread(target=self.camera_loop)
                self.camera_thread.daemon = True
                self.camera_thread.start()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {e}")
        else:
            self.stop_camera()
    
    def stop_camera(self):
        if self.running:
            self.running = False
            if self.pipeline:
                self.pipeline.stop()
            
            # Update UI
            self.start_button.config(text="Start Camera")
            self.screenshot_button.config(state=tk.DISABLED)
            
            # Reset measurements
            self.distance_var.set("Distance: -- m")
            self.height_var.set("Height: -- cm")
            
            # Update status
            self.status_label.config(text="Camera stopped")
    
    def camera_loop(self):
        try:
            # Add trackbars for object detection parameters
            cv2.namedWindow('Controls')
            cv2.createTrackbar('Threshold', 'Controls', 120, 255, lambda x: None)
            cv2.createTrackbar('Min Area', 'Controls', 1000, 10000, lambda x: None)
            
            while self.running:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Apply filters to depth frame
                filtered_depth = utils.apply_filters(aligned_depth_frame, self.filters)
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(filtered_depth.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Get parameters from trackbars
                thresh_val = cv2.getTrackbarPos('Threshold', 'Controls')
                min_area = cv2.getTrackbarPos('Min Area', 'Controls')
                
                # Undistort color image if calibration available
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    undistorted = cv2.undistort(color_image, self.camera_matrix, self.dist_coeffs)
                else:
                    undistorted = color_image.copy()
                gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Method 1: Canny edge detection
                edges = cv2.Canny(blurred, 50, 150)
                
                # Method 2: Adaptive threshold
                binary = cv2.adaptiveThreshold(blurred, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
                
                # Combine methods
                combined = cv2.bitwise_or(edges, binary)
                
                # Morphological operations to clean up
                kernel = np.ones((5,5), np.uint8)
                processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area and aspect ratio
                valid_contours = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        aspect_ratio = float(w)/h
                        if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratios
                            valid_contours.append(cnt)
                
                # Process the largest valid contour
                if valid_contours:
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                            # Make a copy for display
                display_image = undistorted.copy()
                
                # Convert to grayscale for processing
                gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                
                # Create a debug info image for threshold visualization
                debug_image = np.zeros_like(undistorted)
                
                # Try multiple threshold methods for better object detection
                # Method 1: Binary threshold
                _, binary1 = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                
                # Method 2: Otsu's threshold (automatic)
                _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
                # Method 3: Adaptive threshold
                binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
                
                # Combine methods (try all three and see which works best)
                binary = binary1.copy()
                
                # Apply morphological operations
                kernel = np.ones((5, 5), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # Add binary image to debug visualization
                debug_image[:,:,0] = binary  # Show binary mask in blue channel
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw all contours in debug image (green)
                cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 1)
                
                # Flag to track if object detected
                object_detected = False
                
                # Process largest contour (object)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    # Add area info to debug
                    cv2.putText(debug_image, f"Largest area: {area}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if area > min_area:  # Minimum area threshold
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Draw bounding box
                        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Draw contour
                        cv2.drawContours(display_image, [largest_contour], 0, (255, 0, 255), 2)
                        
                        # Get depth intrinsics
                        depth_intrin = filtered_depth.profile.as_video_stream_profile().intrinsics
                        
                        # Find top and bottom points
                        top = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                        bottom = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
                        
                        # Draw points
                        cv2.circle(display_image, top, 7, (255, 0, 0), -1)  # Blue for top
                        cv2.putText(display_image, "Top", (top[0]+10, top[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        cv2.circle(display_image, bottom, 7, (0, 0, 255), -1)  # Red for bottom
                        cv2.putText(display_image, "Bottom", (bottom[0]+10, bottom[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Draw line between top and bottom
                        cv2.line(display_image, top, bottom, (0, 255, 0), 2)
                        
                        # Get center point
                        center_x, center_y = x + w//2, y + h//2
                        cv2.circle(display_image, (center_x, center_y), 7, (0, 255, 255), -1)  # Yellow for center
                        cv2.putText(display_image, "Center", (center_x+10, center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Ensure coordinates are within bounds
                        height, width = depth_image.shape[:2]
                        
                        center_x = min(max(center_x, 0), width-1)
                        center_y = min(max(center_y, 0), height-1)
                        top_x = min(max(top[0], 0), width-1)
                        top_y = min(max(top[1], 0), height-1)
                        bottom_x = min(max(bottom[0], 0), width-1)
                        bottom_y = min(max(bottom[1], 0), height-1)
                        
                        # Get depth values directly from depth image
                        # Convert raw depth to meters
                        center_dist = depth_image[center_y, center_x] * self.depth_scale
                        top_dist = depth_image[top_y, top_x] * self.depth_scale
                        bottom_dist = depth_image[bottom_y, bottom_x] * self.depth_scale
                        
                        # Draw depth values next to points
                        cv2.putText(display_image, f"{top_dist:.3f}m", (top[0]-70, top[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        cv2.putText(display_image, f"{bottom_dist:.3f}m", (bottom[0]-70, bottom[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        cv2.putText(display_image, f"{center_dist:.3f}m", (center_x-70, center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Check if depth values are valid (non-zero)
                        if all(d > 0 for d in [center_dist, top_dist, bottom_dist]):
                            # Convert 2D points to 3D coordinates
                            top_point = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [top_x, top_y], top_dist)
                            bottom_point = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [bottom_x, bottom_y], bottom_dist)
                            
                            # Calculate height in 3D space
                            height_meters = np.sqrt(
                                (top_point[0] - bottom_point[0])**2 + 
                                (top_point[1] - bottom_point[1])**2 + 
                                (top_point[2] - bottom_point[2])**2)
                            
                            # Add to history for temporal filtering
                            self.distance_history.append(center_dist)
                            self.height_history.append(height_meters)
                            
                            if len(self.distance_history) > self.history_size:
                                self.distance_history.pop(0)
                            if len(self.height_history) > self.history_size:
                                self.height_history.pop(0)
                            
                            # Use median filtering for stability
                            filtered_distance = np.median(self.distance_history)
                            filtered_height = np.median(self.height_history)
                            
                            # Update measurements in UI thread
                            self.root.after(0, lambda d=filtered_distance, h=filtered_height: 
                                        self.update_measurements(d, h))
                            
                            # Display measurements on image
                            # Put measurements in a black background box for better readability
                            # Draw semi-transparent background
                            overlay = display_image.copy()
                            cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
                            
                            # Draw measurements with larger font
                            cv2.putText(display_image, f"Distance: {filtered_distance:.3f} m", 
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                            cv2.putText(display_image, f"Height: {filtered_height*100:.1f} cm", 
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                            
                            object_detected = True
                
                # If no object detected, reset measurements
                if not object_detected:
                    self.distance_history = []
                    self.height_history = []
                    self.root.after(0, lambda: self.update_measurements(0, 0))
                    
                    # Display "No object detected" message
                    overlay = display_image.copy()
                    cv2.rectangle(overlay, (10, 10), (300, 50), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
                    cv2.putText(display_image, "No object detected", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Create colorized depth map for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Draw calibration status indicator
                if self.camera_matrix is not None:
                    cv2.putText(depth_colormap, "CALIBRATED", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(depth_colormap, "UNCALIBRATED", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert BGR to RGB for TkInter
                display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
                
                # Store last processed images for screenshot
                self._last_display_image = display_image.copy()
                self._last_depth_colormap = depth_colormap.copy()
                
                # Convert to PhotoImage
                display_img = Image.fromarray(display_image_rgb)
                depth_img = Image.fromarray(depth_colormap_rgb)
                
                display_img_tk = ImageTk.PhotoImage(image=display_img)
                depth_img_tk = ImageTk.PhotoImage(image=depth_img)
                
                # Update UI in main thread
                self.root.after(0, lambda c=display_img_tk, d=depth_img_tk: self.update_images(c, d))
                
                # Show debug window
                cv2.imshow('Debug: Threshold & Contours', debug_image)
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            # Properly capture the exception
            error_message = str(e)
            print(f"Camera loop error: {error_message}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda msg=error_message: self.status_label.config(text=f"Error: {msg}"))
        finally:
            if self.running:
                cv2.destroyWindow('Debug: Threshold & Contours')
                cv2.destroyWindow('Controls')
                self.root.after(0, self.stop_camera)
    
    def update_images(self, color_img_tk, depth_img_tk):
        """Update the displayed images"""
        self.color_view.imgtk = color_img_tk
        self.color_view.config(image=color_img_tk)
        
        self.depth_view.imgtk = depth_img_tk
        self.depth_view.config(image=depth_img_tk)
    
    def update_measurements(self, distance, height):
        """Update measurement displays"""
        self.distance_var.set(f"Distance: {distance:.3f} m")
        self.height_var.set(f"Height: {height*100:.1f} cm")
    
    def take_screenshot(self):
        """Save the current view as an image file"""
        if not self.running:
            return
        
        try:
            # Create screenshots directory if it doesn't exist
            if not os.path.exists("screenshots"):
                os.makedirs("screenshots")
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshots/measurement_{timestamp}.png"
            
            # Use the last processed images (which have all annotations)
            if hasattr(self, '_last_display_image') and self._last_display_image is not None and \
            hasattr(self, '_last_depth_colormap') and self._last_depth_colormap is not None:
                
                # Create composite image
                color_img = self._last_display_image
                depth_img = self._last_depth_colormap
                
                h1, w1 = color_img.shape[:2]
                h2, w2 = depth_img.shape[:2]
                
                composite = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
                composite[:h1, :w1] = color_img
                composite[:h2, w1:w1+w2] = depth_img
                
                # Save composite image
                cv2.imwrite(filename, composite)
                
                # Update status
                self.status_label.config(text=f"Screenshot saved: {filename}")
                messagebox.showinfo("Screenshot", f"Screenshot saved to:\n{filename}")
            else:
                messagebox.showerror("Error", "No processed images available for screenshot")
            
        except Exception as e:
            error_message = str(e)
            messagebox.showerror("Error", f"Failed to save screenshot: {error_message}")
            print(f"Screenshot error: {error_message}")
            import traceback
            traceback.print_exc()
        
    def on_close(self):
        """Handle window close event"""
        # Stop camera if running
        if self.running:
            self.stop_camera()
        
        # Close window
        self.root.destroy()

if __name__ == "__main__":
    # Create and run application
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()