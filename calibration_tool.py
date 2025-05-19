import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import time
import os
from PIL import Image, ImageTk
import threading
import realsense_utils as utils

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Calibration Tool")
        self.root.geometry("1340x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize variables
        self.running = False
        self.pipeline = None
        self.align = None
        self.device_product_line = None
        self.device_serial = None
        self.filters = None
        
        self.image_count = 0
        self.required_images = 15
        self.save_image = False
        self.last_save_time = time.time() - 2
        
        # Checkerboard parameters
        self.checkerboard_size = (9, 6)  # Number of internal corners
        self.square_size = 0.023  # 25mm squares
        
        # Calibration data arrays
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Create object points
        self.objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2) * self.square_size
        
        # Corner refinement criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame for camera view
        self.camera_frame = ttk.Frame(self.root)
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera view
        self.camera_view = ttk.Label(self.camera_frame)
        self.camera_view.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Progress indicator
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.control_frame, variable=self.progress_var, length=400)
        self.progress.grid(row=0, column=0, padx=5, pady=5)
        
        # Progress label
        self.progress_label = ttk.Label(self.control_frame, text=f"Images: 0/{self.required_images}")
        self.progress_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Buttons
        self.start_button = ttk.Button(self.control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.capture_button = ttk.Button(self.control_frame, text="Capture (Space)", command=self.capture_image, state=tk.DISABLED)
        self.capture_button.grid(row=0, column=3, padx=5, pady=5)
        
        self.calibrate_button = ttk.Button(self.control_frame, text="Calibrate", command=self.calibrate, state=tk.DISABLED)
        self.calibrate_button.grid(row=0, column=4, padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready. Press 'Start Camera' to begin.", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Key bindings
        self.root.bind("<space>", lambda event: self.capture_image())
        self.root.bind("<Escape>", lambda event: self.on_close())
        
    def start_camera(self):
        if not self.running:
            try:
                # Create directory for calibration images
                if not os.path.exists("calibration_images"):
                    os.makedirs("calibration_images")
                
                # Initialize camera
                self.pipeline, self.align, self.device_product_line, self.device_serial = utils.initialize_camera()
                self.filters = utils.create_filters()
                
                # Display device info
                self.status_label.config(text=f"Camera initialized: {self.device_product_line}, SN: {self.device_serial}")
                
                # Update UI
                self.start_button.config(text="Stop Camera")
                self.capture_button.config(state=tk.NORMAL)
                
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
            self.capture_button.config(state=tk.DISABLED)
            
            # Update status
            self.status_label.config(text="Camera stopped")
    
    def camera_loop(self):
        try:
            while self.running:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Make a copy for display
                display_image = color_image.copy()
                
                # Convert to grayscale for corner detection
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                corner_found, corners = cv2.findChessboardCorners(
                    gray, 
                    self.checkerboard_size, 
                    cv2.CALIB_CB_ADAPTIVE_THRESH + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE + 
                    cv2.CALIB_CB_FAST_CHECK
                )
                
                # Process corners if found
                if corner_found:
                    # Refine corner detection
                    refined_corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), self.criteria
                    )
                    
                    # Draw the corners on display image
                    cv2.drawChessboardCorners(
                        display_image, self.checkerboard_size, refined_corners, corner_found
                    )
                    
                    # Status text (green)
                    status_text = f"Checkerboard found! Press SPACE to capture."
                    cv2.putText(display_image, status_text, (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Process capture request (from UI thread)
                    if self.save_image and time.time() - self.last_save_time > 1.0:
                        # Save data for calibration
                        self.objpoints.append(self.objp)
                        self.imgpoints.append(refined_corners)
                        
                        # Ensure directory exists
                        if not os.path.exists("calibration_images"):
                            os.makedirs("calibration_images")
                        
                        # Save the image for reference
                        img_file = f"calibration_images/calib_{self.image_count:02d}.png"
                        cv2.imwrite(img_file, color_image)
                        
                        # Update counters
                        self.image_count += 1
                        self.last_save_time = time.time()
                        self.save_image = False
                        
                        # Update UI
                        self.root.after(0, self.update_progress)
                else:
                    # Status text (red)
                    status_text = "Move checkerboard in view!"
                    cv2.putText(display_image, status_text, (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Convert BGR to RGB
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage and update display
                img = Image.fromarray(display_image)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update UI
                self.root.after(0, lambda: self.update_image(img_tk))
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            # Update status
            error_message = str(e)
            self.root.after(0, lambda msg=error_message: self.status_label.config(text=f"Error: {msg}"))
        finally:
            if self.running:
                self.root.after(0, self.stop_camera)
    
    def update_image(self, img_tk):
        self.camera_view.imgtk = img_tk
        self.camera_view.config(image=img_tk)
    
    def update_progress(self):
        # Update progress bar
        progress_pct = (self.image_count / self.required_images) * 100
        self.progress_var.set(progress_pct)
        
        # Update label
        self.progress_label.config(text=f"Images: {self.image_count}/{self.required_images}")
        
        # Update status
        self.status_label.config(text=f"Image {self.image_count}/{self.required_images} captured!")
        
        # Enable calibrate button if enough images
        if self.image_count >= self.required_images:
            self.calibrate_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Enough images collected! Press 'Calibrate' to calculate parameters.")
    
    def capture_image(self):
        if self.running:
            self.save_image = True
    
    def calibrate(self):
        try:
            # Stop camera if running
            if self.running:
                self.stop_camera()
            
            # Check if we have enough images
            if len(self.objpoints) < self.required_images:
                messagebox.showwarning("Warning", f"Not enough images. Collected: {len(self.objpoints)}, Required: {self.required_images}")
                return
            
            # Get image size from the first captured image points
            if self.imgpoints and len(self.imgpoints) > 0:
                # Get dimensions from the first image points array
                h = max(self.imgpoints[0][:, 0, 1]) + 100  # Add padding
                w = max(self.imgpoints[0][:, 0, 0]) + 100  # Add padding
                img_size = (int(w), int(h))
            else:
                # Fallback to a default size if no images are available
                messagebox.showerror("Error", "No valid calibration images found")
                return
            
            # Update status
            self.status_label.config(text="Calculating calibration parameters...")
            
            # Progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Calibration in Progress")
            progress_window.geometry("300x100")
            progress_window.resizable(False, False)
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Calculating calibration parameters...\nThis may take a moment.")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            progress_bar.start()
            
            # Process in separate thread
            def calibration_thread():
                try:
                    # Perform camera calibration
                    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                        self.objpoints, self.imgpoints, img_size, None, None)
                    
                    # Calculate reprojection error
                    mean_error = 0
                    for i in range(len(self.objpoints)):
                        imgpoints2, _ = cv2.projectPoints(
                            self.objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                        error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                        mean_error += error
                    
                    reprojection_error = mean_error/len(self.objpoints)
                    
                    # Save calibration parameters
                    filename = utils.save_calibration(
                        camera_matrix, dist_coeffs, img_size, reprojection_error, 
                        self.device_product_line, self.device_serial)
                    
                    # Update UI
                    self.root.after(0, lambda: self.calibration_complete(filename, reprojection_error))
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Calibration failed: {e}"))
                
                finally:
                    # Close progress dialog
                    self.root.after(0, progress_window.destroy)
            
            # Start calibration thread
            cal_thread = threading.Thread(target=calibration_thread)
            cal_thread.daemon = True
            cal_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Calibration setup failed: {e}")
    
    def calibration_complete(self, filename, error):
        # Show results
        messagebox.showinfo(
            "Calibration Complete", 
            f"Calibration parameters saved to:\n{filename}\n\nReprojection error: {error:.6f} pixels"
        )
        
        # Update status
        self.status_label.config(text=f"Calibration complete! Saved to {filename}")
        
        # Ask if user wants to start measurement tool
        if messagebox.askyesno("Calibration Complete", "Start the Measurement Tool now?"):
            self.root.after(100, self.start_measurement_tool)
    
    def start_measurement_tool(self):
        try:
            import measurement_tool
            self.root.destroy()
            root = tk.Tk()
            app = measurement_tool.MeasurementApp(root)
            root.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start Measurement Tool: {e}")
    
    def on_close(self):
        # Stop camera if running
        if self.running:
            self.stop_camera()
        
        # Close window
        self.root.destroy()

if __name__ == "__main__":
    # Create and run application
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()