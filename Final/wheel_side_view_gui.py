import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import json
import numpy as np
from PIL import Image, ImageTk
import sys

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the image processing functions from the Final module
from image_processing import process_frame

class WheelSideViewGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wheel Height Analyzer")
        self.root.geometry("1000x700")
        
        # Variables to store file paths
        self.image_path = tk.StringVar()
        self.depth_path = tk.StringVar()
        self.calib_path = tk.StringVar()
        
        # Set default calibration path if available
        calib_files = [f for f in os.listdir('.') if f.startswith('realsense_calib_') and f.endswith('.json')]
        if calib_files:
            default_calib = os.path.abspath(sorted(calib_files)[-1])  # Get the latest file
            self.calib_path.set(default_calib)
        
        # Store the measurements
        self.current_measurements = {}
        
        # Create GUI components
        self.create_gui_components()
    
    def create_gui_components(self):
        """Create all GUI elements"""
        # Main container frames
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Image display frames
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original image frame
        self.original_frame = tk.LabelFrame(display_frame, text="Original Image", width=450, height=350)
        self.original_frame.grid(row=0, column=0, padx=10, pady=10)
        self.original_frame.grid_propagate(False)
        
        # Processed image frame
        self.processed_frame = tk.LabelFrame(display_frame, text="Processed Image", width=450, height=350)
        self.processed_frame.grid(row=0, column=1, padx=10, pady=10)
        self.processed_frame.grid_propagate(False)
        
        # Image labels (where images will be displayed)
        self.original_image_label = tk.Label(self.original_frame)
        self.original_image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_image_label = tk.Label(self.processed_frame)
        self.processed_image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Measurements display frame
        measurements_frame = tk.LabelFrame(self.root, text="Wheel Measurements")
        measurements_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Height measurement display
        self.height_var = tk.StringVar(value="Height: Not measured")
        self.height_label = tk.Label(measurements_frame, textvariable=self.height_var, 
                                    font=("Arial", 14, "bold"), pady=10)
        self.height_label.pack()
        
        # Status indicator (Pass/Fail)
        self.status_var = tk.StringVar(value="Status: Not evaluated")
        self.status_label = tk.Label(measurements_frame, textvariable=self.status_var, 
                                     font=("Arial", 16, "bold"), pady=10)
        self.status_label.pack()
        
        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Image selection
        tk.Label(file_frame, text="RGB Side Image:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        tk.Entry(file_frame, textvariable=self.image_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_image).grid(row=0, column=2, padx=5, pady=5)
        
        # Depth image selection
        tk.Label(file_frame, text="Depth Image:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        tk.Entry(file_frame, textvariable=self.depth_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_depth).grid(row=1, column=2, padx=5, pady=5)
        
        # Calibration file selection
        tk.Label(file_frame, text="Calibration File:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        tk.Entry(file_frame, textvariable=self.calib_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_calib).grid(row=2, column=2, padx=5, pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Process button
        tk.Button(button_frame, text="Process Image", command=self.process_image, 
                 bg="#4CAF50", fg="white", height=2, width=20, font=("Arial", 12, "bold")).pack(pady=10)
    
    def browse_image(self):
        """Browse for side image file"""
        path = filedialog.askopenfilename(title="Select RGB Side Image", 
                                         filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path.set(os.path.abspath(path))
            self.display_original_image()
    
    def browse_depth(self):
        """Browse for depth image file"""
        path = filedialog.askopenfilename(title="Select Depth Image", 
                                         filetypes=[("Depth image files", "*.png *.tiff *.exr *.pgm")])
        if path:
            self.depth_path.set(os.path.abspath(path))
            # Display depth preview
            self.display_depth_preview()
    
    def browse_calib(self):
        """Browse for calibration file"""
        path = filedialog.askopenfilename(title="Select Calibration File", 
                                         filetypes=[("JSON files", "*.json")])
        if path:
            self.calib_path.set(os.path.abspath(path))
    
    def display_original_image(self):
        """Display the original image in the GUI"""
        path = self.image_path.get()
        if path and os.path.exists(path):
            try:
                # Load image
                img = cv2.imread(path)
                if img is None:
                    messagebox.showerror("Error", "Could not load image")
                    return
                
                # Convert to RGB for display
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                h, w = img_rgb.shape[:2]
                max_size = 440  # Max dimension for preview
                scale = min(max_size/w, max_size/h)
                new_w, new_h = int(w*scale), int(h*scale)
                img_resized = cv2.resize(img_rgb, (new_w, new_h))
                
                # Convert to PhotoImage
                img_pil = Image.fromarray(img_resized)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Update label
                self.original_image_label.configure(image=img_tk)
                self.original_image_label.image = img_tk  # Keep reference
            except Exception as e:
                messagebox.showerror("Error", f"Error displaying image: {e}")
    
    def display_depth_preview(self):
        """Display the depth image preview"""
        path = self.depth_path.get()
        if path and os.path.exists(path):
            try:
                # Load depth image
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    messagebox.showerror("Error", "Could not load depth image")
                    return
                
                # Ensure single channel for depth visualization
                if len(img.shape) > 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Normalize for display
                normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                img_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                h, w = img_rgb.shape[:2]
                max_size = 440  # Max dimension for preview
                scale = min(max_size/w, max_size/h)
                new_w, new_h = int(w*scale), int(h*scale)
                img_resized = cv2.resize(img_rgb, (new_w, new_h))
                
                # Convert to PhotoImage
                img_pil = Image.fromarray(img_resized)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Display in processed image label temporarily
                self.processed_image_label.configure(image=img_tk)
                self.processed_image_label.image = img_tk  # Keep reference
                
                # Show depth statistics
                non_zero = img[img > 0]
                if len(non_zero) > 0:
                    min_val = np.min(non_zero)
                    max_val = np.max(non_zero)
                    mean_val = np.mean(non_zero)
                    self.height_var.set(f"Depth range: {min_val} - {max_val}, Mean: {mean_val:.1f}")
            except Exception as e:
                messagebox.showerror("Error", f"Error displaying depth image: {e}")
    
    def load_calibration(self):
        """Load calibration from file"""
        calib_path = self.calib_path.get()
        if not calib_path or not os.path.exists(calib_path):
            messagebox.showwarning("Warning", "No calibration file selected. Measurements may be inaccurate.")
            return None
        
        try:
            with open(calib_path, 'r') as f:
                calibration_data = json.load(f)
            
            camera_settings = {
                "camera_matrix": np.array(calibration_data.get("camera_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])),
                "dist_coeffs": np.array(calibration_data.get("dist_coeffs", [0, 0, 0, 0, 0])),
                "depth_scale": calibration_data.get("depth_scale", 0.001),
                "camera_height": calibration_data.get("camera_height", 500)  # Camera height in mm
            }
            
            return camera_settings
        except Exception as e:
            messagebox.showerror("Error", f"Error loading calibration file: {e}")
            return None
    
    def process_image(self):
        """Process the selected image and display results"""
        # Check if image is selected
        image_path = self.image_path.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        
        # Load calibration
        camera_settings = self.load_calibration()
        
        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", "Could not load image")
                return
            
            # Check for depth image
            depth_image = None
            depth_path = self.depth_path.get()
            if depth_path and os.path.exists(depth_path):
                # Load depth image
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Load depth as-is
                if depth_image is None:
                    messagebox.showwarning("Warning", "Could not load depth image. Processing without depth.")
                elif len(depth_image.shape) > 2:
                    # Convert multi-channel depth to single channel if needed
                    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
                
                print(f"Loaded depth image: {depth_path}")
                print(f"Depth image shape: {depth_image.shape if depth_image is not None else 'None'}")
                
                # Show depth statistics if available
                if depth_image is not None:
                    non_zero = depth_image[depth_image > 0]
                    if len(non_zero) > 0:
                        print(f"Depth image stats - Min: {np.min(non_zero)}, Max: {np.max(non_zero)}, Mean: {np.mean(non_zero):.1f}")
            
            # Process the image with specialized function to handle depth input
            if depth_image is not None:
                # First process frame to get mask and bounding box
                result = process_frame(frame, is_top_view=False, camera_settings=camera_settings)
                
                # If mask was found, use it with depth image
                from image_processing import measure_wheel_height_from_depth
                
                # Extract mask from process_frame result if available
                if isinstance(result, tuple) and len(result) == 2:
                    processed_img, measurements = result
                    
                    # Check if we have a mask in the measurements
                    if "mask" in measurements:
                        mask = measurements["mask"]
                        box = measurements.get("box", None)
                        
                        # Get more accurate measurements with depth
                        processed_img, measurements = measure_wheel_height_from_depth(
                            depth_image, frame, mask, box, camera_settings
                        )
                    else:
                        # Fall back to regular processing
                        processed_img, measurements = result
                else:
                    # Fall back to regular processing
                    processed_img, measurements = result
            else:
                # Process without depth
                processed_img, measurements = process_frame(frame, is_top_view=False, camera_settings=camera_settings)
            
            # Store measurements
            self.current_measurements = measurements
            
            # Display processed image
            if processed_img is not None:
                # Convert to RGB for display
                processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                h, w = processed_rgb.shape[:2]
                max_size = 440  # Max dimension for preview
                scale = min(max_size/w, max_size/h)
                new_w, new_h = int(w*scale), int(h*scale)
                processed_resized = cv2.resize(processed_rgb, (new_w, new_h))
                
                # Convert to PhotoImage
                img_pil = Image.fromarray(processed_resized)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Update label
                self.processed_image_label.configure(image=img_tk)
                self.processed_image_label.image = img_tk  # Keep reference
            
            # Update measurements display
            if "height_mm" in measurements:
                height_mm = measurements["height_mm"]
                self.height_var.set(f"Height: {height_mm:.1f} mm")
                
                # Update status if available
                if "is_ok" in measurements:
                    is_ok = measurements["is_ok"]
                    status_text = "PASS" if is_ok else "FAIL"
                    status_color = "#4CAF50" if is_ok else "#F44336"  # Green for pass, red for fail
                    self.status_var.set(f"Status: {status_text}")
                    self.status_label.config(fg=status_color)
                else:
                    self.status_var.set("Status: Not evaluated")
                    self.status_label.config(fg="black")
            else:
                self.height_var.set("Height: Not measured")
                self.status_var.set("Status: Not evaluated")
                self.status_label.config(fg="black")
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing image: {e}")

def main():
    root = tk.Tk()
    app = WheelSideViewGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
