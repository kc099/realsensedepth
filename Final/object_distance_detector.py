import cv2
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

class RealsenseDistanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intel RealSense Distance Detector")
        self.root.geometry("1280x720")
        
        # Initialize variables
        self.pipeline = None
        self.align = None
        self.running = False
        self.thread = None
        self.depth_scale = 0.001  # Default depth scale, will be updated
        
        # MobileNet SSD variables
        self.net = None
        self.detection_active = tk.BooleanVar(value=False)
        self.conf_threshold = 0.5
        
        # COCO class names for MobileNet SSD
        self.classes = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", 
                       "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", 
                       "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", 
                       "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", 
                       "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", 
                       "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
                       "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", 
                       "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
                       "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
                       "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", 
                       "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
                       "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", 
                       "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        
        # Create UI
        self.create_widgets()
        
        # Load MobileNet SSD model
        self.load_model()
    
    def create_widgets(self):
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)
        
        # Start/Stop button
        self.start_button_var = tk.StringVar(value="Start Camera")
        self.start_button = ttk.Button(control_frame, textvariable=self.start_button_var, command=self.toggle_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Detection toggle
        detection_check = ttk.Checkbutton(control_frame, text="Enable Object Detection", variable=self.detection_active)
        detection_check.pack(side=tk.LEFT, padx=20)
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
        confidence_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, length=200, 
                                    orient=tk.HORIZONTAL, value=0.5,
                                    command=self.update_confidence)
        confidence_scale.pack(side=tk.LEFT)
        
        self.confidence_label = ttk.Label(control_frame, text="0.50")
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        # Image displays frame
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Color image frame
        color_frame = ttk.LabelFrame(display_frame, text="RGB Image")
        color_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.color_canvas = tk.Canvas(color_frame, width=640, height=480, bg="black")
        self.color_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Depth image frame
        depth_frame = ttk.LabelFrame(display_frame, text="Depth Image")
        depth_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.depth_canvas = tk.Canvas(depth_frame, width=640, height=480, bg="black")
        self.depth_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for resizing
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Bottom frame for information
        info_frame = ttk.Frame(self.root, padding=10)
        info_frame.pack(fill=tk.X)
        
        # Distance information
        self.distance_var = tk.StringVar(value="Distance: -- m")
        distance_label = ttk.Label(info_frame, textvariable=self.distance_var, font=("Arial", 14, "bold"))
        distance_label.pack(side=tk.LEFT, padx=20)
        
        # Status information
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(info_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def load_model(self):
        """Load MobileNet SSD model"""
        try:
            # Load the pre-trained model
            self.status_var.set("Loading MobileNet SSD model...")
            
            # Path to model files - update these paths to your model location
            model_weights = "MobileNetSSD_deploy.caffemodel"
            model_config = "MobileNetSSD_deploy.prototxt"
            
            # Load the model using OpenCV's DNN module
            self.net = cv2.dnn.readNetFromCaffe(model_config, model_weights)
            
            self.status_var.set("MobileNet SSD model loaded")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            self.detection_active.set(False)
            
            # Create model download instructions
            message = (
                "Error loading the MobileNet SSD model. Please download the model files:\n\n"
                "1. Create a 'models' folder in the same directory as this script\n"
                "2. Download these files:\n"
                "   - MobileNetSSD_deploy.caffemodel\n"
                "   - MobileNetSSD_deploy.prototxt\n\n"
                "You can download them from OpenCV's GitHub or use a pre-trained model."
            )
            tk.messagebox.showerror("Model Not Found", message)
    
    def update_confidence(self, value):
        """Update confidence threshold from slider"""
        self.conf_threshold = float(value)
        self.confidence_label.config(text=f"{self.conf_threshold:.2f}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the Intel RealSense camera"""
        try:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable streams
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            profile = self.pipeline.start(config)
            
            # Get depth sensor and its depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Try to set high accuracy preset
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy preset
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            # Update UI
            self.running = True
            self.start_button_var.set("Stop Camera")
            self.status_var.set("Camera running")
            
            # Start processing thread
            self.thread = threading.Thread(target=self.process_frames)
            self.thread.daemon = True
            self.thread.start()
            
        except Exception as e:
            self.status_var.set(f"Error starting camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera stream"""
        if self.running:
            self.running = False
            
            # Wait for thread to finish
            if self.thread:
                self.thread.join(timeout=1.0)
            
            # Stop the pipeline
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
                self.align = None
            
            # Update UI
            self.start_button_var.set("Start Camera")
            self.status_var.set("Camera stopped")
            
            # Reset distance display
            self.distance_var.set("Distance: -- m")
    
    def process_frames(self):
        """Main loop for processing camera frames"""
        try:
            while self.running:
                # Wait for a coherent pair of frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Apply colormap to depth image
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Process with MobileNet SSD if enabled
                if self.detection_active.get() and self.net is not None:
                    # Create a copy of color image for drawing
                    display_image = color_image.copy()
                    
                    # Get image dimensions
                    h, w = color_image.shape[:2]
                    
                    # Create a blob from the image
                    blob = cv2.dnn.blobFromImage(
                        cv2.resize(color_image, (300, 300)), 
                        0.007843, (300, 300), 127.5)
                    
                    # Set the input to the network
                    self.net.setInput(blob)
                    
                    # Run forward pass to get detections
                    detections = self.net.forward()
                    
                    # Process detections
                    closest_distance = float('inf')
                    has_detection = False
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        
                        # Filter by confidence threshold
                        if confidence > self.conf_threshold:
                            # Get the class ID
                            class_id = int(detections[0, 0, i, 1])
                            
                            # Get the class name
                            if class_id < len(self.classes):
                                class_name = self.classes[class_id]
                            else:
                                class_name = "unknown"
                            
                            # Get bounding box coordinates
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype(int)
                            
                            # Draw the bounding box
                            cv2.rectangle(display_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            
                            # Calculate center point of the bounding box
                            center_x = (startX + endX) // 2
                            center_y = (startY + endY) // 2
                            
                            # Get the distance at the center point
                            center_dist = depth_frame.get_distance(center_x, center_y)
                            
                            # If we got a valid distance and it's closer than previous detections
                            if center_dist > 0 and center_dist < closest_distance:
                                closest_distance = center_dist
                                has_detection = True
                            
                            # Put text above the bounding box
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(display_image, label, (startX, y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Draw center point
                            cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                            
                            # Add distance text
                            distance_text = f"{center_dist:.2f}m"
                            cv2.putText(display_image, distance_text, (center_x + 10, center_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Update distance in UI
                    if has_detection:
                        self.root.after(0, lambda: self.distance_var.set(f"Distance: {closest_distance:.2f} m"))
                    else:
                        self.root.after(0, lambda: self.distance_var.set("Distance: -- m"))
                else:
                    display_image = color_image
                
                # Convert images to the format needed for display
                color_img = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
                depth_img = Image.fromarray(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
                
                # Resize images to fit canvas if needed
                color_img = self.resize_image_to_canvas(color_img, self.color_canvas)
                depth_img = self.resize_image_to_canvas(depth_img, self.depth_canvas)
                
                # Convert to PhotoImage
                color_img_tk = ImageTk.PhotoImage(image=color_img)
                depth_img_tk = ImageTk.PhotoImage(image=depth_img)
                
                # Update UI with new images
                self.root.after(0, lambda: self.update_image(self.color_canvas, color_img_tk))
                self.root.after(0, lambda: self.update_image(self.depth_canvas, depth_img_tk))
                
                # Short delay to reduce CPU usage
                time.sleep(0.03)
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            print(f"Error in process_frames: {str(e)}")
    
    def update_image(self, canvas, img_tk):
        """Update canvas with new image"""
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk  # Keep a reference
    
    def resize_image_to_canvas(self, img, canvas):
        """Resize image to fit the canvas while maintaining aspect ratio"""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas size is not yet determined, return original image
        if canvas_width <= 1 or canvas_height <= 1:
            return img
            
        img_width, img_height = img.size
        
        # Calculate scale factors
        width_factor = canvas_width / img_width
        height_factor = canvas_height / img_height
        
        # Use the smaller factor to ensure the entire image fits
        scale_factor = min(width_factor, height_factor)
        
        # Calculate new dimensions
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        
        # Resize the image
        return img.resize((new_width, new_height), Image.LANCZOS)

if __name__ == "__main__":
    # Create main window
    root = tk.Tk()
    app = RealsenseDistanceApp(root)
    root.mainloop()