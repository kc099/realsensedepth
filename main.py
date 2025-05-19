import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intel RealSense Toolkit")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        
        # Create main frame
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Intel RealSense D435 Toolkit", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Buttons
        calibration_button = ttk.Button(main_frame, text="Camera Calibration Tool", 
                                       command=self.launch_calibration_tool, width=30)
        calibration_button.pack(pady=10)
        
        measurement_button = ttk.Button(main_frame, text="Distance & Height Measurement Tool", 
                                       command=self.launch_measurement_tool, width=30)
        measurement_button.pack(pady=10)
        
        # Exit button
        exit_button = ttk.Button(main_frame, text="Exit", command=root.destroy, width=30)
        exit_button.pack(pady=20)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Arial", 9))
        status_label.pack(pady=10)
        
        # Check for calibration files
        self.check_calibration_files()
    
    def check_calibration_files(self):
        """Check if calibration files exist"""
        calib_files = [f for f in os.listdir('.') if f.startswith('realsense_calib_') and f.endswith('.json')]
        if calib_files:
            self.status_var.set(f"Found {len(calib_files)} calibration file(s)")
        else:
            self.status_var.set("No calibration files found. Please run calibration first.")
    
    def launch_calibration_tool(self):
        """Launch the calibration tool"""
        try:
            self.root.destroy()
            import calibration_tool
            root = tk.Tk()
            app = calibration_tool.CalibrationApp(root)
            root.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Calibration Tool: {e}")
            sys.exit(1)
    
    def launch_measurement_tool(self):
        """Launch the measurement tool"""
        try:
            self.root.destroy()
            import measurement_tool
            root = tk.Tk()
            app = measurement_tool.MeasurementApp(root)
            root.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Measurement Tool: {e}")
            sys.exit(1)

if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()