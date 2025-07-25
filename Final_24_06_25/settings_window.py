import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import json
from datetime import datetime
import sqlite3
import os

# Import centralized debug system
from debug_utils import debug_print

# Import app icon module
try:
    from app_icon import set_app_icon
except ImportError:
    def set_app_icon(window):
        pass  # Fallback if module not available

def init_settings_db():
    """Initialize database for operator and model change tracking"""
    conn = sqlite3.connect('wheel_inspection.db')
    c = conn.cursor()
    
    # Create operators table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS operators
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create model_changes table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS model_changes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  model_name TEXT,
                  operator_id INTEGER,
                  changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(operator_id) REFERENCES operators(id))''')
    
    conn.commit()
    conn.close()

def show_settings_window(root, config_manager, update_model_parameters):
    """
    Show the settings window with enhanced bidirectional calibration synchronization.
    
    Features:
    - Loads latest intrinsics from camera_intrinsics.json on window open
    - RealSense intrinsics acquired from device at app initialization
    - Bidirectional sync: changes in settings.json ↔ camera_intrinsics.json
    - Top camera intrinsics remain constant throughout app
    - Real-time refresh capability from intrinsics file
    - Password-protected calibration editing
    """
    init_settings_db()  # Ensure database tables exist
    
    # Get current settings (already loaded by ConfigManager)
    current_settings = config_manager.settings
    WHEEL_MODELS = config_manager.wheel_models
    
    # Declare entry dictionaries early for function scope access
    top_intrinsics_entries = {}
    realsense_intrinsics_entries = {}
    calib_entries = {}
    
    # Function to refresh all calibration values from unified intrinsics (define early)
    def refresh_calibration_values():
        """Refresh all calibration values from camera_intrinsics.json"""
        try:
            # Force reload from intrinsics file
            config_manager.load_intrinsics_to_settings()
            
            # Update Top Camera intrinsics (editable)
            updated_top_intrinsics = config_manager.get_camera_intrinsics_for_display("top_camera")
            for key, entry in top_intrinsics_entries.items():
                if key in updated_top_intrinsics:
                    current_state = entry.cget('state')
                    entry.config(state='normal')
                    entry.delete(0, tk.END)
                    entry.insert(0, str(updated_top_intrinsics[key]))
                    entry.config(state=current_state)
            
            # Update RealSense intrinsics (editable)
            updated_realsense_intrinsics = config_manager.get_camera_intrinsics_for_display("realsense")
            for key, entry in realsense_intrinsics_entries.items():
                if key in updated_realsense_intrinsics:
                    current_state = entry.cget('state')
                    entry.config(state='normal')
                    entry.delete(0, tk.END)
                    entry.insert(0, str(updated_realsense_intrinsics[key]))
                    entry.config(state=current_state)
            
            # Update measurement calibration entries
            latest_calibration = config_manager.get_all_intrinsics_for_display()
            for key, entry in calib_entries.items():
                if key in latest_calibration:
                    current_state = entry.cget('state')
                    entry.config(state='normal')
                    entry.delete(0, tk.END)
                    entry.insert(0, str(latest_calibration[key]))
                    entry.config(state=current_state)
                
            messagebox.showinfo("Success", "All calibration values refreshed from camera_intrinsics.json")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh calibration values: {e}")

    # Debug: Print calibration values at the start
    # print(f"Settings window opened with calibration: {current_settings.get('calibration', 'No calibration found')}")
    
    settings_window = tk.Toplevel(root)
    settings_window.title("System Settings")
    settings_window.geometry("1100x800")
    settings_window.configure(background="#FFFFFF")
    
    # Set custom icon for settings window
    set_app_icon(settings_window)
    
    # Keep window on top initially
    settings_window.lift()
    settings_window.attributes('-topmost', True)
    settings_window.attributes('-topmost', False)  # Remove topmost but keep it focused
    
    # Notebook for tabs
    notebook = ttk.Notebook(settings_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # User Settings Tab
    user_frame = ttk.Frame(notebook)
    notebook.add(user_frame, text="User Settings")
    
    # Operator Section
    operator_frame = ttk.LabelFrame(user_frame, text="Operator Management")
    operator_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Operator dropdown
    ttk.Label(operator_frame, text="Current Operator:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    operator_combo = ttk.Combobox(operator_frame, state="readonly")
    operator_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    
    # Buttons for operator management
    operator_button_frame = ttk.Frame(operator_frame)
    operator_button_frame.grid(row=0, column=2, padx=10)
    
    def refresh_operators():
        conn = sqlite3.connect('wheel_inspection.db')
        c = conn.cursor()
        c.execute("SELECT name FROM operators ORDER BY name")
        operators = [row[0] for row in c.fetchall()]
        conn.close()
        operator_combo['values'] = operators
        if operators:
            operator_combo.set(operators[0])
    
    def add_operator():
        name = simpledialog.askstring("Add Operator", "Enter operator name:")
        if name:
            try:
                conn = sqlite3.connect('wheel_inspection.db')
                c = conn.cursor()
                c.execute("INSERT INTO operators (name) VALUES (?)", (name,))
                conn.commit()
                conn.close()
                refresh_operators()
                messagebox.showinfo("Success", "Operator added successfully")
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Operator already exists")
    
    def delete_operator():
        operator = operator_combo.get()
        if operator and messagebox.askyesno("Confirm", f"Delete operator {operator}?"):
            conn = sqlite3.connect('wheel_inspection.db')
            c = conn.cursor()
            c.execute("DELETE FROM operators WHERE name=?", (operator,))
            conn.commit()
            conn.close()
            refresh_operators()
    
    ttk.Button(operator_button_frame, text="Add", command=add_operator).pack(side=tk.LEFT, padx=2)
    ttk.Button(operator_button_frame, text="Delete", command=delete_operator).pack(side=tk.LEFT, padx=2)
    refresh_operators()
    
    # Model Change History
    history_frame = ttk.LabelFrame(user_frame, text="Model Change History")
    history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Treeview for history
    history_tree = ttk.Treeview(history_frame, columns=("model", "operator", "timestamp"), show="headings")
    history_tree.heading("model", text="Model")
    history_tree.heading("operator", text="Operator")
    history_tree.heading("timestamp", text="Date/Time")
    history_tree.column("model", width=150)
    history_tree.column("operator", width=150)
    history_tree.column("timestamp", width=200)
    history_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Scrollbar for history
    scrollbar = ttk.Scrollbar(history_tree)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    history_tree.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=history_tree.yview)
    
    def refresh_history():
        for item in history_tree.get_children():
            history_tree.delete(item)
        
        conn = sqlite3.connect('wheel_inspection.db')
        c = conn.cursor()
        c.execute('''SELECT m.model_name, o.name, m.changed_at 
                      FROM model_changes m
                      JOIN operators o ON m.operator_id = o.id
                      ORDER BY m.changed_at DESC''')
        for row in c.fetchall():
            history_tree.insert("", "end", values=row)
        conn.close()
    
    refresh_history()
    
    # Camera Settings
    camera_frame = ttk.LabelFrame(user_frame, text="Camera Settings")
    camera_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Configure grid weights for better layout
    camera_frame.grid_columnconfigure(1, weight=1)
    
    # Add COM Port Selection
    ttk.Label(camera_frame, text="COM Port:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    com_port_combo = ttk.Combobox(camera_frame, state="readonly", width=20)
    com_port_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    
    # Add Baud Rate Selection
    ttk.Label(camera_frame, text="Baud Rate:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    baud_rate_combo = ttk.Combobox(camera_frame, state="readonly", width=10)
    baud_rate_combo['values'] = ['9600', '19200', '38400', '57600', '115200']
    baud_rate_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
    # Set default or current value
    baud_rate_combo.set(str(current_settings.get("baud_rate", "19200")))
    
    def refresh_com_ports():
        try:
            import serial.tools.list_ports
            ports = [port.device for port in serial.tools.list_ports.comports()]
            com_port_combo['values'] = ports
            # Set current value if it exists in settings
            if "com_port" in current_settings and current_settings["com_port"] in ports:
                com_port_combo.set(current_settings["com_port"])
            elif ports:
                com_port_combo.set(ports[0])
        except ImportError:
            com_port_combo['values'] = ["Serial port not available"]
            com_port_combo.set("Serial port not available")
    
    # Add refresh button for COM ports
    refresh_com_button = ttk.Button(camera_frame, text="Refresh", command=refresh_com_ports)
    refresh_com_button.grid(row=0, column=4, padx=5, pady=5)
    
    # Initial COM port refresh
    refresh_com_ports()
    
    # Add Modbus Slave ID setting
    ttk.Label(camera_frame, text="Modbus Slave ID:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    slave_id_var = tk.StringVar(value=str(current_settings.get("modbus_slave_id", "1")))
    slave_id_entry = ttk.Entry(camera_frame, textvariable=slave_id_var, width=10)
    slave_id_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
    
    # Add validation for slave ID entry
    def validate_slave_id(*args):
        try:
            value = slave_id_var.get()
            if value:  # Only validate if there's a value
                num = int(value)
                if not (1 <= num <= 247):
                    slave_id_var.set("1")  # Reset to default if invalid
        except ValueError:
            slave_id_var.set("1")  # Reset to default if not a number
    
    # Add trace to validate slave ID as user types
    slave_id_var.trace_add("write", validate_slave_id)
    
    ttk.Label(camera_frame, text="Top Camera URL:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    top_url_entry = ttk.Entry(camera_frame, width=50)
    top_url_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
    top_url_entry.delete(0, tk.END)
    top_url_entry.insert(0, current_settings["top_camera_url"])
    top_url_entry.config(state='readonly')  # Make readonly after inserting
    
    ttk.Label(camera_frame, text="Side Camera URL:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
    side_url_entry = ttk.Entry(camera_frame, width=50)
    side_url_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
    side_url_entry.delete(0, tk.END)
    side_url_entry.insert(0, current_settings["side_camera_url"])
    side_url_entry.config(state='readonly')  # Make readonly after inserting
    
    # Edit Camera URLs button - place it to the right of the URLs
    def edit_camera_urls():
        password = simpledialog.askstring("Authentication", "Enter password to edit camera URLs:", show='*')
        if password == "admin123":
            top_url_entry.config(state='normal')
            side_url_entry.config(state='normal')
            edit_camera_button.config(text="Save URLs", command=save_camera_urls)
        else:
            messagebox.showerror("Error", "Incorrect password")
    
    def save_camera_urls():
        current_settings["top_camera_url"] = top_url_entry.get()
        current_settings["side_camera_url"] = side_url_entry.get()
        
        # Save to file
        with open("settings.json", "w") as f:
            json.dump(current_settings, f, indent=4)
        
        # Make readonly again
        top_url_entry.config(state='readonly')
        side_url_entry.config(state='readonly')
        edit_camera_button.config(text="Edit URLs", command=edit_camera_urls)
        
        messagebox.showinfo("Success", "Camera URLs saved successfully")
    
    edit_camera_button = ttk.Button(camera_frame, text="Edit URLs", command=edit_camera_urls)
    edit_camera_button.grid(row=2, column=4, rowspan=2, padx=5, pady=5, sticky=tk.NS)
    
    ttk.Label(camera_frame, text="Capture Interval (s):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
    interval_entry = ttk.Entry(camera_frame, width=10)
    interval_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
    # Safely retrieve capture_interval with a default of 5 seconds
    capture_interval = current_settings.get("capture_interval", 5)
    interval_entry.insert(0, str(capture_interval))
    
    # Model Selection with password protection
    model_frame = ttk.LabelFrame(user_frame, text="Wheel Model Selection")
    model_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Label(model_frame, text="Current Model:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    current_model_label = ttk.Label(model_frame, text=current_settings["selected_model"])
    current_model_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    
    # Update to show model-specific tolerance
    ttk.Label(model_frame, text="Model Tolerance (mm):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    tolerance_label = ttk.Label(model_frame, text="N/A")
    tolerance_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
    
    # Set initial tolerance if model exists in WHEEL_MODELS
    if current_settings["selected_model"] in WHEEL_MODELS:
        model_data = WHEEL_MODELS[current_settings["selected_model"]]
        if isinstance(model_data, dict) and "tolerance" in model_data:
            tolerance_label.config(text=str(model_data["tolerance"]))
    
    def change_model():
        if not operator_combo.get():
            messagebox.showerror("Error", "Please select an operator first")
            return
            
        password = simpledialog.askstring("Authentication", "Enter password to change model:", show='*')
        if password == "admin123":  # Replace with secure password check
            model_dialog = tk.Toplevel()
            model_dialog.title("Select Model")
            model_dialog.geometry("400x300")
            
            # Keep dialog on top
            model_dialog.transient(settings_window)
            model_dialog.grab_set()
            
            # Use grid for all widgets in the model dialog
            ttk.Label(model_dialog, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            model_combo = ttk.Combobox(model_dialog, values=list(WHEEL_MODELS.keys()), state="readonly")
            model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            model_combo.set(current_settings["selected_model"])
            
            # Add tolerance entry for model-specific settings
            ttk.Label(model_dialog, text="Model Tolerance (mm):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            tolerance_entry = ttk.Entry(model_dialog)
            tolerance_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
            
            # Pre-fill tolerance if model has it
            def update_tolerance(*args):
                selected_model = model_combo.get()
                if selected_model in WHEEL_MODELS:
                    model_data = WHEEL_MODELS[selected_model]
                    if isinstance(model_data, dict) and "tolerance" in model_data:
                        tolerance_entry.delete(0, tk.END)
                        tolerance_entry.insert(0, str(model_data["tolerance"]))
            
            model_combo.bind('<<ComboboxSelected>>', update_tolerance)
            update_tolerance()  # Initial update

            # Add camera height parameter
            ttk.Label(model_dialog, text="Top Camera to Base Height (mm):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
            height_entry = ttk.Entry(model_dialog)
            height_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
            height_entry.insert(0, str(current_settings["calibration"]["base_height"]))

            def save_model_changes():
                new_model = model_combo.get()
                try:
                    new_tolerance = float(tolerance_entry.get())
                    new_height = float(height_entry.get())
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers")
                    return

                # Update current settings in local scope
                current_settings["selected_model"] = new_model
                current_settings["calibration"]["base_height"] = new_height
                
                # CRITICAL: Update config manager's internal state directly
                config_manager._settings["selected_model"] = new_model
                config_manager._settings["calibration"]["base_height"] = new_height
                
                # Update model tolerance if model exists
                if new_model in WHEEL_MODELS:
                    if isinstance(WHEEL_MODELS[new_model], dict):
                        WHEEL_MODELS[new_model]["tolerance"] = new_tolerance
                    else:
                        # Convert old format to new format
                        min_dia, max_dia, height = WHEEL_MODELS[new_model]
                        WHEEL_MODELS[new_model] = {
                            "min_dia": min_dia,
                            "max_dia": max_dia,
                            "height": height,
                            "tolerance": new_tolerance
                        }
                
                # Update config manager's wheel models
                config_manager._wheel_models = WHEEL_MODELS

                # Update UI in settings window
                current_model_label.config(text=new_model)
                tolerance_label.config(text=str(new_tolerance))
                
                # Force cache invalidation and save settings
                config_manager._last_load_time = 0  # Force reload on next access
                config_manager.save_settings()
                
                # Update model parameters in main window AFTER saving
                update_model_parameters()
                
                # Log the change in database
                operator = operator_combo.get()
                conn = sqlite3.connect('wheel_inspection.db')
                c = conn.cursor()
                c.execute("SELECT id FROM operators WHERE name=?", (operator,))
                operator_id = c.fetchone()[0]
                c.execute("INSERT INTO model_changes (model_name, operator_id) VALUES (?, ?)",
                         (new_model, operator_id))
                conn.commit()
                conn.close()
                
                refresh_history()
                model_dialog.destroy()
                messagebox.showinfo("Success", f"Model changed to {new_model} successfully")
                
                debug_print(f"Model changed to: {new_model} with tolerance: {new_tolerance} and height: {new_height}", "ui")
            
            ttk.Button(model_dialog, text="Save", command=save_model_changes).grid(row=3, column=0, columnspan=2, pady=10)
        else:
            messagebox.showerror("Error", "Incorrect password")
    
    ttk.Button(model_frame, text="Change Model", command=change_model).grid(row=0, column=2, padx=10)
    
    # Model Management Section - Enhanced to show all parameters
    model_management_frame = ttk.LabelFrame(user_frame, text="Model Management")
    model_management_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create Treeview to display model details
    model_tree = ttk.Treeview(model_management_frame, height=5, 
                             columns=("name", "min_dia", "max_dia", "height", "tolerance"), 
                             show="headings")
    model_tree.heading("name", text="Model Name")
    model_tree.heading("min_dia", text="Min Dia (in)")
    model_tree.heading("max_dia", text="Max Dia (in)")
    model_tree.heading("height", text="Height (mm)")
    model_tree.heading("tolerance", text="Tolerance (mm)")
    
    model_tree.column("name", width=100)
    model_tree.column("min_dia", width=80)
    model_tree.column("max_dia", width=80)
    model_tree.column("height", width=80)
    model_tree.column("tolerance", width=100)
    
    model_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Scrollbar for model tree
    model_scrollbar = ttk.Scrollbar(model_management_frame, orient=tk.VERTICAL, command=model_tree.yview)
    model_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    model_tree.configure(yscrollcommand=model_scrollbar.set)
    
    # Refresh model list
    def refresh_model_list():
        for item in model_tree.get_children():
            model_tree.delete(item)
        
        for model_name, model_data in WHEEL_MODELS.items():
            if isinstance(model_data, dict):
                # New format with all parameters
                values = (model_name, 
                         model_data.get("min_dia", 0),
                         model_data.get("max_dia", 0),
                         model_data.get("height", 0),
                         model_data.get("tolerance", 3.0))
            else:
                # Old format - convert to display
                min_dia, max_dia, height = model_data
                values = (model_name, min_dia, max_dia, height, 3.0)  # Default tolerance
            
            model_tree.insert("", "end", values=values)
    
    # Add model button
    def add_model():
        password = simpledialog.askstring("Authentication", "Enter password to add model:", show='*')
        if password == "admin123":
            # Create a dialog for all model information
            model_dialog = tk.Toplevel()
            model_dialog.title("Add New Model")
            model_dialog.geometry("450x300")
            
            # Keep dialog on top
            model_dialog.transient(settings_window)
            model_dialog.grab_set()
            
            # Model information fields
            fields = [
                ("Model Name:", "name"),
                ("Min Diameter (inches):", "min_dia"),
                ("Max Diameter (inches):", "max_dia"),
                ("Height (mm):", "height"),
                ("Tolerance (mm):", "tolerance")
            ]
            
            entries = {}
            for row, (label_text, field_name) in enumerate(fields):
                ttk.Label(model_dialog, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
                entry = ttk.Entry(model_dialog, width=30)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
                entries[field_name] = entry
                
                # Set default tolerance
                if field_name == "tolerance":
                    entry.insert(0, "3.0")
            
            def save_new_model():
                try:
                    name = entries["name"].get().strip()
                    if not name:
                        messagebox.showerror("Error", "Please enter a model name")
                        return
                    
                    min_dia = float(entries["min_dia"].get())
                    max_dia = float(entries["max_dia"].get())
                    height = float(entries["height"].get())
                    tolerance = float(entries["tolerance"].get())
                    
                    if min_dia >= max_dia:
                        messagebox.showerror("Error", "Min diameter must be less than max diameter")
                        return
                    
                    # Store model with all parameters
                    WHEEL_MODELS[name] = {
                        "min_dia": min_dia,
                        "max_dia": max_dia,
                        "height": height,
                        "tolerance": tolerance
                    }
                    
                    refresh_model_list()
                    config_manager.save_settings()
                    model_dialog.destroy()
                    messagebox.showinfo("Success", "Model added successfully")
                    
                    # Update model parameters in main window
                    update_model_parameters()
                    
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers for all fields")
            
            # Buttons
            button_frame = ttk.Frame(model_dialog)
            button_frame.grid(row=len(fields), column=0, columnspan=2, pady=10)
            
            ttk.Button(button_frame, text="Save", command=save_new_model).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=model_dialog.destroy).pack(side=tk.LEFT, padx=5)
            
        else:
            messagebox.showerror("Error", "Incorrect password")
    
    # Delete model button
    def delete_model():
        password = simpledialog.askstring("Authentication", "Enter password to delete model:", show='*')
        if password == "admin123":
            selection = model_tree.selection()
            if selection:
                model_name = model_tree.item(selection[0])['values'][0]
                if messagebox.askyesno("Confirm", f"Delete model {model_name}?"):
                    if model_name in WHEEL_MODELS:
                        del WHEEL_MODELS[model_name]
                        refresh_model_list()
                        config_manager.save_settings()
                        messagebox.showinfo("Success", "Model deleted successfully")
                        
                        # Update model parameters in main window
                        update_model_parameters()
            else:
                messagebox.showwarning("No Selection", "Please select a model to delete")
        else:
            messagebox.showerror("Error", "Incorrect password")
    
    # Button frame
    model_button_frame = ttk.Frame(model_management_frame)
    model_button_frame.pack(side=tk.RIGHT, padx=5)
    
    ttk.Button(model_button_frame, text="Add", command=add_model).pack(fill=tk.X, pady=2)
    ttk.Button(model_button_frame, text="Delete", command=delete_model).pack(fill=tk.X, pady=2)
    
    refresh_model_list()
    
    def save_user_settings():
        """Save only user-accessible settings."""
        try:
            # Save capture interval
            current_settings["capture_interval"] = float(interval_entry.get())
            
            # Save COM port and baud rate settings
            selected_port = com_port_combo.get()
            if selected_port and selected_port != "Serial port not available":
                debug_print(f"Saving COM port: {selected_port}", "ui")
                current_settings["com_port"] = selected_port
                current_settings["baud_rate"] = int(baud_rate_combo.get())
            else:
                debug_print("No valid COM port selected", "ui")
                messagebox.showwarning("Warning", "No COM port selected. Signal detection will not be available.")
            
            # Save Modbus slave ID with validation
            try:
                slave_id = int(slave_id_var.get())
                if 1 <= slave_id <= 247:  # Valid Modbus slave ID range
                    current_settings["modbus_slave_id"] = slave_id
                    debug_print(f"Saving Modbus slave ID: {slave_id}", "ui")
                else:
                    raise ValueError("Slave ID must be between 1 and 247")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid Modbus Slave ID: {str(e)}")
                return
            
            # Save to file using settings_manager
            config_manager.save_settings()
            
            debug_print(f"Settings saved with COM port: {current_settings.get('com_port')}", "ui")
            messagebox.showinfo("Settings Saved", "User settings have been saved successfully.")
            
            # Update model parameters in main window
            update_model_parameters()
            
            # Ensure settings window stays on top
            settings_window.lift()
            settings_window.focus_force()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid settings value: {str(e)}")
    
    save_user_button = ttk.Button(user_frame, text="Save User Settings", command=save_user_settings)
    save_user_button.pack(pady=10)
    
    # Calibration Settings Tab
    calib_frame = ttk.Frame(notebook)
    notebook.add(calib_frame, text="Calibration Settings")
    
    # Create canvas for scrolling
    calib_canvas = tk.Canvas(calib_frame)
    scrollbar = ttk.Scrollbar(calib_frame, orient="vertical", command=calib_canvas.yview)
    scrollable_frame = ttk.Frame(calib_canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: calib_canvas.configure(
            scrollregion=calib_canvas.bbox("all")
        )
    )
    
    calib_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    calib_canvas.configure(yscrollcommand=scrollbar.set)
    
    calib_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Top Camera Intrinsics Frame (Editable)
    top_intrinsics_frame = ttk.LabelFrame(scrollable_frame, text="Top Camera Intrinsics")
    top_intrinsics_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Get Top Camera intrinsics directly from camera_intrinsics.json (cached)
    top_intrinsics = config_manager.get_camera_intrinsics_for_display("top_camera")
    
    # Get RealSense intrinsics for later use (cached)
    realsense_intrinsics = config_manager.get_camera_intrinsics_for_display("realsense")
    
    # Get measurement calibration data once to avoid repeated calls
    measurement_calibration = config_manager.get_all_intrinsics_for_display()

    # Top camera intrinsics (editable)
    def create_top_intrinsics_entry(frame, label, key, row, unit=""):
        ttk.Label(frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        entry = ttk.Entry(frame)  # Editable entry
        entry.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Get value from cached top_intrinsics
        value = top_intrinsics.get(key, 0)
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
        entry.config(state='readonly')  # Start as readonly, enable during edit
        
        if unit:
            ttk.Label(frame, text=unit).grid(row=row, column=2, padx=5, pady=5, sticky=tk.W)
        
        top_intrinsics_entries[key] = entry
        return entry
    
    # Create editable top camera intrinsics entries
    create_top_intrinsics_entry(top_intrinsics_frame, "Focal Length X (fx):", "fx", 0, "px")
    create_top_intrinsics_entry(top_intrinsics_frame, "Focal Length Y (fy):", "fy", 1, "px")
    create_top_intrinsics_entry(top_intrinsics_frame, "Principal Point X (cx):", "cx", 2, "px")
    create_top_intrinsics_entry(top_intrinsics_frame, "Principal Point Y (cy):", "cy", 3, "px")
    
    # Add resolution and note
    resolution_label = ttk.Label(top_intrinsics_frame, text="Resolution: 1280x720", font=("Arial", 8))
    resolution_label.grid(row=4, column=0, columnspan=3, padx=5, pady=2, sticky=tk.W)
    
    note_label = ttk.Label(top_intrinsics_frame, text="Note: Changes will update camera_intrinsics.json and be used throughout app", 
                          font=("Arial", 8, "italic"))
    note_label.grid(row=5, column=0, columnspan=3, padx=5, pady=2, sticky=tk.W)
    
    # Measurement Calibration function (for editable settings) - now uses cached data
    def create_calib_entry(frame, label, key, row, unit=""):
        ttk.Label(frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        entry = ttk.Entry(frame)  # Create as normal entry first
        entry.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Use cached measurement_calibration data instead of calling get_all_intrinsics_for_display repeatedly
        value = measurement_calibration.get(key, current_settings["calibration"].get(key, 0))
        
        entry.delete(0, tk.END)  # Clear any existing content
        entry.insert(0, str(value))
        entry.config(state='readonly')  # Make readonly after inserting value
        
        if unit:
            ttk.Label(frame, text=unit).grid(row=row, column=2, padx=5, pady=5, sticky=tk.W)
        
        calib_entries[key] = entry
        return entry
    
    # Top View Calibration
    top_calib_frame = ttk.LabelFrame(scrollable_frame, text="Top View Calibration")
    top_calib_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Debug: Print current calibration settings
    # print(f"Creating top view entries with current settings: {current_settings['calibration']}")
    
    top_dia_entry = create_calib_entry(top_calib_frame, "Reference Diameter:", "ref_diameter", 0, "mm")
    top_pix_entry = create_calib_entry(top_calib_frame, "Reference Pixels:", "ref_diameter_pixels", 1, "px")
    top_height_entry = create_calib_entry(top_calib_frame, "Height from Base:", "base_height", 2, "mm")
    
    # Side View Calibration
    side_calib_frame = ttk.LabelFrame(scrollable_frame, text="Side View Calibration")
    side_calib_frame.pack(fill=tk.X, padx=10, pady=10)
    
    side_pix_entry = create_calib_entry(side_calib_frame, "Reference Pixels:", "side_ref_pixels", 0, "px")
    side_height_entry = create_calib_entry(side_calib_frame, "Height from Base:", "side_camera_height", 1, "mm")
    
    # Add height parameter instead of thickness parameters
    side_wheel_height_entry = create_calib_entry(side_calib_frame, "Standard Wheel Height:", "wheel_height", 2, "mm")
    
    # RealSense Camera Intrinsics Section
    realsense_calib_frame = ttk.LabelFrame(scrollable_frame, text="RealSense Camera Intrinsics")
    realsense_calib_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Create RealSense intrinsics entries in 2x2 grid
    intrinsics_frame = ttk.Frame(realsense_calib_frame)
    intrinsics_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
    
    # Create RealSense entries manually using cached realsense_intrinsics
    ttk.Label(intrinsics_frame, text="Fx:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    fx_entry = ttk.Entry(intrinsics_frame, width=15)
    fx_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    fx_entry.insert(0, str(realsense_intrinsics.get("fx", current_settings["calibration"].get("fx", 0))))
    fx_entry.config(state='readonly')
    
    ttk.Label(intrinsics_frame, text="Fy:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    fy_entry = ttk.Entry(intrinsics_frame, width=15)
    fy_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
    fy_entry.insert(0, str(realsense_intrinsics.get("fy", current_settings["calibration"].get("fy", 0))))
    fy_entry.config(state='readonly')
    
    ttk.Label(intrinsics_frame, text="Cx:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    cx_entry = ttk.Entry(intrinsics_frame, width=15)
    cx_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
    cx_entry.insert(0, str(realsense_intrinsics.get("cx", current_settings["calibration"].get("cx", 0))))
    cx_entry.config(state='readonly')
    
    ttk.Label(intrinsics_frame, text="Cy:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
    cy_entry = ttk.Entry(intrinsics_frame, width=15)
    cy_entry.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
    cy_entry.insert(0, str(realsense_intrinsics.get("cy", current_settings["calibration"].get("cy", 0))))
    cy_entry.config(state='readonly')
    
    # Store entries for access (update the pre-declared dictionary)
    realsense_intrinsics_entries.update({
        'fx': fx_entry,
        'fy': fy_entry,
        'cx': cx_entry,
        'cy': cy_entry
    })
    
    # Add resolution label and note
    ttk.Label(intrinsics_frame, text="Resolution: 1280x720").grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
    ttk.Label(intrinsics_frame, text="Note: Acquired from device at startup, editable", 
              font=("Arial", 8, "italic")).grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky=tk.W)
    
    # Button to get RealSense intrinsics from device
    def get_realsense_from_device():
        """Get RealSense intrinsics from device and update display"""
        try:
            # Use config manager to get intrinsics from device
            device_intrinsics = config_manager._get_realsense_intrinsics_from_device()
            
            if device_intrinsics:
                # Save to both files
                config_manager._save_realsense_intrinsics_to_both_files(device_intrinsics)
                
                # Update UI immediately
                for key, entry in realsense_intrinsics_entries.items():
                    if key in device_intrinsics:
                        current_state = entry.cget('state')
                        entry.config(state='normal')
                        entry.delete(0, tk.END)
                        entry.insert(0, str(device_intrinsics[key]))
                        entry.config(state=current_state)
                
                messagebox.showinfo("Success", "RealSense intrinsics acquired from device and saved!")
            else:
                messagebox.showerror("Error", "Failed to get RealSense intrinsics from device. Check camera connection.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error getting RealSense intrinsics from device: {e}")
    
    # Buttons row
    buttons_frame = ttk.Frame(realsense_calib_frame)
    buttons_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky=tk.W)
    
    ttk.Button(buttons_frame, text="Get from Device", command=get_realsense_from_device).pack(side=tk.LEFT, padx=(0, 5))
    refresh_button = ttk.Button(buttons_frame, text="Refresh from File", command=refresh_calibration_values)
    refresh_button.pack(side=tk.LEFT, padx=(0, 5))
    
    # RealSense Specific Settings
    realsense_frame = ttk.LabelFrame(scrollable_frame, text="RealSense Settings")
    realsense_frame.pack(fill=tk.X, padx=10, pady=10)
    
    create_calib_entry(realsense_frame, "Depth Scale:", "depth_scale", 0, "")
    create_calib_entry(realsense_frame, "Depth Units:", "depth_units", 1, "mm")
    create_calib_entry(realsense_frame, "Depth Min:", "depth_min", 2, "mm")
    create_calib_entry(realsense_frame, "Depth Max:", "depth_max", 3, "mm")
    
    # Edit Calibration button (password protected)
    def edit_calibration():
        password = simpledialog.askstring("Authentication", "Enter password to edit calibration:", show='*')
        if password == "admin123":
            # Enable all measurement calibration entries for editing
            for entry in calib_entries.values():
                entry.config(state='normal')
            
            # Enable Top Camera intrinsics for editing
            for entry in top_intrinsics_entries.values():
                entry.config(state='normal')
                
            # Enable RealSense intrinsics for editing
            for entry in realsense_intrinsics_entries.values():
                entry.config(state='normal')
            
            # Change button text and command
            edit_button.config(text="Save Calibration", command=save_calibration)
        else:
            messagebox.showerror("Error", "Incorrect password")
    
    # Save calibration values
    def save_calibration():
        try:
            # Update all measurement calibration values from UI entries
            for key, entry in calib_entries.items():
                try:
                    config_manager.update_calibration_value(key, float(entry.get()))
                except ValueError:
                    messagebox.showerror("Error", f"Invalid number format for {key}: {entry.get()}")
                    return
            
            # Update Top Camera intrinsics to camera_intrinsics.json
            top_camera_data = {
                'fx': float(top_intrinsics_entries['fx'].get()),
                'fy': float(top_intrinsics_entries['fy'].get()),
                'cx': float(top_intrinsics_entries['cx'].get()),
                'cy': float(top_intrinsics_entries['cy'].get()),
                'width': 1280,
                'height': 720
            }
            config_manager.update_camera_intrinsics("top_camera", top_camera_data)
            
            # Update RealSense intrinsics from entries to both settings.json and camera_intrinsics.json
            realsense_data = {
                'fx': float(realsense_intrinsics_entries['fx'].get()),
                'fy': float(realsense_intrinsics_entries['fy'].get()),
                'cx': float(realsense_intrinsics_entries['cx'].get()),
                'cy': float(realsense_intrinsics_entries['cy'].get()),
                'width': 1280,
                'height': 720
            }
            
            # Save RealSense intrinsics to camera_intrinsics.json
            config_manager.update_camera_intrinsics("realsense", realsense_data)
            
            # Also update settings.json calibration section for RealSense
            config_manager.update_calibration_value("fx", realsense_data['fx'])
            config_manager.update_calibration_value("fy", realsense_data['fy'])
            config_manager.update_calibration_value("cx", realsense_data['cx'])
            config_manager.update_calibration_value("cy", realsense_data['cy'])
            
            # Disable editing again for all entries
            for key, entry in calib_entries.items():
                entry.config(state='readonly')
            
            for key, entry in top_intrinsics_entries.items():
                entry.config(state='readonly')
                
            for key, entry in realsense_intrinsics_entries.items():
                entry.config(state='readonly')
            
            # Change button back
            edit_button.config(text="Edit Calibration", command=edit_calibration)
            messagebox.showinfo("Success", "All calibration settings saved and synced to camera_intrinsics.json!")
            
            debug_print(f"Saved Top Camera intrinsics: {top_camera_data}", "ui")
            debug_print(f"Saved RealSense intrinsics: {realsense_data}", "ui")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save calibration settings: {e}")
    
    # Edit/Save button
    edit_button = ttk.Button(scrollable_frame, text="Edit Calibration", command=edit_calibration)
    edit_button.pack(pady=10)
    
    # Return the window for reference
    return settings_window