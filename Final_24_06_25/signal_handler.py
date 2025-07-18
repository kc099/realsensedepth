import threading
import time
import traceback
import json

# Import centralized debug system
from debug_utils import debug_print

# Try importing pyserial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    debug_print("pyserial library not found - 24V signal detection will not be available", "errors")

class SignalHandler:
    """Handler for external 24V signal detection and communication"""
    
    def __init__(self, signal_callback=None):
        """
        Initialize signal handler
        
        Args:
            signal_callback: Function to call when 24V signal is detected
        """
        self.signal_callback = signal_callback
        self.thread = None
        self.stop_flag = False
        self.is_running = False
        self.serial_port = None
        self.settings = self._load_settings()
        self.is_processing_signal = False  # Flag to track signal processing state
        self.last_callback_time = 0 
        self.min_callback_interval = 2.0      # Track last callback time globally
        
    def _load_settings(self):
        """Load settings from settings.json"""
        from config_manager import ConfigManager
        config = ConfigManager()
        return {
            "com_port": config.settings.get("com_port"),
            "baud_rate": config.settings.get("baud_rate", 19200),
            "modbus_slave_id": config.settings.get("modbus_slave_id", 1)
        }
        
    def start_detection(self):
        """Start 24V signal detection thread"""
        if not SERIAL_AVAILABLE:
            debug_print("24V signal detection not available (pyserial not installed)", "signal")
            return False
            
        if self.is_running:
            debug_print("Signal detection already running", "signal")
            return True
            
        # Check if we have a valid COM port
        com_port = self.settings.get("com_port")
        
        # If no COM port is selected, try to auto-detect
        if not com_port:
            try:
                available_ports = [port.device for port in serial.tools.list_ports.comports()]
                if available_ports:
                    com_port = available_ports[0]  # Use the first available port
                    debug_print(f"Auto-detected COM port: {com_port}", "signal")
                    # Update settings with the detected port
                    self.settings["com_port"] = com_port
                #  available_ports = [port.device for port in serial.tools.list_ports.comports()]
                # for port in available_ports:
                #     if "Arduino" in port.description or 'USB' in port.description:
                #         com_port = port
                #     # com_port = available_ports[0]  # Use the first available port
                #     print(f"Auto-detected COM port: {com_port}")
                #     # Update settings with the detected port
                #     self.settings["com_port"] = com_port
                #     # Save the updated settings    # Save the updated settings
                   
                else:
                    debug_print("No COM ports available", "errors")
                    return False
            except Exception as e:
                debug_print(f"Error checking available ports: {e}", "errors")
                return False
            
        # Verify port exists
        try:
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if com_port not in available_ports:
                debug_print(f"Configured COM port {com_port} not found. Available ports: {available_ports}", "errors")
                return False
        except Exception as e:
            debug_print(f"Error checking available ports: {e}", "errors")
            return False
            
        self.stop_flag = False
        self.thread = threading.Thread(target=self._detect_signal_thread, daemon=True)
        self.thread.start()
        self.is_running = True
        return True
        
    def stop_detection(self):
        """Stop 24V signal detection thread"""
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1.0)
        self.is_running = False
        self.is_processing_signal = False  # Reset processing state
        
    def reset_processing_state(self):
        """Manually reset the processing state (useful for debugging)"""
        self.is_processing_signal = False
        debug_print("Processing state reset manually", "signal")
        
    def _detect_signal_thread(self):
        """Thread for monitoring 24V signal with frame buffering and cooldown"""
        if not SERIAL_AVAILABLE:
            return
            
        try:
            # Get COM port and baud rate from settings
            com_port = self.settings.get("com_port")
            baud_rate = self.settings.get("baud_rate", 19200)
            slave_id = self.settings.get("modbus_slave_id", 1)
            debug_print(f"Starting signal detection with slave ID: {slave_id}", "signal")
            
            if not com_port:
                debug_print("No COM port selected in settings", "errors")
                return
                
            # Verify port exists
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if com_port not in available_ports:
                debug_print(f"Selected COM port {com_port} not found", "errors")
                return
            
            # Open the serial port with improved settings
            self.serial_port = serial.Serial(
                port=com_port,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,        # Shorter timeout for non-blocking behavior
                write_timeout=0.3,  # Add write timeout
                inter_byte_timeout=None  # Let data accumulate in buffer
            )
            debug_print(f"Connected to {com_port} at {baud_rate} baud for 24V signal detection", "signal")
            
            # Clear any existing data in the buffer
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            debug_print("Serial buffers cleared", "signal")
            
            # Initialize frame buffering variables
            data_buffer = bytearray()
            frame_size = 8  # Expected Modbus frame size
            
            debug_print("Modbus frame detection started - waiting for signals...", "signal")
            
            # Main detection loop with proper frame buffering
            while not self.stop_flag:
                try:
                    # Check if there's data available
                    bytes_waiting = self.serial_port.in_waiting
                    if bytes_waiting > 0:
                        # Read all available bytes
                        new_data = self.serial_port.read(bytes_waiting)
                        if len(new_data) > 0:
                            data_buffer.extend(new_data)
                            debug_print(f"📥 Received {len(new_data)} bytes: {new_data.hex().upper()}", "signal")
                            debug_print(f"📦 Buffer now contains {len(data_buffer)} bytes total", "signal")
                            
                            # Process complete frames
                            while len(data_buffer) >= frame_size:
                                # Extract potential frame
                                potential_frame = data_buffer[:frame_size]
                                debug_print(f"🔍 Processing potential 8-byte frame: {potential_frame.hex().upper()}", "signal")
                                
                                # Process the frame
                                if self._process_modbus_data(potential_frame, slave_id):
                                    # Valid frame found, remove it from buffer
                                    data_buffer = data_buffer[frame_size:]
                                    debug_print(f"✅ Valid frame processed, {len(data_buffer)} bytes remaining in buffer", "signal")
                                    break  # Process one frame at a time to avoid callback flooding
                                else:
                                    # No valid frame at start of buffer, shift by 1 byte and try again
                                    discarded_byte = data_buffer[0]
                                    data_buffer = data_buffer[1:]
                                    debug_print(f"❌ Invalid frame start, discarded byte: 0x{discarded_byte:02X}, {len(data_buffer)} bytes remaining", "signal")
                            
                            # Clear buffer if it gets too large (prevent memory issues)
                            if len(data_buffer) > 100:
                                debug_print(f"⚠️ Buffer too large ({len(data_buffer)} bytes), clearing...", "signal")
                                data_buffer.clear()
                    
                    # Small delay to prevent CPU hogging
                    time.sleep(0.01)
                    
                except Exception as e:
                    debug_print(f"Error in signal detection: {e}", "errors")
                    traceback.print_exc()
                    time.sleep(0.1)  # Brief pause after error
                    continue
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
                debug_print("Serial port closed", "signal")

        except Exception as e:
            debug_print(f"Error in 24V signal detection: {e}", "errors")
            traceback.print_exc()


    def _process_modbus_data(self, data, expected_slave_id):
        """Process received data to find valid Modbus frames
        
        Args:
            data: Received data bytes (should be exactly 8 bytes)
            expected_slave_id: Expected slave ID from settings
            
        Returns:
            bool: True if valid Modbus frame was found and processed, False otherwise
        """
        try:
            # Ensure we have exactly 8 bytes
            if len(data) != 8:
                debug_print(f"Invalid frame length: {len(data)} bytes (expected 8)", "signal")
                return False
            
            # Extract slave ID and function code from first two bytes
            slave_id = data[0]
            function_code = data[1]
            
            debug_print(f"Frame analysis: Slave ID: {slave_id} (expected: {expected_slave_id}), Function: 0x{function_code:02X}", "signal")
            
            # Check for read holding registers request (0x03) to configured slave ID
            if slave_id == expected_slave_id and function_code == 0x03:
                debug_print(f"✓ VALID MODBUS FRAME DETECTED - Slave ID: {slave_id}, Function: 0x03", "signal")
                debug_print(f"  Complete frame: {data.hex().upper()}", "signal")
                
                # Trigger the callback function if it exists
                if self.signal_callback:
                    self.signal_callback(signal_type="MODBUS_FRAME")
                return True
            
            # Check for other common Modbus function codes to the correct slave
            elif slave_id == expected_slave_id and function_code in [0x01, 0x02, 0x04, 0x05, 0x06, 0x0F, 0x10]:
                debug_print(f"⚠ Modbus frame for correct slave but different function - Slave: {slave_id}, Function: 0x{function_code:02X}", "signal")
                # Optionally trigger callback for any valid Modbus frame to correct slave
                # if self.signal_callback:
                #     self.signal_callback(signal_type="MODBUS_FRAME")
                return True  # Valid frame structure, even if not the expected function
            
            # Wrong slave ID or invalid function code
            else:
                if slave_id != expected_slave_id:
                    debug_print(f"✗ Wrong slave ID: {slave_id} (expected: {expected_slave_id})", "signal")
                else:
                    debug_print(f"✗ Unsupported function code: 0x{function_code:02X}", "signal")
                return False
            
        except Exception as e:
            debug_print(f"Error processing Modbus data: {e}", "errors")
            traceback.print_exc()
            return False
            
    def send_measurement_data(self, model_name, diameter_mm, height_mm, wheel_models=None):
        """Send measurement data via enhanced Modbus frame with pass/fail result and proceed status
        
        Frame structure (12 bytes total):
        - Slave ID (1 byte): From settings
        - Function code (1 byte): 0x03
        - Height (4 bytes): IEEE754 float, MSB first (big-endian) - RAW VALUE
        - Diameter (4 bytes): IEEE754 float, MSB first (big-endian) - RAW VALUE
        - Model number (1 byte): 1=10-13, 2=13-16, 3=16-19, 4=19-22, 5=22-25
        - Pass/Fail result (1 byte): 1 for PASS, 0 for FAIL
        - Proceed status (1 byte): 1 if valid measurement, 0 if no measurement/error
        
        Args:
            model_name: Name of the wheel model (e.g., "10-13", "13-16", "16-19")
            diameter_mm: Measured diameter in mm
            height_mm: Measured height in mm
            wheel_models: Dictionary of wheel model specifications for pass/fail check
            
        Returns:
            bool: True if data was sent successfully, False otherwise
        """
        if not SERIAL_AVAILABLE:
            debug_print("Cannot send measurement data - pyserial not installed", "modbus")
            return False
        
        # Check if we have an open serial connection
        if self.serial_port is None or not self.serial_port.is_open:
            debug_print("Cannot send measurement data - serial port not open", "modbus")
            debug_print("The signal detection system must be running to send data", "modbus")
            return False
            
        try:
            import struct
            
            # Get Modbus parameters
            SLAVE_ADDRESS = self.settings.get("modbus_slave_id", 1)
            FUNCTION_CODE = 0x03  # Read Holding Registers
            
            # Convert model name to numeric code
            model_code = self._get_model_code(model_name)
            
            # Use raw measurements (no multiplication by 100)
            height_raw = height_mm
            diameter_raw = diameter_mm
            
            # Determine pass/fail result based on model specifications
            pass_fail_result = self._determine_pass_fail(
                model_name, diameter_mm, height_mm, wheel_models
            )
            
            # Convert raw values to 4-byte IEEE754 float, MSB first (big-endian)
            height_bytes = struct.pack('>f', float(height_raw))
            diameter_bytes = struct.pack('>f', float(diameter_raw))
            
            # Determine if we have valid measurements
            has_valid_measurements = all(v is not None for v in [height_mm, diameter_mm, model_code])
            proceed_byte = 1 if has_valid_measurements else 0
            
            # If no valid measurements, send all zeros for the data fields
            if not has_valid_measurements:
                height_bytes = bytes(4)  # 4 zero bytes for height
                diameter_bytes = bytes(4)  # 4 zero bytes for diameter
                model_code = 0
                pass_fail_result = 0
            
            # Create the complete 12-byte frame
            frame = bytes([
                SLAVE_ADDRESS,    # Byte 0: Slave ID
                FUNCTION_CODE     # Byte 1: Function code (0x03)
            ]) + height_bytes + diameter_bytes + bytes([
                model_code,       # Byte 10: Model code (1-5)
                pass_fail_result, # Byte 11: Pass/Fail result (1=PASS, 0=FAIL)
                proceed_byte      # Byte 12: Proceed status (1=valid, 0=invalid)
            ])
            
            # Send the frame
            self.serial_port.write(frame)
            
            # Print detailed information about the sent data
            result_text = "PASS" if pass_fail_result == 1 else "FAIL"
            debug_print(f"Sent measurement data via Modbus:", "modbus")
            debug_print(f"  Model: {model_name} -> Code: {model_code}", "modbus")
            debug_print(f"  Height: {height_mm:.2f} mm (raw value)", "modbus")
            debug_print(f"  Diameter: {diameter_mm:.2f} mm (raw value)", "modbus")
            debug_print(f"  Result: {result_text} ({pass_fail_result})", "modbus")
            debug_print(f"  Slave ID: {SLAVE_ADDRESS}", "modbus")
            debug_print(f"  Frame size: {len(frame)} bytes", "modbus")
            
            # Print hexadecimal representation for debugging
            hex_repr = ' '.join([f'{b:02X}' for b in frame])
            
            debug_print(f"  Frame (hex): {hex_repr}", "modbus")
            
            # Print byte-by-byte breakdown
            debug_print(f"  Breakdown:", "modbus")
            debug_print(f"    Slave ID: 0x{frame[0]:02X} ({frame[0]})", "modbus")
            debug_print(f"    Function: 0x{frame[1]:02X} ({frame[1]})", "modbus")
            debug_print(f"    Height: {' '.join([f'{b:02X}' for b in frame[2:6]])} ({height_raw if has_valid_measurements else 0:.2f} mm)", "modbus")
            debug_print(f"    Diameter: {' '.join([f'{b:02X}' for b in frame[6:10]])} ({diameter_raw if has_valid_measurements else 0:.2f} mm)", "modbus")
            debug_print(f"    Model code: 0x{frame[10]:02X} ({model_code if has_valid_measurements else 0})", "modbus")
            debug_print(f"    Pass/Fail: 0x{frame[11]:02X} ({result_text if has_valid_measurements else 'N/A'})", "modbus")
            debug_print(f"    Proceed: 0x{frame[12]:02X} ({'Valid' if proceed_byte == 1 else 'Invalid'} measurement)", "modbus")
            
            return True
            
        except Exception as e:
            debug_print(f"Error sending measurement data: {e}", "errors")
            traceback.print_exc()
            return False

    def _get_model_code(self, model_name):
        """Convert model name to numeric code
        
        Args:
            model_name: Model name like "10-13", "13-16", etc.
            
        Returns:
            int: Model code (1-5) where:
                1 = "10-13"
                2 = "13-16" 
                3 = "16-19"
                4 = "19-22"
                5 = "22-25"
        """
        # Define model mapping
        model_mapping = {
            "10-13": 1,
            "13-16": 2,
            "16-19": 3,
            "19-22": 4,
            "22-25": 5
        }
        
        # Direct lookup
        if model_name in model_mapping:
            return model_mapping[model_name]
        if model_name is None:
            model_name = "Unknown"
        elif isinstance(model_name, str):
            model_name = model_name.strip()
            if not model_name:
                model_name = "Unknown"
        else:
            model_name = str(model_name)
        # Try to handle variations in formatting
        normalized_name = model_name.strip().replace(" ", "")
        for key, value in model_mapping.items():
            if normalized_name == key.replace("-", "").replace(" ", ""):
                return value
        
        # If no match found, try to parse and determine range
        try:
            import re
            numbers = re.findall(r'\d+', model_name)
            if len(numbers) >= 2:
                min_val = int(numbers[0])
                max_val = int(numbers[1])
                
                # Determine model code based on range
                if min_val == 10 and max_val == 13:
                    return 1
                elif min_val == 13 and max_val == 16:
                    return 2
                elif min_val == 16 and max_val == 19:
                    return 3
                elif min_val == 19 and max_val == 22:
                    return 4
                elif min_val == 22 and max_val == 25:
                    return 5
        except Exception as e:
            print(f"Error parsing model name for code: {e}")
        
        # Default fallback
        print(f"Warning: Unknown model '{model_name}', using default code 1")
        return 1

    def _determine_pass_fail(self, model_name, diameter_mm, height_mm, wheel_models):
        INCH_TO_MM = 25.4
        try:
            if not wheel_models or model_name not in wheel_models:
                print(f"Warning: No model specifications found for '{model_name}' in wheel_models, defaulting to FAIL")
                return 0

            model_spec = wheel_models[model_name]

            # --- Diameter Check ---
            diameter_ok = False
            # Get diameter tolerance from model_spec, default to 3.0mm if not found in spec
            diameter_tolerance = model_spec.get('diameter_tolerance_mm', 3.0)

            parsed_from_model_name = False
            try:
                if '-' in model_name:
                    parts = model_name.split('-')
                    if len(parts) == 2:
                        min_inch_str, max_inch_str = parts
                        min_inch = float(min_inch_str)
                        max_inch = float(max_inch_str)

                        if min_inch > 0 and max_inch > 0 and min_inch <= max_inch:
                            base_min_dia_mm = min_inch * INCH_TO_MM
                            base_max_dia_mm = max_inch * INCH_TO_MM

                            lower_bound = base_min_dia_mm - diameter_tolerance
                            upper_bound = base_max_dia_mm + diameter_tolerance
                            
                            diameter_ok = lower_bound <= diameter_mm <= upper_bound
                            print(f"Diameter check (from model name '{model_name}'): {diameter_mm:.1f}mm against range [{lower_bound:.1f}mm - {upper_bound:.1f}mm] (Base: {base_min_dia_mm:.1f}-{base_max_dia_mm:.1f}mm, Tol: ±{diameter_tolerance:.1f}mm) -> {'PASS' if diameter_ok else 'FAIL'}")
                            parsed_from_model_name = True
                        else:
                            print(f"Warning: Parsed invalid inch range from model_name '{model_name}'. Falling back to spec.")
                    # else: model_name not in X-Y format, fall through
                # else: model_name not in X-Y format, fall through
            except ValueError:
                print(f"Warning: Could not parse min/max inches from model_name '{model_name}'. Attempting fallback to spec.")

            if not parsed_from_model_name:
                min_spec_dia_mm = model_spec.get('min_dia')
                max_spec_dia_mm = model_spec.get('max_dia')

                if min_spec_dia_mm is not None and max_spec_dia_mm is not None:
                    lower_bound = min_spec_dia_mm - diameter_tolerance
                    upper_bound = max_spec_dia_mm + diameter_tolerance
                    diameter_ok = lower_bound <= diameter_mm <= upper_bound
                    print(f"Diameter check (from model_spec min_dia/max_dia for '{model_name}'): {diameter_mm:.1f}mm against range [{lower_bound:.1f}mm - {upper_bound:.1f}mm] (Base: {min_spec_dia_mm:.1f}-{max_spec_dia_mm:.1f}mm, Tol: ±{diameter_tolerance:.1f}mm) -> {'PASS' if diameter_ok else 'FAIL'}")
                else:
                    expected_dia_mm = model_spec.get('diameter_mm')
                    if expected_dia_mm is not None:
                        lower_bound = expected_dia_mm - diameter_tolerance
                        upper_bound = expected_dia_mm + diameter_tolerance
                        diameter_ok = lower_bound <= diameter_mm <= upper_bound
                        print(f"Diameter check (from model_spec diameter_mm for '{model_name}'): {diameter_mm:.1f}mm against range [{lower_bound:.1f}mm - {upper_bound:.1f}mm] (Expected: {expected_dia_mm:.1f}mm, Tol: ±{diameter_tolerance:.1f}mm) -> {'PASS' if diameter_ok else 'FAIL'}")
                    else:
                        print(f"Warning: No usable diameter specification found for model '{model_name}'. Assuming PASS for diameter.")
                        diameter_ok = True

            # --- Height Check ---
            height_ok = False
            expected_height_mm = model_spec.get('height_mm')
            
            height_tolerance_from_spec = model_spec.get('height_tolerance_mm')
            if height_tolerance_from_spec is None:
                height_tolerance = 1.0
                print(f"Height tolerance not found in model_spec for '{model_name}', using default: {height_tolerance:.1f}mm")
            else:
                height_tolerance = height_tolerance_from_spec
            
            if expected_height_mm is not None:
                height_error = abs(height_mm - expected_height_mm)
                height_ok = height_error <= height_tolerance
                print(f"Height check (for model '{model_name}'): {height_mm:.1f}mm vs {expected_height_mm:.1f}mm ± {height_tolerance:.1f}mm -> {'PASS' if height_ok else 'FAIL'}")
            else:
                print(f"Warning: No height specification found for model '{model_name}'. Assuming PASS for height.")
                height_ok = True

            overall_result = diameter_ok and height_ok
            result_text = "PASS" if overall_result else "FAIL"
            print(f"Overall quality check result for model '{model_name}': {result_text}")
            
            return 1 if overall_result else 0

        except Exception as e:
            print(f"CRITICAL ERROR in _determine_pass_fail for model '{model_name}': {e}")
            import traceback
            traceback.print_exc()
            return 0

    
    