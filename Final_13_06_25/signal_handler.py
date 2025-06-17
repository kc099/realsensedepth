import threading
import time
import traceback
import json


# Try importing pyserial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("pyserial library not found - 24V signal detection will not be available")

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
        self.last_callback_time = 0       # Track last callback time globally
        
    def _load_settings(self):
        """Load settings from settings.json"""
        try:
            with open("settings.json", "r") as f:
                data = json.load(f)
                
                # Get settings from the nested structure
                if isinstance(data, dict):
                    settings = data.get("settings", data)
                    return {
                        "com_port": settings.get("com_port"),
                        "baud_rate": settings.get("baud_rate", 19200),
                        "modbus_slave_id": settings.get("modbus_slave_id", 1)
                    }
                    
                    # # Print loaded settings for debugging
                    # print(f"Loaded settings - COM port: {settings['com_port']}, "
                    #       f"Baud rate: {settings['baud_rate']}, "
                    #       f"Slave ID: {settings['modbus_slave_id']}")
                    # return settings
                    
        except Exception as e:
            print(f"Error loading settings: {e}")
            traceback.print_exc()
            
        # Return default settings if loading fails
        return {"com_port": None, "modbus_slave_id": 1, "baud_rate": 19200}
        
    def start_detection(self):
        """Start 24V signal detection thread"""
        if not SERIAL_AVAILABLE:
            print("24V signal detection not available (pyserial not installed)")
            return False
            
        if self.is_running:
            print("Signal detection already running")
            return True
            
        # Check if we have a valid COM port
        com_port = self.settings.get("com_port")
        
        # If no COM port is selected, try to auto-detect
        if not com_port:
            try:
                available_ports = [port.device for port in serial.tools.list_ports.comports()]
                if available_ports:
                    com_port = available_ports[0]  # Use the first available port
                    print(f"Auto-detected COM port: {com_port}")
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
                    print("No COM ports available")
                    return False
            except Exception as e:
                print(f"Error checking available ports: {e}")
                return False
            
        # Verify port exists
        try:
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if com_port not in available_ports:
                print(f"Configured COM port {com_port} not found. Available ports: {available_ports}")
                return False
        except Exception as e:
            print(f"Error checking available ports: {e}")
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
        print("Processing state reset manually")
        
    def _detect_signal_thread(self):
        """Thread for monitoring 24V signal with frame buffering and cooldown"""
        if not SERIAL_AVAILABLE:
            return
            
        try:
            # Get COM port and baud rate from settings
            com_port = self.settings.get("com_port")
            baud_rate = self.settings.get("baud_rate", 19200)
            slave_id = self.settings.get("modbus_slave_id", 1)
            print(f"Starting signal detection with slave ID: {slave_id}")
            
            if not com_port:
                print("No COM port selected in settings")
                return
                
            # Verify port exists
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if com_port not in available_ports:
                print(f"Selected COM port {com_port} not found")
                return
            
            # Open the serial port with improved settings
            self.serial_port = serial.Serial(
                port=com_port,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,        # Increased timeout to prevent blocking
                write_timeout=0.5,  # Add write timeout
                inter_byte_timeout=0.1  # Timeout between bytes
            )
            print(f"Connected to {com_port} at {baud_rate} baud for 24V signal detection")
            
            # Clear any existing data in the buffer
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            print("Serial buffers cleared")
            
            # Initialize frame buffering variables
            # data_buffer = bytearray()
            # while not self.stop_flag:
            #     try:
            #         bytes_waiting = self.serial_port.in_waiting
            #         if bytes_waiting > 0:
            #             new_data = self.serial_port.read(bytes_waiting)
            #             if len(new_data) > 0:
            #                 data_buffer.extend(new_data)
            #                 print(f"Received {len(new_data)} bytes")
            #                 print(f"Buffer now contains {len(data_buffer)} bytes")

            #                 # Process valid Modbus frame only if it's exactly 8 bytes
            #                 if len(data_buffer) >= 8:
            #                     self._process_modbus_data(data_buffer, slave_id)
            #                     data_buffer.clear()

            #         time.sleep(0.01)
            #     except Exception as e:
            #         print(f"Error reading signal data: {e}")
            #         traceback.print_exc()
            #         time.sleep(0.1)
             # Main detection loop
            while not self.stop_flag:
                try:
                    # Wait for exactly 8 bytes (blocking read)
                    frame = self.serial_port.read(8)
                    
                    if len(frame) == 8:
                        print(f"Received complete 8-byte frame: {frame.hex()}")
                        self._process_modbus_data(frame, slave_id)
                    else:
                        if frame:  # Partial frame received
                            print(f"Received incomplete frame ({len(frame)} bytes), discarding")
                        continue
                        
                except Exception as e:
                    print(f"Error in signal detection: {e}")
                    time.sleep(0.1)  # Brief pause after error
                    continue
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
                print("Serial port closed")

        except Exception as e:
            print(f"Error in 24V signal detection: {e}")
            traceback.print_exc()


    def _process_modbus_data(self, data, expected_slave_id):
        """Process received data to find valid Modbus frames"""
        try:
            # Look for potential Modbus frames in the received data
            for i in range(len(data) - 1):  # Need at least 2 bytes for slave ID and function code
                if i + 7 < len(data):  # Check if we have enough bytes for a minimum frame
                    potential_frame = data[i:i+8]  # Try 8-byte frame first
                    
                    slave_id = potential_frame[0]
                    function_code = potential_frame[1]
                    
                    # print(f"Checking potential frame at offset {i}: "
                    #     f"Slave ID: {slave_id}, Function: 0x{function_code:02X}")
                    
                    # Check for read holding registers request (0x03) to configured slave ID
                    if slave_id == expected_slave_id and function_code == 0x03:
                        # print(f"Valid Modbus read request detected - Slave ID: {slave_id}, Function: 0x03")
                        # Trigger the callback function if it exists
                        if self.signal_callback:
                            self.signal_callback(signal_type="MODBUS_FRAME")
                        return  # Found valid frame, exit processing
                    
                    # Check for other common Modbus function codes
                    elif slave_id == expected_slave_id and function_code in [0x01, 0x02, 0x04, 0x05, 0x06, 0x0F, 0x10]:
                        print(f"Modbus frame for correct slave ID - Slave: {slave_id}, Function: 0x{function_code:02X}")
                        # Could trigger callback for any valid Modbus frame to slave
                        # if self.signal_callback:
                        #     self.signal_callback(signal_type="MODBUS_FRAME")
                    
            # If no valid frame found, just log the data
            print(f"No valid Modbus frame found in {len(data)} bytes")
            
        except Exception as e:
            print(f"Error processing Modbus data: {e}")
            traceback.print_exc()

    def test_serial_connection(self):
        """Test method to verify serial connection is working"""
        if not self.serial_port or not self.serial_port.is_open:
            print("Serial port not open for testing")
            return False
        
        try:
            # Send a test command (query device status)
            # This is a simple Modbus read request - adjust as needed
            test_command = bytes([self.settings.get("modbus_slave_id", 1), 0x03, 0x00, 0x00, 0x00, 0x01])
            
            # Calculate CRC for the test command (simplified)
            crc = self._calculate_modbus_crc(test_command)
            crc_bytes = struct.pack('<H', crc)
            complete_command = test_command + crc_bytes
            
            print(f"Sending test command: {' '.join([f'{b:02X}' for b in complete_command])}")
            self.serial_port.write(complete_command)
            
            # Wait for response
            time.sleep(0.5)
            if self.serial_port.in_waiting > 0:
                response = self.serial_port.read(self.serial_port.in_waiting)
                print(f"Received test response: {' '.join([f'{b:02X}' for b in response])}")
                return True
            else:
                print("No response to test command")
                return False
                
        except Exception as e:
            print(f"Error in serial test: {e}")
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
            print("Cannot send measurement data - pyserial not installed")
            return False
        
        # Check if we have an open serial connection
        if self.serial_port is None or not self.serial_port.is_open:
            print("Cannot send measurement data - serial port not open")
            print("The signal detection system must be running to send data")
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
            print(f"Sent measurement data via Modbus:")
            print(f"  Model: {model_name} -> Code: {model_code}")
            print(f"  Height: {height_mm:.2f} mm (raw value)")
            print(f"  Diameter: {diameter_mm:.2f} mm (raw value)")
            print(f"  Result: {result_text} ({pass_fail_result})")
            print(f"  Slave ID: {SLAVE_ADDRESS}")
            print(f"  Frame size: {len(frame)} bytes")
            
            # Print hexadecimal representation for debugging
            hex_repr = ' '.join([f'{b:02X}' for b in frame])
            
            print(f"  Frame (hex): {hex_repr}")
            
            # Print byte-by-byte breakdown
            print(f"  Breakdown:")
            print(f"    Slave ID: 0x{frame[0]:02X} ({frame[0]})")
            print(f"    Function: 0x{frame[1]:02X} ({frame[1]})")
            print(f"    Height: {' '.join([f'{b:02X}' for b in frame[2:6]])} ({height_raw if has_valid_measurements else 0:.2f} mm)")
            print(f"    Diameter: {' '.join([f'{b:02X}' for b in frame[6:10]])} ({diameter_raw if has_valid_measurements else 0:.2f} mm)")
            print(f"    Model code: 0x{frame[10]:02X} ({model_code if has_valid_measurements else 0})")
            print(f"    Pass/Fail: 0x{frame[11]:02X} ({result_text if has_valid_measurements else 'N/A'})")
            print(f"    Proceed: 0x{frame[12]:02X} ({'Valid' if proceed_byte == 1 else 'Invalid'} measurement)")
            
            return True
            
        except Exception as e:
            print(f"Error sending measurement data: {e}")
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

    def _parse_model_name(self, model_name):
        """Parse model name to extract min and max values
        
        Args:
            model_name: Model name like "10-13", "16-19", etc.
            
        Returns:
            tuple: (min_value, max_value) as integers
        """
        try:
            # Split on hyphen and convert to integers
            if '-' in model_name:
                parts = model_name.split('-')
                if len(parts) == 2:
                    min_val = int(parts[0])
                    max_val = int(parts[1])
                    
                    # Ensure values are within byte range (0-255)
                    min_val = max(0, min(255, min_val))
                    max_val = max(0, min(255, max_val))
                    
                    return min_val, max_val
            
            # If parsing fails, try to extract numbers from the string
            import re
            numbers = re.findall(r'\d+', model_name)
            if len(numbers) >= 2:
                min_val = int(numbers[0])
                max_val = int(numbers[1])
                
                # Ensure values are within byte range
                min_val = max(0, min(255, min_val))
                max_val = max(0, min(255, max_val))
                
                return min_val, max_val
                
        except Exception as e:
            print(f"Error parsing model name '{model_name}': {e}")
        
        # Default fallback
        print(f"Warning: Could not parse model name '{model_name}', using default values")
        return 0, 0
    
    def _calculate_modbus_crc(self, data):
        """Calculate Modbus RTU CRC-16 checksum
        
        Args:
            data: Bytes to calculate CRC for
            
        Returns:
            int: 16-bit CRC value
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001  # Polynomial for Modbus CRC-16
                else:
                    crc >>= 1
        return crc
