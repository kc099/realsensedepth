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
        self.is_processing_signal = False  # Add flag to track signal processing state
        
    def _load_settings(self):
        """Load settings from settings.json"""
        try:
            with open("settings.json", "r") as f:
                data = json.load(f)
                
                # Get settings from the nested structure
                if isinstance(data, dict):
                    if "settings" in data:
                        settings = data["settings"]
                    else:
                        settings = data  # Handle case where settings are at root level
                        
                    # Ensure we have all required settings with defaults
                    settings = {
                        "com_port": settings.get("com_port"),
                        "baud_rate": settings.get("baud_rate", 19200),
                        "modbus_slave_id": settings.get("modbus_slave_id", 1)
                    }
                    
                    # Print loaded settings for debugging
                    print(f"Loaded settings - COM port: {settings['com_port']}, "
                          f"Baud rate: {settings['baud_rate']}, "
                          f"Slave ID: {settings['modbus_slave_id']}")
                    return settings
                    
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
                    try:
                        with open("settings.json", "r") as f:
                            data = json.load(f)
                        if "settings" in data:
                            data["settings"]["com_port"] = com_port
                            with open("settings.json", "w") as f:
                                json.dump(data, f, indent=4)
                            print(f"Updated settings.json with auto-detected COM port: {com_port}")
                    except Exception as e:
                        print(f"Error saving auto-detected COM port: {e}")
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
        
    def _detect_signal_thread(self):
        """Thread for monitoring 24V signal with improved debugging and robustness"""
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
            
            # Open the serial port with selected baud rate
            self.serial_port = serial.Serial(
                port=com_port,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1  # Short timeout to prevent blocking
            )
            print(f"Connected to {com_port} at {baud_rate} baud for 24V signal detection")
            
            # Clear any existing data in the buffer
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            print("Serial buffers cleared")
            
            # Add variables for status tracking
            # last_status_time = time.time()
            # status_interval = 30  # Print status every 30 seconds
            # total_bytes_received = 0
            # frame_count = 0
            
            print("Modbus monitoring started - waiting for frames...")
            
            while not self.stop_flag:
                try:
                    # current_time = time.time()
                    
                    # # Print periodic status to show the loop is running
                    # if current_time - last_status_time >= status_interval:
                    #     print(f"Modbus monitor active - Bytes received: {total_bytes_received}, "
                    #         f"Frames processed: {frame_count}, Port: {com_port}")
                    #     last_status_time = current_time
                    
                    # Check if there's any data on the serial port
                    bytes_waiting = self.serial_port.in_waiting
                    if bytes_waiting > 0:
                        print(f"Data available: {bytes_waiting} bytes")
                        
                        # Read available data (up to reasonable buffer size)
                        max_read = min(bytes_waiting, 256)  # Limit read size
                        data = self.serial_port.read(max_read)
                        # total_bytes_received += len(data)
                        
                        if len(data) > 0:
                            # Print raw data in hex for debugging
                            hex_data = ' '.join([f'{b:02X}' for b in data])
                            # print(f"Received {len(data)} bytes (hex): {hex_data}")
                            if self.signal_callback:
                                self.signal_callback(signal_type="MODBUS_FRAME")
                            # # Process the data to find valid Modbus frames
                            # self._process_modbus_data(data, slave_id)
                            # frame_count += 1
                        else:
                            print("No data read despite bytes_waiting > 0")
                    else:
                        # Short sleep to prevent CPU overuse when no data
                        time.sleep(0.01)
                        
                except serial.SerialTimeoutException:
                    # This is normal - just means no data within timeout period
                    continue
                except serial.SerialException as e:
                    print(f"Serial port error: {e}")
                    break
                except Exception as e:
                    print(f"Error reading signal data: {e}")
                    traceback.print_exc()
                    time.sleep(0.1)  # Wait a bit before retrying
                
            # Clean up
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
            
    def send_measurement_data(self, model_name, diameter_mm, height_mm):
        """Send measurement data via simplified Modbus frame
        
        Frame structure (12 bytes total):
        - Slave ID (1 byte): From settings
        - Function code (1 byte): 0x03
        - Height (4 bytes): IEEE754 float, MSB first (big-endian)
        - Diameter (4 bytes): IEEE754 float, MSB first (big-endian)  
        - Model min value (1 byte): e.g., 10 for "10-13"
        - Model max value (1 byte): e.g., 13 for "10-13"
        
        Args:
            model_name: Name of the wheel model (e.g., "10-13", "16-19")
            diameter_mm: Measured diameter in mm
            height_mm: Measured height in mm
            
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
            
            # Parse model name to extract min and max values
            model_min, model_max = self._parse_model_name(model_name)
            
            # Convert height to 4-byte IEEE754 float, MSB first (big-endian)
            height_bytes = struct.pack('>f', float(height_mm))
            
            # Convert diameter to 4-byte IEEE754 float, MSB first (big-endian)
            diameter_bytes = struct.pack('>f', float(diameter_mm))
            
            # Create the complete 12-byte frame
            frame = bytes([
                SLAVE_ADDRESS,    # Byte 0: Slave ID
                FUNCTION_CODE     # Byte 1: Function code (0x03)
            ]) + height_bytes + diameter_bytes + bytes([
                model_min,        # Byte 10: Model min value
                model_max         # Byte 11: Model max value
            ])
            
            # Send the frame
            self.serial_port.write(frame)
            
            # print(f"Sent measurement data as simplified Modbus frame:")
            # print(f"  Model: {model_name} -> Min: {model_min}, Max: {model_max}")
            # print(f"  Height: {height_mm:.2f} mm")
            # print(f"  Diameter: {diameter_mm:.2f} mm")
            # print(f"  Slave ID: {SLAVE_ADDRESS}")
            # print(f"  Frame size: {len(frame)} bytes")
            
            # # For debugging, print hexadecimal representation
            # hex_repr = ' '.join([f'{b:02X}' for b in frame])
            # print(f"  Frame (hex): {hex_repr}")
            
            # # Also print byte-by-byte breakdown
            # print(f"  Breakdown:")
            print(f"    Slave ID: 0x{frame[0]:02X} ({frame[0]})")
            print(f"    Function: 0x{frame[1]:02X} ({frame[1]})")
            print(f"    Height bytes: {' '.join([f'{b:02X}' for b in frame[2:6]])} ({height_mm:.2f})")
            print(f"    Diameter bytes: {' '.join([f'{b:02X}' for b in frame[6:10]])} ({diameter_mm:.2f})")
            print(f"    Model min: 0x{frame[10]:02X} ({frame[10]})")
            print(f"    Model max: 0x{frame[11]:02X} ({frame[11]})")
            
            return True
            
        except Exception as e:
            print(f"Error sending measurement data: {e}")
            traceback.print_exc()
            return False

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
