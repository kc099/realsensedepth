import threading
import time
import traceback

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
        
    def start_detection(self):
        """Start 24V signal detection thread"""
        if not SERIAL_AVAILABLE:
            print("24V signal detection not available (pyserial not installed)")
            return False
            
        if self.is_running:
            print("Signal detection already running")
            return True
            
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
        """Thread for monitoring 24V signal"""
        if not SERIAL_AVAILABLE:
            return
            
        try:
            # Try to auto-detect Arduino port
            ports = list(serial.tools.list_ports.comports())
            arduino_port = None
            for port in ports:
                if 'Arduino' in port.description or 'USB' in port.description:
                    arduino_port = port.device
                    break
            
            if not arduino_port:
                print("No Arduino device found for 24V signal detection")
                return
            
            # Open the serial port and store it in the class instance
            self.serial_port = serial.Serial(arduino_port, 19200, timeout=1)
            print(f"Connected to {arduino_port} for 24V signal detection")
            
            while not self.stop_flag:
                if self.serial_port.in_waiting:
                    # Don't try to decode as UTF-8, just check if any data is present
                    data = self.serial_port.readline()
                    if data:  # If any data is received, trigger the callback
                        print("Modbus frame received")
                        if self.signal_callback:
                            self.signal_callback(signal_type="MODBUS_FRAME")
                time.sleep(0.1)
                
            # Clean up
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
                print("Serial port closed")
        except Exception as e:
            print(f"Error in 24V signal detection: {e}")
            traceback.print_exc()
            
    def send_measurement_data(self, model_name, diameter_mm, height_mm):
        """Send measurement data via modbus frame
        
        Args:
            model_name: Name of the wheel model
            diameter_mm: Measured diameter in mm
            height_mm: Measured height in mm
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
            
            # Format data as binary according to Modbus protocol requirements
            # Convert model_name to a fixed 8-byte field
            # For model_name, take first 8 chars or pad with zeros if shorter
            model_bytes = model_name.encode('ascii', 'ignore')
            if len(model_bytes) > 8:
                model_bytes = model_bytes[:8]  # Truncate to 8 bytes
            else:
                model_bytes = model_bytes.ljust(8, b'\x00')  # Pad with zeros
                
            # Convert diameter to 4-byte float (32-bit)
            import struct
            diameter_bytes = struct.pack('>f', float(diameter_mm))  # Big-endian float
            
            # Convert height to 4-byte float (32-bit)
            height_bytes = struct.pack('>f', float(height_mm))  # Big-endian float
            
            # Combine all bytes into a single message
            data_bytes = model_bytes + diameter_bytes + height_bytes
            
            # Send binary data
            self.serial_port.write(data_bytes)
            print(f"Sent measurement data as binary: Model={model_name}, Diameter={diameter_mm:.1f}mm, Height={height_mm:.1f}mm")
            return True
            
        except Exception as e:
            print(f"Error sending measurement data: {e}")
            traceback.print_exc()
            return False
        finally:
            pass
            # We don't close the serial port here since it's shared with the detection thread
            # Serial port will be closed when the application terminates
