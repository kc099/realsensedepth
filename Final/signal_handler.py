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
    """Handler for external 24V signal detection"""
    
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
                
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            print(f"Connected to {arduino_port} for 24V signal detection")
            
            while not self.stop_flag:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8').strip()
                    if line == "24V_ON":
                        # When 24V signal is received, call the callback
                        print("24V signal received")
                        if self.signal_callback:
                            self.signal_callback()
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error in 24V detection: {e}")
            print(traceback.format_exc())
        finally:
            if 'ser' in locals():
                ser.close()
