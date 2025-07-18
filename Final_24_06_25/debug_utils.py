# ========================================
# CENTRALIZED DEBUG SYSTEM
# ========================================
# This module provides a centralized debug system for all application modules
# Import this in any module that needs debug output control

# ========================================
# DEBUG CONFIGURATION
# ========================================
# Set DEBUG_MODE = True to enable all debug messages
# Set DEBUG_MODE = False to disable all debug messages (production mode)
DEBUG_MODE = False

# Individual debug categories (when DEBUG_MODE = True)
DEBUG_STARTUP = True         # AI model loading, initialization messages
DEBUG_CAMERAS = True         # Camera start/stop, streaming messages  
DEBUG_PROCESSING = True      # Image processing, measurement cycle messages
DEBUG_DATABASE = True        # Database operations, save confirmations
DEBUG_MODBUS = True          # Modbus communication messages
DEBUG_PERFORMANCE = True     # Performance timing and cycle metrics
DEBUG_THERMAL = True         # Camera heat management messages
DEBUG_ERRORS = True          # Error messages and warnings
DEBUG_UI = True              # UI updates and model parameter changes
DEBUG_UTILS = True           # Utility function messages
DEBUG_SIGNAL_HANDLER = True

# Performance optimization: Pre-compute debug states
_DEBUG_STATES = {
    'startup': DEBUG_MODE and DEBUG_STARTUP,
    'cameras': DEBUG_MODE and DEBUG_CAMERAS,
    'processing': DEBUG_MODE and DEBUG_PROCESSING,
    'database': DEBUG_MODE and DEBUG_DATABASE,
    'modbus': DEBUG_MODE and DEBUG_MODBUS,
    'performance': DEBUG_MODE and DEBUG_PERFORMANCE,
    'thermal': DEBUG_MODE and DEBUG_THERMAL,
    'errors': DEBUG_MODE and DEBUG_ERRORS,
    'ui': DEBUG_MODE and DEBUG_UI,
    'utils': DEBUG_MODE and DEBUG_UTILS,
    'signal': DEBUG_MODE and DEBUG_SIGNAL_HANDLER
}

def debug_print(message, category="general"):
    """Optimized debug print function with minimal overhead when disabled"""
    # Fast path: if debug is disabled, return immediately
    if not DEBUG_MODE:
        return
    
    # Check if this category is enabled
    if not _DEBUG_STATES.get(category, False):
        return
    
    # Only format the message if debug is enabled
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{category.upper()}] {message}")
    except Exception:
        # Fallback if timestamp fails
        print(f"[{category.upper()}] {message}")

def enable_debug_mode():
    """Enable debug mode and update states"""
    global DEBUG_MODE, _DEBUG_STATES
    DEBUG_MODE = True
    _DEBUG_STATES = {
        'startup': DEBUG_STARTUP,
        'cameras': DEBUG_CAMERAS,
        'processing': DEBUG_PROCESSING,
        'database': DEBUG_DATABASE,
        'modbus': DEBUG_MODBUS,
        'performance': DEBUG_PERFORMANCE,
        'thermal': DEBUG_THERMAL,
        'errors': DEBUG_ERRORS,
        'ui': DEBUG_UI,
        'utils': DEBUG_UTILS,
        'signal': DEBUG_SIGNAL_HANDLER
    }
    print("Debug mode enabled")

def disable_debug_mode():
    """Disable debug mode and update states"""
    global DEBUG_MODE, _DEBUG_STATES
    DEBUG_MODE = False
    _DEBUG_STATES = {k: False for k in _DEBUG_STATES}
    print("Debug mode disabled")

def toggle_debug_mode():
    """Toggle debug mode"""
    if DEBUG_MODE:
        disable_debug_mode()
    else:
        enable_debug_mode()

def debug_on():
    """Alias for enable_debug_mode"""
    enable_debug_mode()

def debug_off():
    """Alias for disable_debug_mode"""
    disable_debug_mode()

def debug_toggle():
    """Alias for toggle_debug_mode"""
    toggle_debug_mode()

def set_debug_category(category, enabled):
    """Enable/disable specific debug categories"""
    global DEBUG_STARTUP, DEBUG_CAMERAS, DEBUG_PROCESSING, DEBUG_DATABASE
    global DEBUG_MODBUS, DEBUG_PERFORMANCE, DEBUG_THERMAL, DEBUG_ERRORS
    global DEBUG_UI, DEBUG_UTILS, DEBUG_SIGNAL_HANDLER
    
    category_mapping = {
        "startup": "DEBUG_STARTUP",
        "cameras": "DEBUG_CAMERAS",
        "processing": "DEBUG_PROCESSING",
        "database": "DEBUG_DATABASE",
        "modbus": "DEBUG_MODBUS",
        "performance": "DEBUG_PERFORMANCE",
        "thermal": "DEBUG_THERMAL",
        "errors": "DEBUG_ERRORS",
        "ui": "DEBUG_UI",
        "utils": "DEBUG_UTILS",
        "signal": "DEBUG_SIGNAL_HANDLER"
    }
    
    if category in category_mapping:
        globals()[category_mapping[category]] = enabled
        debug_print(f"Debug category '{category}' {'enabled' if enabled else 'disabled'}")
    else:
        debug_print(f"Unknown debug category: {category}", "errors")

def get_debug_status():
    """Get current debug configuration status"""
    status = {
        "debug_mode": DEBUG_MODE,
        "categories": {
            "startup": DEBUG_STARTUP,
            "cameras": DEBUG_CAMERAS,
            "processing": DEBUG_PROCESSING,
            "database": DEBUG_DATABASE,
            "modbus": DEBUG_MODBUS,
            "performance": DEBUG_PERFORMANCE,
            "thermal": DEBUG_THERMAL,
            "errors": DEBUG_ERRORS,
            "ui": DEBUG_UI,
            "utils": DEBUG_UTILS,
            "signal": DEBUG_SIGNAL_HANDLER
        }
    }
    return status

# Backward compatibility - allow importing debug_print directly
__all__ = ['debug_print', 'enable_debug_mode', 'disable_debug_mode', 'toggle_debug_mode', 
           'debug_on', 'debug_off', 'debug_toggle', 'set_debug_category', 'get_debug_status'] 