# Debug Message Control System

The Wheel Inspection System now has a comprehensive debug message control system that allows you to enable/disable debug messages as needed.

## Quick Control (Production Mode)

By **default**, all debug messages are **DISABLED** (production mode). The terminal will be clean with no debug output.

## How to Enable Debug Messages

### Method 1: Edit the debug_utils.py file
1. Open `debug_utils.py`
2. Change `DEBUG_MODE = False` to `DEBUG_MODE = True`
3. Restart the application

### Method 2: Use Python Console (while app is running)
```python
# Enable all debug messages
from debug_utils import debug_on
debug_on()

# Disable all debug messages  
from debug_utils import debug_off
debug_off()

# Toggle debug mode on/off
from debug_utils import debug_toggle
debug_toggle()
```

### Method 3: Direct function calls (in code)
```python
from debug_utils import enable_debug_mode, disable_debug_mode
enable_debug_mode()   # Turn on debug messages
disable_debug_mode()  # Turn off debug messages
```

## Debug Categories

When debug mode is enabled, you can see different types of messages:

- **[STARTUP]** - AI model loading, initialization messages
- **[CAMERAS]** - Camera start/stop, streaming messages  
- **[PROCESSING]** - Image processing, measurement cycle messages
- **[DATABASE]** - Database operations, save confirmations
- **[MODBUS]** - Modbus communication messages
- **[PERFORMANCE]** - Performance timing and cycle metrics
- **[THERMAL]** - Camera heat management messages
- **[ERRORS]** - Error messages and warnings
- **[UI]** - UI updates and model parameter changes
- **[SIGNAL]** - Signal detection and processing messages

## Selective Category Control (Advanced)

You can enable/disable specific categories:

```python
from debug_utils import set_debug_category

# Enable only error messages
set_debug_category("errors", True)
set_debug_category("startup", False)

# Enable only signal detection messages
set_debug_category("signal", True)
```

## Production Deployment

For production deployment:
1. Ensure `DEBUG_MODE = False` in `debug_utils.py`
2. This eliminates all debug output and improves performance
3. Debug can still be enabled later if needed for troubleshooting

## Benefits

- **Performance**: Debug messages are completely bypassed when disabled
- **Clean Terminal**: No debug clutter in production
- **Flexible Control**: Enable only what you need
- **Easy Toggle**: Quick enable/disable without code changes
- **Categorized Output**: Organized debug information by type

## Files Modified

- `wheel_main.py` - Main application debug messages controlled
- `utils.py` - Utility function debug messages controlled  
- `signal_handler.py` - Signal detection debug messages controlled
- `debug_utils.py` - Centralized debug control system (NEW)

## Quick Reference

| Action | Code |
|--------|------|
| Enable debug | `from debug_utils import debug_on; debug_on()` |
| Disable debug | `from debug_utils import debug_off; debug_off()` |
| Toggle debug | `from debug_utils import debug_toggle; debug_toggle()` |
| Check status | `from debug_utils import get_debug_status; print(get_debug_status())` 