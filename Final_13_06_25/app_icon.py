"""
This module provides functions to apply custom icons to Tkinter windows.
"""
import os
import tkinter as tk
from PIL import Image, ImageTk

def set_app_icon(window, icon_path=None):
    """
    Set a custom icon for a Tkinter window.
    
    Args:
        window: The Tkinter window object
        icon_path: Path to the icon file (.ico for Windows or .png/.gif)
    
    Returns:
        True if successful, False otherwise
    """
    # If no icon path specified, use default
    if not icon_path:
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taurus.ico")
    
    # If the icon doesn't exist, try a PNG as fallback
    if not os.path.exists(icon_path):
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taurus.png")
    
    if not os.path.exists(icon_path):
        print(f"Warning: App icon not found at {icon_path}")
        return False
    
    try:
        # Different handling for Windows (.ico) vs other platforms
        if icon_path.lower().endswith('.ico'):
            window.iconbitmap(icon_path)
        else:
            # For other image formats
            icon_img = Image.open(icon_path)
            icon_photo = ImageTk.PhotoImage(icon_img)
            window.tk.call('wm', 'iconphoto', window._w, icon_photo)
        
        return True
    except Exception as e:
        print(f"Error setting app icon: {e}")
        return False
