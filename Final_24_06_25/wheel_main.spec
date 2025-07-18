# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

# Get the current directory
current_dir = Path('.').absolute()

# Define data files to include
data_files = [
    # Configuration files
    ('camera_intrinsics.json', '.'),
    ('settings.json', '.'),
    ('wheel_inspection.db', '.'),
    
    # AI Model (critical for operation)
    ('maskrcnn_wheel_best.pth', '.'),
    
    # Icons and images
    ('taurus.ico', '.'),
    ('taurus.png', '.'),
    ('Taurus_logo.PNG', '.'),
    ('wheel_icon.ico', '.'),
    ('wheel_icon1.ico', '.'),
    
    # Create captured frames directory
    ('captured_frames1', 'captured_frames1'),
]

# Hidden imports for complex dependencies
hidden_imports = [
    # PyTorch and related
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchvision.models',
    'torchvision.models.detection',
    'torchvision.models.detection.mask_rcnn',
    'torchvision.models.detection.faster_rcnn',
    'torchvision.transforms',
    'torchvision.transforms.functional',
    
    # RealSense
    'pyrealsense2',
    
    # OpenCV
    'cv2',
    
    # Serial communication
    'serial',
    'serial.tools',
    'serial.tools.list_ports',
    
    # Database
    'sqlite3',
    
    # GUI
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.simpledialog',
    'tkcalendar',
    
    # Image processing
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    
    # Scientific computing
    'numpy',
    'scipy',
    'matplotlib',
    
    # System utilities
    'threading',
    'concurrent.futures',
    'queue',
    'gc',
    'traceback',
    'datetime',
    'time',
    'os',
    'sys',
    'math',
    'json',
    
    # Application modules
    'config_manager',
    'camera_streams',
    'image_processing',
    'signal_handler',
    'database',
    'utils',
    'settings_window',
    'reports_window',
    'app_icon',
    'camera_utils',
    'debug_utils',
]

# Collect binaries (DLLs and shared libraries)
binaries = []

# PyTorch CUDA libraries (if available)
try:
    import torch
    torch_dir = Path(torch.__file__).parent
    
    # Add CUDA libraries if available
    cuda_libs = list(torch_dir.glob("**/*.dll"))
    for lib in cuda_libs:
        if any(name in lib.name.lower() for name in ['cuda', 'cublas', 'curand', 'cusparse']):
            binaries.append((str(lib), '.'))
except ImportError:
    pass

# OpenCV libraries
try:
    import cv2
    cv2_dir = Path(cv2.__file__).parent
    opencv_libs = list(cv2_dir.glob("**/*.dll"))
    for lib in opencv_libs:
        binaries.append((str(lib), '.'))
except ImportError:
    pass

# RealSense libraries
try:
    import pyrealsense2 as rs
    rs_dir = Path(rs.__file__).parent
    rs_libs = list(rs_dir.glob("**/*.dll"))
    for lib in rs_libs:
        binaries.append((str(lib), '.'))
except ImportError:
    pass

a = Analysis(
    ['wheel_main.py'],
    pathex=[str(current_dir)],
    binaries=binaries,
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'matplotlib.tests',
        'numpy.tests',
        'torch.test',
        'PIL.tests',
        'cv2.tests',
    ],
    noarchive=False,
    optimize=0,
)

# Remove duplicate binaries
a.binaries = list(set(a.binaries))

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WheelInspectionSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX for better compatibility
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Disable console for production (set to True for debugging)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='taurus.ico',  # Set application icon
)
