# Wheel Inspection System - Build & Deployment Guide

## 🎯 Overview

This guide will help you create a standalone executable (.exe) of the Wheel Inspection System that can run on any Windows computer without requiring Python or dependencies to be pre-installed.

## 📋 Prerequisites

### Development Environment
- **Windows 10/11** (64-bit)
- **Python 3.8-3.11** (PyInstaller works best with these versions)
- **8GB+ RAM** (16GB recommended for large AI model packaging)
- **20GB+ free disk space** (for build artifacts and final executable)

### Hardware Dependencies
- **NVIDIA GPU** (optional but recommended for AI performance)
- **USB 3.0 ports** for cameras
- **RealSense D435 camera** support

## 🛠️ Step-by-Step Build Process

### Step 1: Environment Setup

1. **Clone/Download the project**
   ```bash
   # Navigate to your project directory
   cd /path/to/Final_19_06_25
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv wheel_inspection_env
   wheel_inspection_env\Scripts\activate
   ```

3. **Install build dependencies**
   ```bash
   # Option A: Install minimal dependencies for building
   pip install -r requirements_build.txt
   
   # Option B: Install all dependencies (larger but more compatible)
   pip install -r requirements.txt
   
   # Option C: Manual essential installation
   pip install PyInstaller torch torchvision opencv-python pyrealsense2 numpy scipy pyserial tkcalendar Pillow
   ```

### Step 2: GPU Support (Optional but Recommended)

For CUDA GPU acceleration:
```bash
# Install PyTorch with CUDA support (replace cu118 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

To check CUDA availability:
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

### Step 3: Pre-Build Validation

1. **Test the application**
   ```bash
   python wheel_main.py
   ```

2. **Verify all files are present**
   - All Python modules (.py files)
   - AI model (`maskrcnn_wheel_best.pth`)
   - Configuration files (`*.json`)
   - Icons (`*.ico`, `*.png`)
   - Database (`wheel_inspection.db`)

### Step 4: Build the Executable

Run the automated build script:
```bash
python build_exe.py
```

This script will:
- ✅ Check all dependencies
- ✅ Validate required files
- ✅ Clean previous build artifacts
- ✅ Build the executable with PyInstaller
- ✅ Validate the output
- ✅ Create deployment package

**Manual build (alternative):**
```bash
# Clean previous builds
rmdir /s build dist

# Build with PyInstaller
python -m PyInstaller --clean wheel_main.spec
```

### Step 5: Build Output

After successful build, you'll find:

```
📁 dist/
├── 📄 WheelInspectionSystem.exe    (Main executable ~500-800MB)
├── 📄 camera_intrinsics.json       (Camera calibration data)
├── 📄 settings.json                (Application settings)
├── 📄 maskrcnn_wheel_best.pth      (AI model ~168MB)
├── 📄 wheel_inspection.db          (Database)
├── 📄 taurus.ico                   (Application icon)
└── 📁 captured_frames1/            (Image capture directory)

📁 deployment/
├── 📄 WheelInspectionSystem.exe    (Copy of executable)
└── 📄 README_DEPLOYMENT.md         (End-user guide)
```

## 🚀 Deployment

### For Target Machines

**System Requirements:**
- Windows 10/11 (64-bit)
- 8GB+ RAM (16GB recommended)
- USB 3.0 ports for cameras
- 2GB+ free disk space

**Deployment Steps:**
1. Copy the entire `deployment/` folder to target machine
2. Run `WheelInspectionSystem.exe`
3. Configure cameras and settings through the GUI

### First Run Checklist
- [ ] Application starts without errors
- [ ] Camera connections are detected
- [ ] AI model loads successfully (check console output)
- [ ] Settings window opens and functions
- [ ] Database operations work
- [ ] Serial/Modbus communication functions

## 🔧 Troubleshooting

### Common Build Issues

**1. PyInstaller not found**
```bash
pip install PyInstaller
```

**2. CUDA/GPU errors during build**
- Build will continue with CPU-only version
- GPU features will be disabled but application will still work

**3. "Module not found" errors**
- Check that all dependencies are installed
- Run `pip list` to verify installations

**4. Large executable size (>1GB)**
- This is normal due to PyTorch and AI model
- Use `--exclude-module` in spec file to reduce size if needed

**5. Missing DLL errors on target machine**
- Install Microsoft Visual C++ Redistributable
- Ensure all required DLLs are bundled in spec file

### Runtime Issues on Target Machines

**1. Application won't start**
- Check Windows Event Viewer for detailed errors
- Ensure antivirus isn't blocking the executable
- Try running as Administrator

**2. Camera not detected**
- Install RealSense SDK on target machine
- Check USB 3.0 connectivity
- Verify camera drivers

**3. GPU acceleration not working**
- This is expected on machines without compatible NVIDIA GPUs
- Application will fall back to CPU processing

## ⚙️ Build Customization

### Reducing Executable Size

Edit `wheel_main.spec` to exclude unnecessary modules:

```python
excludes=[
    'matplotlib.tests',
    'numpy.tests', 
    'torch.test',
    'PIL.tests',
    'cv2.tests',
    'scipy.sparse',  # If not used
    'scipy.stats',   # If not used
],
```

### Debug vs Production Build

**Debug Build (console enabled):**
```python
console=True,   # Shows console for debugging
debug=True,     # More verbose output
```

**Production Build (no console):**
```python
console=False,  # Hide console window
debug=False,    # Minimal output
```

### Including Additional Files

Add to `data_files` in `wheel_main.spec`:
```python
data_files = [
    ('your_additional_file.txt', '.'),
    ('config_folder', 'config_folder'),
]
```

## 📊 Performance Considerations

### Executable Startup Time
- **First run**: 10-30 seconds (AI model loading)
- **Subsequent runs**: 5-15 seconds
- **GPU vs CPU**: GPU reduces processing time by 60-75%

### Memory Usage
- **Minimum**: 2GB RAM
- **Recommended**: 8GB RAM
- **With large models**: 16GB RAM

### Storage Requirements
- **Executable**: 500-800MB
- **Runtime data**: 100-500MB
- **Captured images**: Variable (depends on usage)

## 📝 Final Notes

1. **Test thoroughly** on different machines before deployment
2. **Document any specific requirements** for target environments
3. **Keep source code** for future updates and rebuilds
4. **Consider code signing** for enterprise deployment
5. **Plan for updates** - rebuilding is required for code changes

## 🆘 Support

If you encounter issues:
1. Check this guide first
2. Review console output and error messages
3. Test with minimal configuration
4. Verify hardware compatibility
5. Check Windows compatibility (some older versions may have issues)

---

**Last Updated**: January 2025
**Compatible with**: Windows 10/11, Python 3.8-3.11, PyInstaller 5.0+ 