#!/usr/bin/env python3
"""
Wheel Inspection System - Executable Builder
============================================

This script automates the creation of a standalone executable for the 
Wheel Inspection System that can run on any Windows system without 
requiring Python or dependencies to be pre-installed.

Usage:
    python build_exe.py

Requirements:
    - PyInstaller
    - All project dependencies listed in requirements.txt
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 40)

def check_dependencies():
    """Check if all required dependencies are available"""
    print_step(1, "Checking Dependencies")
    
    required_modules = [
        'PyInstaller', 'torch', 'torchvision', 'cv2', 'pyrealsense2', 
        'tkinter', 'numpy', 'PIL', 'serial'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                import PIL
            elif module == 'serial':
                import serial
            elif module == 'pyrealsense2':
                import pyrealsense2
            elif module == 'PyInstaller':
                import PyInstaller
            else:
                __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Please install missing dependencies using:")
        print("pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are available!")
    return True

def validate_files():
    """Validate that all required files exist"""
    print_step(2, "Validating Required Files")
    
    required_files = [
        'wheel_main.py',
        'wheel_main.spec',
        'maskrcnn_wheel_best.pth',
        'camera_intrinsics.json',
        'settings.json',
        'taurus.ico',
        'config_manager.py',
        'image_processing.py',
        'camera_streams.py',
        'signal_handler.py',
        'database.py',
        'utils.py',
        'settings_window.py',
        'reports_window.py',
        'app_icon.py',
        'camera_utils.py'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            if file == 'maskrcnn_wheel_best.pth':
                size_mb = size / (1024 * 1024)
                print(f"✓ {file} ({size_mb:.1f} MB)")
            else:
                size_kb = size / 1024
                print(f"✓ {file} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("\n✅ All required files are present!")
    return True

def clean_build_directories():
    """Clean previous build artifacts"""
    print_step(3, "Cleaning Build Directories")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            print(f"🗑️  Removing {dir_name}/")
            shutil.rmtree(dir_name)
        else:
            print(f"ℹ️  {dir_name}/ not found (already clean)")
    
    print("\n✅ Build directories cleaned!")

def build_executable():
    """Build the executable using PyInstaller"""
    print_step(4, "Building Executable")
    
    print("🔨 Running PyInstaller...")
    print("This may take several minutes...")
    
    start_time = time.time()
    
    try:
        # Run PyInstaller with the spec file
        result = subprocess.run([
            sys.executable, '-m', 'PyInstaller', 
            '--clean',  # Clean PyInstaller cache
            'wheel_main.spec'
        ], capture_output=True, text=True)
        
        build_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Build completed successfully in {build_time:.1f} seconds!")
            return True
        else:
            print(f"❌ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build failed with exception: {e}")
        return False

def validate_executable():
    """Validate the created executable"""
    print_step(5, "Validating Executable")
    
    exe_path = Path('dist/WheelInspectionSystem.exe')
    
    if not exe_path.exists():
        print("❌ Executable not found!")
        return False
    
    exe_size = exe_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Executable created: {exe_path}")
    print(f"📁 Size: {exe_size:.1f} MB")
    
    # Check if all data files are included
    dist_dir = Path('dist')
    required_data_files = [
        'camera_intrinsics.json',
        'settings.json', 
        'maskrcnn_wheel_best.pth',
        'taurus.ico'
    ]
    
    print("\n📂 Checking bundled data files:")
    for file in required_data_files:
        if (dist_dir / file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
    
    return True

def create_deployment_package():
    """Create a deployment package with additional files"""
    print_step(6, "Creating Deployment Package")
    
    # Create deployment directory
    deploy_dir = Path('deployment')
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # Copy executable
    shutil.copy('dist/WheelInspectionSystem.exe', deploy_dir)
    
    # Copy any additional runtime files that might be needed
    runtime_files = [
        'README_DEPLOYMENT.md',  # We'll create this
    ]
    
    # Create README for deployment
    readme_content = """# Wheel Inspection System - Deployment Guide

## System Requirements
- Windows 10/11 (64-bit)
- Minimum 8GB RAM (16GB recommended)
- USB 3.0 ports for cameras
- At least 2GB free disk space

## Installation
1. Copy all files to your desired installation directory
2. Run WheelInspectionSystem.exe
3. Configure cameras and settings through the application interface

## First Run
- The application will create configuration files automatically
- Ensure cameras are connected before starting inspection
- Check serial/Modbus settings in the Settings window

## Troubleshooting
- If the application fails to start, check Windows Event Viewer for errors
- Ensure no antivirus software is blocking the executable
- For camera issues, verify USB connections and drivers

## Support
Contact your system administrator for technical support.
"""
    
    with open(deploy_dir / 'README_DEPLOYMENT.md', 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Deployment package created in: {deploy_dir}")
    print(f"📁 Package contents:")
    for item in deploy_dir.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   📄 {item.name} ({size_mb:.1f} MB)")
    
    return True

def main():
    """Main build process"""
    print_header("Wheel Inspection System - Executable Builder")
    print("Building standalone executable for Windows deployment...")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return False
    
    # Step 2: Validate files
    if not validate_files():
        return False
    
    # Step 3: Clean build directories
    clean_build_directories()
    
    # Step 4: Build executable
    if not build_executable():
        return False
    
    # Step 5: Validate executable
    if not validate_executable():
        return False
    
    # Step 6: Create deployment package
    create_deployment_package()
    
    print_header("BUILD COMPLETED SUCCESSFULLY!")
    print("🎉 Your Wheel Inspection System executable is ready!")
    print("\n📁 Output locations:")
    print("   • Executable: dist/WheelInspectionSystem.exe")
    print("   • Deployment package: deployment/")
    print("\n🚀 Next steps:")
    print("   1. Test the executable on your development machine")
    print("   2. Copy the deployment package to target machines")
    print("   3. Run WheelInspectionSystem.exe on the target system")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Build process failed!")
        sys.exit(1)
    else:
        print("\n✅ Build process completed successfully!")
        sys.exit(0) 