@echo off
REM Wheel Inspection System - Quick Build Script
REM ============================================

echo.
echo ===============================================
echo  Wheel Inspection System - Quick Build
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8-3.11 and try again
    pause
    exit /b 1
)

echo Python found. Starting build process...
echo.

REM Run the build script
python build_exe.py

if errorlevel 1 (
    echo.
    echo ===============================================
    echo  BUILD FAILED!
    echo ===============================================
    echo Check the error messages above for details.
    pause
    exit /b 1
) else (
    echo.
    echo ===============================================
    echo  BUILD COMPLETED SUCCESSFULLY!
    echo ===============================================
    echo.
    echo Your executable is ready in the 'dist' folder.
    echo Deployment package is in the 'deployment' folder.
    echo.
    echo Press any key to open the dist folder...
    pause >nul
    explorer dist
)

exit /b 0 