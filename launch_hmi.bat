@echo off
rem ===========================================================================
rem  launch_hmi.bat - start the PyQt6 defect-inspection HMI without a console.
rem
rem  Purpose:
rem    Launches hmi\app.py with pythonw.exe so no console window appears.
rem
rem  Requirements:
rem    - Windows
rem    - Python 3 with PyQt6, opencv-python, numpy, onnxruntime installed
rem    - exports\best_fp32.onnx present (the HMI loads it at startup)
rem
rem  Interpreter lookup order:
rem    1) .venv\Scripts\pythonw.exe   (project-local virtual environment)
rem    2) first pythonw.exe found on PATH (via "where pythonw")
rem    3) otherwise print an error and pause
rem
rem  Usage:
rem    Double-click this file, or run it from a terminal.
rem    Startup diagnostics are written to %TEMP%\hmi_launcher.log because
rem    pythonw has no console and its stdout/stderr would otherwise be lost.
rem
rem  Notes:
rem    This file is intentionally ASCII-only. .bat files are parsed with the
rem    OEM/GBK code page on Windows, so non-ASCII comments can be garbled.
rem
rem    "start /b" is required for the log to work. A plain "start" gives the
rem    child its own console and silently drops the redirection, leaving the
rem    log empty. With /b the redirection stays, and since pythonw is a
rem    GUI-subsystem binary it never attaches to a console, so no extra
rem    window appears and this launcher can exit immediately.
rem ===========================================================================

setlocal enabledelayedexpansion

rem --- switch to the directory this script lives in ---------------------------
cd /d "%~dp0"

set "APP=hmi\app.py"
set "LOG=%TEMP%\hmi_launcher.log"

rem --- sanity check: the application entry point must exist -------------------
if not exist "%APP%" (
    echo [ERROR] %APP% not found.
    echo [ERROR] Run this script from the repository root.
    pause
    exit /b 1
)

rem --- locate a pythonw.exe ---------------------------------------------------
set "PYW="

if exist ".venv\Scripts\pythonw.exe" (
    set "PYW=%CD%\.venv\Scripts\pythonw.exe"
    echo [INFO] Using project virtual environment: "!PYW!"
    goto :found
)

rem "where" prints every match, one path per line. Capture it to a temp file
rem and read the first line: simpler and more portable than piping a nested
rem command into "for /f".
set "PYWLIST=%TEMP%\hmi_pythonw.txt"
where pythonw > "%PYWLIST%" 2>nul
for /f "usebackq delims=" %%I in ("%PYWLIST%") do (
    if not defined PYW set "PYW=%%I"
)
del "%PYWLIST%" >nul 2>&1

if not defined PYW (
    echo [ERROR] pythonw.exe was not found.
    echo [ERROR] Looked for .venv\Scripts\pythonw.exe and on PATH.
    echo [ERROR] Install Python 3 and add it to PATH, or create .venv in this
    echo [ERROR] directory and install the requirements.
    pause
    exit /b 1
)
echo [INFO] Using pythonw from PATH: "!PYW!"

:found
rem --- launch -----------------------------------------------------------------
rem pythonw has no console, so stdout/stderr are None inside the process.
rem Redirect both streams to a log file to keep startup errors inspectable.
echo [INFO] Log file: "%LOG%"
echo [INFO] Starting %APP% ...

start "" /b "!PYW!" "%APP%" > "%LOG%" 2>&1

echo [INFO] Launched. If the window does not appear, check "%LOG%".
endlocal
exit /b 0
