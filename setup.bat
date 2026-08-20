@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "PYTHON_CMD=python"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "MODELS_DIR=%PROJECT_DIR%\models"

echo ========================================
echo   Descarga de modelos MediaPipe
echo   (Gesture-control - Windows)
echo ========================================
echo.

:: Verificar Python
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: No se encontro 'python' en el PATH.
    echo Descarga Python 3.12 desde https://www.python.org/downloads/
    echo y marca "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('%PYTHON_CMD% --version') do set "PYTHON_VERSION=%%a"
echo [OK] Python encontrado: %PYTHON_VERSION%

:: Escribir script temporal de descarga
echo.
echo [+] Preparando descarga...
set "TEMP_SCRIPT=%PROJECT_DIR%\_download_models_tmp.py"

(
echo import os, sys
^&echo from pathlib import Path
^&echo from urllib.request import urlretrieve
^&echo.
^&echo models_dir = Path(r"%MODELS_DIR%")
^&echo models_dir.mkdir(parents=True, exist_ok=True)
^&echo.
^&echo MODELS = {
^&echo     "pose_landmarker_lite.task": (
^&echo         "https://storage.googleapis.com/mediapipe-models/"
^&echo         "pose_landmarker/pose_landmarker_lite/float16/latest/"
^&echo         "pose_landmarker_lite.task"
^&echo     ),
^&echo     "pose_landmarker_full.task": (
^&echo         "https://storage.googleapis.com/mediapipe-models/"
^&echo         "pose_landmarker/pose_landmarker_full/float16/latest/"
^&echo         "pose_landmarker_full.task"
^&echo     ),
^&echo     "hand_landmarker.task": (
^&echo         "https://storage.googleapis.com/mediapipe-models/"
^&echo         "hand_landmarker/hand_landmarker/float16/latest/"
^&echo         "hand_landmarker.task"
^&echo     ),
^&echo }
^&echo.
^&echo def report(block_num, block_size, total_size):
^&echo     downloaded = block_num * block_size
^&echo     if total_size ^> 0:
^&echo         pct = min(100, downloaded * 100 // total_size)
^&echo         bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
^&echo         sys.stdout.write(f"\r  [{bar}] {pct}%%")
^&echo         sys.stdout.flush()
^&echo     else:
^&echo         sys.stdout.write(f"\r  Descargado: {downloaded // 1024} KB")
^&echo         sys.stdout.flush()
^&echo.
^&echo for filename, url in MODELS.items():
^&echo     dest = models_dir / filename
^&echo     if dest.exists():
^&echo         print(f"\n✓ {filename} ya existe ({dest.stat().st_size / 1024 / 1024:.1f} MB) — omitiendo")
^&echo         continue
^&echo     print(f"\nDescargando {filename}...")
^&echo     urlretrieve(url, dest, reporthook=report)
^&echo     print(f"\n✓ {filename} descargado ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
^&echo.
^&echo print("\nTodos los modelos estan listos.")
) > "%TEMP_SCRIPT%"

:: Ejecutar descarga
echo [+] Descargando modelos de MediaPipe...
%PYTHON_CMD% "%TEMP_SCRIPT%"
if errorlevel 1 (
    echo ERROR: Fallo la descarga de modelos.
    del "%TEMP_SCRIPT%" >nul 2>&1
    pause
    exit /b 1
)

:: Limpiar script temporal
del "%TEMP_SCRIPT%" >nul 2>&1

:: Verificacion
echo.
echo ========================================
echo   Verificacion
echo ========================================

if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" -c "import mediapipe; print('  MediaPipe:', mediapipe.__version__)" 2>nul || echo   MediaPipe: no instalado en el venv
    "%VENV_DIR%\Scripts\python.exe" -c "import cv2; print('  OpenCV:', cv2.__version__)" 2>nul || echo   OpenCV: no instalado en el venv
    "%VENV_DIR%\Scripts\python.exe" -c "import numpy; print('  NumPy:', numpy.__version__)" 2>nul || echo   NumPy: no instalado en el venv
) else (
    echo   (Virtualenv no encontrado; se omitio la verificacion de dependencias)
)

echo.
echo ========================================
echo   Modelos listos
echo ========================================
echo.

echo Si aun no has creado el entorno virtual, ejecuta estos comandos:
echo.
echo   1. Crear virtualenv con Python 3.12:
echo      python -m venv .venv
echo.
echo   2. Activar el entorno (CMD):
echo      .venv\Scripts\activate.bat
echo.
echo      O en PowerShell:
echo      .venv\Scripts\Activate.ps1
echo.
echo   3. Instalar dependencias:
echo      pip install -r requirements.txt
echo.
echo   4. Ejecutar el proyecto:
echo      python main.py
echo.

pause
