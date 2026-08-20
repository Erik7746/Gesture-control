@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "PYTHON_CMD=python"

echo ========================================
echo   Instalacion de Gesture-control
echo   (Windows)
echo ========================================
echo.

:: Verificar Python
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: No se encontro 'python' en el PATH.
    echo Descarga Python 3.12 desde https://www.python.org/downloads/
    echo y marca "Add Python to PATH" durante la instalacion.
    exit /b 1
)

for /f "tokens=*" %%a in ('%PYTHON_CMD% --version') do set "PYTHON_VERSION=%%a"
echo [OK] Python encontrado: %PYTHON_VERSION%

:: Crear virtualenv
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [OK] Virtualenv ya existe
) else (
    echo [+] Creando virtualenv...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: No se pudo crear el virtualenv
        exit /b 1
    )
    echo [OK] Virtualenv creado
)

:: Instalar dependencias
echo [+] Instalando dependencias...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 (
    echo ERROR: Fallo la instalacion de dependencias
    exit /b 1
)
echo [OK] Dependencias instaladas

:: Descargar modelos
echo [+] Descargando modelos...
"%VENV_DIR%\Scripts\python.exe" "%PROJECT_DIR%\scripts\download_models.py"
if errorlevel 1 (
    echo ERROR: Fallo la descarga de modelos
    exit /b 1
)

:: Verificacion
echo.
echo ========================================
echo   Verificacion
echo ========================================
"%VENV_DIR%\Scripts\python.exe" -c "import mediapipe; print('  MediaPipe:', mediapipe.__version__)"
"%VENV_DIR%\Scripts\python.exe" -c "import cv2; print('  OpenCV:', cv2.__version__)"
"%VENV_DIR%\Scripts\python.exe" -c "import numpy; print('  NumPy:', numpy.__version__)"

echo.
echo ========================================
echo   Instalacion completada
echo ========================================
echo.
echo Para ejecutar:
echo   1. Activa el entorno:   %VENV_DIR%\Scripts\activate.bat
echo   2. Ejecuta:             python main.py
echo.
pause
