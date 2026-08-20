#!/usr/bin/env bash
# Script de instalación del proyecto Gesture-control
# Uso: ./setup.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python3.12"
VENV_DIR="${PROJECT_DIR}/.venv"
MODELS_DIR="${PROJECT_DIR}/models"

echo "========================================"
echo "  Instalación de Gesture-control"
echo "========================================"
echo ""

# ── Verificar Python 3.12 ────────────────────────────────────────────────
if ! command -v "${PYTHON_CMD}" &> /dev/null; then
    echo "ERROR: ${PYTHON_CMD} no está instalado."
    echo "Instálalo con el gestor de paquetes de tu distro, por ejemplo:"
    echo "  sudo pacman -S python312   (Arch)"
    echo "  sudo apt install python3.12 (Debian/Ubuntu)"
    exit 1
fi

PYTHON_VERSION="$(${PYTHON_CMD} --version)"
echo "✓ Python encontrado: ${PYTHON_VERSION}"

# ── Crear virtualenv ─────────────────────────────────────────────────────
if [ -d "${VENV_DIR}" ]; then
    echo "✓ Virtualenv ya existe en ${VENV_DIR}"
else
    echo "→ Creando virtualenv en ${VENV_DIR}..."
    "${PYTHON_CMD}" -m venv "${VENV_DIR}"
    echo "✓ Virtualenv creado"
fi

# ── Activar e instalar dependencias ──────────────────────────────────────
echo "→ Instalando dependencias desde requirements.txt..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"
echo "✓ Dependencias instaladas"

# ── Descargar modelos ────────────────────────────────────────────────────
echo "→ Descargando modelos de MediaPipe..."
"${VENV_DIR}/bin/python" "${PROJECT_DIR}/scripts/download_models.py"

# ── Verificación rápida ──────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Verificación"
echo "========================================"
"${VENV_DIR}/bin/python" -c "import mediapipe; print('  MediaPipe:', mediapipe.__version__)"
"${VENV_DIR}/bin/python" -c "import cv2; print('  OpenCV:', cv2.__version__)"
"${VENV_DIR}/bin/python" -c "import numpy; print('  NumPy:', numpy.__version__)"

echo ""
echo "========================================"
echo "  Instalación completada"
echo "========================================"
echo ""
echo "Para activar el entorno:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Para ejecutar:"
echo "  python main.py"
echo ""
