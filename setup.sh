#!/usr/bin/env bash
# Script de descarga de modelos para Gesture-control (Linux/macOS)
# Uso: ./setup.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python3"
VENV_DIR="${PROJECT_DIR}/.venv"
MODELS_DIR="${PROJECT_DIR}/models"

echo "========================================"
echo "  Descarga de modelos MediaPipe"
echo "  (Gesture-control)"
echo "========================================"
echo ""

# ── Verificar Python ─────────────────────────────────────────────────────
if ! command -v "${PYTHON_CMD}" &> /dev/null; then
    PYTHON_CMD="python"
    if ! command -v "${PYTHON_CMD}" &> /dev/null; then
        echo "ERROR: No se encontro 'python3' ni 'python' en el PATH."
        echo "Instala Python 3.12 desde el gestor de paquetes de tu distro."
        exit 1
    fi
fi

PYTHON_VERSION="$(${PYTHON_CMD} --version)"
echo "[OK] Python encontrado: ${PYTHON_VERSION}"

# ── Descargar modelos ────────────────────────────────────────────────────
echo ""
echo "→ Descargando modelos de MediaPipe..."

mkdir -p "${MODELS_DIR}"

${PYTHON_CMD} << 'PYEOF'
import os, sys
from pathlib import Path
from urllib.request import urlretrieve

models_dir = Path(os.environ.get("MODELS_DIR", "models"))
models_dir.mkdir(parents=True, exist_ok=True)

MODELS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/"
        "hand_landmarker.task"
    ),
}

def report(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
        sys.stdout.write(f"\r  [{bar}] {pct}%")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r  Descargado: {downloaded // 1024} KB")
        sys.stdout.flush()

for filename, url in MODELS.items():
    dest = models_dir / filename
    if dest.exists():
        print(f"\n✓ {filename} ya existe ({dest.stat().st_size / 1024 / 1024:.1f} MB) — omitiendo")
        continue
    print(f"\nDescargando {filename}...")
    urlretrieve(url, dest, reporthook=report)
    print(f"\n✓ {filename} descargado ({dest.stat().st_size / 1024 / 1024:.1f} MB)")

print("\nTodos los modelos estan listos.")
PYEOF

# ── Verificacion ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Verificacion"
echo "========================================"

if [ -f "${VENV_DIR}/bin/python" ]; then
    "${VENV_DIR}/bin/python" -c "import mediapipe; print('  MediaPipe:', mediapipe.__version__)" 2>/dev/null || echo "  MediaPipe: no instalado en el venv"
    "${VENV_DIR}/bin/python" -c "import cv2; print('  OpenCV:', cv2.__version__)" 2>/dev/null || echo "  OpenCV: no instalado en el venv"
    "${VENV_DIR}/bin/python" -c "import numpy; print('  NumPy:', numpy.__version__)" 2>/dev/null || echo "  NumPy: no instalado en el venv"
else
    echo "  (Virtualenv no encontrado; se omitio la verificacion de dependencias)"
fi

echo ""
echo "========================================"
echo "  Modelos listos"
echo "========================================"
echo ""

echo "Si aun no has creado el entorno virtual, ejecuta los siguientes comandos:"
echo ""
echo "  1. Crear virtualenv con Python 3.12:"
echo "     python3.12 -m venv .venv"
echo ""
echo "  2. Activar el entorno:"
echo "     source .venv/bin/activate"
echo ""
echo "  3. Instalar dependencias:"
echo "     pip install -r requirements.txt"
echo ""
echo "  4. Ejecutar el proyecto:"
echo "     python main.py"
echo ""
