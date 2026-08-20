#!/usr/bin/env python3
"""Descarga los modelos .task de MediaPipe necesarios para el proyecto."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# URLs oficiales de Google Cloud Storage
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


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Muestra una barra de progreso simple en consola."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
        sys.stdout.write(f"\r  [{bar}] {percent}%")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r  Descargado: {downloaded // 1024} KB")
        sys.stdout.flush()


def download_models(models_dir: Path | None = None) -> None:
    """Descarga todos los modelos definidos en MODELS."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de modelos: {models_dir}")

    for filename, url in MODELS.items():
        dest_path = models_dir / filename
        if dest_path.exists():
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} ya existe ({size_mb:.1f} MB) — omitiendo")
            continue

        print(f"\nDescargando {filename}...")
        try:
            urlretrieve(url, dest_path, reporthook=_report_progress)
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"\n✓ {filename} descargado ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"\n✗ Error descargando {filename}: {exc}")
            if dest_path.exists():
                dest_path.unlink()
            sys.exit(1)

    print("\nTodos los modelos están listos.")


if __name__ == "__main__":
    download_models()
