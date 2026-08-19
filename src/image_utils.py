from __future__ import annotations

import logging

import cv2
import numpy as np

from mediapipe.tasks.python.vision.core.image import Image as MPImage
from mediapipe.tasks.python.vision.core.image import ImageFormat

logger = logging.getLogger(__name__)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convierte un frame BGR de OpenCV a RGB contiguo."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Se esperaba imagen BGR de 3 canales, se obtuvo {frame.shape}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    return rgb


def create_mp_image(rgb_frame: np.ndarray) -> MPImage:
    """Crea un objeto MediaPipe Image desde un array RGB de NumPy."""
    return MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Extrae una región del frame.

    Args:
        frame: Imagen original.
        roi: (x1, y1, x2, y2) en coordenadas del frame.

    Returns:
        Imagen recortada (copia).
    """
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    # Clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        logger.warning("ROI inválido después de clamp: %s", roi)
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()


def resize_for_model(image: np.ndarray, target_size: int) -> np.ndarray:
    """Redimensiona una imagen a un tamaño cuadrado fijo.

    Args:
        image: Imagen de entrada.
        target_size: Ancho y alto deseados (ej. 224).

    Returns:
        Imagen redimensionada.
    """
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def transform_normalized_landmarks(
    landmarks: list,
    roi: tuple[int, int, int, int],
) -> list[tuple[float, float, float]]:
    """Convierte landmarks normalizados [0,1] del ROI a coordenadas del frame original.

    MediaPipe HandLandmarker devuelve coordenadas normalizadas respecto a la imagen
    de entrada. Si la imagen de entrada es un ROI redimensionado, las coordenadas
    normalizadas se interpretan directamente sobre las dimensiones *originales* del
    ROI antes del resize, porque la normalización es lineal.

    Fórmula:
        x_original = roi_x1 + landmark.x * roi_width
        y_original = roi_y1 + landmark.y * roi_height
        z se mantiene (es relativo a la profundidad de la mano, no al frame).

    Args:
        landmarks: Lista de objetos NormalizedLandmark.
        roi: (x1, y1, x2, y2) del ROI original en el frame.

    Returns:
        Lista de tuplas (x, y, z) en coordenadas absolutas del frame.
    """
    x1, y1, x2, y2 = roi
    roi_w = x2 - x1
    roi_h = y2 - y1
    out = []
    for lm in landmarks:
        out.append(
            (
                x1 + lm.x * roi_w,
                y1 + lm.y * roi_h,
                lm.z * max(roi_w, roi_h),  # escalar z proporcionalmente al ROI
            )
        )
    return out
