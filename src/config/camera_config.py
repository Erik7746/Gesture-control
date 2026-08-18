"""Configuración para la captura de video desde cámara."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraConfig:
    """Parámetros configurables para la captura de cámara.

    Attributes:
        device_index: Índice del dispositivo de video (0 = webcam integrada,
            1, 2, ... = cámaras externas / USB).
        width: Resolución horizontal deseada en píxeles.
        height: Resolución vertical deseada en píxeles.
        fps: Cuadros por segundo deseados.
        backend: Backend de OpenCV a utilizar (p.ej. cv2.CAP_DSHOW).
            Si es None, OpenCV elegirá el backend por defecto del sistema.
    """

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: Optional[int] = None
