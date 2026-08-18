"""Captura de video desde cámara con OpenCV.

Este módulo encapsula la apertura, configuración y lectura de frames
provenientes de dispositivos de video. Permite seleccionar cámaras
externas por índice y ajustar resolución y FPS de forma configurable.
"""

import logging
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from src.config.camera_config import CameraConfig

logger = logging.getLogger(__name__)


class CameraCaptureError(Exception):
    """Error relacionado con la captura de video desde cámara."""


class CameraCapture:
    """Gestiona la captura de video de un dispositivo de cámara.

    La clase abre la cámara una sola vez durante su inicialización,
    aplica la configuración solicitada y permite leer frames de forma
    secuencial. Implementa el protocolo de context manager para garantizar
    la liberación de recursos.
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        """Inicializa la captura con la configuración proporcionada.

        Args:
            config: Configuración de cámara. Si es None, se usan valores por
                defecto.
        """
        self.config = config if config is not None else CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        """Abre el dispositivo de video y aplica la configuración."""
        backend = self.config.backend if self.config.backend is not None else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.config.device_index, backend)

        if not self._cap.isOpened():
            raise CameraCaptureError(
                f"No se pudo abrir la cámara con índice {self.config.device_index}. "
                f"Verifica que el dispositivo esté conectado."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        logger.info(
            "Cámara %d abierta. Resolución real: %dx%d, FPS real: %.2f",
            self.config.device_index,
            self.actual_width,
            self.actual_height,
            self.actual_fps,
        )

    @property
    def is_opened(self) -> bool:
        """Indica si la cámara sigue abierta y disponible."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def actual_width(self) -> int:
        """Resolución horizontal real negociada con la cámara."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        """Resolución vertical real negociada con la cámara."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def actual_fps(self) -> float:
        """Cuadros por segundo real negociados con la cámara."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        """Resolución real como tupla (ancho, alto)."""
        return self.actual_width, self.actual_height

    def read(self) -> Optional[np.ndarray]:
        """Lee un frame de la cámara.

        Returns:
            Imagen capturada como array de NumPy, o None si la lectura
            falló o la cámara fue liberada.
        """
        if not self.is_opened:
            logger.warning("Intento de leer de una cámara cerrada.")
            return None

        success, frame = self._cap.read()
        if not success or frame is None:
            logger.warning("No se pudo leer un frame de la cámara.")
            return None

        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Generador que produce frames mientras la cámara esté abierta.

        Yields:
            Frames capturados desde la cámara.
        """
        while self.is_opened:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def release(self) -> None:
        """Libera el dispositivo de video y los recursos asociados."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Cámara %d liberada.", self.config.device_index)

    def __enter__(self) -> "CameraCapture":
        """Permite usar la clase como context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Libera recursos al salir del context manager."""
        self.release()

    def __del__(self) -> None:
        """Intenta liberar recursos si el objeto es destruido."""
        self.release()


def list_available_cameras(max_index: int = 10) -> List[int]:
    """Detecta los índices de cámaras disponibles en el sistema.

    Útil para descubrir cámaras externas conectadas además de la
    webcam integrada.

    Args:
        max_index: Índice máximo a probar (por defecto 0 a 9).

    Returns:
        Lista de índices numéricos de cámaras que lograron abrirse.
    """
    available: List[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        try:
            if cap.isOpened():
                available.append(index)
        finally:
            cap.release()
    return available
