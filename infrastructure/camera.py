from __future__ import annotations

import cv2
import numpy as np

from config.settings import CameraConfig


class CameraError(RuntimeError):
    """La cámara no se pudo abrir o falló durante la captura."""


class Camera:
    """Wrapper de cv2.VideoCapture con configuración inicial."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self._config.index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"No se pudo abrir la camara {self._config.index}")

        # Forzar MJPEG para maximizar FPS en webcams integradas
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

        self._cap = cap

    def read(self) -> np.ndarray:
        """Devuelve un fotograma BGR."""
        if self._cap is None:
            raise CameraError("La cámara no está abierta; llama antes a open().")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Fallo al leer un fotograma de la cámara.")
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
