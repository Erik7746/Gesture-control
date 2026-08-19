from __future__ import annotations

import logging
from typing import Callable

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.core.image import Image as MPImage

from .config import Config

logger = logging.getLogger(__name__)


class HandDetector:
    """Wrapper de HandLandmarker en modo LIVE_STREAM.

    El resultado llega de forma asíncrona al callback configurado.
    """

    def __init__(self, config: Config, result_callback: Callable) -> None:
        self._config = config
        self._landmarker: HandLandmarker | None = None

        base_options = BaseOptions(model_asset_path=str(config.hand_model_path))
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=2,  # detectar ambas; el tracker elegirá la relevante
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_hand_tracking_confidence,
            result_callback=result_callback,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        logger.info("HandLandmarker inicializado (LIVE_STREAM)")

    def detect_async(self, image: MPImage, timestamp_ms: int) -> None:
        """Envía un frame o ROI al detector asíncrono.

        Args:
            image: Imagen MediaPipe (puede ser un ROI reescalado).
            timestamp_ms: Timestamp monotónicamente creciente en milisegundos.
        """
        if self._landmarker is None:
            raise RuntimeError("HandLandmarker no inicializado.")
        try:
            self._landmarker.detect_async(image, timestamp_ms)
        except Exception as exc:
            logger.warning("Error en detect_async de hand: %s", exc)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("HandLandmarker cerrado.")

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
