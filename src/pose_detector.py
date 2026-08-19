from __future__ import annotations

import logging
from typing import Callable

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.core.image import Image as MPImage

from .config import Config

logger = logging.getLogger(__name__)


class PoseDetector:
    """Wrapper de PoseLandmarker en modo LIVE_STREAM.

    El resultado llega de forma asíncrona al callback configurado.
    """

    def __init__(self, config: Config, result_callback: Callable) -> None:
        self._config = config
        self._landmarker: PoseLandmarker | None = None

        base_options = BaseOptions(model_asset_path=str(config.pose_model_path))
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=config.min_pose_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_pose_tracking_confidence,
            result_callback=result_callback,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        logger.info("PoseLandmarker inicializado (LIVE_STREAM)")

    def detect_async(self, image: MPImage, timestamp_ms: int) -> None:
        """Envía un frame al detector asíncrono.

        Args:
            image: Imagen MediaPipe.
            timestamp_ms: Timestamp monotónicamente creciente en milisegundos.
        """
        if self._landmarker is None:
            raise RuntimeError("PoseLandmarker no inicializado.")
        try:
            self._landmarker.detect_async(image, timestamp_ms)
        except Exception as exc:
            logger.warning("Error en detect_async de pose: %s", exc)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("PoseLandmarker cerrado.")

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
