from __future__ import annotations

import logging
import sys
import time

import cv2

from mediapipe.tasks.python.vision import HandLandmarkerResult

from config.settings import Config, CameraConfig, ModelConfig, ROIConfig, TrackingConfig
from infrastructure.camera import Camera, CameraError
from infrastructure.timing import FPSLimiter, FPSCounter
from detection.image_pipeline import bgr_to_rgb, create_mp_image
from detection.hand_detector import HandDetector
from core.types import HandState
from presentation.visualizer import Visualizer
from utils.logging_config import configure_logging


def main() -> int:
    configure_logging("INFO")
    logger = logging.getLogger(__name__)

    config = Config(
        camera=CameraConfig(),
        models=ModelConfig(),
        roi=ROIConfig(),
        tracking=TrackingConfig(),
    )
    fps_limiter = FPSLimiter(config.camera.target_fps)
    fps_counter = FPSCounter()
    visualizer = Visualizer()

    # Estado compartido entre callback y bucle principal
    hands: list[HandState] = []
    frame_shape = (config.camera.frame_height, config.camera.frame_width, 3)

    def hand_callback(result: HandLandmarkerResult, *args) -> None:
        """Callback que recibe landmarks normalizados [0,1] del frame completo."""
        nonlocal hands
        hands = []

        if not result.hand_landmarks:
            return

        h, w = frame_shape[:2]

        for idx, lm_norm in enumerate(result.hand_landmarks[:2]):  # máximo 2 manos
            score = 0.0
            if result.handedness and idx < len(result.handedness):
                score = result.handedness[idx][0].score

            landmarks = [
                (lm.x * w, lm.y * h, lm.z * max(w, h))
                for lm in lm_norm
            ]

            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
            size = max(max(xs) - min(xs), max(ys) - min(ys))

            hands.append(
                HandState(
                    landmarks=landmarks,
                    confidence=score,
                    center=center,
                    size=size,
                )
            )

    try:
        with Camera(config.camera) as camera:
            with HandDetector(config.models, hand_callback) as hand_detector:
                start_time_ns = time.time_ns()
                logger.info("Test full-frame iniciado | num_hands=2 | sin pose | sin ROI")

                while True:
                    fps_limiter.wait()

                    frame = camera.read()
                    if config.camera.mirror:
                        frame = cv2.flip(frame, 1)

                    frame_shape = frame.shape

                    rgb_full = bgr_to_rgb(frame)
                    mp_image_full = create_mp_image(rgb_full)

                    timestamp_ms = (time.time_ns() - start_time_ns) // 1_000_000
                    if timestamp_ms <= 0:
                        timestamp_ms = 1

                    hand_detector.detect_async(mp_image_full, timestamp_ms)

                    fps_counter.tick()

                    display = visualizer.draw(
                        frame=frame.copy(),
                        hands=hands,
                        pose_result=None,
                        roi=None,
                        tracker_state="TEST",
                        tracking_mode="full_frame",
                        fps=fps_counter.fps,
                    )

                    cv2.imshow("Test Full Frame", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

    except CameraError as exc:
        logger.error("Error de cámara: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Error inesperado: %s", exc)
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
