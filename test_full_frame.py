from __future__ import annotations

import logging
import sys
import time

import cv2

from mediapipe.tasks.python.vision import HandLandmarkerResult

from src.config import Config
from src.camera import Camera, CameraError
from src.fps import FPSLimiter, FPSCounter
from src.image_utils import bgr_to_rgb, create_mp_image
from src.hand_detector import HandDetector
from src.tracker import HandState
from src.visualizer import Visualizer


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    config = Config(camera_index=0)
    fps_limiter = FPSLimiter(config.target_fps)
    fps_counter = FPSCounter()
    visualizer = Visualizer()

    # Estado compartido entre callback y bucle principal
    hands: list[HandState] = []
    frame_shape = (config.frame_height, config.frame_width, 3)

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

            # Frame completo: coordenadas normalizadas se convierten directamente a píxeles
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
        with Camera(config) as camera:
            with HandDetector(config, hand_callback) as hand_detector:
                start_time_ns = time.time_ns()
                logger.info("Test full-frame iniciado | num_hands=2 | sin pose | sin ROI")

                while True:
                    fps_limiter.wait()

                    frame = camera.read()
                    if config.mirror:
                        frame = cv2.flip(frame, 1)

                    frame_shape = frame.shape

                    # Frame completo RGB -> MediaPipe Image
                    rgb_full = bgr_to_rgb(frame)
                    mp_image_full = create_mp_image(rgb_full)

                    timestamp_ms = (time.time_ns() - start_time_ns) // 1_000_000
                    if timestamp_ms <= 0:
                        timestamp_ms = 1

                    # Pasar frame completo directamente al detector
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
