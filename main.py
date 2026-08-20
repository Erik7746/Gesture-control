from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from src.config import Config
from src.camera import Camera, CameraError
from src.fps import FPSLimiter, FPSCounter
from src.image_utils import bgr_to_rgb, create_mp_image, crop_roi, resize_for_model
from src.tracker import Tracker
from src.pose_detector import PoseDetector
from src.hand_detector import HandDetector
from src.visualizer import Visualizer


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sistema de detección de manos a distancia")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Índice de la cámara (default: 0)")
    parser.add_argument("--fps", type=int, default=30, help="FPS objetivo (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Ancho del frame (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Alto del frame (default: 480)")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging (default: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = Config(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.fps,
        log_level=args.log_level,
    )

    logger.info(
        "Iniciando | cámara=%s res=%sx%s fps=%s",
        config.camera_index,
        config.frame_width,
        config.frame_height,
        config.target_fps,
    )

    fps_limiter = FPSLimiter(config.target_fps)
    fps_counter = FPSCounter()
    visualizer = Visualizer()

    tracker = Tracker(config)

    # Callbacks: actualizan el tracker desde hilos internos de MediaPipe
    def pose_callback(result, *cb_args):
        tracker.on_pose_result(result)

    def hand_callback(result, *cb_args):
        tracker.on_hand_result(result)

    try:
        with Camera(config) as camera:
            with PoseDetector(config, pose_callback) as pose_detector:
                with HandDetector(config, hand_callback) as hand_detector:
                    # Timestamp base para monotonicidad
                    start_time_ns = time.time_ns()

                    while True:
                        fps_limiter.wait()

                        frame = camera.read()
                        if config.mirror:
                            frame = cv2.flip(frame, 1)

                        # Preparar imagen RGB para MediaPipe
                        rgb_full = bgr_to_rgb(frame)
                        mp_image_full = create_mp_image(rgb_full)

                        # Timestamp monotónico en ms
                        timestamp_ms = (time.time_ns() - start_time_ns) // 1_000_000
                        if timestamp_ms <= 0:
                            timestamp_ms = 1

                        # ── Pose detection (asíncrono, opcional) ─────────────
                        if tracker.should_run_pose_detection():
                            pose_detector.detect_async(mp_image_full, timestamp_ms)

                        # ── Hand detection sobre ROI ─────────────────────────
                        roi = tracker.get_roi_for_hand_detection(frame.shape)
                        tracker.set_current_hand_roi(roi)

                        # Crop + resize del ROI
                        roi_rgb = crop_roi(rgb_full, roi)
                        roi_resized = resize_for_model(roi_rgb, config.hand_input_size)
                        mp_image_roi = create_mp_image(roi_resized)

                        hand_detector.detect_async(mp_image_roi, timestamp_ms)

                        # ── Visualización (estado conocido hasta ahora) ──────
                        fps_counter.tick()
                        real_fps = fps_counter.fps

                        display = visualizer.draw(
                            frame=frame.copy(),
                            hand_left=tracker.hand_left,
                            hand_right=tracker.hand_right,
                            pose_result=tracker.pose_result,
                            roi=roi,
                            tracker_state=tracker.state,
                            fps=real_fps,
                        )

                        cv2.imshow("Gesture Control", display)

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
