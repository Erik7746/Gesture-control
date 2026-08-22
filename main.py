from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from config.settings import Config, CameraConfig, ModelConfig, ROIConfig, TrackingConfig
from infrastructure.camera import Camera, CameraError
from infrastructure.timing import FPSLimiter, FPSCounter
from detection.image_pipeline import bgr_to_rgb, create_mp_image, crop_roi
from core.state_machine import HandTrackingStateMachine
from detection.pose_detector import PoseDetector
from detection.hand_detector import HandDetector
from presentation.visualizer import Visualizer
from utils.logging_config import configure_logging


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


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        camera=CameraConfig(
            index=args.camera,
            frame_width=args.width,
            frame_height=args.height,
            target_fps=args.fps,
        ),
        models=ModelConfig(),
        roi=ROIConfig(),
        tracking=TrackingConfig(),
        log_level=args.log_level,
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = build_config(args)

    logger.info(
        "Iniciando | cámara=%s res=%sx%s fps=%s",
        config.camera.index,
        config.camera.frame_width,
        config.camera.frame_height,
        config.camera.target_fps,
    )

    fps_limiter = FPSLimiter(config.camera.target_fps)
    fps_counter = FPSCounter()
    visualizer = Visualizer()

    state_machine = HandTrackingStateMachine(
        tracking_config=config.tracking,
        roi_config=config.roi,
        model_config=config.models,
    )

    # Callbacks: actualizan el state_machine desde hilos internos de MediaPipe
    def pose_callback(result, *cb_args):
        state_machine.on_pose_result(result)

    def hand_callback(result, *cb_args):
        state_machine.on_hand_result(result)

    try:
        with Camera(config.camera) as camera:
            with PoseDetector(config.models, pose_callback) as pose_detector:
                with HandDetector(config.models, hand_callback) as hand_detector:
                    # Timestamp base para monotonicidad
                    start_time_ns = time.time_ns()

                    while True:
                        fps_limiter.wait()

                        frame = camera.read()
                        if config.camera.mirror:
                            frame = cv2.flip(frame, 1)

                        # Preparar imagen RGB para MediaPipe
                        rgb_full = bgr_to_rgb(frame)
                        mp_image_full = create_mp_image(rgb_full)

                        # Timestamp monotónico en ms
                        timestamp_ms = (time.time_ns() - start_time_ns) // 1_000_000
                        if timestamp_ms <= 0:
                            timestamp_ms = 1

                        # ── Pose detection (asíncrono, opcional) ─────────────
                        if state_machine.should_run_pose_detection():
                            pose_detector.detect_async(mp_image_full, timestamp_ms)

                        # ── Hand detection sobre ROI ─────────────────────────
                        roi = state_machine.get_roi_for_hand_detection(frame.shape)
                        state_machine.set_current_hand_roi(roi)

                        # Crop del ROI (sin resize forzado para preservar calidad)
                        roi_rgb = crop_roi(rgb_full, roi)
                        mp_image_roi = create_mp_image(roi_rgb)

                        hand_detector.detect_async(mp_image_roi, timestamp_ms)

                        # ── Visualización (estado conocido hasta ahora) ──────
                        fps_counter.tick()
                        real_fps = fps_counter.fps

                        display = visualizer.draw(
                            frame=frame.copy(),
                            hands=state_machine.hands,
                            pose_result=state_machine.pose_result,
                            roi=roi,
                            tracker_state=state_machine.state.name,
                            tracking_mode=state_machine.tracking_mode.name,
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
