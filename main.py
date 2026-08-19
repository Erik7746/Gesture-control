from __future__ import annotations

import argparse
import logging
import sys

import cv2

from src.config import Config
from src.camera import Camera, CameraError
from src.fps import FPSLimiter, FPSCounter


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sistema de reconocimiento de gestos")
    parser.add_argument(
        "--camera", "-c", type=int, default=0, help="Indice de la camara (default: 0)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="FPS objetivo del bucle principal (default: 30)"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Ancho de la camara en pixeles (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Alto de la camara en pixeles (default: 480)"
    )
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
        "Iniciando con camara=%s, resolucion=%sx%s, target_fps=%s",
        config.camera_index,
        config.frame_width,
        config.frame_height,
        config.target_fps,
    )

    fps_limiter = FPSLimiter(config.target_fps)
    fps_counter = FPSCounter()

    try:
        with Camera(config) as camera:
            while True:
                fps_limiter.wait()

                frame = camera.read()

                if config.mirror:
                    frame = cv2.flip(frame, 1)

                fps_counter.tick()
                real_fps = fps_counter.fps

                # Mostrar FPS en pantalla
                cv2.putText(
                    frame,
                    f"FPS: {real_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Gesture Control", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except CameraError as exc:
        logger.error("Error de camara: %s", exc)
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
