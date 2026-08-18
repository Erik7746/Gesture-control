"""Punto de entrada del sistema de reconocimiento de gestos.

Por ahora solo inicializa la captura de cámara y muestra el video en
una ventana. Sirve como base para integrar posteriormente la detección
de manos y el reconocimiento de gestos.
"""

import argparse
import logging

import cv2

from src.camera.camera_capture import CameraCapture, list_available_cameras
from src.config.camera_config import CameraConfig
from src.utils.fps_counter import FpsCounter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Sistema de reconocimiento de gestos de manos."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Índice de la cámara a utilizar (0 = webcam integrada).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Resolución horizontal deseada.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Resolución vertical deseada.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Cuadros por segundo deseados.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Lista las cámaras disponibles y sale.",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta el bucle principal de captura y visualización."""
    args = parse_arguments()

    if args.list_cameras:
        cameras = list_available_cameras()
        logger.info("Cámaras disponibles: %s", cameras)
        return

    config = CameraConfig(
        device_index=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    try:
        with CameraCapture(config) as camera:
            logger.info(
                "Cámara lista. Resolución real: %dx%d, FPS real: %.2f",
                camera.actual_width,
                camera.actual_height,
                camera.actual_fps,
            )

            fps_counter = FpsCounter()

            for frame in camera.frames():
                current_fps = fps_counter.update()

                cv2.putText(
                    frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Gesture Control", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Tecla 'q' presionada. Cerrando...")
                    break

    except Exception as exc:
        logger.error("Error durante la ejecución: %s", exc)
        raise
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
