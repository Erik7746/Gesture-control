from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from mediapipe.tasks.python.vision import PoseLandmarkerResult, HandLandmarkerResult

from .config import Config
from .roi_estimator import ROIEstimator

logger = logging.getLogger(__name__)


@dataclass
class HandState:
    """Estado interno de una mano trackeada."""

    landmarks: list[tuple[float, float, float]]  # 21 landmarks en coords originales
    confidence: float
    center: tuple[float, float]  # centro aproximado en píxeles
    size: float  # tamaño estimado en píxeles


class Tracker:
    """Máquina de estados para seguimiento de hasta 2 manos con fallback a detección.

    Thread-safe: los callbacks de MediaPipe LIVE_STREAM actualizan
    resultados desde hilos internos; el bucle principal consulta el estado
    a través de métodos que adquieren un lock.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._roi_estimator = ROIEstimator(config)
        self._lock = threading.Lock()

        # Estado
        self._state = "DETECTING"  # DETECTING | TRACKING | LOST
        self._pose_result: Optional[PoseLandmarkerResult] = None
        self._hands: list[HandState] = []

        # Contadores
        self._frames_without_any_hand = 0
        self._frames_without_pose = 0
        self._frames_since_pose_recalibration = 0
        self._frame_counter = 0

        # ROI del frame actual (escrito por el hilo principal, leído por callback)
        self._current_hand_roi: tuple[int, int, int, int] = (0, 0, 0, 0)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def on_pose_result(self, result: PoseLandmarkerResult) -> None:
        """Callback cuando llega un resultado de PoseLandmarker."""
        with self._lock:
            self._pose_result = result
            if result.pose_landmarks:
                self._frames_without_pose = 0
            else:
                self._frames_without_pose += 1

    def set_current_hand_roi(self, roi: tuple[int, int, int, int]) -> None:
        """Establece el ROI usado en el frame actual para hand detection."""
        with self._lock:
            self._current_hand_roi = roi

    def on_hand_result(self, result: HandLandmarkerResult, *args) -> None:
        """Callback cuando llega un resultado de HandLandmarker.

        Actualiza la lista interna de manos detectadas (máximo 2).
        No se distingue entre mano derecha o izquierda.
        """
        with self._lock:
            roi = self._current_hand_roi
            if not result.hand_landmarks:
                self._frames_without_any_hand += 1
                logger.debug("HandLandmarker no encontró manos en ROI %s", roi)
                return

            self._frames_without_any_hand = 0
            from .image_utils import transform_normalized_landmarks

            new_hands: list[HandState] = []
            for lm_norm in result.hand_landmarks[:2]:  # máximo 2 manos
                # Tomar la mayor confianza disponible entre las categorías de handedness
                score = 0.0
                idx = len(new_hands)
                if result.handedness and idx < len(result.handedness):
                    score = result.handedness[idx][0].score

                # Transformar a coords originales
                lm_orig = transform_normalized_landmarks(lm_norm, roi)
                xs = [p[0] for p in lm_orig]
                ys = [p[1] for p in lm_orig]
                center = (sum(xs) / len(xs), sum(ys) / len(ys))
                size = max(max(xs) - min(xs), max(ys) - min(ys))

                new_hands.append(
                    HandState(
                        landmarks=lm_orig,
                        confidence=score,
                        center=center,
                        size=size,
                    )
                )

            self._hands = new_hands
            self._state = "TRACKING"
            logger.debug("Manos detectadas: %d", len(self._hands))

    # ── Consultas desde el bucle principal ───────────────────────────────────

    def get_roi_for_hand_detection(self, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Devuelve el ROI sobre el cual ejecutar HandLandmarker en el frame actual."""
        with self._lock:
            self._frame_counter += 1

            # Verificar pérdida de tracking global
            if self._frames_without_any_hand >= self._config.hand_lost_threshold:
                if self._state == "TRACKING":
                    logger.info("Tracking perdido: %d frames sin ninguna mano", self._frames_without_any_hand)
                    self._state = "LOST"
                    self._hands = []

            # Si tenemos al menos una mano trackeada → ROI unificado
            if self._state == "TRACKING" and self._hands:
                rois = [
                    self._roi_estimator.from_hand_position(hand.center, hand.size, frame_shape)
                    for hand in self._hands
                ]
                unified = self._roi_estimator.unify_rois(rois, frame_shape)
                # Si el ROI unificado es casi todo el frame, mejor usar frame completo
                u_w = unified[2] - unified[0]
                u_h = unified[3] - unified[1]
                f_h, f_w = frame_shape[:2]
                if u_w >= f_w * self._config.roi_unify_max_ratio and u_h >= f_h * self._config.roi_unify_max_ratio:
                    return self._roi_estimator.full_frame(frame_shape)
                return unified

            # DETECTING / LOST: usar pose para estimar ROIs de ambos brazos
            if self._pose_result is not None and self._frames_without_pose < self._config.pose_lost_threshold:
                rois = self._roi_estimator.from_pose(self._pose_result, frame_shape)
                if rois:
                    roi_list = list(rois.values())
                    unified = self._roi_estimator.unify_rois(roi_list, frame_shape)
                    u_w = unified[2] - unified[0]
                    u_h = unified[3] - unified[1]
                    f_h, f_w = frame_shape[:2]
                    if u_w >= f_w * self._config.roi_unify_max_ratio and u_h >= f_h * self._config.roi_unify_max_ratio:
                        return self._roi_estimator.full_frame(frame_shape)
                    return unified

            # Fallback absoluto
            return self._roi_estimator.full_frame(frame_shape)

    def should_run_pose_detection(self) -> bool:
        """Indica si en este frame conviene enviar un frame a PoseLandmarker."""
        with self._lock:
            if self._state in ("DETECTING", "LOST"):
                return True
            if self._frames_since_pose_recalibration >= self._config.pose_recalibration_interval:
                self._frames_since_pose_recalibration = 0
                return True
            self._frames_since_pose_recalibration += 1
            return False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def hands(self) -> list[HandState]:
        with self._lock:
            return self._hands.copy()

    @property
    def pose_result(self) -> Optional[PoseLandmarkerResult]:
        with self._lock:
            return self._pose_result
