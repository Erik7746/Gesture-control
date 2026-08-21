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
    """Máquina de estados para seguimiento de hasta 2 manos.

    Funciona en dos modos:
    - wrists: ROI generado a partir de las muñecas detectadas por pose.
    - hands: ROI generado a partir de las 2 manos trackeadas.

    El modo wrists es el puente para encontrar las manos a distancia.
    Solo se pasa a hands cuando se detectan 2 manos; si se pierde alguna,
    se vuelve inmediatamente a wrists.

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
        self._tracking_mode = "wrists"  # wrists | hands
        self._pose_result: Optional[PoseLandmarkerResult] = None
        self._hands: list[HandState] = []

        # Contadores
        self._frames_without_any_hand = 0
        self._frames_without_pose = 0
        self._frames_since_pose_recalibration = 0
        self._frame_counter = 0
        self._frames_with_two_hands = 0
        self._frames_without_two_hands = 0

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

        Actualiza la lista interna de manos detectadas (máximo 2) y
        gestiona la transición entre modos wrists/hands.
        """
        with self._lock:
            roi = self._current_hand_roi
            if not result.hand_landmarks:
                self._frames_without_any_hand += 1
                logger.debug("HandLandmarker no encontró manos en ROI %s", roi)
                self._handle_mode_transition([])
                return

            self._frames_without_any_hand = 0
            from .image_utils import transform_normalized_landmarks

            new_hands: list[HandState] = []
            for idx, lm_norm in enumerate(result.hand_landmarks[:2]):  # máximo 2 manos
                # Tomar la mayor confianza disponible entre las categorías de handedness
                score = 0.0
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
            self._handle_mode_transition(new_hands)

    def _handle_mode_transition(self, new_hands: list[HandState]) -> None:
        """Decide si se mantiene o cambia el modo de tracking usando histeresis.

        Esto evita oscilaciones constantes entre wrists y hands cuando la
        detección fluctúa frame a frame.
        """
        detected_count = len(new_hands)

        # Actualizar contadores de histeresis
        if detected_count == 2:
            self._frames_with_two_hands += 1
            self._frames_without_two_hands = 0
        else:
            self._frames_without_two_hands += 1
            self._frames_with_two_hands = 0

        if self._tracking_mode == "wrists":
            if self._frames_with_two_hands >= self._config.mode_switch_to_hands_threshold:
                self._tracking_mode = "hands"
                self._state = "TRACKING"
                self._frames_with_two_hands = 0
                logger.info(
                    "Modo wrists -> hands: %d frames con 2 manos",
                    self._config.mode_switch_to_hands_threshold,
                )
        else:  # hands
            if self._frames_without_two_hands >= self._config.mode_switch_to_wrists_threshold:
                self._tracking_mode = "wrists"
                self._state = "DETECTING"
                self._hands = []
                self._frames_without_two_hands = 0
                # Forzar pose detection inmediato al volver a wrists
                self._frames_since_pose_recalibration = self._config.pose_recalibration_interval
                logger.info(
                    "Modo hands -> wrists: %d frames sin 2 manos",
                    self._config.mode_switch_to_wrists_threshold,
                )
            elif detected_count == 2:
                self._state = "TRACKING"

        logger.debug(
            "Manos detectadas: %d | modo=%s | estado=%s | hist=(%d, %d)",
            detected_count,
            self._tracking_mode,
            self._state,
            self._frames_with_two_hands,
            self._frames_without_two_hands,
        )

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
                    self._tracking_mode = "wrists"
                    self._hands = []
                    self._frames_since_pose_recalibration = self._config.pose_recalibration_interval

            # Modo hands: ROI unificado a partir de las 2 manos trackeadas
            if self._tracking_mode == "hands" and len(self._hands) == 2:
                rois = [
                    self._roi_estimator.from_hand_position(hand.center, hand.size, frame_shape)
                    for hand in self._hands
                ]
                unified = self._roi_estimator.unify_rois(rois, frame_shape)
                u_w = unified[2] - unified[0]
                u_h = unified[3] - unified[1]
                f_h, f_w = frame_shape[:2]
                if u_w >= f_w * self._config.roi_unify_max_ratio and u_h >= f_h * self._config.roi_unify_max_ratio:
                    return self._roi_estimator.full_frame(frame_shape)
                return unified

            # Modo wrists / DETECTING / LOST: ROI a partir de las muñecas de pose
            if self._pose_result is not None and self._frames_without_pose < self._config.pose_lost_threshold:
                wrist_roi = self._roi_estimator.from_pose(self._pose_result, frame_shape)
                if wrist_roi is not None:
                    return wrist_roi

            # Fallback absoluto
            return self._roi_estimator.full_frame(frame_shape)

    def should_run_pose_detection(self) -> bool:
        """Indica si en este frame conviene enviar un frame a PoseLandmarker."""
        with self._lock:
            # En modo wrists siempre necesitamos pose fresca
            if self._tracking_mode == "wrists":
                return True

            # En modo hands, usar el intervalo configurado
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
    def tracking_mode(self) -> str:
        with self._lock:
            return self._tracking_mode

    @property
    def hands(self) -> list[HandState]:
        with self._lock:
            return self._hands.copy()

    @property
    def pose_result(self) -> Optional[PoseLandmarkerResult]:
        with self._lock:
            return self._pose_result
