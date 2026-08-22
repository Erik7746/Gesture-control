from __future__ import annotations

import logging
import math
import threading
from typing import Optional

from mediapipe.tasks.python.vision import PoseLandmarkerResult, HandLandmarkerResult

from config.settings import TrackingConfig, ROIConfig, ModelConfig
from core.types import HandState, TrackingState, TrackingMode
from core.roi_estimator import ROIEstimator
from detection.image_pipeline import denormalize_landmarks_to_roi

logger = logging.getLogger(__name__)


class HandTrackingStateMachine:
    """Máquina de estados para seguimiento de hasta 2 manos.

    Funciona en dos modos:
    - WRISTS: ROI generado a partir de las muñecas detectadas por pose.
    - HANDS: ROI generado a partir de las 2 manos trackeadas.

    El modo WRISTS es el puente para encontrar las manos a distancia.
    Solo se pasa a HANDS cuando se detectan 2 manos; si se pierde alguna,
    se vuelve inmediatamente a WRISTS.

    Incluye un mini-tracker por proximidad: si una mano no se detecta en
    un frame, se mantiene en su última posición conocida hasta
    hand_max_lost_frames, evitando parpadeo visual.

    Thread-safe: los callbacks de MediaPipe LIVE_STREAM actualizan
    resultados desde hilos internos; el bucle principal consulta el estado
    a través de métodos que adquieren un lock.
    """

    def __init__(
        self,
        tracking_config: TrackingConfig,
        roi_config: ROIConfig,
        model_config: ModelConfig,
    ) -> None:
        self._tracking_config = tracking_config
        self._roi_config = roi_config
        self._model_config = model_config
        self._roi_estimator = ROIEstimator(model_config, roi_config)
        self._lock = threading.Lock()

        # Estado
        self._state = TrackingState.DETECTING
        self._tracking_mode = TrackingMode.WRISTS
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

        Actualiza la lista interna de manos usando mini-tracking por proximidad.
        """
        with self._lock:
            roi = self._current_hand_roi
            if not result.hand_landmarks:
                self._frames_without_any_hand += 1
                self._age_hands()
                logger.debug("HandLandmarker no encontró manos en ROI %s", roi)
                self._handle_mode_transition()
                return

            self._frames_without_any_hand = 0

            # Convertir detecciones del frame actual a HandState
            new_hands: list[HandState] = []
            for idx, lm_norm in enumerate(result.hand_landmarks[:2]):  # máximo 2 manos
                score = 0.0
                if result.handedness and idx < len(result.handedness):
                    score = result.handedness[idx][0].score

                lm_orig = denormalize_landmarks_to_roi(lm_norm, roi)
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
                        lost_frames=0,
                    )
                )

            self._merge_hands(new_hands)
            self._handle_mode_transition()

    def _age_hands(self) -> None:
        """Incrementa lost_frames y descarta manos perdidas demasiado tiempo."""
        for hand in self._hands:
            hand.lost_frames += 1
        self._hands = [
            hand for hand in self._hands
            if hand.lost_frames <= self._tracking_config.hand_max_lost_frames
        ]

    def _merge_hands(self, new_hands: list[HandState]) -> None:
        """Empareja nuevas detecciones con manos trackeadas por proximidad.

        Las manos no emparejadas se mantienen temporalmente (incrementando
        lost_frames). Las nuevas detecciones no emparejadas se añaden.
        El resultado se ordena por posición x para identidad visual estable.
        """
        if not self._hands:
            self._hands = sorted(new_hands[:2], key=lambda h: h.center[0])
            return

        if not new_hands:
            self._age_hands()
            return

        # Calcular todas las distancias entre manos existentes y nuevas
        pairs: list[tuple[float, int, int]] = []
        for i, existing in enumerate(self._hands):
            for j, new in enumerate(new_hands):
                dist = math.hypot(
                    existing.center[0] - new.center[0],
                    existing.center[1] - new.center[1],
                )
                pairs.append((dist, i, j))

        # Ordenar por distancia (emparejamiento greedy)
        pairs.sort(key=lambda x: x[0])

        assigned_existing: set[int] = set()
        assigned_new: set[int] = set()
        updated_hands: list[HandState] = []

        for _dist, i, j in pairs:
            if i in assigned_existing or j in assigned_new:
                continue

            updated_hands.append(
                HandState(
                    landmarks=new_hands[j].landmarks,
                    confidence=new_hands[j].confidence,
                    center=new_hands[j].center,
                    size=new_hands[j].size,
                    lost_frames=0,
                )
            )
            assigned_existing.add(i)
            assigned_new.add(j)

        # Manos existentes no emparejadas: mantener temporalmente
        for i, existing in enumerate(self._hands):
            if i not in assigned_existing:
                existing.lost_frames += 1
                if existing.lost_frames <= self._tracking_config.hand_max_lost_frames:
                    updated_hands.append(existing)

        # Nuevas detecciones no emparejadas: añadir como manos nuevas
        for j, new in enumerate(new_hands):
            if j not in assigned_new:
                updated_hands.append(new)

        # Limitar a 2 manos y ordenar por posición x (izquierda -> derecha)
        self._hands = sorted(updated_hands, key=lambda h: h.center[0])[:2]

    def _handle_mode_transition(self) -> None:
        """Decide si se mantiene o cambia el modo de tracking usando histeresis."""
        tracked_count = len(self._hands)

        if tracked_count == 2:
            self._frames_with_two_hands += 1
            self._frames_without_two_hands = 0
        else:
            self._frames_without_two_hands += 1
            self._frames_with_two_hands = 0

        if self._tracking_mode == TrackingMode.WRISTS:
            if self._frames_with_two_hands >= self._tracking_config.mode_switch_to_hands_threshold:
                self._tracking_mode = TrackingMode.HANDS
                self._state = TrackingState.TRACKING
                self._frames_with_two_hands = 0
                logger.info(
                    "Modo WRISTS -> HANDS: %d frames con 2 manos",
                    self._tracking_config.mode_switch_to_hands_threshold,
                )
        else:  # HANDS
            if self._frames_without_two_hands >= self._tracking_config.mode_switch_to_wrists_threshold:
                self._tracking_mode = TrackingMode.WRISTS
                self._state = TrackingState.DETECTING
                self._hands = []
                self._frames_without_two_hands = 0
                self._frames_since_pose_recalibration = self._tracking_config.pose_recalibration_interval
                logger.info(
                    "Modo HANDS -> WRISTS: %d frames sin 2 manos",
                    self._tracking_config.mode_switch_to_wrists_threshold,
                )
            elif tracked_count == 2:
                self._state = TrackingState.TRACKING

        logger.debug(
            "Manos activas: %d | modo=%s | estado=%s | hist=(%d, %d)",
            tracked_count,
            self._tracking_mode.name,
            self._state.name,
            self._frames_with_two_hands,
            self._frames_without_two_hands,
        )

    # ── Consultas desde el bucle principal ───────────────────────────────────

    def get_roi_for_hand_detection(self, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Devuelve el ROI sobre el cual ejecutar HandLandmarker en el frame actual."""
        with self._lock:
            self._frame_counter += 1

            # Verificar pérdida de tracking global
            if self._frames_without_any_hand >= self._tracking_config.hand_lost_threshold:
                if self._state == TrackingState.TRACKING:
                    logger.info("Tracking perdido: %d frames sin ninguna mano", self._frames_without_any_hand)
                    self._state = TrackingState.LOST
                    self._tracking_mode = TrackingMode.WRISTS
                    self._hands = []
                    self._frames_since_pose_recalibration = self._tracking_config.pose_recalibration_interval

            # Modo HANDS: ROI unificado a partir de las 2 manos trackeadas
            if self._tracking_mode == TrackingMode.HANDS and len(self._hands) == 2:
                rois = [
                    self._roi_estimator.from_hand_position(hand.center, hand.size, frame_shape)
                    for hand in self._hands
                ]
                unified = self._roi_estimator.unify_rois(rois, frame_shape)
                u_w = unified[2] - unified[0]
                u_h = unified[3] - unified[1]
                f_h, f_w = frame_shape[:2]
                if u_w >= f_w * self._roi_config.unify_max_ratio and u_h >= f_h * self._roi_config.unify_max_ratio:
                    return self._roi_estimator.full_frame(frame_shape)
                return unified

            # Modo WRISTS / DETECTING / LOST: ROI a partir de las muñecas de pose
            if self._pose_result is not None and self._frames_without_pose < self._tracking_config.pose_lost_threshold:
                wrist_roi = self._roi_estimator.from_pose(self._pose_result, frame_shape)
                if wrist_roi is not None:
                    return wrist_roi

            # Fallback absoluto
            return self._roi_estimator.full_frame(frame_shape)

    def should_run_pose_detection(self) -> bool:
        """Indica si en este frame conviene enviar un frame a PoseLandmarker."""
        with self._lock:
            if self._tracking_mode == TrackingMode.WRISTS:
                return True

            if self._frames_since_pose_recalibration >= self._tracking_config.pose_recalibration_interval:
                self._frames_since_pose_recalibration = 0
                return True
            self._frames_since_pose_recalibration += 1
            return False

    @property
    def state(self) -> TrackingState:
        with self._lock:
            return self._state

    @property
    def tracking_mode(self) -> TrackingMode:
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
