from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from mediapipe.tasks.python.vision import PoseLandmarkerResult, PoseLandmark

from .config import Config

logger = logging.getLogger(__name__)

# Índices de landmarks relevantes
SHOULDER_L = PoseLandmark.LEFT_SHOULDER.value
ELBOW_L = PoseLandmark.LEFT_ELBOW.value
WRIST_L = PoseLandmark.LEFT_WRIST.value
SHOULDER_R = PoseLandmark.RIGHT_SHOULDER.value
ELBOW_R = PoseLandmark.RIGHT_ELBOW.value
WRIST_R = PoseLandmark.RIGHT_WRIST.value

_ARM_INDICES = [SHOULDER_L, ELBOW_L, WRIST_L, SHOULDER_R, ELBOW_R, WRIST_R]


def _get_landmark(result: PoseLandmarkerResult, person_idx: int, idx: int):
    """Obtiene un landmark específico de una persona."""
    return result.pose_landmarks[person_idx][idx]


def _person_confidence(result: PoseLandmarkerResult, person_idx: int) -> float:
    """Estima la confianza de una persona promediando visibility de landmarks de brazo."""
    visibilities = []
    for idx in _ARM_INDICES:
        lm = _get_landmark(result, person_idx, idx)
        visibilities.append(lm.visibility if lm.visibility is not None else 0.0)
    return float(np.mean(visibilities))


def _best_person(result: PoseLandmarkerResult) -> int:
    """Devuelve el índice de la persona con mayor confianza."""
    best_idx = 0
    best_conf = -1.0
    for i in range(len(result.pose_landmarks)):
        conf = _person_confidence(result, i)
        if conf > best_conf:
            best_conf = conf
            best_idx = i
    return best_idx


def _landmark_to_pixel(
    lm, frame_width: int, frame_height: int
) -> tuple[float, float]:
    return lm.x * frame_width, lm.y * frame_height


def _roi_from_arm(
    shoulder: tuple[float, float],
    elbow: tuple[float, float],
    wrist: tuple[float, float],
    frame_shape: tuple[int, int],
    config: Config,
) -> Optional[tuple[int, int, int, int]]:
    """Calcula un ROI cuadrado alrededor de la región probable de la mano.

    La lógica estima la posición de la mano extendiendo la dirección del
    antebrazo más allá de la muñeca, y genera un ROI cuadrado proporcional
    a la longitud del antebrazo.
    """
    frame_h, frame_w = frame_shape[:2]

    # Vector antebrazo: codo -> muñeca
    v_x = wrist[0] - elbow[0]
    v_y = wrist[1] - elbow[1]
    forearm_len = math.hypot(v_x, v_y)

    if forearm_len < 1e-6:
        logger.debug("Antebrazo de longitud cero; ROI desde muñeca.")
        forearm_len = 20.0  # fallback mínimo en píxeles aproximados

    # Centro estimado de la mano: muñeca + extensión en dirección del antebrazo
    ext = config.roi_forearm_extension
    hand_x = wrist[0] + v_x * ext
    hand_y = wrist[1] + v_y * ext

    # Tamaño del ROI proporcional al antebrazo
    roi_size = int(forearm_len * config.roi_size_factor)
    roi_size = max(roi_size, config.roi_min_size)
    max_size = int(min(frame_w, frame_h) * config.roi_max_size_ratio)
    roi_size = min(roi_size, max_size)

    # Cuadrado centrado en la mano estimada
    x1 = int(hand_x - roi_size / 2)
    y1 = int(hand_y - roi_size / 2)
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Clamp al frame
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    # Asegurar tamaño mínimo después de clamp
    if x2 - x1 < config.roi_min_size:
        x2 = min(frame_w, x1 + config.roi_min_size)
        x1 = max(0, x2 - config.roi_min_size)
    if y2 - y1 < config.roi_min_size:
        y2 = min(frame_h, y1 + config.roi_min_size)
        y1 = max(0, y2 - config.roi_min_size)

    return (x1, y1, x2, y2)


class ROIEstimator:
    """Genera regiones de interés candidatas para las manos."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def from_pose(
        self, result: PoseLandmarkerResult, frame_shape: tuple[int, int]
    ) -> Optional[dict[str, tuple[int, int, int, int]]]:
        """Devuelve diccionario {'left': roi, 'right': roi} desde pose.

        Si no hay poses detectadas o la confianza es muy baja, devuelve None.
        """
        if not result.pose_landmarks:
            return None

        person_idx = _best_person(result)
        conf = _person_confidence(result, person_idx)
        if conf < self._config.min_pose_presence_confidence:
            logger.debug("Confianza de pose insuficiente: %.2f", conf)
            return None

        rois = {}
        frame_h, frame_w = frame_shape[:2]

        # Brazo izquierdo
        s = _landmark_to_pixel(_get_landmark(result, person_idx, SHOULDER_L), frame_w, frame_h)
        e = _landmark_to_pixel(_get_landmark(result, person_idx, ELBOW_L), frame_w, frame_h)
        w = _landmark_to_pixel(_get_landmark(result, person_idx, WRIST_L), frame_w, frame_h)
        roi_l = _roi_from_arm(s, e, w, frame_shape, self._config)
        if roi_l:
            rois["left"] = roi_l

        # Brazo derecho
        s = _landmark_to_pixel(_get_landmark(result, person_idx, SHOULDER_R), frame_w, frame_h)
        e = _landmark_to_pixel(_get_landmark(result, person_idx, ELBOW_R), frame_w, frame_h)
        w = _landmark_to_pixel(_get_landmark(result, person_idx, WRIST_R), frame_w, frame_h)
        roi_r = _roi_from_arm(s, e, w, frame_shape, self._config)
        if roi_r:
            rois["right"] = roi_r

        if not rois:
            return None
        return rois

    def from_hand_position(
        self,
        hand_center: tuple[float, float],
        hand_size: float,
        frame_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Genera un ROI cuadrado alrededor de una posición de mano conocida.

        Args:
            hand_center: (x, y) en píxeles del frame original.
            hand_size: Tamaño estimado de la mano en píxeles (ej. bounding box aproximado).
            frame_shape: (h, w) del frame.

        Returns:
            ROI cuadrado.
        """
        frame_h, frame_w = frame_shape[:2]
        roi_size = int(hand_size * self._config.roi_size_factor)
        roi_size = max(roi_size, self._config.roi_min_size)
        max_size = int(min(frame_w, frame_h) * self._config.roi_max_size_ratio)
        roi_size = min(roi_size, max_size)

        x1 = int(hand_center[0] - roi_size / 2)
        y1 = int(hand_center[1] - roi_size / 2)
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        return (x1, y1, x2, y2)

    def full_frame(self, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Devuelve ROI que cubre todo el frame."""
        frame_h, frame_w = frame_shape[:2]
        return (0, 0, frame_w, frame_h)
