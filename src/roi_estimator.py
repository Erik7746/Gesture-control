from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from mediapipe.tasks.python.vision import PoseLandmarkerResult, PoseLandmark

from .config import Config

logger = logging.getLogger(__name__)

# Índices de landmarks relevantes
WRIST_L = PoseLandmark.LEFT_WRIST.value
WRIST_R = PoseLandmark.RIGHT_WRIST.value

_ARM_INDICES = [
    PoseLandmark.LEFT_SHOULDER.value,
    PoseLandmark.LEFT_ELBOW.value,
    WRIST_L,
    PoseLandmark.RIGHT_SHOULDER.value,
    PoseLandmark.RIGHT_ELBOW.value,
    WRIST_R,
]


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


class ROIEstimator:
    """Genera regiones de interés candidatas para las manos."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def from_pose(
        self, result: PoseLandmarkerResult, frame_shape: tuple[int, int]
    ) -> Optional[tuple[int, int, int, int]]:
        """Devuelve un único ROI que cubre ambas muñecas.

        Si no hay poses detectadas, la confianza es muy baja, o falta alguna
        muñeca, devuelve None (el llamador deberá usar frame completo).
        """
        if not result.pose_landmarks:
            return None

        person_idx = _best_person(result)
        conf = _person_confidence(result, person_idx)
        if conf < self._config.min_pose_presence_confidence:
            logger.debug("Confianza de pose insuficiente: %.2f", conf)
            return None

        frame_h, frame_w = frame_shape[:2]

        wrist_l = _get_landmark(result, person_idx, WRIST_L)
        wrist_r = _get_landmark(result, person_idx, WRIST_R)

        if wrist_l.visibility is None or wrist_l.visibility < self._config.min_pose_presence_confidence:
            logger.debug("Muñeca izquierda no visible")
            return None
        if wrist_r.visibility is None or wrist_r.visibility < self._config.min_pose_presence_confidence:
            logger.debug("Muñeca derecha no visible")
            return None

        x_l, y_l = _landmark_to_pixel(wrist_l, frame_w, frame_h)
        x_r, y_r = _landmark_to_pixel(wrist_r, frame_w, frame_h)

        # Bounding box que cubre ambas muñecas
        x1 = min(x_l, x_r)
        y1 = min(y_l, y_r)
        x2 = max(x_l, x_r)
        y2 = max(y_l, y_r)

        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)

        # Margen configurable alrededor de las muñecas
        margin_x = max(width * self._config.roi_wrist_margin_x, self._config.roi_wrist_min_size / 2)
        margin_y = max(height * self._config.roi_wrist_margin_y, self._config.roi_wrist_min_size / 2)

        x1 -= margin_x
        y1 -= margin_y
        x2 += margin_x
        y2 += margin_y

        # Clamp al frame
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame_w, int(x2))
        y2 = min(frame_h, int(y2))

        # Asegurar tamaño mínimo después de clamp
        if x2 - x1 < self._config.roi_wrist_min_size:
            x2 = min(frame_w, x1 + self._config.roi_wrist_min_size)
            x1 = max(0, x2 - self._config.roi_wrist_min_size)
        if y2 - y1 < self._config.roi_wrist_min_size:
            y2 = min(frame_h, y1 + self._config.roi_wrist_min_size)
            y1 = max(0, y2 - self._config.roi_wrist_min_size)

        return (x1, y1, x2, y2)

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

    def unify_rois(
        self,
        rois: list[tuple[int, int, int, int]],
        frame_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Devuelve un ROI que cubre todos los ROIs dados (bounding box unificado).

        Añade un margen proporcional al tamaño para absorber movimientos.
        """
        if not rois:
            return self.full_frame(frame_shape)

        x1 = min(r[0] for r in rois)
        y1 = min(r[1] for r in rois)
        x2 = max(r[2] for r in rois)
        y2 = max(r[3] for r in rois)

        # Margen del 20% del tamaño del ROI unificado
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        x1 -= margin_x
        y1 -= margin_y
        x2 += margin_x
        y2 += margin_y

        frame_h, frame_w = frame_shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        return (x1, y1, x2, y2)

    def full_frame(self, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Devuelve ROI que cubre todo el frame."""
        frame_h, frame_w = frame_shape[:2]
        return (0, 0, frame_w, frame_h)
