from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from mediapipe.tasks.python.vision import PoseLandmarkerResult, PoseLandmark

from .tracker import HandState

logger = logging.getLogger(__name__)

# Colores BGR
COLOR_ROI = (0, 255, 255)           # Cyan
COLOR_HAND_LEFT = (0, 255, 0)       # Verde
COLOR_HAND_RIGHT = (0, 0, 255)      # Rojo
COLOR_POSE = (255, 0, 0)            # Azul
COLOR_TEXT = (0, 255, 0)            # Verde
COLOR_TEXT_WARN = (0, 165, 255)     # Naranja

# Conexiones de la mano
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),       # Índice
    (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anular
    (0, 17), (17, 18), (18, 19), (19, 20), # Meñique
    (5, 9), (9, 13), (13, 17),            # Base dedos
]

# Conexiones del brazo en pose
_ARM_POSE_CONNECTIONS = [
    (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.LEFT_ELBOW.value),
    (PoseLandmark.LEFT_ELBOW.value, PoseLandmark.LEFT_WRIST.value),
    (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_ELBOW.value),
    (PoseLandmark.RIGHT_ELBOW.value, PoseLandmark.RIGHT_WRIST.value),
]


class Visualizer:
    """Dibuja resultados de detección sobre el frame BGR."""

    def draw(
        self,
        frame: np.ndarray,
        hand_left: Optional[HandState],
        hand_right: Optional[HandState],
        pose_result: Optional[PoseLandmarkerResult],
        roi: Optional[tuple[int, int, int, int]],
        tracker_state: str,
        fps: float,
    ) -> np.ndarray:
        """Dibuja toda la información de debug/visualización sobre el frame."""
        h, w = frame.shape[:2]

        # ROI
        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROI, 2)
            cv2.putText(frame, "ROI", (x1 + 4, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROI, 1)

        # Pose (solo brazos)
        if pose_result is not None and pose_result.pose_landmarks:
            self._draw_pose_arms(frame, pose_result, w, h)

        # Manos
        if hand_left is not None:
            self._draw_hand(frame, hand_left, COLOR_HAND_LEFT)
        if hand_right is not None:
            self._draw_hand(frame, hand_right, COLOR_HAND_RIGHT)

        # HUD
        state_color = COLOR_TEXT
        if tracker_state == "LOST":
            state_color = COLOR_TEXT_WARN
        elif tracker_state == "DETECTING":
            state_color = (255, 255, 0)

        cv2.putText(frame, f"State: {tracker_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        y_offset = 90
        if hand_left is not None:
            info = f"L: {hand_left.handedness} conf={hand_left.confidence:.2f}"
            cv2.putText(frame, info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HAND_LEFT, 2)
            y_offset += 25
        if hand_right is not None:
            info = f"R: {hand_right.handedness} conf={hand_right.confidence:.2f}"
            cv2.putText(frame, info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HAND_RIGHT, 2)

        return frame

    def _draw_pose_arms(
        self,
        frame: np.ndarray,
        result: PoseLandmarkerResult,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Dibuja solo los brazos de la primera persona detectada."""
        if not result.pose_landmarks:
            return
        person = result.pose_landmarks[0]
        for start_idx, end_idx in _ARM_POSE_CONNECTIONS:
            lm1 = person[start_idx]
            lm2 = person[end_idx]
            if lm1.visibility is not None and lm1.visibility < 0.5:
                continue
            if lm2.visibility is not None and lm2.visibility < 0.5:
                continue
            p1 = (int(lm1.x * frame_w), int(lm1.y * frame_h))
            p2 = (int(lm2.x * frame_w), int(lm2.y * frame_h))
            cv2.line(frame, p1, p2, COLOR_POSE, 2)
            cv2.circle(frame, p1, 3, COLOR_POSE, -1)
            cv2.circle(frame, p2, 3, COLOR_POSE, -1)

    def _draw_hand(
        self,
        frame: np.ndarray,
        hand_state: HandState,
        color: tuple[int, int, int],
    ) -> None:
        """Dibuja los 21 landmarks y conexiones de una mano."""
        lm = hand_state.landmarks
        if len(lm) < 21:
            return

        # Puntos
        for i, (x, y, _z) in enumerate(lm):
            px, py = int(x), int(y)
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.circle(frame, (px, py), 4, (0, 0, 0), 1)

        # Líneas
        for a, b in _HAND_CONNECTIONS:
            if a < len(lm) and b < len(lm):
                pa = (int(lm[a][0]), int(lm[a][1]))
                pb = (int(lm[b][0]), int(lm[b][1]))
                cv2.line(frame, pa, pb, color, 1)
