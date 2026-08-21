from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Cámara
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    mirror: bool = True
    target_fps: int = 30
    log_level: str = "INFO"

    # Modelos
    pose_model_path: Path = Path("models/pose_landmarker_full.task")
    hand_model_path: Path = Path("models/hand_landmarker.task")

    # Umbrales de confianza
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_pose_tracking_confidence: float = 0.5
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5

    # ROI (región de interés)
    roi_size_factor: float = 2.2
    """Factor multiplicador del tamaño de la mano para determinar el ROI en modo hands."""
    roi_min_size: int = 96
    """Tamaño mínimo del ROI en píxeles (ancho o alto)."""
    roi_max_size_ratio: float = 0.8
    """Máximo ratio del ROI respecto al frame (0.8 = 80% del frame)."""
    roi_unify_max_ratio: float = 0.75
    """Si un ROI unificado supera este ratio en ancho Y alto, usar frame completo."""

    # ROI basado en muñecas (modo wrists)
    roi_wrist_margin_x: float = 0.25
    """Margen horizontal alrededor de las muñecas como ratio del ancho entre ellas."""
    roi_wrist_margin_y: float = 1
    """Margen vertical alrededor de las muñecas como ratio del alto entre ellas."""
    roi_wrist_min_size: int = 220
    """Tamaño mínimo del ROI de muñecas en píxeles (ancho o alto)."""

    # Tracking
    hand_lost_threshold: int = 5
    """Frames consecutivos sin detección de mano antes de marcar como LOST."""
    pose_lost_threshold: int = 10
    """Frames consecutivos sin detección de pose antes de descartarla."""
    pose_recalibration_interval: int = 30
    """Frames entre redetecciones periódicas con PoseLandmarker durante TRACKING."""
    mode_switch_to_hands_threshold: int = 3
    """Frames consecutivos con 2 manos detectadas para cambiar de wrists a hands."""
    mode_switch_to_wrists_threshold: int = 5
    """Frames consecutivos sin 2 manos detectadas para volver de hands a wrists."""

    # Hand Landmarker input
    hand_input_size: int = 224
    """Tamaño de entrada esperado por Hand Landmarker (resize del ROI)."""
