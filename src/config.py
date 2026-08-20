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
    inference_fps: int = 15
    """FPS a los que se ejecutan los modelos de MediaPipe (inferencia)."""
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
    roi_forearm_extension: float = 0.4
    """Extensión más allá de la muñeca en dirección del antebrazo (0.0 = muñeca, 1.0 = longitud del antebrazo)."""
    roi_size_factor: float = 2.2
    """Factor multiplicador de la longitud del antebrazo para determinar el tamaño del ROI."""
    roi_min_size: int = 96
    """Tamaño mínimo del ROI en píxeles (ancho o alto)."""
    roi_max_size_ratio: float = 0.8
    """Máximo ratio del ROI respecto al frame (0.8 = 80% del frame)."""
    roi_unify_max_ratio: float = 0.75
    """Si un ROI unificado supera este ratio en ancho Y alto, usar frame completo."""

    # Tracking
    hand_lost_threshold: int = 5
    """Frames consecutivos sin detección de mano antes de marcar como LOST."""
    pose_lost_threshold: int = 10
    """Frames consecutivos sin detección de pose antes de descartarla."""
    pose_recalibration_interval: int = 30
    """Frames entre redetecciones periódicas con PoseLandmarker durante TRACKING."""

    # Hand Landmarker input
    hand_input_size: int = 224
    """Tamaño de entrada esperado por Hand Landmarker (resize del ROI)."""
