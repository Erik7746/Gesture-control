from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType

PixelCoord = NewType("PixelCoord", float)
ROI = tuple[int, int, int, int]  # x1, y1, x2, y2


class TrackingState(Enum):
    """Estados posibles del seguimiento global."""

    DETECTING = auto()
    TRACKING = auto()
    LOST = auto()


class TrackingMode(Enum):
    """Modos de generación de ROI."""

    WRISTS = auto()
    HANDS = auto()


@dataclass
class HandState:
    """Estado interno de una mano trackeada."""

    landmarks: list[tuple[PixelCoord, PixelCoord, PixelCoord]]  # 21 landmarks
    confidence: float
    center: tuple[PixelCoord, PixelCoord]
    size: float
    lost_frames: int = 0
