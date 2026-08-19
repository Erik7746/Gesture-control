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
