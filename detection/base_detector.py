from __future__ import annotations

from typing import Protocol

from mediapipe.tasks.python.vision.core.image import Image as MPImage


class AsyncDetector(Protocol):
    """Protocolo común para detectores asíncronos de MediaPipe."""

    def detect_async(self, image: MPImage, timestamp_ms: int) -> None:
        ...

    def close(self) -> None:
        ...
