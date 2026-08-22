from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class FPSLimiter:
    """Limita el bucle principal a un target FPS usando sleep."""

    target_fps: float

    def __post_init__(self) -> None:
        self._frame_duration = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._last_time = time.perf_counter()

    def wait(self) -> None:
        """Espera el tiempo necesario para respetar el target FPS."""
        now = time.perf_counter()
        elapsed = now - self._last_time
        sleep_time = self._frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.perf_counter()


class FPSCounter:
    """Cuenta los frames por segundo reales."""

    def __init__(self) -> None:
        self._start_time = time.perf_counter()
        self._frame_count = 0
        self._fps = 0.0

    def tick(self) -> None:
        self._frame_count += 1
        elapsed = time.perf_counter() - self._start_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = time.perf_counter()

    @property
    def fps(self) -> float:
        return self._fps
