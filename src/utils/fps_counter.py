"""Contador de frames por segundo (FPS).

Utilidad para medir y mostrar la tasa de cuadros procesados en tiempo
real. Se basa en el tiempo transcurrido entre llamadas sucesivas.
"""

import time
from collections import deque
from typing import Optional


class FpsCounter:
    """Calcula el FPS promedio de forma suavizada.

    Mantiene una ventana deslizante con los tiempos de los últimos
    frames para evitar fluctuaciones bruscas en la lectura.
    """

    def __init__(self, window_size: int = 30) -> None:
        """Inicializa el contador.

        Args:
            window_size: Cantidad de frames recientes a promediar.
        """
        self._window_size = window_size
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._last_time: Optional[float] = None

    def update(self) -> float:
        """Registra un nuevo frame y devuelve el FPS estimado.

        Returns:
            FPS promedio en la ventana actual. Si no hay suficientes
            datos, devuelve 0.0.
        """
        now = time.perf_counter()

        if self._last_time is not None:
            self._timestamps.append(now - self._last_time)

        self._last_time = now

        if not self._timestamps:
            return 0.0

        average_delta = sum(self._timestamps) / len(self._timestamps)
        return 1.0 / average_delta if average_delta > 0 else 0.0

    def reset(self) -> None:
        """Reinicia el contador eliminando el historial."""
        self._timestamps.clear()
        self._last_time = None

    @property
    def fps(self) -> float:
        """FPS actual calculado sobre la ventana deslizante."""
        if not self._timestamps:
            return 0.0
        average_delta = sum(self._timestamps) / len(self._timestamps)
        return 1.0 / average_delta if average_delta > 0 else 0.0
