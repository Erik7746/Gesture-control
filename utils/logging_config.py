from __future__ import annotations

import logging
import sys


def configure_logging(level_name: str) -> None:
    """Configura logging básico con formato consistente."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
