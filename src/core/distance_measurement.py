"""3D Distance Measurement — state and computation"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MeasurementResult:
    point1: np.ndarray          # (3,) world coordinates
    point2: np.ndarray          # (3,) world coordinates
    distance: float             # Euclidean distance in mm
    midpoint: np.ndarray        # (3,) world coordinates


class DistanceMeasurement:
    """Manages the two-point measurement workflow."""

    def __init__(self):
        self._points: list = []

    @property
    def point_count(self) -> int:
        return len(self._points)

    @property
    def is_complete(self) -> bool:
        return len(self._points) == 2

    def add_point(self, world_pos: np.ndarray) -> None:
        """Add a picked point. Resets if already 2 points."""
        if len(self._points) == 2:
            self._points = []
        self._points.append(np.asarray(world_pos, dtype=np.float64))

    def get_result(self) -> Optional[MeasurementResult]:
        """Return result if two points are set, else None."""
        if len(self._points) != 2:
            return None
        p1, p2 = self._points[0], self._points[1]
        return MeasurementResult(
            point1=p1,
            point2=p2,
            distance=float(np.linalg.norm(p2 - p1)),
            midpoint=(p1 + p2) / 2.0,
        )

    def reset(self) -> None:
        """Clear all picked points."""
        self._points.clear()
