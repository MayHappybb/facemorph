"""Warping engines for piecewise affine transformation."""

from abc import ABC, abstractmethod
import numpy as np
import cv2


class WarpingEngine(ABC):
    """Abstract base class for warping implementations."""

    @abstractmethod
    def warp(
        self,
        image: np.ndarray,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray,
        triangles: list[tuple[int, int, int]],
        output_size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Warp image from src_landmarks geometry to dst_landmarks geometry.

        Args:
            image: Source image (H, W, C)
            src_landmarks: (N, 2) source landmark coordinates
            dst_landmarks: (N, 2) destination landmark coordinates
            triangles: List of (i, j, k) triangle vertex indices
            output_size: (width, height) for output canvas. If None, uses image size.

        Returns:
            Warped image in destination geometry
        """
        pass
