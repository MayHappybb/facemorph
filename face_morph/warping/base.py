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
        output_size: tuple[int, int] | None = None,
        src_face_center: np.ndarray = None,
        dst_face_center: np.ndarray = None,
        src_face_scale: float = 1.0,
        dst_face_scale: float = 1.0
    ) -> np.ndarray:
        """Warp image from src_landmarks geometry to dst_landmarks geometry.

        Args:
            image: Source image (H, W, C)
            src_landmarks: (N, 2) source landmark coordinates (normalized, face-centered)
            dst_landmarks: (N, 2) destination landmark coordinates (normalized, face-centered)
            triangles: List of (i, j, k) triangle vertex indices
            output_size: (width, height) for output canvas. If None, uses image size.
            src_face_center: (2,) face centroid in source image pixels
            dst_face_center: (2,) face centroid in destination image pixels
            src_face_scale: Scale factor for source face
            dst_face_scale: Scale factor for destination face

        Returns:
            Warped image in destination geometry
        """
        pass
