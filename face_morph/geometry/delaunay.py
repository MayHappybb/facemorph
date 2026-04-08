"""Delaunay triangulation utilities."""

import numpy as np
import cv2
from scipy.spatial import Delaunay


def compute_delaunay_triangles(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Compute Delaunay triangulation of points using SciPy.

    Args:
        points: (N, 2) array of point coordinates

    Returns:
        List of triangles, each as (i, j, k) vertex indices
    """
    # Use SciPy's Delaunay which is more robust than OpenCV's Subdiv2D
    tri = Delaunay(points)
    # tri.simplices is (M, 3) array of vertex indices
    return [tuple(t) for t in tri.simplices]


def find_point_index(points: np.ndarray, target: np.ndarray, tol: float = 1e-3) -> int | None:
    """Find index of target point in points array.

    Args:
        points: (N, 2) array
        target: (2,) target point
        tol: tolerance for coordinate matching

    Returns:
        Index if found, None otherwise
    """
    diff = np.abs(points - target)
    matches = np.all(diff < tol, axis=1)
    if np.any(matches):
        return int(np.argmax(matches))
    return None


def add_frame_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Add anchor points at image corners and edges for boundary coverage.

    The Delaunay triangulation only covers the convex hull of landmarks.
    By adding frame points (corners + edge midpoints), we extend the
    triangulation to cover the entire image, allowing warping of background.

    Args:
        points: (N, 2) landmark points
        width: image width
        height: image height

    Returns:
        (N+8, 2) array with frame points appended
    """
    frame_points = np.array([
        [0, 0],           # top-left corner
        [width // 2, 0],  # top-mid
        [width - 1, 0],   # top-right
        [0, height // 2], # left-mid
        [width - 1, height // 2],  # right-mid
        [0, height - 1],  # bottom-left
        [width // 2, height - 1],  # bottom-mid
        [width - 1, height - 1],   # bottom-right
    ], dtype=np.float64)

    return np.vstack([points, frame_points])


def get_triangle_bounding_box(
    triangle: np.ndarray,
    width: int,
    height: int
) -> tuple[int, int, int, int]:
    """Get integer bounding box of triangle, clamped to image bounds.

    Args:
        triangle: (3, 2) array of vertices
        width: image width
        height: image height

    Returns:
        (x_min, x_max, y_min, y_max)
    """
    x_min = max(0, int(np.floor(triangle[:, 0].min())))
    x_max = min(width - 1, int(np.ceil(triangle[:, 0].max())))
    y_min = max(0, int(np.floor(triangle[:, 1].min())))
    y_max = min(height - 1, int(np.ceil(triangle[:, 1].max())))

    return x_min, x_max, y_min, y_max
