"""Affine transformation utilities."""

import numpy as np


def compute_affine_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute 2x3 affine matrix mapping src_pts -> dst_pts.

    Both src_pts and dst_pts are 3x2 numpy arrays (three 2D points).
    Returns a 2x3 affine transformation matrix.

    The affine transformation in homogeneous coordinates is:
        [x']   [a b c] [x]
        [y'] = [d e f] [y]
                       [1]

    We solve for the 6 unknowns using 3 point correspondences (6 equations).
    """
    src_pts = src_pts.astype(np.float64)
    dst_pts = dst_pts.astype(np.float64)

    # Build homogeneous coordinate matrices (3x3)
    src_h = np.vstack([src_pts.T, [1, 1, 1]])  # 3x3
    dst_h = np.vstack([dst_pts.T, [1, 1, 1]])  # 3x3

    # Solve: dst_h = T @ src_h  =>  T = dst_h @ src_h^-1
    T = dst_h @ np.linalg.inv(src_h)

    return T[:2, :].astype(np.float32)  # Return 2x3 matrix


def compute_inverse_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute inverse affine transform for backward mapping.

    For inverse warping, we need to map from destination back to source.
    This is equivalent to computing the forward transform with swapped arguments.
    """
    return compute_affine_transform(dst_pts, src_pts)


def invert_affine_matrix(T: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine transformation matrix.

    Returns another 2x3 matrix representing the inverse transform.
    """
    # Convert to 3x3 homogeneous form
    T_h = np.vstack([T, [0, 0, 1]])
    T_inv_h = np.linalg.inv(T_h)
    return T_inv_h[:2, :]


def apply_affine_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply affine transform T to points.

    Args:
        T: 2x3 affine matrix
        points: Nx2 array of points

    Returns:
        Nx2 array of transformed points
    """
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((len(points), 1))
    points_h = np.hstack([points, ones])  # Nx3
    return (T @ points_h.T).T
